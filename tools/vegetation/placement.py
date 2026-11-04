"""Ecological placement of vegetation (numpy only).

Habitat model per species from terrain/water fields:
  altitude band (relative height), slope (no trees on cliffs), moisture (water
  proximity + final-terrain drainage + wet bank band), large-scale groves/clearings,
  open valley meadows kept for gameplay sightlines.
Trees: jittered candidates -> species by relative suitability -> Poisson rejection
with species trunk/crown radii (no interpenetrating trunks, no regular rows).
Logs: lie on near-flat ground, pitched to the terrain, with a trunk-free corridor.
Understory uses a canopy-shade map splatted from the placed trees.
"""
from dataclasses import asdict, dataclass
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "terrain"), os.path.join(HERE, "..", "water")]
from noise import Fractal  # noqa: E402
from terrain_core import gaussian_blur, smoothstep  # noqa: E402
from water_core import bilinear  # noqa: E402


@dataclass
class VegConfig:
    seed: int = 42
    tree_density: float = 1.0          # global multiplier
    pine_per_ha: float = 700.0
    oak_per_ha: float = 280.0
    birch_per_ha: float = 170.0
    dead_per_ha: float = 30.0
    logs_per_ha: float = 6.0
    bush_per_ha: float = 900.0
    fern_per_m2: float = 0.35
    grass_spacing_m: float = 0.55      # jitter grid for grass candidates (render budget)
    flower_fraction: float = 0.025
    max_tree_slope_deg: float = 42.0   # no trees above this (cliffs / scree)
    meadow_strength: float = 1.0       # how open the valley floor is kept


# Trunk spacing radius (m) and crown radius for canopy (m), per tree species.
TREE_R = {"pine": 2.4, "oak": 4.5, "birch": 2.0, "dead": 2.2}
CROWN_R = {"pine": 2.6, "oak": 5.5, "birch": 2.6, "dead": 0.6}


class Fields:
    """Habitat fields on a working grid (half the mesh resolution is plenty)."""

    def __init__(self, cfg, H, size_m, wet, drainage, water, step=2):
        self.size = size_m
        Hs = H[::step, ::step]
        self.H = Hs
        res = Hs.shape[0]
        self.cell = size_m / (res - 1)
        lin = np.linspace(-size_m / 2, size_m / 2, res)
        self.X, self.Y = np.meshgrid(lin, lin)
        gy, gx = np.gradient(Hs, self.cell)
        self.gx, self.gy = gx, gy
        self.slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        floor = np.percentile(Hs, 2)
        self.rel = np.clip((Hs - floor) / (Hs.max() - floor), 0, 1)
        w = water[::step, ::step].astype(float)
        self.water = w > 0.5
        c = self.cell
        self.near_water = gaussian_blur(w, max(int(1.5 / c), 1)) > 0.02          # within ~1.5 m
        prox6 = np.clip(gaussian_blur(w, int(6 / c)) * 4, 0, 1)
        prox20 = np.clip(gaussian_blur(w, int(20 / c)) * 3, 0, 1)
        dr = drainage[::step, ::step]
        self.moisture = np.clip(0.5 * prox6 + 0.6 * prox20 + 0.6 * dr ** 2 + 0.3 * wet[::step, ::step], 0, 1)
        fr = Fractal(cfg.seed + 700)
        groves = fr.fbm(self.X, self.Y, 1.0 / 110.0, 3)
        self.forest = smoothstep(-0.45, 0.2, groves)                              # clearings where low
        meadow_n = fr.fbm(self.X + 300, self.Y, 1.0 / 70.0, 3)
        valley = (1 - smoothstep(0.10, 0.22, self.rel)) * (1 - smoothstep(10, 20, self.slope))
        # Open meadows over ~half the valley floor (sightlines / combat space), groves between.
        self.meadow = 1 - cfg.meadow_strength * valley * smoothstep(-0.35, 0.05, meadow_n)
        self.patch = fr.fbm(self.X, self.Y - 200, 1.0 / 12.0, 3)                   # small-scale patchiness
        self.dry = ~self.near_water

    def sample(self, field, x, y):
        return bilinear(field.astype(float), self.size, x, y)


def tree_densities(cfg, F):
    """Trees per m^2 for each species on the field grid."""
    so = 1 - smoothstep(cfg.max_tree_slope_deg - 8, cfg.max_tree_slope_deg - 1, F.slope)
    base = so * F.forest * F.meadow * F.dry * cfg.tree_density
    rel, m, sl = F.rel, F.moisture, F.slope
    d = {
        # Treeline: thins out toward the summits (no single-file rows along ridge crests).
        "pine": cfg.pine_per_ha / 1e4 * smoothstep(0.06, 0.28, rel) * (1 - smoothstep(0.72, 0.9, rel)) * (1 - 0.4 * m),
        "oak": cfg.oak_per_ha / 1e4 * (1 - smoothstep(0.14, 0.32, rel)) * (1 - 0.6 * m) * (1 - smoothstep(16, 26, sl)),
        "birch": cfg.birch_per_ha / 1e4 * (0.2 + m) * (1 - smoothstep(0.3, 0.55, rel)),
        "dead": cfg.dead_per_ha / 1e4 * (0.3 + smoothstep(0.45, 0.8, rel) + smoothstep(26, 34, sl)) * (1 - 0.8 * smoothstep(0.85, 0.97, rel)),
    }
    return {k: v * base for k, v in d.items()}


class SpatialHash:
    def __init__(self, cell):
        self.cell = cell
        self.g = {}

    def _k(self, x, y):
        return int(np.floor(x / self.cell)), int(np.floor(y / self.cell))

    def add(self, x, y, r, tag=None):
        self.g.setdefault(self._k(x, y), []).append((x, y, r, tag))

    def near(self, x, y, reach):
        cx, cy = self._k(x, y)
        n = int(np.ceil(reach / self.cell))
        for i in range(cx - n, cx + n + 1):
            for j in range(cy - n, cy + n + 1):
                yield from self.g.get((i, j), ())


def jitter_grid(size_m, spacing, rng):
    n = int(size_m / spacing)
    g = (np.arange(n) + 0.5) * spacing - size_m / 2
    X, Y = np.meshgrid(g, g)
    X = X.ravel() + rng.uniform(-0.5, 0.5, X.size) * spacing
    Y = Y.ravel() + rng.uniform(-0.5, 0.5, Y.size) * spacing
    keep = (np.abs(X) < size_m / 2 - 0.5) & (np.abs(Y) < size_m / 2 - 0.5)
    return X[keep], Y[keep]


def place_trees(cfg, F, rng, variants):
    dens = tree_densities(cfg, F)
    names = list(dens)
    spacing = 2.0
    x, y = jitter_grid(F.size, spacing, rng)
    D = np.stack([F.sample(dens[k], x, y) for k in names], 1)
    tot = D.sum(1)
    acc = rng.random(len(x)) < np.clip(tot * spacing * spacing, 0, 1)
    x, y, D, tot = x[acc], y[acc], D[acc], tot[acc]
    cum = np.cumsum(D / tot[:, None], 1)
    sp_idx = (rng.random(len(x))[:, None] > cum).sum(1)
    # Hard constraints (density is smooth; these are not): no trees on cliffs or in/at water.
    hard = ((F.sample(F.slope, x, y) <= cfg.max_tree_slope_deg)
            & (F.sample(F.near_water.astype(float), x, y) < 0.01))
    x, y, sp_idx = x[hard], y[hard], sp_idx[hard]
    order = rng.permutation(len(x))
    hsh = SpatialHash(5.0)
    out = {k: [] for k in names}
    maxr = max(TREE_R.values())
    for i in order:
        sp = names[sp_idx[i]]
        r = TREE_R[sp]
        ok = True
        for (qx, qy, qr, _) in hsh.near(x[i], y[i], maxr):
            if (qx - x[i]) ** 2 + (qy - y[i]) ** 2 < (0.5 * (r + qr)) ** 2:
                ok = False
                break
        if ok:
            hsh.add(x[i], y[i], r, sp)
            out[sp].append((x[i], y[i]))
    trees = {}
    for sp, pts in out.items():
        if not pts:
            continue
        p = np.array(pts)
        n = len(p)
        rel = F.sample(F.rel, p[:, 0], p[:, 1])
        sl = F.sample(F.slope, p[:, 0], p[:, 1])
        z = bilinear(F.H, F.size, p[:, 0], p[:, 1])
        scale = rng.uniform(0.8, 1.15, n) * (1 - 0.3 * smoothstep(0.5, 1.0, rel))   # stunted high up
        sink = 0.15 + np.tan(np.radians(sl)) * 0.45 * scale
        rot = np.column_stack([rng.normal(0, 0.03, n), rng.normal(0, 0.03, n), rng.uniform(0, 2 * np.pi, n)])
        trees[sp] = {"pos": np.column_stack([p, z - sink]), "rot": rot, "scale": scale,
                     "variant": rng.integers(0, variants[sp], n)}
    return trees, hsh


def place_logs(cfg, F, rng, trees_hash, lengths):
    dens = tree_densities(cfg, F)
    tree_d = sum(dens.values())
    area = F.size ** 2
    target = int(cfg.logs_per_ha * area / 1e4)
    pos, rot, var, out_len = [], [], [], []
    tries = 0
    while len(pos) < target and tries < target * 60:
        tries += 1
        x, y = rng.uniform(-F.size / 2 + 12, F.size / 2 - 12, 2)
        if F.sample(tree_d, np.array([x]), np.array([y]))[0] < 0.006:
            continue
        v = int(rng.integers(len(lengths)))
        L = lengths[v]
        yaw = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(yaw), np.sin(yaw)])
        ts = np.linspace(-0.5, 0.5, 7) * L
        px, py = x + d[0] * ts, y + d[1] * ts
        if F.sample(F.slope, px, py).max() > 18 or (F.sample(F.dry.astype(float), px, py) < 0.99).any():
            continue
        z = bilinear(F.H, F.size, px, py)
        line = np.linspace(z[0], z[-1], len(z))
        if np.abs(z - line).max() > 0.35:           # would bridge a hollow or bury in a bump
            continue
        clear = True
        for qx_, qy_ in zip(px, py):
            for (qx, qy, qr, tag) in trees_hash.near(qx_, qy_, 2.0):
                if (qx - qx_) ** 2 + (qy - qy_) ** 2 < 1.3 ** 2:
                    clear = False
                    break
            if not clear:
                break
        if not clear:
            continue
        pitch = -np.arctan2(z[-1] - z[0], L)
        pos.append((x, y, (z[0] + z[-1]) / 2 - 0.12))
        rot.append((0.0, pitch, yaw))
        var.append(v)
        out_len.append(L)
        for qx_, qy_ in zip(px, py):
            trees_hash.add(qx_, qy_, 0.8, "log")
    n = len(pos)
    return {"pos": np.array(pos).reshape(n, 3), "rot": np.array(rot).reshape(n, 3),
            "scale": np.ones(n), "variant": np.array(var, int)}


def canopy_map(F, trees):
    """Crown coverage 0..1 (+ per-type litter) splatted on the field grid."""
    res = F.H.shape[0]
    cov = {"needle": np.zeros((res, res)), "leaf": np.zeros((res, res))}
    for sp, t in trees.items():
        kind = "needle" if sp in ("pine", "dead") else "leaf"
        for (x, y, _), s in zip(t["pos"], t["scale"]):
            R = CROWN_R[sp] * s
            c0 = int((x + F.size / 2) / F.cell)
            r0 = int((y + F.size / 2) / F.cell)
            n = int(R / F.cell) + 1
            rs, cs = np.mgrid[max(r0 - n, 0):min(r0 + n + 1, res), max(c0 - n, 0):min(c0 + n + 1, res)]
            d = np.hypot((cs - c0) * F.cell, (rs - r0) * F.cell) / R
            cov[kind][rs, cs] += np.clip(1 - d ** 2, 0, 1)
    for k in cov:
        cov[k] = np.clip(gaussian_blur(cov[k], 2), 0, 1)
    canopy = np.clip(cov["needle"] + cov["leaf"], 0, 1)
    return canopy, cov


def place_ground(cfg, F, rng, canopy, trees_hash, variants):
    """Bushes, ferns, grass and flowers."""
    so = lambda lim: 1 - smoothstep(lim - 6, lim, F.slope)   # noqa: E731
    out = {}
    # Bushes: forest edges and light understory.
    edge = 4 * canopy * (1 - canopy)
    d_bush = cfg.bush_per_ha / 1e4 * (0.75 * edge + 0.25 * canopy) * so(34) * F.dry * (1 - 0.6 * F.rel)
    x, y = jitter_grid(F.size, 1.5, rng)
    acc = rng.random(len(x)) < np.clip(F.sample(d_bush, x, y) * 2.25, 0, 1)
    out["bush"] = _thin(x[acc], y[acc], trees_hash, 1.0, 1.4, rng)
    # Ferns: shaded, moist forest floor.
    d_fern = cfg.fern_per_m2 * canopy * (0.25 + F.moisture) * so(36) * F.dry * (1 - smoothstep(0.55, 0.8, F.rel))
    d_fern *= smoothstep(-0.3, 0.3, F.patch)                       # clumped colonies
    x, y = jitter_grid(F.size, 0.9, rng)
    acc = rng.random(len(x)) < np.clip(F.sample(d_fern, x, y) * 0.81, 0, 1)
    out["fern"] = _thin(x[acc], y[acc], trees_hash, 0.5, None, rng)
    # Grass: open, not too steep, not rocky; patchy. Flowers in meadows.
    clump = smoothstep(-0.55, 0.25, F.patch)                              # bare patches between clumps
    d_grass = (1 - canopy) ** 2 * so(32) * F.dry * (1 - smoothstep(0.55, 0.75, F.rel)) * (0.2 + 0.8 * clump)
    x, y = jitter_grid(F.size, cfg.grass_spacing_m, rng)
    acc = rng.random(len(x)) < np.clip(F.sample(d_grass, x, y), 0, 1)
    x, y = x[acc], y[acc]
    meadow = F.sample((1 - F.meadow) + (1 - canopy) * (F.rel < 0.12), x, y)
    fl = rng.random(len(x)) < cfg.flower_fraction * np.clip(meadow, 0, 1) * 2
    out["grass"] = (x[~fl], y[~fl])
    out["flowers"] = (x[fl], y[fl])
    inst = {}
    for sp, (px, py) in out.items():
        n = len(px)
        z = bilinear(F.H, F.size, px, py)
        gx = F.sample(F.gx, px, py)
        gy = F.sample(F.gy, px, py)
        # Lean ground cover partly into the slope normal.
        rot = np.column_stack([np.arctan(gy) * 0.6, -np.arctan(gx) * 0.6, rng.uniform(0, 2 * np.pi, n)])
        lo, hi = {"bush": (0.7, 1.3), "fern": (0.7, 1.3), "grass": (0.8, 1.7), "flowers": (0.8, 1.3)}[sp]
        inst[sp] = {"pos": np.column_stack([px, py, z - 0.03]), "rot": rot, "scale": rng.uniform(lo, hi, n),
                    "variant": rng.integers(0, variants[sp], n)}
    return inst


def _thin(x, y, trees_hash, trunk_clear, self_min, rng):
    """Keep points clear of trunks/logs and (optionally) of each other."""
    keep_x, keep_y = [], []
    own = {}
    cell = max(self_min or 1.0, 1.0)
    for i in rng.permutation(len(x)):
        ok = True
        for (qx, qy, qr, tag) in trees_hash.near(x[i], y[i], 1.5):
            lim = trunk_clear + (0.3 if tag in ("oak",) else 0.0)
            if (qx - x[i]) ** 2 + (qy - y[i]) ** 2 < lim ** 2:
                ok = False
                break
        if ok and self_min:
            k = (int(x[i] // cell), int(y[i] // cell))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for (ox, oy) in own.get((k[0] + di, k[1] + dj), ()):
                        if (ox - x[i]) ** 2 + (oy - y[i]) ** 2 < self_min ** 2:
                            ok = False
            if ok:
                own.setdefault(k, []).append((x[i], y[i]))
        if ok:
            keep_x.append(x[i])
            keep_y.append(y[i])
    return np.array(keep_x), np.array(keep_y)


def terrain_layers(F, canopy, cov, grass_xy=None):
    """Ground-material masks: leaf litter, needle litter, moss, grass sward."""
    moss = np.clip(F.moisture * canopy * 1.3 + smoothstep(30, 42, F.slope) * F.moisture * 0.9, 0, 1)
    moss *= 1 - smoothstep(0.8, 0.95, F.rel)
    out = {"canopy": canopy, "litter_leaf": cov["leaf"], "litter_needle": cov["needle"], "moss": moss}
    if grass_xy is not None and len(grass_xy[0]):
        res = F.H.shape[0]
        g = np.zeros((res, res))
        c = np.clip(((grass_xy[0] + F.size / 2) / F.cell).astype(int), 0, res - 1)
        r = np.clip(((grass_xy[1] + F.size / 2) / F.cell).astype(int), 0, res - 1)
        np.add.at(g, (r, c), 1.0)
        g = gaussian_blur(g, 2)
        out["grass_cover"] = np.clip(g / max(np.percentile(g[g > 0], 90), 1e-6), 0, 1)
    return out


def config_dict(cfg):
    return asdict(cfg)
