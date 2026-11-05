"""Ecological placement of vegetation (numpy only).

Habitat fields from terrain/water: altitude band (relative height), slope (no trees
on cliffs), aspect (sunny / shady slopes), moisture (water proximity + final-terrain
drainage + wet bank band), large-scale groves/clearings, open valley meadows kept
for gameplay sightlines.
Forest stands (zones): each cell is assigned a stand type (riparian, broadleaf,
mixed, pine, spruce, deadwood) by habitat suitability plus low-frequency noise,
so stands have clear but irregular borders; a narrow soft band mixes neighbours.
Each stand has its own tree density and species mix.
Trees: jittered candidates -> species by stand mix -> Poisson rejection
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
    tree_density: float = 1.0          # global multiplier on every stand's density
    zone_noise: float = 0.5            # how much low-frequency noise breaks up stand borders
    mountain_forest: float = 1.0       # strength of the large forests climbing some mountains (0 = off)
    zone_softness: float = 0.02        # width of the mixed band between stands (score units)
    logs_per_ha: float = 6.0
    bush_per_ha: float = 900.0
    fern_per_m2: float = 0.35
    grass_spacing_m: float = 0.55      # jitter grid for grass candidates (render budget)
    flower_fraction: float = 0.025
    max_tree_slope_deg: float = 42.0   # no trees above this (cliffs / scree)
    meadow_strength: float = 1.0       # how open the valley floor is kept


# Trunk spacing radius (m) and crown radius for canopy (m), per tree species.
TREE_R = {"pine": 2.4, "spruce": 2.2, "oak": 4.5, "maple": 3.8, "alder": 2.4, "birch": 2.0, "dead": 2.2}
CROWN_R = {"pine": 2.6, "spruce": 2.4, "oak": 5.5, "maple": 4.6, "alder": 2.8, "birch": 2.6, "dead": 0.6}
TREE_SPECIES = tuple(TREE_R)
NEEDLE_SPECIES = ("pine", "spruce", "dead")    # litter kind under the crown

# Forest stands: id (index + 1 in the zone map; 0 = water), density (trees/ha before the
# grove/meadow/slope mask) and species mix (fractions, sum 1).
ZONES = {
    "riparian": {"label": "河岸林", "per_ha": 520.0, "mix": {"alder": 0.5, "birch": 0.38, "maple": 0.12}},
    "broadleaf": {"label": "闊葉林", "per_ha": 330.0, "mix": {"oak": 0.5, "maple": 0.35, "birch": 0.15}},
    "mixed": {"label": "混交林", "per_ha": 500.0, "mix": {"pine": 0.3, "spruce": 0.15, "oak": 0.2, "maple": 0.2, "birch": 0.15}},
    "pine": {"label": "松林", "per_ha": 600.0, "mix": {"pine": 0.86, "birch": 0.08, "dead": 0.06}},
    "spruce": {"label": "雲杉林", "per_ha": 650.0, "mix": {"spruce": 0.82, "pine": 0.12, "dead": 0.06}},
    "deadwood": {"label": "枯木林", "per_ha": 240.0, "mix": {"dead": 0.7, "birch": 0.22, "pine": 0.08}},
}
ZONE_NAMES = tuple(ZONES)


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
        # Aspect: +1 on south-facing slopes (Blender +Y is north, ground rises northward), -1 north-facing.
        steep = smoothstep(4, 14, self.slope)
        self.sunny = np.clip(gy / np.maximum(np.hypot(gx, gy), 1e-6), -1, 1) * steep
        # Mountain forests: a few mountains (large-scale noise) are wooded far up their slopes,
        # others stay bare rock and scree. 0..1 on the slopes of the chosen mountains.
        mf = smoothstep(0.02, 0.16, fr.fbm(self.X + 2500, self.Y - 1800, 1.0 / 260.0, 2))
        self.mountain = cfg.mountain_forest * mf * smoothstep(0.22, 0.42, self.rel)
        self.zone_w = zone_weights(cfg, self, fr)
        self.zone = np.where(self.water, 0, np.argmax(self.zone_w, 0) + 1).astype(np.uint8)

    def sample(self, field, x, y):
        return bilinear(field.astype(float), self.size, x, y)


def _band(x, centre, half):
    """1 at centre, 0 beyond centre +- half (smooth)."""
    return 1 - smoothstep(0, half, np.abs(x - centre))


def zone_weights(cfg, F, fr):
    """Stand weights (zones x res x res), summing to 1 per cell.

    Suitability per stand from habitat, plus independent low-frequency noise so borders
    wander; a softmax with a small temperature keeps stands mostly pure with a narrow
    mixed band at the border."""
    rel, m, sl, sun = F.rel, F.moisture, F.slope, F.sunny
    score = {
        "riparian": smoothstep(0.15, 0.45, m) * (1 - smoothstep(0.2, 0.36, rel)) * 1.15,
        "broadleaf": (1 - smoothstep(0.16, 0.34, rel)) * (1 - smoothstep(18, 28, sl)) * (1 - 0.5 * m),
        "mixed": _band(rel, 0.3, 0.2) * 0.72,
        "pine": _band(rel, 0.48, 0.28) * (0.75 + 0.35 * sun) * (1 - 0.6 * m),
        "spruce": smoothstep(0.38, 0.62, rel) * (0.75 - 0.35 * sun) * (1 - 0.3 * smoothstep(0.85, 0.97, rel)) + 0.6 * F.mountain,
        # Windthrow / old burn: a few irregular patches on the slopes.
        "deadwood": smoothstep(0.1, 0.35, fr.fbm(F.X - 900, F.Y + 400, 1.0 / 120.0, 3)) * _band(rel, 0.45, 0.35) * 1.3,
    }
    S = []
    for i, name in enumerate(ZONE_NAMES):
        n = fr.fbm(F.X + 1700 * (i + 1), F.Y - 900 * (i + 1), 1.0 / 90.0, 3)
        S.append(score[name] + cfg.zone_noise * n)
    S = np.stack(S)
    W = np.exp((S - S.max(0, keepdims=True)) / max(cfg.zone_softness, 1e-3))
    W /= W.sum(0, keepdims=True)
    return W


def tree_densities(cfg, F):
    """Trees per m^2 for each species on the field grid (sum over stands of weight x density x mix)."""
    lim = cfg.max_tree_slope_deg + 8 * F.mountain           # forested mountains hold trees on steeper ground
    so = 1 - smoothstep(lim - 8, lim - 1, F.slope)
    # Treeline: thins out toward the summits (no single-file rows along ridge crests); higher on forested mountains.
    treeline = 1 - 0.85 * smoothstep(0.82 + 0.1 * F.mountain, 0.97 + 0.03 * F.mountain, F.rel)
    # Forested mountains: continuous cover (no grove clearings) and denser stands.
    forest = np.maximum(F.forest, F.mountain)
    base = so * forest * F.meadow * F.dry * treeline * cfg.tree_density * (1 + 1.6 * F.mountain)
    d = {sp: np.zeros_like(base) for sp in TREE_SPECIES}
    for i, name in enumerate(ZONE_NAMES):
        z = ZONES[name]
        for sp, frac in z["mix"].items():
            d[sp] += F.zone_w[i] * z["per_ha"] / 1e4 * frac
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
    hard = ((F.sample(F.slope, x, y) <= cfg.max_tree_slope_deg + 8 * F.sample(F.mountain, x, y))
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
        # Fallen timber concentrates in windthrow / deadwood stands.
        dead_w = F.sample(F.zone_w[ZONE_NAMES.index("deadwood")], np.array([x]), np.array([y]))[0]
        if rng.random() > 0.3 + 0.7 * dead_w:
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
        kind = "needle" if sp in NEEDLE_SPECIES else "leaf"
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
