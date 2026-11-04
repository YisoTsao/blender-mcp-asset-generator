"""Geologically motivated rock placement (numpy only).

  cliff      bedrock plates on slopes > ~37 deg, long axis along the contour, aligned
             to the slope plane and sunk so no edge floats (exposed bedrock / cliff band)
  mountain   upright angular outcrops on convex ridges, high and steep-ish ground
  boulder    talus: below cliffs where the slope relaxes (< 30 deg, stable); larger
             blocks run out further from the cliff foot. Sparse erratics in forest
  weathered  half-buried, rounded, stratified rocks on forested mid slopes
  moss       mossy boulders in shaded, moist forest
  river      water-worn rocks: in the bed where flow is fast (riffles / rapids), along
             the banks, densely in the tributary gorge and plunge pool; long axis
             along the flow, imbricated (tilted upstream)
  pebble     gravel on wet shores and shallow bed
Free-standing rocks only on stable slopes; every rock is sunk into the ground
(fraction per type) and checked against floating edges. Trunks are never inside rocks.
"""
from dataclasses import asdict, dataclass
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "vegetation"), os.path.join(HERE, "..", "terrain")]
from placement import SpatialHash, jitter_grid  # noqa: E402
from terrain_core import gaussian_blur, smoothstep  # noqa: E402


@dataclass
class RockConfig:
    seed: int = 42
    density: float = 1.0
    cliff_per_ha: float = 180.0          # on cliff ground only
    mountain_per_ha: float = 90.0
    talus_per_ha: float = 600.0
    erratic_per_ha: float = 8.0
    weathered_per_ha: float = 40.0
    moss_per_ha: float = 70.0
    river_bed_per_m2: float = 0.08       # x local flow speed factor
    bank_per_m2: float = 0.05
    stream_per_m2: float = 0.22
    pebble_per_m2: float = 0.9
    max_free_rock_slope_deg: float = 30.0


def euler_xyz(M):
    """Rotation matrices (n,3,3) -> Blender XYZ Euler (R = Rz Ry Rx)."""
    y = np.arcsin(np.clip(-M[:, 2, 0], -1, 1))
    x = np.arctan2(M[:, 2, 1], M[:, 2, 2])
    z = np.arctan2(M[:, 1, 0], M[:, 0, 0])
    return np.column_stack([x, y, z])


def frames(z_axis, x_hint):
    z = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)
    x = x_hint - z * (x_hint * z).sum(1, keepdims=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=2)            # columns = local axes in world


def rot_about(M, axis_col, ang):
    """Rotate frames M about their own local axis (0,1,2) by ang."""
    c, s = np.cos(ang), np.sin(ang)
    R = np.zeros((len(ang), 3, 3))
    i, j = [a for a in (0, 1, 2) if a != axis_col]
    R[:, axis_col, axis_col] = 1
    R[:, i, i], R[:, i, j], R[:, j, i], R[:, j, j] = c, -s, s, c
    return M @ R


class RockField:
    def __init__(self, F, canopy, depth, underwater, river_pts, river_vel, stream_pts):
        self.F = F
        c = F.cell
        self.canopy = canopy
        self.depth = depth
        self.under = underwater
        steep = (F.slope > 40).astype(float)
        self.cliffness = gaussian_blur(steep, int(7 / c))
        hb = gaussian_blur(F.H, int(6 / c))
        lap = (np.roll(hb, 1, 0) + np.roll(hb, -1, 0) + np.roll(hb, 1, 1) + np.roll(hb, -1, 1) - 4 * hb) / c ** 2
        self.convex = np.clip(-lap * 8, 0, 1)
        self.river_pts, self.river_vel = river_pts, river_vel
        self.stream_pts = stream_pts
        self.normal = np.stack([-F.gx, -F.gy, np.ones_like(F.gx)], -1)
        self.normal /= np.linalg.norm(self.normal, axis=-1, keepdims=True)

    def s(self, field, x, y):
        return self.F.sample(field, x, y)

    def normal_at(self, x, y):
        n = np.column_stack([self.s(self.normal[..., i], x, y) for i in range(3)])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def h(self, x, y):
        return self.F.sample(self.F.H, x, y)


def nearest_on(pts, x, y):
    d = (x[:, None] - pts[None, :, 0]) ** 2 + (y[:, None] - pts[None, :, 1]) ** 2
    i = d.argmin(1)
    return i, np.sqrt(d[np.arange(len(x)), i])


def candidates(F, density, spacing, rng, mult=1.0):
    x, y = jitter_grid(F.size, spacing, rng)
    keep = (np.abs(x) < F.size / 2 - 3) & (np.abs(y) < F.size / 2 - 3)
    x, y = x[keep], y[keep]
    acc = rng.random(len(x)) < np.clip(F.sample(density, x, y) * spacing * spacing * mult, 0, 1)
    return x[acc], y[acc]


def place_all(cfg, RF, rng, dims, trunks):
    """dims: kind -> (nvar, 3) asset bounding dims. trunks: (n,2) tree xy.
    Returns kind -> dict(pos, rot, scale, variant, moss, wet) and footprints (x, y, r)."""
    F = RF.F
    out = {}
    rock_hash = SpatialHash(6.0)
    tree_hash = SpatialHash(4.0)
    for x, y in trunks:
        tree_hash.add(x, y, 0.0)
    moist_shade = np.clip(F.moisture * 0.8 + RF.canopy * 0.7 - 0.25, 0, 1)
    D = cfg.density

    def accept(kind, x, y, scale, var, allow_overlap=0.0, trunk_clear=True):
        """Poisson test against previous rocks and trunks; returns kept indices."""
        keep = []
        for i in rng.permutation(len(x)):
            r = 0.5 * max(dims[kind][var[i]][:2]) * scale[i]
            ok = True
            if trunk_clear:
                for (qx, qy, _, _) in tree_hash.near(x[i], y[i], r + 1):
                    if (qx - x[i]) ** 2 + (qy - y[i]) ** 2 < (r * 0.8 + 0.4) ** 2:
                        ok = False
                        break
            if ok:
                for (qx, qy, qr, _) in rock_hash.near(x[i], y[i], r + 6):
                    if (qx - x[i]) ** 2 + (qy - y[i]) ** 2 < ((r + qr) * (1 - allow_overlap)) ** 2:
                        ok = False
                        break
            if ok:
                rock_hash.add(x[i], y[i], r, kind)
                keep.append(i)
        return np.array(sorted(keep), int)

    def finish(kind, x, y, M, scale, var, sink, moss, wet, lift=None):
        n = len(x)
        z = RF.h(x, y)
        if lift is not None:
            z = z + lift
        h = np.array([dims[kind][v][2] for v in var]) * scale
        # Sink along the local up axis by `sink` of the rock height.
        up = M[:, :, 2]
        pos = np.column_stack([x, y, z]) - up * (sink * h)[:, None]
        out[kind] = {"pos": pos, "rot": euler_xyz(M), "scale": scale, "variant": var,
                     "moss": np.clip(moss, 0, 1), "wet": np.clip(wet, 0, 1)}
        return n

    def nvar(kind, n):
        return rng.integers(0, len(dims[kind]), n)

    # ---- cliffs: plates in the slope plane, long axis along the contour
    d = cfg.cliff_per_ha / 1e4 * smoothstep(37, 47, F.slope) * D
    x, y = candidates(F, d, 3.0, rng)
    var = nvar("cliff", len(x))
    scale = rng.uniform(0.55, 1.25, len(x))
    k = accept("cliff", x, y, scale, var, allow_overlap=0.55, trunk_clear=False)
    x, y, var, scale = x[k], y[k], var[k], scale[k]
    n = RF.normal_at(x, y)
    contour = np.column_stack([-n[:, 1], n[:, 0], np.zeros(len(x))])
    contour[np.linalg.norm(contour, axis=1) < 1e-6] = (1, 0, 0)
    # Face tilted steeper than the slope (outward): slabs jut out as ledges / rock walls.
    out_h = np.column_stack([n[:, 0], n[:, 1], np.zeros(len(x))])
    out_h /= np.maximum(np.linalg.norm(out_h, axis=1, keepdims=True), 1e-6)
    M = frames(n + out_h * rng.uniform(0.35, 0.7, len(x))[:, None], contour)
    M = rot_about(M, 2, rng.normal(0, 0.35, len(x)))
    n = M[:, :, 2]
    # Sink: base so ~40% of the plate thickness shows; extra where the slope curves away.
    dm = np.array([dims["cliff"][v] for v in var]) * scale[:, None]
    zc = RF.h(x, y)
    gap = np.zeros(len(x))
    for a, b in ((0.45, 0), (-0.45, 0), (0, 0.45), (0, -0.45), (0.3, 0.3), (-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3)):
        q = np.column_stack([x, y, zc]) + M[:, :, 0] * (a * dm[:, 0])[:, None] + M[:, :, 1] * (b * dm[:, 1])[:, None]
        gap = np.maximum(gap, (q[:, 2] - RF.h(q[:, 0], q[:, 1])) * n[:, 2])
    ok = gap < 0.7 * dm[:, 2]
    x, y, var, scale, M, gap, dm = x[ok], y[ok], var[ok], scale[ok], M[ok], gap[ok], dm[ok]
    sink = 0.5 + gap / dm[:, 2]
    moss = RF.s(moist_shade, x, y) * 0.6 + RF.s(F.moisture, x, y) * 0.25
    finish("cliff", x, y, M, scale, var, sink, moss, np.zeros(len(x)))

    # ---- mountain outcrops: convex ridges, upper slopes
    d = (cfg.mountain_per_ha / 1e4 * smoothstep(0.3, 0.55, F.rel) * smoothstep(18, 30, F.slope)
         * (1 - smoothstep(44, 52, F.slope)) * (0.25 + RF.convex) * D)
    x, y = candidates(F, d, 4.0, rng)
    var = nvar("mountain", len(x))
    scale = rng.uniform(0.6, 1.5, len(x))
    k = accept("mountain", x, y, scale, var, allow_overlap=0.2)
    x, y, var, scale = x[k], y[k], var[k], scale[k]
    n = RF.normal_at(x, y)
    upv = n * 0.4 + np.array([0, 0, 1.0]) * 0.6
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    M = frames(upv, np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(x))]))
    sl = RF.s(F.slope, x, y)
    finish("mountain", x, y, M, scale, var, 0.25 + 0.006 * sl, RF.s(moist_shade, x, y) * 0.5, np.zeros(len(x)))

    # ---- talus boulders below cliffs (stable ground only) + forest erratics
    stable = 1 - smoothstep(cfg.max_free_rock_slope_deg - 6, cfg.max_free_rock_slope_deg, F.slope)
    talus = smoothstep(0.03, 0.2, RF.cliffness) * stable
    d = (cfg.talus_per_ha * talus + cfg.erratic_per_ha * F.forest * (F.rel < 0.5)) / 1e4 * stable * F.dry * D
    x, y = candidates(F, d, 3.0, rng)
    cl = RF.s(RF.cliffness, x, y)
    # Run-out sorting: big blocks travel further from the cliff foot.
    scale = rng.uniform(0.35, 0.8, len(x)) + 0.7 * (1 - np.clip(cl / 0.5, 0, 1)) * rng.random(len(x))
    var = nvar("boulder", len(x))
    k = accept("boulder", x, y, scale, var, allow_overlap=0.15)
    x, y, var, scale = x[k], y[k], var[k], scale[k]
    sl = RF.s(F.slope, x, y)
    ok = sl <= cfg.max_free_rock_slope_deg
    x, y, var, scale = x[ok], y[ok], var[ok], scale[ok]
    n = RF.normal_at(x, y)
    upv = n * 0.7 + np.array([0, 0, 1.0]) * 0.3
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    M = frames(upv, np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(x))]))
    finish("boulder", x, y, M, scale, var, rng.uniform(0.18, 0.32, len(x)), RF.s(moist_shade, x, y), np.zeros(len(x)))

    # ---- weathered, half-buried rocks on forested mid slopes
    d = (cfg.weathered_per_ha / 1e4 * F.forest * smoothstep(0.08, 0.2, F.rel) * (1 - smoothstep(0.55, 0.7, F.rel))
         * stable * F.dry * D)
    x, y = candidates(F, d, 4.0, rng)
    var = nvar("weathered", len(x))
    scale = rng.uniform(0.6, 1.3, len(x))
    k = accept("weathered", x, y, scale, var)
    x, y, var, scale = x[k], y[k], var[k], scale[k]
    n = RF.normal_at(x, y)
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    M = frames(n * 0.8 + [0, 0, 0.2], np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(x))]))
    finish("weathered", x, y, M, scale, var, rng.uniform(0.35, 0.5, len(x)), RF.s(moist_shade, x, y) * 0.8 + 0.1, np.zeros(len(x)))

    # ---- moss-covered boulders: shaded moist forest floor
    d = cfg.moss_per_ha / 1e4 * smoothstep(0.2, 0.6, RF.canopy) * smoothstep(0.05, 0.35, F.moisture) * stable * F.dry * D
    x, y = candidates(F, d, 3.0, rng)
    var = nvar("boulder", len(x))
    scale = rng.uniform(0.35, 0.85, len(x))
    k = accept("moss", x, y, scale, var, allow_overlap=0.1) if len(x) else np.zeros(0, int)
    x, y, var, scale = x[k], y[k], var[k], scale[k]
    n = RF.normal_at(x, y)
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    M = frames(n * 0.7 + [0, 0, 0.3], np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(x))]))
    finish("moss", x, y, M, scale, var, rng.uniform(0.25, 0.4, len(x)), rng.uniform(0.85, 1.0, len(x)), np.zeros(len(x)))

    # ---- river rocks: fast water, banks, gorge
    under = RF.under > 0.5
    shallow = under & (RF.depth < 0.9)
    vel = np.zeros_like(F.H)
    if len(RF.river_pts):
        i, dist = nearest_on(RF.river_pts, F.X.ravel()[shallow.ravel()], F.Y.ravel()[shallow.ravel()])
        vel.ravel()[np.flatnonzero(shallow)] = RF.river_vel[i]
    bank = F.near_water & ~under
    stream_band = np.zeros_like(F.H, bool)
    if len(RF.stream_pts):
        _, ds = nearest_on(RF.stream_pts, F.X.ravel(), F.Y.ravel())
        stream_band = (ds.reshape(F.H.shape) < 3.0)
    d = (cfg.river_bed_per_m2 * shallow * smoothstep(0.45, 1.3, vel) + cfg.bank_per_m2 * bank
         + cfg.stream_per_m2 * stream_band) * D
    x, y = candidates(F, d, 1.0, rng)
    var = nvar("river", len(x))
    in_stream = RF.s(stream_band.astype(float), x, y) > 0.5
    scale = rng.uniform(0.35, 1.0, len(x)) * np.where(in_stream, 1.25, 1.0)
    k = accept("river", x, y, scale, var, allow_overlap=0.25)
    x, y, var, scale, in_stream = x[k], y[k], var[k], scale[k], in_stream[k]
    # Flow direction: tangent of the nearest river / stream polyline point.
    tang = np.zeros((len(x), 3))
    for pts, sel in ((RF.river_pts, ~in_stream), (RF.stream_pts, in_stream)):
        if len(pts) < 2 or not sel.any():
            continue
        i, _ = nearest_on(pts, x[sel], y[sel])
        t = np.gradient(pts[:, :2], axis=0)[i]
        tang[sel, :2] = t / np.linalg.norm(t, axis=1, keepdims=True)
    tang[np.linalg.norm(tang, axis=1) < 0.5] = (1, 0, 0)
    n = RF.normal_at(x, y)
    M = frames(n * 0.5 + [0, 0, 0.5], tang)
    M = rot_about(M, 2, rng.normal(0, 0.3, len(x)))
    M = rot_about(M, 1, rng.uniform(0.12, 0.3, len(x)))          # imbrication
    wetv = np.clip(RF.s(under.astype(float), x, y) + RF.s(F.near_water.astype(float), x, y) + in_stream, 0, 1)
    finish("river", x, y, M, scale, var, rng.uniform(0.3, 0.45, len(x)), (1 - wetv) * 0.4, wetv * 0.9)

    # ---- pebbles on wet shores / shallow bed (clustered)
    patch = smoothstep(-0.2, 0.35, F.patch)
    d = cfg.pebble_per_m2 * (bank + shallow * (RF.depth < 0.4) * 0.5 + stream_band * 0.6) * patch * D
    x, y = candidates(F, d, 0.6, rng)
    var = nvar("pebble", len(x))
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    n = RF.normal_at(x, y)
    M = frames(n, np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(x))]))
    scale = rng.uniform(0.7, 2.6, len(x))
    finish("pebble", x, y, M, scale, var, np.full(len(x), 0.35), np.zeros(len(x)),
           np.clip(RF.s(F.near_water.astype(float), x, y) + RF.s(under.astype(float), x, y), 0, 1) * 0.8)

    scree = np.clip(talus * 1.4, 0, 1)
    return out, scree


def config_dict(cfg):
    return asdict(cfg)
