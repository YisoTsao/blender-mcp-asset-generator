"""Forest Valley water generator (numpy only).

Consumes a Phase 1 heightfield + river centreline and produces:
  * a river design along a smoothed spline: width, depth, water-surface profile
    (monotonic downstream, with pool / riffle / rapid sequences), flow speed
  * a carved terrain: asymmetric river bed (thalweg shifted to the outer bank on
    bends), graded banks, and a natural lake basin with an irregular shoreline
  * per-cell maps: water mask, water depth, flow vector, foam, wetness

Coordinates: x east, y north, metres. Arrays H[row, col], row 0 = south.
Visual water (meshes + shader + flow map) is separated from any physical
simulation, as WORLD_DESIGN section 5 requires.
"""
from dataclasses import asdict, dataclass, field
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "terrain"))
from noise import Fractal  # noqa: E402
from terrain_core import gaussian_blur, smoothstep  # noqa: E402


@dataclass
class WaterConfig:
    seed: int = 42
    spline_spacing_m: float = 2.0
    river_width_m: float = 12.0          # mean wetted width
    width_variation: float = 0.25        # +- fraction from noise
    pool_depth_m: float = 2.2            # deepest pools (outer bends)
    riffle_depth_m: float = 0.45         # shallow crossings between bends
    rapids: int = 2                      # number of rapid reaches
    rapid_length_m: float = 28.0
    rapid_drop_fraction: float = 0.5     # share of total fall concentrated in rapids
    bank_width_m: float = 9.0            # graded bank zone outside the waterline
    freeboard_m: float = 0.6             # min bank height above the water surface
    discharge_m3s: float = 4.5           # sets flow speed via continuity Q = v*A
    lake: bool = True
    lake_radius_m: float = 30.0
    lake_max_depth_m: float = 3.5
    lake_shore_irregularity: float = 0.45
    lake_elongation: float = 1.5         # ellipse axis ratio
    lake_connector: bool = True          # backwater channel joining lake and river
    connector_half_width_m: float = 2.2
    connector_depth_m: float = 0.7
    tributary: bool = True               # mountain stream with a waterfall
    stream_half_width_m: float = 1.6
    stream_discharge_m3s: float = 0.5
    waterfall_height_m: float = 6.0
    gorge_gradient: float = 0.04         # stream gradient below the fall (knickpoint retreat gorge)


@dataclass
class WaterResult:
    config: WaterConfig
    heights: np.ndarray                  # carved terrain (metres)
    spline: dict                         # per-sample river attributes
    lake: dict                           # level, centre, mask etc. (or None)
    maps: dict                           # water_mask, depth, flow_x, flow_y, speed, foam, wet
    channels: list = field(default_factory=list)   # connector / tributary designs
    timings: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers

def chaikin(pts, iterations):
    for _ in range(iterations):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def resample_polyline(pts, spacing):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.arange(0, cum[-1], spacing)
    s = np.append(s, cum[-1]) if cum[-1] - s[-1] > 1e-6 else s
    return np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]), s


def smooth1d(v, sigma_samples):
    r = int(3 * sigma_samples)
    if r < 1:
        return v
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def bilinear(H, size_m, x, y):
    res = H.shape[0]
    fx = np.clip((x + size_m / 2) / size_m * (res - 1), 0, res - 1.0001)
    fy = np.clip((y + size_m / 2) / size_m * (res - 1), 0, res - 1.0001)
    c, r = fx.astype(int), fy.astype(int)
    tx, ty = fx - c, fy - r
    return (H[r, c] * (1 - tx) * (1 - ty) + H[r, c + 1] * tx * (1 - ty)
            + H[r + 1, c] * (1 - tx) * ty + H[r + 1, c + 1] * tx * ty)


def project_to_polyline(px, py, pts, chunk=32768):
    """Distance, arc length s, and signed lateral offset (left of flow = +)."""
    a = pts[:-1]
    ab = pts[1:] - a
    L = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    ab2 = np.maximum(L ** 2, 1e-9)
    fx, fy = px.ravel(), py.ravel()
    n = fx.size
    d = np.empty(n)
    s = np.empty(n)
    lat = np.empty(n)
    for i in range(0, n, chunk):
        qx = fx[i:i + chunk, None]
        qy = fy[i:i + chunk, None]
        t = np.clip(((qx - a[:, 0]) * ab[:, 0] + (qy - a[:, 1]) * ab[:, 1]) / ab2, 0, 1)
        dx = qx - (a[:, 0] + t * ab[:, 0])
        dy = qy - (a[:, 1] + t * ab[:, 1])
        d2 = dx * dx + dy * dy
        k = np.argmin(d2, axis=1)
        rows = np.arange(k.size)
        dd = np.sqrt(d2[rows, k])
        cross = ab[k, 0] * dy[rows, k] - ab[k, 1] * dx[rows, k]
        d[i:i + chunk] = dd
        s[i:i + chunk] = cum[k] + t[rows, k] * L[k]
        lat[i:i + chunk] = np.sign(cross) * dd
    return d.reshape(px.shape), s.reshape(px.shape), lat.reshape(px.shape)


# --------------------------------------------------------------------------- river design

def design_river(cfg, H, size_m, centerline_xy, frac, rng):
    half = size_m / 2.0
    pts = np.asarray(centerline_xy, float)
    pts = chaikin(pts, 3)
    # Clip to the map (keep 1 m past the edge so the water reaches the border).
    keep = np.abs(pts[:, 1]) <= half + 1.0
    pts = pts[keep]
    pts, s = resample_polyline(pts, cfg.spline_spacing_m)
    L = s[-1]
    n = len(s)
    sig = 4.0 / cfg.spline_spacing_m

    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])          # left of flow
    heading = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
    kappa = smooth1d(np.gradient(heading, s), 5.0 / cfg.spline_spacing_m)   # +: turning left
    kref = np.percentile(np.abs(kappa), 90) + 1e-6
    bend = np.clip(kappa / kref, -1, 1)                          # -1..1 signed bend strength

    # Width: noise + wider on bends (point bars).
    wn = frac.fbm(s / 55.0, np.full(n, 7.3), 1.0, 3)
    half_w = 0.5 * cfg.river_width_m * (1 + cfg.width_variation * wn) * (1 + 0.18 * np.abs(bend))
    half_w = smooth1d(half_w, sig)

    # Rapids: on the straightest reaches, spaced apart and away from the ends.
    straight = smooth1d(1 - np.abs(bend), 10.0 / cfg.spline_spacing_m)
    cand = np.argsort(-straight)
    centres = []
    for i in cand:
        si = s[i]
        if si < 0.15 * L or si > 0.85 * L:
            continue
        if all(abs(si - c) > L / (cfg.rapids + 1) * 0.8 for c in centres):
            centres.append(si)
        if len(centres) >= cfg.rapids:
            break
    centres.sort()
    rapid = np.zeros(n)
    for c in centres:
        rapid = np.maximum(rapid, 1 - smoothstep(0, cfg.rapid_length_m * 0.6, np.abs(s - c)))

    # Depth: pools on bends, riffles at crossings, shallow and fast in rapids.
    pool = np.abs(bend) ** 0.7
    depth = cfg.riffle_depth_m + (cfg.pool_depth_m - cfg.riffle_depth_m) * pool
    depth *= 1 + 0.15 * frac.fbm(s / 25.0, np.full(n, 1.7), 1.0, 2)
    depth = depth * (1 - 0.55 * rapid)
    depth = smooth1d(depth, sig)

    # Water surface: bank-limited, monotonic, fall concentrated at rapids.
    off = half_w + cfg.bank_width_m * 0.7
    bl = bilinear(H, size_m, pts[:, 0] + normal[:, 0] * off, pts[:, 1] + normal[:, 1] * off)
    br = bilinear(H, size_m, pts[:, 0] - normal[:, 0] * off, pts[:, 1] - normal[:, 1] * off)
    bank = smooth1d(np.minimum(bl, br), 15.0 / cfg.spline_spacing_m)
    z0 = bank[: max(3, n // 20)].min() - cfg.freeboard_m
    z1 = bank[-max(3, n // 20):].min() - cfg.freeboard_m
    z1 = min(z1, z0 - 0.002 * L)                                  # guarantee downstream fall
    fall = z0 - z1
    steps = np.zeros(n)
    for c in centres:
        steps += smoothstep(c - cfg.rapid_length_m / 2, c + cfg.rapid_length_m / 2, s)
    steps /= max(len(centres), 1)
    f = cfg.rapid_drop_fraction if centres else 0.0
    surf = z0 - fall * ((1 - f) * s / L + f * steps)
    surf = np.minimum(surf, bank - cfg.freeboard_m)
    surf = np.minimum.accumulate(surf)                           # never flows uphill

    # Flow speed from continuity, cross-section ~ 0.62 * width * depth (parabolic).
    area = 0.62 * 2 * half_w * depth
    speed = cfg.discharge_m3s / np.maximum(area, 0.5)
    slope = -np.gradient(surf, s)
    foam = np.clip(rapid * 1.2 + np.clip((slope - 0.01) * 40, 0, 1), 0, 1)

    return {
        "xy": pts, "s": s, "length_m": float(L), "tangent": tang, "normal": normal,
        "curvature": kappa, "bend": bend, "half_width": half_w, "depth": depth,
        "surface_z": surf, "bed_z": surf - depth, "speed": speed, "rapid": rapid,
        "foam": foam, "rapid_centres_s": centres, "bank_z": bank,
    }


# --------------------------------------------------------------------------- carving

def carve_river(cfg, H, X, Y, sp, frac):
    d, s, lat = project_to_polyline(X, Y, sp["xy"])
    S = sp["s"]
    hw = np.interp(s, S, sp["half_width"])
    dep = np.interp(s, S, sp["depth"])
    surf = np.interp(s, S, sp["surface_z"])
    bend = np.interp(s, S, sp["bend"])
    # Thalweg shifts toward the outer bank (left turn -> right bank, lat < 0).
    o = -0.35 * hw * bend
    side_w = np.where(lat >= o, hw - o, hw + o)
    u = np.clip(np.abs(lat - o) / np.maximum(side_w, 0.5), 0, 1.5)
    # Asymmetric section: steep cut bank on the outside, gentle point bar inside.
    outer = np.sign(lat - o) == -np.sign(bend + 1e-9)
    p = np.where(outer, 2.4, 1.6)
    rough = 0.12 * frac.fbm(X, Y, 1.0 / 4.0, 2)
    lip = 0.25
    # Flat-bottomed section with steeper sides; the lip only rises at the very edge.
    bed = (surf - dep * np.clip(1 - u ** p, 0, 1) ** 0.75
           + lip * smoothstep(0.85, 1.0, u) + rough * (u < 1))

    Hn = H.copy()
    inside = u < 1.0
    Hn[inside] = bed[inside]
    # Graded bank: join the terrain, never lower than surface + lip (no leaks).
    t = np.clip((np.abs(lat) - hw) / cfg.bank_width_m, 0, 1)
    bank_zone = (~inside) & (t < 1)
    blend = surf + lip + (H - surf - lip) * smoothstep(0, 1, t)
    # Minimum bank height near the water (no leaks), fading out so it meets the
    # terrain continuously at the outer edge of the bank zone (no steps/levees).
    rim = surf + lip + 0.5 * t
    lift = np.maximum(rim - blend, 0.0) * (1 - smoothstep(0.35, 1.0, t))
    Hn[bank_zone] = (blend + lift)[bank_zone]

    water = inside & (Hn < surf)
    depth_map = np.where(water, surf - Hn, 0.0)
    prof = np.clip(1 - np.clip(u, 0, 1) ** 2, 0, 1) ** 0.5
    spd = np.interp(s, S, sp["speed"]) * (0.35 + 0.65 * prof)
    tx = np.interp(s, S, sp["tangent"][:, 0])
    ty = np.interp(s, S, sp["tangent"][:, 1])
    foam = np.clip(np.interp(s, S, sp["foam"]) + smoothstep(0.75, 1.0, u) * 0.5, 0, 1)
    wet = np.clip(1 - np.clip((np.abs(lat) - hw) / (cfg.bank_width_m * 0.6), 0, 1), 0, 1)
    return Hn, {
        "water": water, "depth": depth_map, "surface": np.where(water, surf, np.nan),
        "flow_x": np.where(water, tx * spd, 0), "flow_y": np.where(water, ty * spd, 0),
        "speed": np.where(water, spd, 0), "foam": np.where(water, foam, 0), "wet": wet,
        "dist": d,
        # Watertightness: dry cells within the bank zone must stand above the surface.
        "leaks": int(((~inside) & (np.abs(lat) < hw + cfg.bank_width_m) & (Hn < surf)).sum()),
    }


def find_lake_site(cfg, H, X, Y, river_dist, size_m, cell):
    half = size_m / 2
    R = cfg.lake_radius_m
    # Local relief over the lake footprint: std of height in a ~1.3R window.
    rad = int(R * 1.3 / cell)
    mu = gaussian_blur(H, rad)
    relief = np.sqrt(np.maximum(gaussian_blur(H * H, rad) - mu * mu, 0))
    gy, gx = np.gradient(mu, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    rel = (H - H.min()) / np.ptp(H)
    edge = np.maximum(np.abs(X), np.abs(Y))
    ok = ((river_dist > R + 25) & (river_dist < R + 160) & (rel < 0.3)
          & (edge < half - R - 40))
    score = np.where(ok, -relief - 0.1 * slope - 0.004 * river_dist, -np.inf)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return float(X[r, c]), float(Y[r, c]), float(slope[r, c])


def lake_shape(cfg, X, Y, cx, cy, frac):
    """Elongated, irregular basin footprint. rn < 1 is inside the design shoreline."""
    R = cfg.lake_radius_m
    dx, dy = X - cx, Y - cy
    # Elongated along a seeded axis (glacial/oxbow lakes are rarely round).
    th = (cfg.seed * 0.618) % 1.0 * np.pi
    e = np.sqrt(cfg.lake_elongation)
    ax = (dx * np.cos(th) + dy * np.sin(th)) / e
    ay = (-dx * np.sin(th) + dy * np.cos(th)) * e
    r = np.hypot(ax, ay)
    ang = np.arctan2(ay, ax)
    shore = 1 + cfg.lake_shore_irregularity * (
        0.6 * frac.fbm(np.cos(ang) * 2.0 + 5, np.sin(ang) * 2.0 - 3, 1.0, 3)
        + 0.4 * frac.fbm(X, Y, 1.0 / 18.0, 2))
    return r, r / (R * shore), shore


def lake_max_level(H, rn):
    """Highest level the rim can hold without spilling."""
    ring = (rn >= 1.0) & (rn <= 1.3)
    return float(H[ring].min()) - 0.35


def carve_lake(cfg, H, X, Y, cx, cy, frac, level=None):
    R = cfg.lake_radius_m
    r, rn, shore = lake_shape(cfg, X, Y, cx, cy, frac)
    lmax = lake_max_level(H, rn)
    level = lmax if level is None else min(level, lmax)
    # Depth from a smoothed inside-distance field (not from rn, whose angular
    # shoreline noise would converge into radial ridges at the centre).
    cell = X[0, 1] - X[0, 0]
    inside = (rn < 1.0).astype(float)
    f = gaussian_blur(inside, max(int(R * 0.35 / cell), 2))
    f = np.clip((f - 0.5) * 2, 0, 1) * inside
    bowl = level + 0.05 - (cfg.lake_max_depth_m + 0.05) * f ** 0.6
    bowl += 0.25 * frac.fbm(X, Y, 1.0 / 9.0, 2) * (rn < 1)
    Hn = np.where(rn < 1.0, np.minimum(H, bowl), H)
    # Shore slope cap (~16-24 deg, noisy) above the waterline, blended out
    # smoothly so there are no carved walls or terraces.
    beyond = np.maximum(r - R * shore, 0.0)
    ang_cap = np.radians(20 + 4 * frac.fbm(X, Y, 1.0 / 15.0, 2))
    cap = level + 0.3 + np.tan(ang_cap) * beyond
    capped = np.minimum(Hn, np.maximum(cap, bowl))
    w = 1 - smoothstep(1.5, 2.6, rn)
    Hn = Hn * (1 - w) + capped * w
    Hn = np.where(rn < 1.0, np.minimum(Hn, bowl), Hn)
    water = (rn < 1.3) & (Hn < level)
    # Keep only the component connected to the centre (no stray puddles).
    water = _connected(water, np.unravel_index(np.argmin(np.where(water, rn, np.inf)), rn.shape))
    depth = np.where(water, level - Hn, 0.0)
    wet = np.clip(1 - (rn - 1.0) / 0.25, 0, 1)
    return Hn, {"level": level, "level_max": lmax, "leaks": _ring_leaks(Hn, water, level),
                "center": (cx, cy), "water": water, "depth": depth, "wet": wet}


def _ring_leaks(Hn, water, level, extra_water=None):
    ring1 = np.zeros_like(water)
    ring1[1:] |= water[:-1]; ring1[:-1] |= water[1:]; ring1[:, 1:] |= water[:, :-1]; ring1[:, :-1] |= water[:, 1:]
    ring1 &= ~water
    if extra_water is not None:
        ring1 &= ~extra_water
    return int((Hn[ring1] < level).sum())


def _connected(mask, seed):
    out = np.zeros_like(mask)
    stack = [seed]
    rows, cols = mask.shape
    while stack:
        r, c = stack.pop()
        if r < 0 or c < 0 or r >= rows or c >= cols or out[r, c] or not mask[r, c]:
            continue
        out[r, c] = True
        stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    return out


# --------------------------------------------------------------------------- channels

def make_spline(pts, spacing, smooth=2):
    pts = chaikin(np.asarray(pts, float), smooth)
    xy, s = resample_polyline(pts, spacing)
    tang = np.gradient(xy, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    return {"xy": xy, "s": s, "length_m": float(s[-1]), "tangent": tang,
            "normal": np.column_stack([-tang[:, 1], tang[:, 0]])}


def carve_channel(H, X, Y, ch, frac, protect):
    """Generic channel carve (connector, stream, gorge). Never fills existing water."""
    cell = X[0, 1] - X[0, 0]
    reach = float((ch["half_width"] + ch["bank_width"]).max()) + 3 * cell
    x0, x1 = ch["xy"][:, 0].min() - reach, ch["xy"][:, 0].max() + reach
    y0, y1 = ch["xy"][:, 1].min() - reach, ch["xy"][:, 1].max() + reach
    cs = np.nonzero((X[0] >= x0) & (X[0] <= x1))[0]
    rs = np.nonzero((Y[:, 0] >= y0) & (Y[:, 0] <= y1))[0]
    sl = (slice(rs[0], rs[-1] + 1), slice(cs[0], cs[-1] + 1))
    Xs, Ys, Hs, Ps = X[sl], Y[sl], H[sl], protect[sl]
    d, s, lat = project_to_polyline(Xs, Ys, ch["xy"])
    S = ch["s"]
    hw = np.interp(s, S, ch["half_width"])
    dep = np.interp(s, S, ch["depth"])
    surf = np.interp(s, S, ch["surface_z"])
    bw = np.interp(s, S, ch["bank_width"])
    u = np.abs(lat) / np.maximum(hw, 0.3)
    lip = 0.2
    rough = 0.08 * frac.fbm(Xs, Ys, 1.0 / 3.0, 2)
    bed = surf - dep * np.clip(1 - u ** 2, 0, 1) ** 0.75 + lip * smoothstep(0.85, 1.0, u) + rough * (u < 1)
    inside = u < 1.0
    Hn = Hs.copy()
    Hn[inside] = np.minimum(Hs, bed)[inside]
    t = np.clip((np.abs(lat) - hw) / np.maximum(bw, 0.5), 0, 1)
    bank = (~inside) & (t < 1) & (~Ps)
    blend = surf + lip + (Hs - surf - lip) * smoothstep(0, 1, t)
    rim = surf + lip + 0.4 * t
    lift = np.maximum(rim - blend, 0.0) * (1 - smoothstep(0.35, 1.0, t))
    # Banks are graded toward the channel (gorge walls) and only lifted near the water.
    Hn[bank] = (blend + lift)[bank]
    water = inside & (Hn < surf)
    prof = np.clip(1 - np.clip(u, 0, 1) ** 2, 0, 1) ** 0.5
    spd = np.interp(s, S, ch["speed"]) * (0.35 + 0.65 * prof)
    out = {k: np.zeros_like(H) for k in ("depth", "flow_x", "flow_y", "speed", "foam", "wet")}
    out["water"] = np.zeros(H.shape, bool)
    out["water"][sl] = water
    out["depth"][sl] = np.where(water, surf - Hn, 0)
    out["flow_x"][sl] = np.where(water, np.interp(s, S, ch["tangent"][:, 0]) * spd, 0)
    out["flow_y"][sl] = np.where(water, np.interp(s, S, ch["tangent"][:, 1]) * spd, 0)
    out["speed"][sl] = np.where(water, spd, 0)
    out["foam"][sl] = np.where(water, np.clip(np.interp(s, S, ch["foam"]) + 0.4 * smoothstep(0.75, 1, u), 0, 1), 0)
    out["wet"][sl] = np.clip(1 - np.clip((np.abs(lat) - hw) / 2.5, 0, 1), 0, 1)
    out["leaks"] = int((bank & (t < 0.6) & (Hn < surf)).sum())
    Hfull = H.copy()
    Hfull[sl] = Hn
    return Hfull, out


def plan_connector(cfg, sp, lake_c, level_max, rng):
    """Backwater channel: lake level = river surface at the junction."""
    xy, sz = sp["xy"], sp["surface_z"]
    ok = sz <= level_max - 0.02
    dist = np.linalg.norm(xy - np.asarray(lake_c), axis=1)
    i = int(np.argmin(np.where(ok, dist, np.inf)))
    level = float(sz[i])
    a, b = np.asarray(lake_c), xy[i]
    mid = (a + b) / 2
    perp = np.array([-(b - a)[1], (b - a)[0]])
    ctrl = mid + perp * rng.uniform(-0.25, 0.25)
    t = np.linspace(0, 1, 40)[:, None]
    pts = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
    ch = make_spline(pts, 1.0, smooth=1)
    n = len(ch["s"])
    wn = 1 + 0.2 * np.sin(ch["s"] / 7.0 + rng.uniform(0, 6))
    ch.update({
        "name": "Lake_Connector", "kind": "connector", "river_index": i,
        "half_width": cfg.connector_half_width_m * wn, "depth": np.full(n, cfg.connector_depth_m),
        "surface_z": np.full(n, level), "bank_width": np.full(n, 4.0),
        "speed": np.full(n, 0.06), "foam": np.zeros(n),
    })
    return level, ch


def d8_drainage(H, cell, outlet_mask):
    """D8 drainage on a depression-filled surface (priority-flood from outlets).

    Returns receivers (flat index, self for outlets), drainage area (m^2), filled heights.
    """
    import heapq
    rows, cols = H.shape
    n = rows * cols
    rr, cc = np.divmod(np.arange(n), cols)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    nb = np.full((n, 8), -1, np.int64)
    nd = np.array([np.hypot(dr, dc) * cell for dr, dc in offs])
    for k, (dr, dc) in enumerate(offs):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
        nb[ok, k] = r2[ok] * cols + c2[ok]
    nb_list = [tuple(j for j in row if j >= 0) for row in nb.tolist()]
    hl = H.ravel().astype(float).tolist()
    outlets = np.flatnonzero(outlet_mask.ravel())
    closed = bytearray(n)
    heap = [(hl[o], int(o)) for o in outlets]
    for o in outlets:
        closed[o] = 1
    heapq.heapify(heap)
    while heap:
        z, i = heapq.heappop(heap)
        for j in nb_list[i]:
            if not closed[j]:
                closed[j] = 1
                if hl[j] <= z:
                    hl[j] = z + 1e-4
                heapq.heappush(heap, (hl[j], j))
    hf = np.array(hl)
    safe = np.where(nb >= 0, nb, 0)
    drop = (hf[:, None] - np.where(nb >= 0, hf[safe], np.inf)) / nd
    kb = np.argmax(drop, axis=1)
    is_out = outlet_mask.ravel()
    has = (drop[np.arange(n), kb] > 0) & ~is_out
    rec = np.where(has, safe[np.arange(n), kb], np.arange(n))
    order = np.argsort(hf, kind="stable")[::-1].tolist()
    rec_l = rec.tolist()
    a = [cell * cell] * n
    for i in order:
        r = rec_l[i]
        if r != i:
            a[r] += a[i]
    return rec, np.array(a).reshape(rows, cols), hf.reshape(rows, cols)


def trace_tributary(H, X, Y, river_water, avoid_water, size_m, min_area_m2=600.0, step=4):
    """Pick a mountain stream on the *final* terrain's drainage network.

    Drainage is solved on a coarse grid (every `step` cells) with the river and the
    map border as outlets; candidates must drain into the river, stay off the map
    edge and lakes, and the best one maximises fall + steepest reach (waterfall site).
    """
    Hc = H[::step, ::step]
    Xc, Yc = X[::step, ::step], Y[::step, ::step]
    rows, cols = Hc.shape
    cell = Xc[0, 1] - Xc[0, 0]
    half = size_m / 2
    rw = river_water[::step, ::step].copy()
    for dr in range(step):                      # don't miss thin river cells when subsampling
        for dc in range(step):
            sub = river_water[dr::step, dc::step]
            rw[:sub.shape[0], :sub.shape[1]] |= sub[:rows, :cols]
    border = np.zeros_like(rw)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    rec, area, _ = d8_drainage(Hc, cell, rw | border)
    avoid = avoid_water[::step, ::step] & ~rw
    rel = (Hc - Hc.min()) / np.ptp(Hc)
    edge = np.maximum(np.abs(Xc), np.abs(Yc))
    cand = np.flatnonzero(((area >= min_area_m2) & (rel > 0.18) & (rel < 0.8) & (edge < half - 15)).ravel())
    best = None
    rwf, avf, edf, hcf = rw.ravel(), avoid.ravel(), edge.ravel(), Hc.ravel()
    for start in cand:
        path = [int(start)]
        i = int(start)
        ok = True
        while rec[i] != i:
            i = int(rec[i])
            path.append(i)
            if avf[i] or edf[i] > half - 15 or len(path) * cell > 450:
                ok = False
                break
        if not ok or not rwf[i] or len(path) * cell < 120:
            continue
        z = hcf[path]
        w = max(int(10 / cell), 2)
        steep = max(z[j - w] - z[j + w] for j in range(w, len(z) - w)) / (2 * w * cell) if len(z) > 2 * w + 1 else 0
        sc = (z[0] - z[-1]) + 40 * steep
        if best is None or sc > best[0]:
            best = (sc, path)
    if best is None:
        return None
    p = np.array(best[1])
    return np.column_stack([Xc.ravel()[p], Yc.ravel()[p]])


def design_stream(cfg, H, size_m, path_xy, river_surface, frac):
    # The D8 path is a grid staircase: resample and low-pass the coordinates first.
    p, _ = resample_polyline(np.asarray(path_xy, float), 2.0)
    p = np.column_stack([smooth1d(p[:, 0], 3.0), smooth1d(p[:, 1], 3.0)])
    ch = make_spline(p, 1.0, smooth=2)
    xy, s = ch["xy"], ch["s"]
    n = len(s)
    T = smooth1d(bilinear(H, size_m, xy[:, 0], xy[:, 1]), 3.0)
    T = np.minimum.accumulate(T)
    # Waterfall site: the gorge below the lip must meet the natural profile again
    # within `max_gorge` metres (a knickpoint inside the steep reach, never at the
    # mountain foot where a gorge would trench across the flat valley floor).
    Hf, g, max_gorge = cfg.waterfall_height_m, cfg.gorge_gradient, 40.0
    best_i, best_len = None, np.inf
    for i in range(int(15 / 1.0), n - 5):
        lip_i = T[i] - 0.25
        j = np.nonzero((s > s[i]) & (s <= s[i] + max_gorge))[0]
        if j.size == 0:
            continue
        meet = (T[j] - 0.25) <= lip_i - Hf - g * (s[j] - s[i])
        if meet.any():
            glen = s[j[np.argmax(meet)]] - s[i]
            if glen < best_len:
                best_i, best_len = i, glen
    if best_i is None:                                   # not steep enough anywhere: no fall
        best_i = int(n * 0.4)
        Hf = 0.0
    i_f = best_i
    surf = T - 0.25
    lip = surf[i_f]
    down = lip - Hf - g * (s - s[i_f])
    surf[i_f + 1:] = np.minimum(surf[i_f + 1:], down[i_f + 1:])
    # Join the river at its surface; end just inside the river.
    rz = bilinear(np.nan_to_num(river_surface, nan=-1e6), size_m, xy[:, 0], xy[:, 1])
    inside = np.nonzero(rz > -1e5)[0]
    J = int(inside[0]) if inside.size else n - 1
    zj = rz[J] if inside.size else surf[-1]
    k0 = max(J - 10, i_f + 2)
    if J > k0:
        surf[k0:J] = np.minimum(surf[k0:J], np.linspace(surf[k0], zj, J - k0))
    surf[J:] = zj - 0.03
    end = min(J + 3, n)
    surf = np.minimum.accumulate(surf)
    for key in ("xy", "s", "tangent", "normal"):
        ch[key] = ch[key][:end]
    s, T, surf = s[:end], T[:end], surf[:end]
    n = end
    ch["length_m"] = float(s[-1])
    pool = np.where(s > s[i_f], 1 - smoothstep(0, 9.0, s - s[i_f] - 1.0), 0.0)
    hw = cfg.stream_half_width_m * (1 + 0.2 * frac.fbm(s / 15.0, np.full(n, 2.2), 1.0, 2)) * (1 + 0.9 * pool)
    depth = 0.35 + 1.3 * pool
    slope = -np.gradient(surf, s)
    slope[i_f:i_f + 2] = 0.0                                        # the fall itself is the sheet mesh
    speed = cfg.stream_discharge_m3s / np.maximum(0.62 * 2 * hw * depth, 0.2) * (1 + 3 * np.clip(slope, 0, 0.3))
    foam = np.clip((slope - 0.05) * 6, 0, 1) + 1.0 * pool
    bank_w = 2.0 + 0.75 * np.maximum(T - surf, 0)                   # gorge walls ~53 deg
    ch.update({
        "name": "Tributary_Stream", "kind": "stream", "half_width": hw, "depth": depth,
        "surface_z": surf, "bank_width": bank_w, "speed": speed, "foam": np.clip(foam, 0, 1),
        "terrain_z": T, "fall_index": i_f,
        "fall": {"lip_xyz": [float(xy[i_f, 0]), float(xy[i_f, 1]), float(surf[i_f])],
                 "pool_z": float(surf[i_f + 1]), "height_m": float(surf[i_f] - surf[i_f + 1]),
                 "direction": [float(ch["tangent"][i_f, 0]), float(ch["tangent"][i_f, 1])],
                 "width_m": float(2 * hw[i_f]), "s_m": float(s[i_f])},
    })
    return ch


def carve_fall_notch(H, X, Y, st):
    """Vertical lip + undercut plunge basin so the falling sheet clears the rock face."""
    f = st["fall"]
    if f["height_m"] <= 0.5:
        return H, 0
    lip = np.array(f["lip_xyz"][:2])
    d = np.array(f["direction"])
    nrm = np.array([-d[1], d[0]])
    hw = f["width_m"] / 2
    along = (X - lip[0]) * d[0] + (Y - lip[1]) * d[1]
    lat = (X - lip[0]) * nrm[0] + (Y - lip[1]) * nrm[1]
    zone = (along > 0.1) & (along < 3.5) & (np.abs(lat) < hw + 0.4)
    # Pool floor deepens toward the impact point; sides stay as the gorge walls.
    floor = f["pool_z"] - 1.2 * (1 - np.clip(np.abs(lat) / (hw + 0.4), 0, 1) ** 2)
    Hn = H.copy()
    Hn[zone] = np.minimum(H, floor)[zone]
    return Hn, int(zone.sum())


def upsample2_cubic(a):
    """2x-1 upsample with 4-point cubic (Catmull-Rom) midpoints; originals preserved."""
    def up1(v, axis):
        v = np.moveaxis(v, axis, 0)
        p = np.concatenate([v[:1], v, v[-1:]], 0)
        mid = (-p[:-3] + 9 * p[1:-2] + 9 * p[2:-1] - p[3:]) / 16.0
        out = np.empty((2 * v.shape[0] - 1,) + v.shape[1:], v.dtype)
        out[0::2] = v
        out[1::2] = mid
        return np.moveaxis(out, 0, axis)
    return up1(up1(a.astype(np.float64), 0), 1)


# --------------------------------------------------------------------------- main

def generate_water(cfg: WaterConfig, H, size_m, centerline_xy) -> WaterResult:
    t0 = time.perf_counter()
    timings = {}
    rng = np.random.default_rng(cfg.seed + 500)
    frac = Fractal(cfg.seed + 500)
    res = H.shape[0]
    cell = size_m / (res - 1)
    lin = np.linspace(-size_m / 2, size_m / 2, res)
    X, Y = np.meshgrid(lin, lin)

    sp = design_river(cfg, H, size_m, centerline_xy, frac, rng)
    timings["river_design"] = time.perf_counter() - t0
    t = time.perf_counter()
    Hn, rm = carve_river(cfg, H, X, Y, sp, frac)
    timings["river_carve"] = time.perf_counter() - t

    water = rm["water"].copy()
    depth = rm["depth"].copy()
    wet = rm["wet"].copy()
    fx, fy, spd, foam = rm["flow_x"].copy(), rm["flow_y"].copy(), rm["speed"].copy(), rm["foam"].copy()
    checks = {"river_leaks": rm["leaks"]}
    channels = []

    def merge(o):
        nonlocal water, depth, wet, fx, fy, spd, foam
        new = o["water"] & ~water
        water |= o["water"]
        depth = np.where(new, o["depth"], depth)
        fx = np.where(new, o["flow_x"], fx)
        fy = np.where(new, o["flow_y"], fy)
        spd = np.where(new, o["speed"], spd)
        foam = np.where(new, o["foam"], foam)
        wet = np.maximum(wet, o["wet"])

    lake = None
    t = time.perf_counter()
    if cfg.lake:
        cx, cy, site_slope = find_lake_site(cfg, Hn, X, Y, rm["dist"], size_m, cell)
        level = None
        if cfg.lake_connector:
            _, rn, _ = lake_shape(cfg, X, Y, cx, cy, frac)
            level, conn = plan_connector(cfg, sp, (cx, cy), lake_max_level(Hn, rn), rng)
        Hn, lake = carve_lake(cfg, Hn, X, Y, cx, cy, frac, level)
        lake["site_slope_deg"] = site_slope
        water |= lake["water"]
        depth = np.where(lake["water"], lake["depth"], depth)
        wet = np.maximum(wet, lake["wet"])
        if cfg.lake_connector:
            Hn, co = carve_channel(Hn, X, Y, conn, frac, protect=water)
            merge(co)
            conn["leaks"] = co["leaks"]
            channels.append(conn)
        lake["leaks"] = _ring_leaks(Hn, lake["water"], lake["level"], extra_water=water)
        checks["lake_leaks"] = lake["leaks"]
    timings["lake"] = time.perf_counter() - t

    t = time.perf_counter()
    if cfg.tributary:
        step = max(1, int(round(2.0 / cell)))                     # ~2 m drainage grid
        path = trace_tributary(Hn, X, Y, rm["water"], water, size_m, step=step)
        if path is not None:
            rsurf = np.full(Hn.shape, np.nan)
            rsurf[rm["water"]] = rm["surface"][rm["water"]]
            st = design_stream(cfg, Hn, size_m, path, rsurf, frac)
            Hn, so = carve_channel(Hn, X, Y, st, frac, protect=water)
            Hn, st["notch_cells"] = carve_fall_notch(Hn, X, Y, st)
            merge(so)
            st["leaks"] = so["leaks"]
            channels.append(st)
            checks["stream_leaks"] = so["leaks"]
    timings["tributary"] = time.perf_counter() - t

    wet = np.maximum(wet, water.astype(float))

    # Drainage area of the *final* terrain (moisture input for vegetation).
    t = time.perf_counter()
    step = max(1, int(round(1.0 / cell)))
    Hc = Hn[::step, ::step]
    wc = water[::step, ::step].copy()
    wc[0, :] = wc[-1, :] = wc[:, 0] = wc[:, -1] = True
    _, area_c, _ = d8_drainage(Hc, cell * step, wc)
    la = np.log1p(area_c)
    for _ in range(int(round(np.log2(step)))):
        la = upsample2_cubic(la)
    la = la[:res, :res]
    drainage = np.clip(la / la.max(), 0, 1)
    timings["drainage"] = time.perf_counter() - t

    maps = {"water_mask": water, "depth": depth, "flow_x": fx, "flow_y": fy,
            "speed": spd, "foam": foam, "wet": wet, "river_water": rm["water"],
            "drainage": drainage}
    timings["total"] = time.perf_counter() - t0
    return WaterResult(cfg, Hn.astype(np.float32), sp, lake, maps, channels, timings, checks)


def config_dict(cfg):
    return asdict(cfg)
