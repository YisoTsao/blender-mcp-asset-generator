"""Forest Valley terrain heightfield generator (numpy only).

Pipeline
  1. River valley spine: a meandering polyline crossing the map north -> south.
  2. Mountain ranges: N spines that start near the map border and run inward,
     tapering toward the river so tributary valleys form between them.
  3. Multi-scale noise: domain-warped low-frequency hills, ridged multifractal
     detail masked to the ranges, and fine surface roughness.
  4. Valley carve: a flat-ish valley floor with a downstream gradient and a
     reserved river channel.
  5. Erosion: vectorised droplet hydraulic erosion + thermal (talus) erosion.

Coordinates: x east, y north, metres, map centred on the origin.
Array layout: H[row, col] with row 0 = south edge (y = -size/2).
"""
from dataclasses import dataclass, field, asdict
import heapq
import time

import numpy as np

from noise import Fractal


@dataclass
class TerrainConfig:
    seed: int = 42
    size_m: float = 500.0          # world extent (square)
    resolution: int = 513          # vertices per side (heightmap pixels)
    max_height_m: float = 110.0    # target highest peak above river floor
    mountain_ranges: int = 4       # 3..5 recommended
    detail: float = 1.0            # 0.5 (smooth) .. 1.5 (very rough)
    erosion: float = 1.0           # 0 disables erosion
    valley_floor_half_width_m: float = 32.0
    valley_wall_half_width_m: float = 120.0
    river_channel_half_width_m: float = 7.0
    river_channel_depth_m: float = 2.5
    river_floor_north_m: float = 9.0   # river enters high (north) ...
    river_floor_south_m: float = 3.0   # ... and leaves low (south)
    macro_downsample: int = 4          # fluvial erosion grid = (res-1)/n + 1 -> valley spacing
    fluvial_k: float = 0.04            # stream-power erodibility (higher = deeper valleys)
    fluvial_iterations: int = 60
    hillslope_diffusion: float = 0.12  # rounds crests (temperate, soil-mantled hills)

    def validate(self):
        assert self.resolution >= 65
        assert 1 <= self.mountain_ranges <= 8
        assert self.size_m > 0 and self.max_height_m > 0


@dataclass
class TerrainResult:
    config: TerrainConfig
    heights: np.ndarray            # float32 metres, shape (res, res)
    river_polyline: np.ndarray     # (n, 3) x, y, z of river centre line
    range_spines: list             # list of (n, 2) arrays
    river_mask: np.ndarray         # 0..1, 1 inside the reserved channel
    timings: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- utils

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def distance_to_polyline(px, py, pts, chunk=65536):
    """Return (distance, normalised arc-length param 0..1) to a 2D polyline."""
    a = pts[:-1]
    ab = pts[1:] - a
    seg_len = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    ab2 = np.maximum(seg_len ** 2, 1e-9)
    n = px.size
    dist = np.empty(n)
    param = np.empty(n)
    fx, fy = px.ravel(), py.ravel()
    for s in range(0, n, chunk):
        qx = fx[s:s + chunk, None]
        qy = fy[s:s + chunk, None]
        t = np.clip(((qx - a[:, 0]) * ab[:, 0] + (qy - a[:, 1]) * ab[:, 1]) / ab2, 0.0, 1.0)
        dx = qx - (a[:, 0] + t * ab[:, 0])
        dy = qy - (a[:, 1] + t * ab[:, 1])
        d2 = dx * dx + dy * dy
        k = np.argmin(d2, axis=1)
        rows = np.arange(k.size)
        dist[s:s + chunk] = np.sqrt(d2[rows, k])
        param[s:s + chunk] = (cum[k] + t[rows, k] * seg_len[k]) / total
    return dist.reshape(px.shape), param.reshape(px.shape)


def gaussian_blur(h, radius):
    """Separable gaussian blur with edge clamping (radius in cells)."""
    if radius <= 0:
        return h
    sigma = radius / 2.0
    k = np.arange(-radius, radius + 1)
    w = np.exp(-(k * k) / (2 * sigma * sigma))
    w /= w.sum()
    p = np.pad(h, radius, mode="edge")
    tmp = sum(w[i] * p[:, i:i + h.shape[1]] for i in range(w.size))
    out = sum(w[i] * tmp[i:i + h.shape[0], :] for i in range(w.size))
    return out


# --------------------------------------------------------------------------- layout

def make_river(cfg, rng, frac):
    half = cfg.size_m / 2.0
    ys = np.linspace(half + 20.0, -half - 20.0, 90)
    a1 = rng.uniform(28.0, 42.0)
    a2 = rng.uniform(8.0, 16.0)
    l1 = rng.uniform(280.0, 380.0)
    l2 = rng.uniform(90.0, 140.0)
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    x0 = rng.uniform(-25.0, 25.0)
    xs = (x0 + a1 * np.sin(2 * np.pi * ys / l1 + p1)
          + a2 * np.sin(2 * np.pi * ys / l2 + p2)
          + 10.0 * frac.fbm(ys * 0.0 + 3.3, ys, 1.0 / 120.0, 3))
    return np.column_stack([xs, ys])


def make_ranges(cfg, rng, river):
    """Mountain spines: start near the border, run inward, stop short of the valley."""
    half = cfg.size_m / 2.0
    n = cfg.mountain_ranges
    buffer = cfg.valley_wall_half_width_m * 0.55
    # Alternate sides of the river; spread along y so valleys open between them.
    sides = np.array([(-1) ** i for i in range(n)])
    rng.shuffle(sides)
    spines, peaks = [], []
    for side in (-1, 1):
        idx = np.where(sides == side)[0]
        k = len(idx)
        if k == 0:
            continue
        band = cfg.size_m / k
        for j, i in enumerate(idx):
            yc = -half + band * (j + 0.5) + rng.uniform(-0.18, 0.18) * band
            # Start beyond the outer edge so the range reads as part of a larger massif.
            start = np.array([side * rng.uniform(half * 0.95, half * 1.25),
                              yc + rng.uniform(-60, 60)])
            # Head toward the valley at an oblique angle.
            target_x = np.interp(yc, river[::-1, 1], river[::-1, 0])
            end = np.array([target_x + side * buffer * rng.uniform(1.0, 1.5),
                            yc + rng.uniform(-110, 110)])
            m = 24
            t = np.linspace(0, 1, m)[:, None]
            pts = start + t * (end - start)
            # Bend the spine with a perpendicular sine + jitter.
            d = end - start
            nrm = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
            bend = rng.uniform(-45, 45) * np.sin(np.pi * t) + rng.uniform(-14, 14) * np.sin(
                2 * np.pi * t * rng.uniform(1.5, 2.5) + rng.uniform(0, 6))
            pts = pts + bend * nrm
            spines.append(pts)
            peaks.append(rng.uniform(0.72, 1.0))
    order = np.argsort(rng.uniform(size=len(peaks)))
    peaks = np.array(peaks)
    peaks[order[0]] = 1.0  # guarantee one full-height massif
    return spines, peaks


# --------------------------------------------------------------------------- erosion

def hydraulic_erosion(h, cell, rng, droplets, batch=8192, max_steps=72,
                      inertia=0.08, capacity=6.0, min_capacity=0.01,
                      erode_rate=0.35, deposit_rate=0.25, evaporate=0.02,
                      gravity=9.81, radius=3):
    """Vectorised particle hydraulic erosion (after H. Beyer 2015).

    Thousands of droplets advance in lock-step; erosion/deposition is applied
    with bincount so concurrent droplets sharing a cell accumulate correctly.
    """
    rows, cols = h.shape
    flat = h.ravel()  # view: writes update h
    oy, ox = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    bw = np.maximum(0.0, radius - np.sqrt(ox ** 2 + oy ** 2))
    keep = bw > 0
    ox, oy, bw = ox[keep], oy[keep], bw[keep]
    bw = bw / bw.sum()

    def sample(px, py):
        x0 = px.astype(np.int64)
        y0 = py.astype(np.int64)
        fx = px - x0
        fy = py - y0
        i = y0 * cols + x0
        h00, h10 = flat[i], flat[i + 1]
        h01, h11 = flat[i + cols], flat[i + cols + 1]
        gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
        gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
        hh = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
              + h01 * (1 - fx) * fy + h11 * fx * fy)
        return hh, gx / cell, gy / cell

    margin = radius + 2
    done = 0
    while done < droplets:
        n = min(batch, droplets - done)
        done += n
        px = rng.uniform(margin, cols - margin - 1, n)
        py = rng.uniform(margin, rows - margin - 1, n)
        dx = np.zeros(n)
        dy = np.zeros(n)
        speed = np.ones(n)
        water = np.ones(n)
        sed = np.zeros(n)
        alive = np.ones(n, bool)
        for _ in range(max_steps):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            x, y = px[idx], py[idx]
            hh, gx, gy = sample(x, y)
            ndx = dx[idx] * inertia - gx * (1 - inertia)
            ndy = dy[idx] * inertia - gy * (1 - inertia)
            ln = np.sqrt(ndx * ndx + ndy * ndy)
            flat_dir = ln < 1e-9
            if flat_dir.any():
                a = rng.uniform(0, 2 * np.pi, flat_dir.sum())
                ndx[flat_dir], ndy[flat_dir] = np.cos(a), np.sin(a)
                ln[flat_dir] = 1.0
            ndx /= ln
            ndy /= ln
            nx, ny = x + ndx, y + ndy
            inside = (nx > margin) & (nx < cols - margin - 1) & (ny > margin) & (ny < rows - margin - 1)
            nh = np.where(inside, sample(np.where(inside, nx, x), np.where(inside, ny, y))[0], hh)
            dh = nh - hh
            cap = np.maximum(-dh * speed[idx] * water[idx] * capacity, min_capacity)
            s = sed[idx]
            depositing = (s > cap) | (dh > 0)
            # Deposit (bilinear to the 4 corners of the old position).
            dep = np.where(dh > 0, np.minimum(dh, s), (s - cap) * deposit_rate)
            dep = np.where(depositing, dep, 0.0)
            # Erode (radius brush around the old position), never digging a pit.
            ero = np.where(depositing, 0.0, np.minimum((cap - s) * erode_rate, -dh))
            x0 = x.astype(np.int64)
            y0 = y.astype(np.int64)
            fx, fy = x - x0, y - y0
            base = y0 * cols + x0
            if dep.any():
                tgt = np.concatenate([base, base + 1, base + cols, base + cols + 1])
                amt = np.concatenate([dep * (1 - fx) * (1 - fy), dep * fx * (1 - fy),
                                      dep * (1 - fx) * fy, dep * fx * fy])
                flat += np.bincount(tgt, amt, flat.size)
            if ero.any():
                tgt = (base[:, None] + oy[None, :] * cols + ox[None, :]).ravel()
                amt = (ero[:, None] * bw[None, :]).ravel()
                flat -= np.bincount(tgt, amt, flat.size)
            sed[idx] = s + ero - dep
            speed[idx] = np.sqrt(np.maximum(speed[idx] ** 2 - dh * gravity / cell * 0.1, 0.0))
            water[idx] *= (1 - evaporate)
            px[idx], py[idx] = nx, ny
            dx[idx], dy[idx] = ndx, ndy
            alive[idx] = inside
    return h


def thermal_erosion(h, cell, iterations, talus_deg=36.0, rate=0.22):
    """Move material down any slope steeper than the talus angle (scree formation)."""
    talus = np.tan(np.radians(talus_deg))
    for _ in range(iterations):
        p = np.pad(h, 1, mode="edge")
        delta = np.zeros_like(h)
        for (dy, dx), dist in (((0, 1), 1.0), ((0, -1), 1.0), ((1, 0), 1.0), ((-1, 0), 1.0),
                               ((1, 1), 1.414), ((1, -1), 1.414), ((-1, 1), 1.414), ((-1, -1), 1.414)):
            nb = p[1 + dy:1 + dy + h.shape[0], 1 + dx:1 + dx + h.shape[1]]
            excess = (h - nb) - talus * dist * cell
            move = np.where(excess > 0, excess * rate / 8.0, 0.0)
            delta -= move
            # The neighbour receives what we lose: shift back.
            recv = np.zeros_like(p)
            recv[1 + dy:1 + dy + h.shape[0], 1 + dx:1 + dx + h.shape[1]] = move
            delta += recv[1:-1, 1:-1]
        h = h + delta
    return h


def resample(h, new_res):
    """Bilinear resample of a square grid (corner-aligned)."""
    old = h.shape[0]
    t = np.linspace(0, old - 1, new_res)
    i0 = np.clip(np.floor(t).astype(int), 0, old - 2)
    f = t - i0
    rows = h[i0] * (1 - f)[:, None] + h[i0 + 1] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def stream_power_evolution(h, uplift, cell, outlets, iterations, k=0.01, m=0.5,
                           diffusion=0.08):
    """Landscape evolution: uplift + implicit stream-power incision + hillslope diffusion.

    Braun & Willett (2013) O(n) implicit solver on a D8 drainage tree. Depressions
    are filled each step with an epsilon priority-flood seeded at the outlets, so
    every cell drains to an outlet and the result has a connected dendritic
    valley network instead of noise pits. Returns (heights, drainage_area_m2).
    """
    rows, cols = h.shape
    n = rows * cols
    rr, cc = np.divmod(np.arange(n), cols)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    nb = np.full((n, 8), -1, np.int64)
    nd = np.array([np.hypot(dr, dc) * cell for dr, dc in offs])
    for k8, (dr, dc) in enumerate(offs):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < rows) & (c2 >= 0) & (c2 < cols)
        nb[ok, k8] = r2[ok] * cols + c2[ok]
    nb_list = [tuple(j for j in row if j >= 0) for row in nb.tolist()]
    nb_safe = np.where(nb >= 0, nb, 0)
    is_outlet = np.zeros(n, bool)
    is_outlet[outlets] = True
    U = np.where(is_outlet, 0.0, uplift.ravel())
    hf = h.ravel().astype(np.float64).copy()
    area = np.full(n, cell * cell)
    for _ in range(iterations):
        hf += U
        # Priority-flood depression filling (Barnes et al. 2014, epsilon variant).
        hl = hf.tolist()
        closed = bytearray(n)
        heap = []
        for o in outlets:
            closed[o] = 1
            heap.append((hl[o], o))
        heapq.heapify(heap)
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            z, i = pop(heap)
            for j in nb_list[i]:
                if not closed[j]:
                    closed[j] = 1
                    hj = hl[j]
                    if hj <= z:
                        hj = z + 1e-4
                        hl[j] = hj
                    push(heap, (hj, j))
        hf = np.array(hl)
        # Steepest-descent receivers.
        drop = (hf[:, None] - np.where(nb >= 0, hf[nb_safe], np.inf)) / nd
        kbest = np.argmax(drop, axis=1)
        has_rec = (drop[np.arange(n), kbest] > 0) & ~is_outlet
        rec = np.where(has_rec, nb_safe[np.arange(n), kbest], np.arange(n))
        dist = np.where(has_rec, nd[kbest], 1.0)
        order = np.argsort(hf, kind="stable")
        # Drainage area: accumulate from the top of each tree down.
        rec_l = rec.tolist()
        a_l = [cell * cell] * n
        for i in order[::-1].tolist():
            r = rec_l[i]
            if r != i:
                a_l[r] += a_l[i]
        area = np.array(a_l)
        # Implicit incision, outlets upward.
        F = (k * area ** m / dist).tolist()
        hl = hf.tolist()
        for i in order.tolist():
            r = rec_l[i]
            if r != i:
                f = F[i]
                hl[i] = (hl[i] + f * hl[r]) / (1.0 + f)
        hf = np.array(hl).reshape(rows, cols)
        # Hillslope diffusion rounds knife-edge crests.
        p = np.pad(hf, 1, mode="edge")
        lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25 - hf
        hf = (hf + diffusion * lap).ravel()
        hf[is_outlet] = h.ravel()[is_outlet]
    return hf.reshape(rows, cols), area.reshape(rows, cols)


# --------------------------------------------------------------------------- main

def build_uplift(cfg, frac, MX, MY, river, spines, peaks):
    """Tectonic uplift field (0..~1): ranges + boundary rim + low foothills, zero along the river."""
    half = cfg.size_m / 2.0
    WX, WY = frac.warp(MX, MY, 1.0 / 230.0, 45.0)
    ranges = np.zeros_like(MX)
    for k, (pts, peak) in enumerate(zip(spines, peaks)):
        d, s = distance_to_polyline(WX, WY, pts)
        along = frac.fbm(s * 7.0 + 17.0 * k, np.full_like(s, 3.1 * k), 1.0, 3)
        crest = peak * (0.82 + 0.25 * along) * (1.0 - 0.55 * smoothstep(0.45, 1.0, s))
        width = (135.0 + 40.0 * frac.fbm(MX + 400 * k, MY, 1.0 / 170.0, 2)) * (1.0 - 0.3 * s)
        body = crest * (1.0 - smoothstep(0.0, 1.0, d / width)) ** 0.8
        ranges = np.maximum(ranges, body) + 0.2 * np.minimum(ranges, body)
    hills = frac.fbm(WX, WY, 1.0 / 150.0, 4)
    # Wavy natural boundary instead of a square frame.
    edge = np.maximum(np.abs(WX), np.abs(WY)) / half + 0.08 * frac.fbm(MX, MY, 1.0 / 110.0, 3)
    rim = smoothstep(0.74, 1.05, edge) * (0.45 + 0.2 * hills)
    lowland = 0.10 + 0.22 * np.clip(hills + 0.25, 0.0, None)
    u = np.maximum(np.maximum(ranges, rim), lowland)
    u *= 0.82 + 0.3 * frac.fbm(MX, MY, 1.0 / 70.0, 3)
    rd, _ = distance_to_polyline(MX, MY, river)
    u *= smoothstep(cfg.valley_floor_half_width_m * 0.5, cfg.valley_wall_half_width_m, rd) ** 0.7
    return np.clip(u, 0.0, None), ranges


def generate(cfg: TerrainConfig) -> TerrainResult:
    cfg.validate()
    timings = {}
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    frac = Fractal(cfg.seed)
    frac_detail = Fractal(cfg.seed + 101)
    half = cfg.size_m / 2.0
    res = cfg.resolution
    cell = cfg.size_m / (res - 1)
    D = cfg.detail
    Hmax = cfg.max_height_m

    # ---- layout
    river = make_river(cfg, rng, frac)
    spines, peaks = make_ranges(cfg, rng, river)
    timings["layout"] = time.perf_counter() - t0

    # ---- macro landform: uplift + fluvial erosion on a coarse grid
    t = time.perf_counter()
    mres = (res - 1) // cfg.macro_downsample + 1
    mcell = cfg.size_m / (mres - 1)
    mlin = np.linspace(-half, half, mres)
    MX, MY = np.meshgrid(mlin, mlin)
    uplift, ranges = build_uplift(cfg, frac, MX, MY, river, spines, peaks)
    h0 = Hmax * 0.55 * uplift + 3.0 * frac.fbm(MX, MY, 1.0 / 35.0, 3) + rng.uniform(0, 0.05, MX.shape)
    mrd, _ = distance_to_polyline(MX, MY, river)
    outlets = np.nonzero((mrd.ravel() < cfg.river_channel_half_width_m * 2.0)
                         & (MY.ravel() <= -half + 1e-6))[0]
    h0.ravel()[outlets] = 0.0
    timings["uplift"] = time.perf_counter() - t
    t = time.perf_counter()
    iters = int(round(cfg.fluvial_iterations * max(cfg.erosion, 0.15)))
    macro, area = stream_power_evolution(h0, Hmax * 0.03 * uplift, mcell, outlets, iters,
                                         k=cfg.fluvial_k, diffusion=cfg.hillslope_diffusion)
    macro -= macro.min()
    macro *= Hmax / np.percentile(macro, 99.95)
    timings["fluvial_erosion"] = time.perf_counter() - t

    # ---- upsample + multi-scale detail
    t = time.perf_counter()
    lin = np.linspace(-half, half, res)
    X, Y = np.meshgrid(lin, lin)  # X[row, col] = x(col), Y[row, col] = y(row)
    h = gaussian_blur(resample(macro, res), max(1, cfg.macro_downsample // 2))
    area = resample(area, res)
    rel = np.clip(h / Hmax, 0, 1)
    gy, gx = np.gradient(h, cell)
    slope = np.hypot(gx, gy)
    rocky = smoothstep(0.25, 0.75, rel) * smoothstep(0.3, 0.9, slope)
    WX, WY = frac_detail.warp(X, Y, 1.0 / 90.0, 14.0)
    ridged = frac_detail.ridged(WX, WY, 1.0 / 55.0, int(round(3 + 2 * D)), gain=2.0)
    mid = frac_detail.fbm(WX, WY, 1.0 / 40.0, 3)
    fine = frac.fbm(X, Y, 1.0 / 9.0, int(round(1 + 2 * D)))
    sloped = smoothstep(0.12, 0.6, slope)
    rills = frac.ridged(WX * 1.0, WY * 1.0, 1.0 / 28.0, 3, gain=1.8)
    h = (h
         + D * Hmax * 0.035 * (ridged - 0.45) * rocky
         + D * 2.2 * (rills - 0.5) * sloped * (0.5 + rel)   # rills/spurs on all slopes
         + D * 1.6 * mid * (0.4 + rel)
         + D * 0.35 * fine)
    timings["detail"] = time.perf_counter() - t

    # ---- valley floor: flatten toward a downstream-sloping floor (combat space + river basin)
    t = time.perf_counter()
    rd, rs = distance_to_polyline(X, Y, river)
    rd_w = rd + 8.0 * frac.fbm(X, Y, 1.0 / 60.0, 3)
    # Floor gradient from northing (smooth), not arc length (jumps between meanders).
    ns = np.clip((half - Y) / cfg.size_m, 0.0, 1.0)
    floor = cfg.river_floor_north_m + (cfg.river_floor_south_m - cfg.river_floor_north_m) * ns
    floor = floor + 1.0 * mid * smoothstep(cfg.river_channel_half_width_m * 1.5,
                                            cfg.valley_floor_half_width_m, rd_w)
    wall = smoothstep(cfg.valley_floor_half_width_m * 0.6, cfg.valley_wall_half_width_m, rd_w)
    h = floor + np.maximum(h - floor, 0.0) * wall ** 0.9
    # Micro-relief so the valley floor is not a carpet: swales, hummocks, terraces.
    swale = frac_detail.fbm(WX, WY, 1.0 / 32.0, 4)
    knoll = frac_detail.fbm(X, Y, 1.0 / 14.0, 2)
    hummock = frac.fbm(X, Y, 1.0 / 6.0, 2)
    away = smoothstep(cfg.river_channel_half_width_m, cfg.river_channel_half_width_m * 3.0, rd_w)
    h += D * (1.6 * swale + 0.6 * knoll + 0.3 * hummock) * away * (1.0 - 0.5 * wall)
    timings["valley"] = time.perf_counter() - t

    t = time.perf_counter()
    if cfg.erosion > 0:
        h = thermal_erosion(h, cell, int(25 * cfg.erosion))
    timings["thermal_erosion"] = time.perf_counter() - t

    # ---- reserve the river channel
    river_mask = 1.0 - smoothstep(cfg.river_channel_half_width_m * 0.6,
                                  cfg.river_channel_half_width_m * 1.6, rd_w)
    bed = floor - cfg.river_channel_depth_m
    h = h * (1 - river_mask) + np.minimum(h, bed) * river_mask
    shift = h.min()
    h = h - shift

    zs = np.interp(np.linspace(0, 1, len(river)), [0, 1],
                   [cfg.river_floor_north_m, cfg.river_floor_south_m]) - shift
    river3 = np.column_stack([river, zs])
    timings["total"] = time.perf_counter() - t0
    res_ = TerrainResult(cfg, h.astype(np.float32), river3, spines, river_mask.astype(np.float32), timings)
    res_.drainage_area = area.astype(np.float32)
    return res_


def config_dict(cfg):
    return asdict(cfg)
