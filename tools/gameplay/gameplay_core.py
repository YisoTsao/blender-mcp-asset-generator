"""FPS gameplay analysis (numpy only): walkability / navigation grid, obstacle raster,
line-of-sight exposure, cover detection, spawn / objective / pickup placement,
combat & kill zones, route validation.

Grid: 1 m cells (nav), 4 m cells (visibility analysis).
"""
from dataclasses import asdict, dataclass
import heapq

import numpy as np


@dataclass
class GameConfig:
    seed: int = 42
    nav_cell_m: float = 1.0
    max_walk_slope_deg: float = 40.0        # UE default walkable floor ~44 deg; keep margin
    max_wade_depth_m: float = 0.7
    agent_radius_m: float = 0.4
    team_spawns: int = 8
    ffa_spawns: int = 24
    ffa_spawn_min_sep_m: float = 28.0
    vis_cell_m: float = 4.0
    vantage_points: int = 400
    max_sight_m: float = 250.0
    eye_h_m: float = 1.7
    run_speed_ms: float = 5.0
    kill_zone_exposure: float = 0.6        # fraction of the max vantage count
    kill_zone_max_cover_per_100m2: float = 0.5


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class Grid:
    def __init__(self, H, size_m, cell):
        self.size = size_m
        self.cell = cell
        n = int(round(size_m / cell)) + 1
        self.n = n
        lin = np.linspace(-size_m / 2, size_m / 2, n)
        self.X, self.Y = np.meshgrid(lin, lin)
        res = H.shape[0]
        fx = (self.X + size_m / 2) / size_m * (res - 1)
        fy = (self.Y + size_m / 2) / size_m * (res - 1)
        self.H = _bilinear_idx(H, fx, fy)

    def rc(self, x, y):
        c = np.clip(np.round((np.asarray(x) + self.size / 2) / self.cell).astype(int), 0, self.n - 1)
        r = np.clip(np.round((np.asarray(y) + self.size / 2) / self.cell).astype(int), 0, self.n - 1)
        return r, c

    def sample(self, A, x, y):
        fx = (np.asarray(x, float) + self.size / 2) / self.cell
        fy = (np.asarray(y, float) + self.size / 2) / self.cell
        return _bilinear_idx(A.astype(float), fx, fy)


def _bilinear_idx(A, fx, fy):
    n0, n1 = A.shape
    fx = np.clip(fx, 0, n1 - 1.001)
    fy = np.clip(fy, 0, n0 - 1.001)
    c, r = fx.astype(int), fy.astype(int)
    tx, ty = fx - c, fy - r
    return (A[r, c] * (1 - tx) * (1 - ty) + A[r, c + 1] * tx * (1 - ty)
            + A[r + 1, c] * (1 - tx) * ty + A[r + 1, c + 1] * tx * ty)


# --------------------------------------------------------------------------- obstacles

def raster_obstacles(G, discs, boxes):
    """discs: (x, y, r, h); boxes: (cx, cy, z0, sx, sy, h, yaw). Returns blocked mask and
    obstacle-top height field (for LOS)."""
    blocked = np.zeros((G.n, G.n), bool)
    top = np.zeros((G.n, G.n))
    for x, y, r, h in discs:
        r0, c0 = G.rc(x, y)
        k = int(np.ceil(r / G.cell)) + 1
        rs, cs = np.mgrid[max(r0 - k, 0):min(r0 + k + 1, G.n), max(c0 - k, 0):min(c0 + k + 1, G.n)]
        inside = (G.X[rs, cs] - x) ** 2 + (G.Y[rs, cs] - y) ** 2 <= r * r
        blocked[rs[inside], cs[inside]] = True
        top[rs[inside], cs[inside]] = np.maximum(top[rs[inside], cs[inside]], h)
    for cx, cy, z0, sx, sy, h, yaw in boxes:
        R = max(sx, sy) / 2 + G.cell
        r0, c0 = G.rc(cx, cy)
        k = int(np.ceil(R / G.cell)) + 1
        rs, cs = np.mgrid[max(r0 - k, 0):min(r0 + k + 1, G.n), max(c0 - k, 0):min(c0 + k + 1, G.n)]
        dx, dy = G.X[rs, cs] - cx, G.Y[rs, cs] - cy
        u = dx * np.cos(yaw) + dy * np.sin(yaw)
        v = -dx * np.sin(yaw) + dy * np.cos(yaw)
        inside = (np.abs(u) <= sx / 2 + 0.2) & (np.abs(v) <= sy / 2 + 0.2)
        ground = G.H[rs, cs]
        # Only obstacles that rise into the body (not decks / floors at ground level).
        rel_top = z0 + h - ground
        rel_bot = z0 - ground
        solid = inside & (rel_top > 0.45) & (rel_bot < 1.6)
        blocked[rs[solid], cs[solid]] = True
        top[rs[inside], cs[inside]] = np.maximum(top[rs[inside], cs[inside]], np.clip(rel_top[inside], 0, None))
    return blocked, top


# --------------------------------------------------------------------------- navigation

def label_components(mask):
    """4-connected labelling (iterative flood fill with a stack)."""
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    n0, n1 = mask.shape
    for start in zip(*np.nonzero(mask)):
        if lab[start]:
            continue
        cur += 1
        stack = [start]
        lab[start] = cur
        while stack:
            r, c = stack.pop()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < n0 and 0 <= cc < n1 and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = cur
                    stack.append((rr, cc))
    return lab, cur


def dilate(mask, k):
    m = mask.copy()
    for _ in range(k):
        n = m.copy()
        n[1:] |= m[:-1]
        n[:-1] |= m[1:]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def nav_grid(cfg, G, depth, blocked, walk_override):
    gy, gx = np.gradient(G.H, G.cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    walk = (slope <= cfg.max_walk_slope_deg) & (depth <= cfg.max_wade_depth_m)
    walk |= walk_override                                  # bridge decks
    walk &= ~dilate(blocked, int(round(cfg.agent_radius_m / G.cell)))
    lab, n = label_components(walk)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = lab == sizes.argmax()
    return main, slope, {"walkable_cells": int(walk.sum()), "components": int(n),
                         "main_component_cells": int(main.sum()), "islands_removed_cells": int(walk.sum() - main.sum())}


def shortest_path(nav, cost, a, b):
    """Dijkstra on the nav grid (8-neighbour); a, b = (r, c). Returns list of (r, c) or None."""
    n0, n1 = nav.shape
    dist = np.full(nav.shape, np.inf)
    prev = -np.ones(nav.shape + (2,), np.int32)
    dist[a] = 0
    pq = [(0.0, a)]
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if (r, c) == b:
            break
        if d > dist[r, c]:
            continue
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n0 and 0 <= cc < n1 and nav[rr, cc]:
                nd = d + cost[rr, cc] * (1.4142 if dr and dc else 1.0)
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    prev[rr, cc] = (r, c)
                    heapq.heappush(pq, (nd, (rr, cc)))
    if not np.isfinite(dist[b]):
        return None
    path = [b]
    while path[-1] != a:
        path.append(tuple(prev[path[-1]]))
    return path[::-1]


def nearest_nav(nav, G, x, y, max_r=30):
    r0, c0 = G.rc(x, y)
    for k in range(max_r):
        rs, cs = np.mgrid[max(r0 - k, 0):min(r0 + k + 1, G.n), max(c0 - k, 0):min(c0 + k + 1, G.n)]
        ok = nav[rs, cs]
        if ok.any():
            d = (rs - r0) ** 2 + (cs - c0) ** 2
            d = np.where(ok, d, 1 << 30)
            i = np.unravel_index(d.argmin(), d.shape)
            return int(rs[i]), int(cs[i])
    return None


# --------------------------------------------------------------------------- visibility

def exposure_map(cfg, V, block_h, nav_v, rng, extra_vantage=()):
    """Number of vantage points (on walkable high ground + towers) with line of sight to
    each 4 m cell's standing player. V: Grid at vis resolution, block_h: occluder heights
    above ground (canopy / structures) at vis resolution."""
    cand = np.argwhere(nav_v)
    hgt = V.H[nav_v]
    # Bias vantage points to high, open ground (where shooters would set up).
    w = (hgt - hgt.min() + 2.0) ** 1.5 * (block_h[nav_v] < 2.0)
    w = w / w.sum()
    idx = rng.choice(len(cand), min(cfg.vantage_points, len(cand)), replace=False, p=w)
    vant = [(V.X[r, c], V.Y[r, c], V.H[r, c] + cfg.eye_h_m) for r, c in cand[idx]] + list(extra_vantage)
    occl = V.H + block_h
    exp = np.zeros(V.H.shape)
    tx, ty, tz = V.X.ravel(), V.Y.ravel(), V.H.ravel() + 1.2
    steps = 40
    t = np.linspace(0.04, 0.96, steps)
    for (vx, vy, vz) in vant:
        dx, dy, dz = tx - vx, ty - vy, tz - vz
        dist = np.hypot(dx, dy)
        ok = dist < cfg.max_sight_m
        if not ok.any():
            continue
        px = vx + dx[ok, None] * t
        py = vy + dy[ok, None] * t
        pz = vz + dz[ok, None] * t
        oz = V.sample(occl, px, py)
        visible = (oz < pz - 0.1).all(1)
        e = np.zeros(tx.size, bool)
        e[np.flatnonzero(ok)[visible]] = True
        exp += e.reshape(exp.shape)
    return exp, np.array(vant)


def visible(V, block_h, p, q):
    occl = V.H + block_h
    t = np.linspace(0.03, 0.97, 80)
    px, py = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
    pz = p[2] + (q[2] - p[2]) * t
    return bool((V.sample(occl, px, py) < pz - 0.1).all())


def config_dict(cfg):
    return asdict(cfg)
