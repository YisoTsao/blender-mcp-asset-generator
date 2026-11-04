"""Environment layout (numpy only): site selection, road / trail routing, terrain grading.

Sites are chosen from terrain logic, not fixed coordinates, so any seed works:
  HQ_South / HQ_North  military outposts at the two valley ends (team bases)
  Bridge               river crossing on a straight, narrow, calm reach (not at rapids)
  Farmhouse            abandoned farm on open, flat valley floor
  Ruins                old stone ruin on a knoll (local prominence = commanding position)
  Lumber_Camp          cabins + sawmill in a forest clearing (cleared trees -> stumps)
  Bunker               concrete pillbox above the bridge with sight of the crossing
  Trench               zig-zag trench line covering the bridge from the far bank
Roads: least-cost paths (Dijkstra, 8-neighbour, slope/water/forest costs) on a 2 m grid,
Chaikin-smoothed; terrain graded to a smoothed longitudinal profile with no cross-slope.
"""
from dataclasses import asdict, dataclass
import heapq
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "terrain"), os.path.join(HERE, "..", "water")]
from terrain_core import gaussian_blur, smoothstep  # noqa: E402
from water_core import bilinear, chaikin, project_to_polyline, resample_polyline, smooth1d  # noqa: E402


@dataclass
class EnvConfig:
    seed: int = 42
    road_half_width_m: float = 2.4
    trail_half_width_m: float = 0.75
    road_max_grade: float = 0.12
    road_blend_m: float = 3.5
    site_min_separation_m: float = 70.0
    hq_radius_m: float = 22.0
    bridge_clearance_m: float = 1.2
    trench_depth_m: float = 1.5
    trench_half_width_m: float = 0.6


class Site:
    def __init__(self, name, x, y, radius, yaw=0.0, **kw):
        self.name, self.x, self.y, self.radius, self.yaw = name, float(x), float(y), float(radius), float(yaw)
        self.z = 0.0
        self.extra = kw

    def as_dict(self):
        return {"name": self.name, "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
                "radius": self.radius, "yaw_deg": round(float(np.degrees(self.yaw)), 1),
                **{k: v for k, v in self.extra.items() if isinstance(v, (int, float, str))}}


class Layout:
    def __init__(self, cfg, H, size_m, water, canopy, river_pts, river_s, river_w, rapids_s):
        self.cfg = cfg
        self.H = H                                    # full-res heights (modified in place by grading)
        self.size = size_m
        self.res = H.shape[0]
        self.cell = size_m / (self.res - 1)
        lin = np.linspace(-size_m / 2, size_m / 2, self.res)
        self.X, self.Y = np.meshgrid(lin, lin)
        self.water = water
        self.canopy = canopy                          # on the same grid as H
        self.river_pts, self.river_s, self.river_w, self.rapids_s = river_pts, river_s, river_w, rapids_s
        self.sites = {}
        self.paths = {}
        self.protect = np.zeros_like(H, bool)
        self._refresh()

    def _refresh(self):
        gy, gx = np.gradient(self.H, self.cell)
        self.slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.water_dist = self._dist(self.water)

    def _dist(self, mask):
        """Approximate distance (m) to mask via repeated dilation on a 2 m grid."""
        step = max(int(round(2.0 / self.cell)), 1)
        m = mask[::step, ::step].copy()
        d = np.where(m, 0.0, np.inf)
        cur = m.copy()
        for i in range(1, 80):
            nb = cur.copy()
            nb[1:] |= cur[:-1]
            nb[:-1] |= cur[1:]
            nb[:, 1:] |= cur[:, :-1]
            nb[:, :-1] |= cur[:, 1:]
            new = nb & ~cur
            d[new] = i * self.cell * step
            cur = nb
            if cur.all():
                break
        d[np.isinf(d)] = 999.0
        up = np.repeat(np.repeat(d, step, 0), step, 1)[:self.res, :self.res]
        return up

    def h(self, x, y):
        return bilinear(self.H, self.size, np.asarray(x, float), np.asarray(y, float))

    # ------------------------------------------------------------------ sites
    def flatness(self, radius_m):
        return gaussian_blur(self.slope, max(int(radius_m / self.cell / 2), 1))

    def pick(self, name, radius, score, region):
        """Highest score inside region, at least min separation from existing sites."""
        sc = np.where(region, score, -np.inf)
        for s in self.sites.values():
            far = np.hypot(self.X - s.x, self.Y - s.y) > max(self.cfg.site_min_separation_m, s.radius + radius + 10)
            sc = np.where(far, sc, -np.inf)
        edge = (np.abs(self.X) < self.size / 2 - radius - 8) & (np.abs(self.Y) < self.size / 2 - radius - 8)
        sc = np.where(edge, sc, -np.inf)
        i = int(np.argmax(sc))
        if not np.isfinite(sc.ravel()[i]):
            raise RuntimeError(f"no valid site for {name}")
        r, c = divmod(i, self.res)
        site = Site(name, self.X[r, c], self.Y[r, c], radius)
        self.sites[name] = site
        return site

    def river_x_at(self, y):
        o = np.argsort(self.river_pts[:, 1])
        return np.interp(y, self.river_pts[o, 1], self.river_pts[o, 0])

    def choose_sites(self, rng):
        cfg = self.cfg
        X, Y = self.X, self.Y
        half = self.size / 2
        valley = 1 - smoothstep(4, 18, self.H - np.percentile(self.H, 3))
        dry = self.water_dist > 12
        rx = self.river_x_at(Y)
        # Team bases at the valley ends: flat, dry, valley floor.
        for name, sel in (("HQ_South", Y < -half * 0.62), ("HQ_North", Y > half * 0.62)):
            f = self.flatness(cfg.hq_radius_m)
            score = -f - 0.04 * np.abs(Y) * 0 + 6 * valley - 0.02 * np.abs(X - rx)
            s = self.pick(name, cfg.hq_radius_m, score, sel & dry & (f < 9) & (self.water_dist > cfg.hq_radius_m + 6))
            s.yaw = np.arctan2(-s.y, -s.x * 0.3)                       # gate faces the map centre
        # Bridge: straight, narrow, calm reach away from rapids and the lake connector.
        pts, ss = self.river_pts, self.river_s
        t = np.gradient(pts[:, :2], axis=0)
        ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        curv = np.abs(np.gradient(smooth1d(ang, 2)))
        far_rapid = np.min(np.abs(ss[:, None] - np.asarray(self.rapids_s)[None]), axis=1) > 45
        mid = (ss > ss[-1] * 0.25) & (ss < ss[-1] * 0.8)
        dists_hq = np.min([np.hypot(pts[:, 0] - s.x, pts[:, 1] - s.y) for s in self.sites.values()], axis=0)
        bank = np.array([self._bank_height(i) for i in range(len(pts))])
        score = -curv * 40 - self.river_w * 0.5 + far_rapid * 10 + mid * 10 + (dists_hq > 120) * 10 - np.abs(bank) * 0.5
        lake_avoid = np.array([self.water_dist[self._rc(p[0], p[1])] for p in pts])   # always 0 on river; keep
        del lake_avoid
        i = int(np.argmax(score))
        ty, tx = t[i, 1], t[i, 0]
        yaw = np.arctan2(ty, tx) + np.pi / 2                              # deck runs across the flow
        b = Site("Bridge", pts[i, 0], pts[i, 1], self.river_w[i] / 2 + 8, yaw, span=float(self.river_w[i] + 7.0),
                 water_z=float(pts[i, 2]), river_index=i)
        self.sites["Bridge"] = b
        # Farmhouse: open flat meadow, a little away from the river.
        f = self.flatness(14)
        score = -f + 8 * valley * (1 - self.canopy) - 0.01 * np.abs(Y)
        self.pick("Farmhouse", 14, score, dry & (f < 6) & (np.abs(Y) < half * 0.55) & (self.water_dist > 30))
        # Ruins: knoll with prominence over the surroundings, buildable top.
        # Knoll on the valley side: prominent but low enough to be reachable on foot.
        prom = self.H - gaussian_blur(self.H, int(30 / self.cell))
        f = self.flatness(11)
        above = self.H - np.percentile(self.H, 3)
        access = gaussian_blur((self.slope < 22).astype(float), int(20 / self.cell))
        score = prom * 1.5 - f * 0.5 + 6 * access
        self.pick("Ruins", 12, score, dry & (f < 11) & (prom > 0.8) & (above > 5) & (above < 24) & (access > 0.6)
                  & (np.abs(Y) < half * 0.75))
        # Lumber camp: flat spot surrounded by forest.
        f = self.flatness(13)
        ring = gaussian_blur(self.canopy, int(30 / self.cell))
        score = ring * 10 - f * 0.4
        self.pick("Lumber_Camp", 13, score, dry & (f < 9) & (ring > 0.12))
        # Bunker: 25-50 m from the bridge, higher than the deck, sight to the bridge.
        d_b = np.hypot(X - b.x, Y - b.y)
        f = self.flatness(5)
        rise = self.H - b.extra["water_z"]
        score = np.clip(rise, 0, 12) - f * 0.4 - np.abs(d_b - 35) * 0.2
        old = self.cfg.site_min_separation_m
        self.cfg.site_min_separation_m = 25.0
        bk = self.pick("Bunker", 5, score, dry & (d_b > 22) & (d_b < 55) & (f < 14) & (self.water_dist > 8))
        bk.yaw = np.arctan2(b.y - bk.y, b.x - bk.x)                     # embrasure faces the bridge
        # Trench: on the far bank from the bunker, perpendicular to the bridge approach.
        side_bunker = np.sign((bk.x - b.x) * np.cos(b.yaw) + (bk.y - b.y) * np.sin(b.yaw))
        d = -side_bunker * np.array([np.cos(b.yaw), np.sin(b.yaw)])
        cx, cy = np.array([b.x, b.y]) + d * (b.extra["span"] / 2 + 24)
        tr = Site("Trench", cx, cy, 16, b.yaw + np.pi / 2)
        self.sites["Trench"] = tr
        self.cfg.site_min_separation_m = old
        for s in self.sites.values():
            s.z = float(self.h(s.x, s.y))
        return self.sites

    def _rc(self, x, y):
        c = int(np.clip(round((x + self.size / 2) / self.cell), 0, self.res - 1))
        r = int(np.clip(round((y + self.size / 2) / self.cell), 0, self.res - 1))
        return r, c

    def _bank_height(self, i):
        p = self.river_pts[i]
        t = self.river_pts[min(i + 1, len(self.river_pts) - 1), :2] - self.river_pts[max(i - 1, 0), :2]
        n = np.array([-t[1], t[0]]) / max(np.linalg.norm(t), 1e-6)
        w = self.river_w[i] / 2 + 3
        za = self.h(p[0] + n[0] * w, p[1] + n[1] * w)
        zb = self.h(p[0] - n[0] * w, p[1] - n[1] * w)
        return float(za - zb)

    # ------------------------------------------------------------------ routing
    def route(self, a, b, kind="road", avoid_sites=()):
        """Least-cost path a->b (xy) on a 2 m grid; returns smoothed (n,2) polyline."""
        step = max(int(round(2.0 / self.cell)), 1)
        sl = self.slope[::step, ::step]
        wd = self.water_dist[::step, ::step]
        can = self.canopy[::step, ::step]
        n = sl.shape[0]
        c2 = self.cell * step
        lim = 16.0 if kind == "road" else 24.0
        cost = 1 + (sl / (9.0 if kind == "road" else 13.0)) ** 2 + 400 * (sl > lim) + 2.5 * can
        cost += 1e4 * (wd < 1.0) + 8 * (wd < 4) + 3 * (wd < 9)
        Xs, Ys = self.X[::step, ::step], self.Y[::step, ::step]
        for s in avoid_sites:
            cost += 50 * (np.hypot(Xs - s.x, Ys - s.y) < s.radius * 0.9)
        edge = (np.abs(Xs) > self.size / 2 - 4) | (np.abs(Ys) > self.size / 2 - 4)
        cost[edge] += 40

        def idx(p):
            return (int(np.clip(round((p[1] + self.size / 2) / c2), 0, n - 1)),
                    int(np.clip(round((p[0] + self.size / 2) / c2), 0, n - 1)))

        s0, s1 = idx(a), idx(b)
        dist = np.full((n, n), np.inf)
        prev = np.full((n, n, 2), -1, np.int32)
        dist[s0] = 0
        pq = [(0.0, s0)]
        nbr = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) == s1:
                break
            if d > dist[r, c]:
                continue
            for dr, dc in nbr:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nd = d + 0.5 * (cost[r, c] + cost[rr, cc]) * (1.4142 if dr and dc else 1.0)
                    if nd < dist[rr, cc]:
                        dist[rr, cc] = nd
                        prev[rr, cc] = (r, c)
                        heapq.heappush(pq, (nd, (rr, cc)))
        path = [s1]
        while path[-1] != s0:
            r, c = prev[path[-1]]
            if r < 0:
                raise RuntimeError("route failed")
            path.append((int(r), int(c)))
        path = path[::-1]
        xy = np.array([(c * c2 - self.size / 2, r * c2 - self.size / 2) for r, c in path])
        xy[0], xy[-1] = a, b
        xy = xy[::2] if len(xy) > 6 else xy
        xy = np.vstack([xy[:1], xy[1:-1], xy[-1:]])
        xy = chaikin(xy, 3)
        xy, _ = resample_polyline(xy, 1.0)
        return xy

    # ------------------------------------------------------------------ grading
    def grade_path(self, name, xy, half_w, z_fix=None, blend=None, sink=0.06):
        """Flatten a corridor to a smoothed longitudinal profile. z_fix: {index: z} anchors."""
        blend = self.cfg.road_blend_m if blend is None else blend
        z = self.h(xy[:, 0], xy[:, 1])
        # Longitudinal profile: smoothed terrain, cut / fill bounded (the route itself already
        # avoids steep ground, so grading only removes bumps and cross-slope).
        prof = smooth1d(z, 6)
        lim = 1.2 if half_w > 1 else 0.6
        for _ in range(4):
            prof = smooth1d(np.clip(prof, z - lim, z + lim), 3)
        prof = np.clip(prof, z - lim, z + lim)
        if z_fix:
            for i, zz in z_fix.items():
                w = np.exp(-0.5 * ((np.arange(len(xy)) - i) / 8.0) ** 2)
                prof = prof * (1 - w) + zz * w
        R = half_w + blend + 1
        near = self._near_mask(xy, R)
        rr, cc = np.nonzero(near)
        d, s, lat = project_to_polyline(self.X[rr, cc], self.Y[rr, cc], xy)
        si = np.clip(s, 0, len(prof) - 1)
        pz = np.interp(si, np.arange(len(prof)), prof) - sink
        w = 1 - smoothstep(half_w + 0.3, half_w + blend, d)
        # Keep water and building pads untouched.
        w *= ~self.water[rr, cc] & ~self.protect[rr, cc]
        self.H[rr, cc] = self.H[rr, cc] * (1 - w) + pz * w
        self.paths[name] = {"xy": xy, "z": prof - sink, "half_width": half_w}
        return d, rr, cc

    def _near_mask(self, xy, R):
        m = np.zeros_like(self.H, bool)
        n = int(R / self.cell) + 1
        for x, y in xy[::2]:
            r0, c0 = self._rc(x, y)
            m[max(r0 - n, 0):r0 + n + 1, max(c0 - n, 0):c0 + n + 1] = True
        return m

    def pad(self, x, y, radius, blend=6.0, z=None):
        """Level a building pad (median height) and blend it into the surrounding slope."""
        d = np.hypot(self.X - x, self.Y - y)
        inside = d < radius
        if z is None:
            z = float(np.median(self.H[inside]))
        w = 1 - smoothstep(radius, radius + blend, d)
        w *= ~self.water
        self.H[:] = self.H * (1 - w) + z * w
        self.protect |= d < radius
        return z

    def trench(self, xy, depth, half_w):
        """Cut a vertical-walled trench (walls get timber revetment meshes)."""
        near = self._near_mask(xy, half_w + 2)
        rr, cc = np.nonzero(near)
        d, s, _ = project_to_polyline(self.X[rr, cc], self.Y[rr, cc], xy)
        floor = np.interp(np.clip(s, 0, len(xy) - 1), np.arange(len(xy)), smooth1d(self.h(xy[:, 0], xy[:, 1]), 3)) - depth
        w = 1 - smoothstep(half_w + 0.05, half_w + 1.0, d)     # > 2 cells: no aliasing spikes
        w *= ~self.protect[rr, cc]
        self.H[rr, cc] = np.minimum(self.H[rr, cc], self.H[rr, cc] * (1 - w) + floor * w)
        # Spoil berm (parapet) on the enemy side is built from sandbags; slight earth mound both sides.
        berm = np.exp(-0.5 * ((d - half_w - 1.2) / 0.6) ** 2) * 0.35 * (d > half_w + 0.3)
        self.H[rr, cc] += berm
        return floor

    def finish(self):
        self._refresh()


def config_dict(cfg):
    return asdict(cfg)
