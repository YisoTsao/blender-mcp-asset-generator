"""Camera placement that guarantees an unobstructed view of a target (numpy only).

Candidates around the target (headings x distances) are scored by:
  terrain occlusion along the sight line, trunks / rocks near the line, canopy at the
  camera (inside a crown), and clearance around the camera position.
"""
import numpy as np


class ViewFinder:
    def __init__(self, height_fn, obstacles_xy, obstacles_r, canopy_fn=None, size_m=500.0):
        self.h = height_fn                     # (x, y arrays) -> ground z
        self.ob = np.asarray(obstacles_xy, float).reshape(-1, 2)
        self.obr = np.asarray(obstacles_r, float).reshape(-1)
        self.canopy = canopy_fn
        self.half = size_m / 2 - 5

    def score(self, cam, tgt):
        cam, tgt = np.asarray(cam, float), np.asarray(tgt, float)
        if np.abs(cam[:2]).max() > self.half:
            return -1e9
        t = np.linspace(0.03, 0.97, 60)
        ray = cam[None] + (tgt - cam)[None] * t[:, None]
        ground = self.h(ray[:, 0], ray[:, 1])
        occl = np.clip(ground + 0.3 - ray[:, 2], 0, None).sum()
        s = -40.0 * occl
        if len(self.ob):
            d = cam[:2] - tgt[:2]
            L = max(np.linalg.norm(d), 1e-6)
            u = -d / L
            rel = self.ob - cam[:2]
            along = rel @ u
            perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
            block = (along > 0.5) & (along < L - 2) & (perp < self.obr + 0.8)
            # Only obstacles tall enough to cut the line matter (approx: line < 12 m above ground).
            if block.any():
                p = cam[:2] + u * along[block, None]
                zl = cam[2] + (tgt[2] - cam[2]) * (along[block] / L)
                tall = zl - self.h(p[:, 0], p[:, 1]) < 26
                s -= 40.0 * tall.sum()
            near = np.hypot(rel[:, 0], rel[:, 1]) - self.obr
            s -= 60.0 * (near < 2.5).sum()
        if self.canopy is not None:
            s -= 80.0 * float(self.canopy(np.array([cam[0]]), np.array([cam[1]]))[0])
        return s

    def find(self, target_xy, target_dz, dists, heights, headings=24, prefer_heading=None, look_down=False):
        tx, ty = target_xy
        tz = float(self.h(np.array([tx]), np.array([ty]))[0]) + target_dz
        best, best_s = None, -1e18
        for a in np.linspace(0, 2 * np.pi, headings, endpoint=False):
            for d in dists:
                for hgt in heights:
                    cx, cy = tx + np.cos(a) * d, ty + np.sin(a) * d
                    cz = float(self.h(np.array([cx]), np.array([cy]))[0]) + hgt
                    s = self.score((cx, cy, cz), (tx, ty, tz))
                    s -= 0.05 * d + 0.3 * hgt
                    if look_down:
                        s -= 8.0 * max(0.0, tz + 1.0 - cz)
                    if prefer_heading is not None:
                        s -= 10 * (1 - np.cos(a - prefer_heading))
                    if s > best_s:
                        best_s, best = s, ((cx, cy, cz), (tx, ty, tz))
        return best, best_s
