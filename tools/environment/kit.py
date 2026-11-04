"""Geometry kit for man-made structures (numpy): boxes, beams, cylinders, prisms,
jagged ruined walls with openings. Everything in metres, local frame, +Z up.

A Geo accumulates parts with a material slot name each; `to_mesh` turns it into
one Blender mesh with per-face material indices and box-projected UVs.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "common")]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_xyz(rx, ry, rz):
    cx, sx, cy, sy = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rot_z(rz) @ Ry @ Rx


BOX_F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


class Geo:
    def __init__(self):
        self.v, self.f, self.m = [], [], []
        self.nv = 0
        self.slots = []
        self.colliders = []          # (kind, center, size, yaw) simple collision primitives
        self.cover = []              # (x, y, z_top, kind) cover points (local)

    def slot(self, name):
        if name not in self.slots:
            self.slots.append(name)
        return self.slots.index(name)

    def add(self, verts, faces, mat):
        mi = self.slot(mat)
        self.v.append(np.asarray(verts, float))
        self.f.extend([list(np.asarray(f) + self.nv) for f in faces])
        self.m.extend([mi] * len(faces))
        self.nv += len(verts)

    def box(self, c, size, mat, R=None, yaw=0.0, collide=True, taper=0.0):
        sx, sy, sz = np.asarray(size, float) / 2
        corners = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])
        if taper:
            corners[4:, :2] *= (1 - taper)
        R = rot_z(yaw) if R is None else R
        self.add(corners @ R.T + np.asarray(c, float), BOX_F, mat)
        if collide:
            self.colliders.append(("box", list(np.asarray(c, float)), [2 * sx, 2 * sy, 2 * sz], float(yaw)))

    def beam(self, p0, p1, w, h, mat, up=(0, 0, 1), collide=False):
        """Rectangular timber from p0 to p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        x = d / L
        upv = np.asarray(up, float)
        if abs(x @ upv) > 0.95:
            upv = np.array([1.0, 0, 0])
        y = np.cross(upv, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        R = np.column_stack([x, y, z])
        self.box((p0 + p1) / 2, (L, w, h), mat, R=R, collide=collide)

    def cyl(self, p0, p1, r, mat, seg=10, caps=True, r1=None, collide=False):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        x = d / L
        ref = np.array([0, 0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(x, ref)
        u /= np.linalg.norm(u)
        w = np.cross(x, u)
        th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        r1 = r if r1 is None else r1
        ring0 = p0 + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
        ring1 = p1 + r1 * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
        verts = np.vstack([ring0, ring1])
        faces = [[i, (i + 1) % seg, seg + (i + 1) % seg, seg + i] for i in range(seg)]
        if caps:
            faces.append(list(range(seg))[::-1])
            faces.append(list(range(seg, 2 * seg)))
        self.add(verts, faces, mat)
        if collide:
            self.colliders.append(("capsule", list((p0 + p1) / 2), [max(r, r1), L], 0.0))

    def prism(self, c, length, width, height, mat, yaw=0.0, collide=False):
        """Triangular prism (gable) along local x, apex up."""
        L, W = length / 2, width / 2
        v = np.array([[-L, -W, 0], [L, -W, 0], [L, W, 0], [-L, W, 0], [-L, 0, height], [L, 0, height]])
        faces = [[0, 1, 5, 4], [1, 2, 5], [2, 3, 4, 5], [3, 0, 4], [0, 3, 2, 1]]
        self.add(v @ rot_z(yaw).T + np.asarray(c, float), faces, mat)

    def roof(self, c, length, width, pitch_h, thick, mat, overhang=0.4, yaw=0.0, missing=None, rng=None):
        """Two-slope roof as thick panels; `missing` fraction removes panel strips (ruins)."""
        R = rot_z(yaw)
        L = length / 2 + overhang
        W = width / 2 + overhang
        n = 6
        xs = np.linspace(-L, L, n + 1)
        for side in (-1, 1):
            ang = np.arctan2(pitch_h, W)
            slope_len = np.hypot(W, pitch_h)
            for i in range(n):
                if missing and rng is not None and rng.random() < missing:
                    continue
                cx = (xs[i] + xs[i + 1]) / 2
                cy = side * W / 2
                cz = pitch_h / 2
                Rp = R @ rot_xyz(-side * ang, 0, 0)
                self.box(R @ np.array([cx, cy, cz]) + np.asarray(c, float), (xs[1] - xs[0] + 0.01, slope_len, thick),
                         mat, R=Rp, collide=False)

    def wall(self, p0, p1, height, thick, mat, openings=(), ruin=0.0, rng=None, block=0.45, base_z=0.0,
             collide=True):
        """Straight wall from p0 to p1 (xy) with rectangular openings [(u0, u1, z0, z1)] in metres
        along the wall; `ruin` 0..1 lowers the top in jagged steps (masonry-like breaks)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        yaw = np.arctan2(d[1], d[0])
        R = rot_z(yaw)
        cuts = sorted(set([0.0, L] + [u for o in openings for u in o[:2]]))
        # Subdivide into masonry-width strips so ruined tops are jagged.
        strips = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            k = max(int(np.ceil((b - a) / block)), 1)
            e = np.linspace(a, b, k + 1)
            strips.extend(zip(e[:-1], e[1:]))
        top = np.full(len(strips), float(height))
        if ruin > 0 and rng is not None:
            # Smooth random damage profile (several breaches) + per-strip jitter.
            prof = np.cumsum(rng.normal(0, 1, len(strips)))
            prof = np.convolve(np.pad(prof, 2, mode="edge"), np.ones(5) / 5, mode="valid")
            prof = (prof - prof.min()) / max(np.ptp(prof), 1e-6)
            top = height * (1 - ruin * prof) - rng.uniform(0, 0.18, len(strips)) * ruin * height * 0.4
            top = np.maximum(top, 0.25)
        # Heights at strip boundaries -> sloped, continuous broken top line (not stair steps).
        hb = np.concatenate([[top[0]], (top[:-1] + top[1:]) / 2, [top[-1]]])
        if ruin > 0 and rng is not None:
            hb = hb + rng.normal(0, 0.08 * ruin * height / 3, len(hb))
        for k, ((a, b), tz) in enumerate(zip(strips, top)):
            um = (a + b) / 2
            solid = [(0.0, tz)]
            for (u0, u1, z0, z1) in openings:
                if u0 <= um <= u1:
                    new = []
                    for (s0, s1) in solid:
                        if z1 <= s0 or z0 >= s1:
                            new.append((s0, s1))
                            continue
                        if z0 > s0:
                            new.append((s0, z0))
                        if z1 < s1:
                            new.append((z1, s1))
                    solid = new
            for (s0, s1) in solid:
                if s1 - s0 < 0.05:
                    continue
                if abs(s1 - tz) < 1e-6 and ruin > 0:
                    zl, zr = max(hb[k], s0 + 0.05), max(hb[k + 1], s0 + 0.05)
                else:
                    zl = zr = s1
                self.slab(p0, R, a, b - a + 0.002, thick, base_z + s0, base_z + zl, base_z + zr, mat)
        if collide:
            mid = (p0 + p1) / 2
            self.colliders.append(("box", [mid[0], mid[1], base_z + float(top.mean()) / 2],
                                   [L, thick, float(top.mean())], float(yaw)))
        return top

    def slab(self, p0, R, u, length, thick, z0, z1l, z1r, mat):
        """Wall piece from u to u+length along the wall, bottom z0, sloped top z1l -> z1r."""
        t = thick / 2
        loc = np.array([[u, -t, z0], [u + length, -t, z0], [u + length, t, z0], [u, t, z0],
                        [u, -t, z1l], [u + length, -t, z1r], [u + length, t, z1r], [u, t, z1l]])
        self.add(loc @ R.T + np.array([p0[0], p0[1], 0.0]), BOX_F, mat)

    def transformed(self, yaw, offset):
        """Return arrays transformed into world space (for merging / instance export)."""
        R = rot_z(yaw)
        V = np.vstack(self.v) @ R.T + np.asarray(offset, float)
        return V

    def merge(self, other, yaw=0.0, offset=(0, 0, 0)):
        R = rot_z(yaw)
        off = np.asarray(offset, float)
        V = np.vstack(other.v) @ R.T + off if other.v else np.zeros((0, 3))
        remap = [self.slot(s) for s in other.slots]
        self.v.append(V)
        self.f.extend([list(np.asarray(f) + self.nv) for f in other.f])
        self.m.extend([remap[i] for i in other.m])
        self.nv += len(V)
        for kind, c, size, y in other.colliders:
            cc = R @ np.asarray(c, float) + off
            self.colliders.append((kind, list(cc), size, y + yaw))
        for (x, y, z, k) in other.cover:
            p = R @ np.array([x, y, z]) + off
            self.cover.append((p[0], p[1], p[2], k))

    def arrays(self):
        V = np.vstack(self.v) if self.v else np.zeros((0, 3))
        return V, self.f, np.array(self.m, np.int32)
