"""Procedural foliage textures (RGBA, numpy only): needles, leaves, grass, fern, flowers.

Each texture is drawn analytically at 2x and box-downsampled for anti-aliasing.
Transparent pixels get their colour bled in from neighbours so mip-mapping and
alpha-dither edges don't pick up dark fringes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "terrain"))
from png_io import write_rgba8  # noqa: E402


class Canvas:
    def __init__(self, size, ss=2):
        self.n = size * ss
        self.ss = ss
        self.rgb = np.zeros((self.n, self.n, 3))
        self.a = np.zeros((self.n, self.n))

    def _bbox(self, pts, pad):
        lo = np.clip(np.floor(pts.min(0) * self.n - pad), 0, self.n - 1).astype(int)
        hi = np.clip(np.ceil(pts.max(0) * self.n + pad), 0, self.n - 1).astype(int)
        return lo, hi

    def leaf(self, base, tip, width, color, shape="ovate", lobes=0, serr=0.0, vein=0.25, rng=None):
        """Leaf in [0,1]^2 coords: base->tip midrib, max half-width `width` (fraction of length)."""
        base, tip = np.asarray(base, float), np.asarray(tip, float)
        L = np.linalg.norm(tip - base)
        if L < 1e-4:
            return
        ax = (tip - base) / L
        nr = np.array([-ax[1], ax[0]])
        corners = np.array([base - nr * width * L, base + nr * width * L, tip - nr * width * L, tip + nr * width * L])
        lo, hi = self._bbox(corners, 2)
        ys, xs = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        px = (xs + 0.5) / self.n - base[0]
        py = (ys + 0.5) / self.n - base[1]
        u = (px * ax[0] + py * ax[1]) / L
        v = (px * nr[0] + py * nr[1]) / L
        uc = np.clip(u, 0, 1)
        if shape == "needle":
            w = width * (1 - uc) ** 0.3
        elif shape == "blade":
            w = width * (1 - uc) ** 0.8
        elif shape == "lanceolate":
            w = width * np.sin(np.pi * uc) ** 0.9 * (1 - 0.3 * uc)
        else:  # ovate
            w = width * np.sin(np.pi * uc ** 0.8) ** 0.7
        if lobes:
            w = w * (0.72 + 0.28 * np.abs(np.sin(np.pi * lobes * uc)) ** 0.6)
        if serr:
            w = w * (1 - serr * (np.abs(np.sin(np.pi * 28 * uc)) ** 2))
        inside = (u >= 0) & (u <= 1) & (np.abs(v) <= w)
        if not inside.any():
            return
        # Shading: darker margins, light midrib, faint lateral veins.
        edge = np.clip(np.abs(v) / np.maximum(w, 1e-6), 0, 1)
        shade = 1.0 - 0.25 * edge ** 2
        mid = np.exp(-(v / (0.035 * width + 0.004)) ** 2) * vein
        lat = vein * 0.4 * (np.abs(np.sin((uc * 9 - np.abs(v) / max(width, 1e-3) * 2.5) * np.pi)) < 0.12)
        col = np.asarray(color)[None, None, :] * shade[..., None] * (1 + 0.6 * mid[..., None]) * (1 - 0.15 * lat[..., None])
        self._comp(ys, xs, inside, col)

    def line(self, p0, p1, w0, w1, color):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        lo, hi = self._bbox(np.array([p0, p1]), max(w0, w1) * self.n + 2)
        ys, xs = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        p = np.stack([(xs + 0.5) / self.n, (ys + 0.5) / self.n], -1)
        d = p1 - p0
        t = np.clip(((p - p0) @ d) / max(d @ d, 1e-9), 0, 1)
        dist = np.linalg.norm(p - (p0 + t[..., None] * d), axis=-1)
        inside = dist <= (w0 + (w1 - w0) * t)
        col = np.broadcast_to(np.asarray(color, float), inside.shape + (3,)) * (1 - 0.3 * (dist / max(w0, 1e-6)))[..., None].clip(0.6, 1)
        self._comp(ys, xs, inside, col)

    def disc(self, c, r, color):
        c = np.asarray(c, float)
        lo, hi = self._bbox(np.array([c - r, c + r]), 2)
        ys, xs = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        d = np.hypot((xs + 0.5) / self.n - c[0], (ys + 0.5) / self.n - c[1]) / r
        inside = d <= 1
        col = np.asarray(color)[None, None, :] * (1 - 0.35 * d ** 2)[..., None]
        self._comp(ys, xs, inside, col)

    def _comp(self, ys, xs, mask, col):
        self.rgb[ys[mask], xs[mask]] = col[mask] if col.ndim == 3 else col
        self.a[ys[mask], xs[mask]] = 1.0

    def save(self, path, bleed=12):
        ss, n = self.ss, self.n // self.ss
        a = self.a.reshape(n, ss, n, ss).mean((1, 3))
        rgb = (self.rgb * self.a[..., None]).reshape(n, ss, n, ss, 3).sum((1, 3))
        cnt = self.a.reshape(n, ss, n, ss).sum((1, 3))[..., None]
        rgb = np.where(cnt > 0, rgb / np.maximum(cnt, 1e-9), 0)
        # Colour bleed into transparent texels.
        filled = cnt[..., 0] > 0
        for _ in range(bleed):
            acc = np.zeros_like(rgb)
            num = np.zeros(filled.shape)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                acc += np.roll(np.roll(rgb * filled[..., None], dy, 0), dx, 1)
                num += np.roll(np.roll(filled, dy, 0), dx, 1)
            new = (~filled) & (num > 0)
            rgb[new] = acc[new] / num[new, None]
            filled |= new
        img = np.concatenate([rgb, a[..., None]], -1)
        # Image row 0 = top; our canvas y grows upward -> flip.
        write_rgba8(path, (np.clip(img, 0, 1) * 255).astype(np.uint8)[::-1])


def _jit(rng, c, amt):
    return np.clip(np.asarray(c) * (1 + rng.uniform(-amt, amt, 3)), 0, 1)


def pine_needles(path, rng, size=1024):
    """A conifer spray: central twig, side twigs, dense needle pairs."""
    cv = Canvas(size)
    bark = (0.20, 0.13, 0.08)
    base_col = np.array([0.09, 0.17, 0.07])
    main = [np.array([0.5, 0.02]), np.array([0.52, 0.98])]
    twigs = [main]
    for i in range(7):
        t = 0.15 + i * 0.11
        p = main[0] + (main[1] - main[0]) * t
        side = 1 if i % 2 else -1
        ang = np.radians(90 - side * rng.uniform(35, 55))
        ln = 0.42 * (1 - t * 0.6)
        twigs.append([p, p + ln * np.array([np.cos(ang), np.sin(ang)])])
    for p0, p1 in twigs:
        cv.line(p0, p1, 0.006, 0.003, bark)
    for p0, p1 in twigs:
        d = p1 - p0
        L = np.linalg.norm(d)
        ax = d / L
        nr = np.array([-ax[1], ax[0]])
        for t in np.linspace(0.02, 1.0, int(L * 140)):
            p = p0 + d * t
            for side in (-1, 1):
                a = np.radians(rng.uniform(40, 70))
                dirv = ax * np.cos(a) + side * nr * np.sin(a)
                ln = rng.uniform(0.07, 0.11) * (1 - 0.4 * t)
                cv.leaf(p, p + dirv * ln, 0.035, _jit(rng, base_col, 0.25), shape="needle", vein=0.1)
    cv.save(path)


def broadleaf_cluster(path, rng, size=1024, kind="oak"):
    """Leafy twig cluster: oak (lobed), birch (small serrate, pendulous), bush (dense ovate)."""
    cv = Canvas(size)
    bark = (0.22, 0.17, 0.11) if kind != "birch" else (0.30, 0.22, 0.16)
    if kind == "oak":
        col, n_leaf, lsize, width, lobes, serr = np.array([0.13, 0.24, 0.06]), 75, (0.12, 0.18), 0.32, 5, 0.0
    elif kind == "birch":
        col, n_leaf, lsize, width, lobes, serr = np.array([0.22, 0.33, 0.07]), 170, (0.055, 0.085), 0.38, 0, 0.12
    else:
        col, n_leaf, lsize, width, lobes, serr = np.array([0.12, 0.22, 0.06]), 150, (0.07, 0.1), 0.4, 0, 0.05
    # Twig skeleton.
    segs = []
    root = np.array([0.5, 0.03])
    stack = [(root, np.radians(90 + rng.uniform(-6, 6)), 0.8, 0)]
    while stack:
        p, ang, ln, depth = stack.pop()
        if kind == "birch" and depth > 0:
            ang -= np.radians(25)                     # pendulous
        q = p + ln * np.array([np.cos(ang), np.sin(ang)])
        segs.append((p, q, depth))
        if depth < 2:
            for k in range(3 if depth == 0 else 2):
                s = (-1) ** k
                stack.append((p + (q - p) * rng.uniform(0.2 + 0.2 * k, 0.45 + 0.2 * k),
                              ang + s * np.radians(rng.uniform(30, 55)), ln * rng.uniform(0.45, 0.6), depth + 1))
    for p, q, dpt in segs:
        cv.line(p, q, 0.007 / (dpt + 1), 0.003 / (dpt + 1), bark)
    for _ in range(n_leaf):
        p, q, _d = segs[rng.integers(len(segs))]
        b = p + (q - p) * rng.uniform(0.3, 1.0)
        ang = rng.uniform(0, 2 * np.pi) if kind != "birch" else np.radians(-90 + rng.uniform(-60, 60))
        ln = rng.uniform(*lsize)
        tip = np.clip(b + ln * np.array([np.cos(ang), np.sin(ang)]), 0.02, 0.98)
        b = np.clip(b, 0.03, 0.97)
        cv.leaf(b, tip, width, _jit(rng, col, 0.22), lobes=lobes, serr=serr, vein=0.3)
    cv.save(path)


def grass_tuft(path, rng, size=512, flowers=False):
    cv = Canvas(size)
    for _ in range(46):
        x0 = 0.5 + rng.normal(0, 0.12)
        h = rng.uniform(0.45, 0.95)
        bend = rng.uniform(-0.25, 0.25)
        green = np.array([0.18, 0.30, 0.08]) * rng.uniform(0.75, 1.15)
        dry = np.array([0.42, 0.38, 0.20])
        col = green * (1 - 0.35 * rng.random()) + dry * 0.35 * rng.random()
        pts = [np.array([x0, 0.0])]
        for k in range(1, 5):
            t = k / 4
            pts.append(np.array([x0 + bend * t * t, h * t]))
        for k in range(4):   # continuous taper along the blade
            cv.line(pts[k], pts[k + 1], 0.009 * (1 - k / 4) + 0.0012, 0.009 * (1 - (k + 1) / 4) + 0.0008,
                    np.clip(col * (0.85 + 0.1 * k), 0, 1))
    if flowers:
        palette = [(0.92, 0.90, 0.85), (0.95, 0.80, 0.15), (0.55, 0.35, 0.75), (0.85, 0.25, 0.2)]
        for _ in range(9):
            x = 0.5 + rng.normal(0, 0.15)
            h = rng.uniform(0.4, 0.85)
            cv.line((x, 0), (x + rng.uniform(-0.05, 0.05), h), 0.004, 0.003, (0.2, 0.32, 0.1))
            c = palette[rng.integers(len(palette))]
            for k in range(6):
                a = k * np.pi / 3
                cv.disc((x + 0.02 * np.cos(a), h + 0.02 * np.sin(a)), 0.016, c)
            cv.disc((x, h), 0.012, (0.85, 0.7, 0.1))
    cv.save(path)


def fern_frond(path, rng, size=1024):
    cv = Canvas(size)
    col = np.array([0.12, 0.26, 0.06])
    pts = [np.array([0.5 + 0.08 * (t ** 2) * 0, 0.02 + 0.95 * t]) for t in np.linspace(0, 1, 40)]
    pts = [p + np.array([0.06 * np.sin(np.pi * i / 39) * 0.5, 0]) for i, p in enumerate(pts)]
    for i in range(len(pts) - 1):
        cv.line(pts[i], pts[i + 1], 0.006 * (1 - i / 40) + 0.002, 0.006 * (1 - (i + 1) / 40) + 0.002, (0.18, 0.24, 0.08))
    for i in range(3, len(pts) - 1):
        t = i / len(pts)
        ln = 0.42 * np.sin(np.pi * min(t * 1.15, 1.0)) ** 0.8 * (1 - t * 0.5)
        for side in (-1, 1):
            ang = np.radians(90 - side * 70 + rng.uniform(-6, 6))
            b = pts[i]
            tip = b + ln * np.array([np.cos(ang), np.sin(ang)])
            cv.leaf(b, tip, 0.13, _jit(rng, col, 0.15), shape="lanceolate", lobes=9, vein=0.35)
    cv.save(path)


def generate_all(out_dir, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed + 900)
    files = {
        "pine_needles": lambda p: pine_needles(p, rng),
        "oak_leaves": lambda p: broadleaf_cluster(p, rng, kind="oak"),
        "birch_leaves": lambda p: broadleaf_cluster(p, rng, kind="birch"),
        "bush_leaves": lambda p: broadleaf_cluster(p, rng, kind="bush"),
        "grass": lambda p: grass_tuft(p, rng),
        "grass_flowers": lambda p: grass_tuft(p, rng, flowers=True),
        "fern": lambda p: fern_frond(p, rng),
    }
    paths = {}
    for name, fn in files.items():
        path = os.path.join(out_dir, f"T_{name}.png")
        fn(path)
        paths[name] = path
    return paths
