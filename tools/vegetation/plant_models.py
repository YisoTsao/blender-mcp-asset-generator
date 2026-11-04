"""Procedural plant meshes (bpy): trees, logs, bushes, ferns, grass.

Geometry is accumulated in numpy by a small MeshBuilder (tubes for wood, cards
for foliage) and turned into one mesh per asset variant with two material slots:
0 = bark/wood, 1 = foliage. Assets are built with the base at the origin, +Z up,
in metres. Poly budgets are kept low enough for thousands of instances.
"""
import bpy
import numpy as np


class MeshBuilder:
    def __init__(self):
        self.v = []          # vertex arrays
        self.f = []          # face index arrays (all quads)
        self.uv = []         # per-loop uv arrays (nf, 4, 2)
        self.mat = []        # material index per face
        self.nv = 0

    def _add(self, verts, faces, uvs, mat):
        self.v.append(verts)
        self.f.append(faces + self.nv)
        self.uv.append(uvs)
        self.mat.append(np.full(len(faces), mat, np.int32))
        self.nv += len(verts)

    def tube(self, pts, radii, ring=8, mat=0, cap=False):
        """Swept circle along pts with parallel-transport frames. UV in metres."""
        pts = np.asarray(pts, float)
        n = len(pts)
        tang = np.gradient(pts, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        nrm = np.cross(tang[0], ref)
        nrm /= np.linalg.norm(nrm)
        frames = []
        for i in range(n):
            if i:
                nrm = nrm - tang[i] * (nrm @ tang[i])
                nrm /= max(np.linalg.norm(nrm), 1e-9)
            frames.append((nrm, np.cross(tang[i], nrm)))
        th = np.linspace(0, 2 * np.pi, ring, endpoint=False)
        verts = np.array([pts[i] + radii[i] * (np.cos(t) * frames[i][0] + np.sin(t) * frames[i][1])
                          for i in range(n) for t in th])
        idx = np.arange(n * ring).reshape(n, ring)
        nxt = np.roll(idx, -1, axis=1)
        faces = np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], -1).reshape(-1, 4)
        seglen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        circ = 2 * np.pi * np.maximum(np.mean(radii), 0.01)
        uu = np.arange(ring + 1) / ring * circ
        uvs = []
        for i in range(n - 1):
            for j in range(ring):
                uvs.append([[uu[j], seglen[i]], [uu[j + 1], seglen[i]], [uu[j + 1], seglen[i + 1]], [uu[j], seglen[i + 1]]])
        self._add(verts, faces, np.array(uvs), mat)
        if cap:   # close the far end with a fan of quads (degenerate last vertex)
            c = self.nv
            self.v.append(pts[-1:])
            self.nv += 1
            last = idx[-1] + (self.nv - 1 - n * ring)
            fc = np.array([[last[j], last[(j + 1) % ring], c, c] for j in range(ring)])
            self.f.append(fc)
            self.uv.append(np.zeros((ring, 4, 2)))
            self.mat.append(np.full(ring, mat, np.int32))

    def card(self, base, up, right, w, h, mat=1, uv=(0, 0, 1, 1)):
        """Quad from `base` (bottom centre) along unit `up`, width along unit `right`."""
        base, up, right = (np.asarray(x, float) for x in (base, up, right))
        a = base - right * w / 2
        b = base + right * w / 2
        verts = np.array([a, b, b + up * h, a + up * h])
        u0, v0, u1, v1 = uv
        self._add(verts, np.array([[0, 1, 2, 3]]), np.array([[[u0, v0], [u1, v0], [u1, v1], [u0, v1]]]), mat)

    def bent_card(self, base, d0, d1, right, w, h0, h1, mat=1):
        """Two-segment card (arching frond / grass): first segment along d0, second along d1."""
        base, d0, d1, right = (np.asarray(x, float) for x in (base, d0, d1, right))
        p0, p1 = base, base + d0 * h0
        p2 = p1 + d1 * h1
        r = right * w / 2
        verts = np.array([p0 - r, p0 + r, p1 + r * 0.9, p1 - r * 0.9, p2 + r * 0.5, p2 - r * 0.5])
        t = h0 / (h0 + h1)
        faces = np.array([[0, 1, 2, 3], [3, 2, 4, 5]])
        uvs = np.array([[[0, 0], [1, 0], [1, t], [0, t]], [[0, t], [1, t], [1, 1], [0, 1]]])
        self._add(verts, faces, uvs, mat)

    def build(self, name, materials):
        verts = np.concatenate(self.v)
        faces = np.concatenate(self.f)
        uvs = np.concatenate(self.uv).reshape(-1, 2)
        mats = np.concatenate(self.mat)
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(verts))
        me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        nf = len(faces)
        me.loops.add(nf * 4)
        me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
        me.polygons.add(nf)
        me.polygons.foreach_set("loop_start", (np.arange(nf) * 4).astype(np.int32))
        me.polygons.foreach_set("material_index", mats)
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", uvs.astype(np.float32).ravel())
        me.update(calc_edges=True)
        me.validate(clean_customdata=False)
        for m in materials:
            me.materials.append(m)
        # Wood smooth, cards flat.
        smooth = (mats == 0)
        me.polygons.foreach_set("use_smooth", smooth)
        ob = bpy.data.objects.new(name, me)
        return ob


# --------------------------------------------------------------------------- helpers

def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def dir_from(az, el):
    return np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])


def wobbly_path(p0, d, length, n, rng, wob=0.05, droop=0.0):
    """Points from p0 along d with gentle random curvature and optional gravity droop."""
    pts = [np.asarray(p0, float)]
    d = unit(d)
    step = length / (n - 1)
    for i in range(1, n):
        d = unit(d + rng.normal(0, wob, 3) - np.array([0, 0, droop]))
        pts.append(pts[-1] + d * step)
    return np.array(pts)


def add_roots(mb, r0, rng, count, reach, depth=0.5):
    for k in range(count):
        az = 2 * np.pi * (k + rng.uniform(-0.3, 0.3)) / count
        p0 = np.array([0, 0, r0 * 0.9])
        pts = np.array([p0 + np.array([np.cos(az), np.sin(az), 0]) * reach * t + np.array([0, 0, -depth * t ** 1.5])
                        for t in np.linspace(0, 1, 5)])
        mb.tube(pts, np.linspace(r0 * 0.55, r0 * 0.08, 5), ring=6)


def trunk(mb, H, r0, rng, flare=0.8, lean=0.03, n=12, ring=10, top_r=0.03):
    z = np.linspace(-0.4, H, n)
    leanv = rng.normal(0, lean, 2)
    bend = rng.normal(0, 0.15, 2)
    pts = np.column_stack([leanv[0] * z + bend[0] * np.sin(z / H * np.pi),
                           leanv[1] * z + bend[1] * np.sin(z / H * np.pi), z])
    r = np.maximum(r0 * (1 - np.clip(z, 0, None) / H) ** 0.85, top_r)
    r *= 1 + flare * np.exp(-np.clip(z, 0, None) / 0.6)
    mb.tube(pts, r, ring=ring)
    return pts, r


def point_on(pts, z):
    i = int(np.clip(np.searchsorted(pts[:, 2], z), 1, len(pts) - 1))
    t = (z - pts[i - 1, 2]) / max(pts[i, 2] - pts[i - 1, 2], 1e-6)
    return pts[i - 1] + (pts[i] - pts[i - 1]) * t


# --------------------------------------------------------------------------- species

def make_pine(name, rng, mats):
    mb = MeshBuilder()
    H = rng.uniform(15, 22)
    r0 = H * rng.uniform(0.014, 0.018)
    pts, _ = trunk(mb, H, r0, rng, flare=0.6, lean=0.015)
    add_roots(mb, r0, rng, 5, r0 * 4)
    crown0 = H * rng.uniform(0.32, 0.45)
    # Dead stubs below the crown.
    for z in rng.uniform(2.0, crown0, rng.integers(6, 11)):
        az, L = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.0)
        p = point_on(pts, z)
        mb.tube(wobbly_path(p, dir_from(az, rng.uniform(-0.2, 0.3)), L, 3, rng, 0.1), np.linspace(0.04, 0.01, 3), ring=4)
    z = crown0
    while z < H - 0.6:
        k = rng.integers(4, 7)
        az0 = rng.uniform(0, 2 * np.pi)
        frac = (H - z) / (H - crown0)
        for j in range(k):
            az = az0 + 2 * np.pi * j / k + rng.uniform(-0.3, 0.3)
            L = max(0.5, (H - z) * 0.30 * rng.uniform(0.85, 1.1) + 0.3)
            p = point_on(pts, z)
            bp = wobbly_path(p, dir_from(az, np.radians(rng.uniform(-5, 15))), L, 4, rng, 0.05, droop=0.06 * frac)
            mb.tube(bp, np.linspace(0.025 + 0.04 * frac, 0.008, 4), ring=4)
            ncard = max(3, int(L / 0.32))
            for c in range(ncard):
                t = 0.25 + 0.75 * c / max(ncard - 1, 1)
                i = min(int(t * 3), 2)
                base = bp[i] + (bp[i + 1] - bp[i]) * (t * 3 - i)
                fwd = unit(bp[-1] - bp[0])
                side = unit(np.cross(fwd, [0, 0, 1]))
                roll = np.radians(rng.uniform(-25, 25))
                right = unit(side * np.cos(roll) + np.array([0, 0, 1]) * np.sin(roll))
                sz = rng.uniform(1.3, 1.8) * (0.6 + 0.5 * min(L / 3, 1))
                mb.card(base - fwd * 0.25, unit(fwd + rng.normal(0, 0.15, 3) - [0, 0, 0.1]), right, sz * 1.1, sz)
                # Crossed card (always) so sprays read as volume from every angle.
                mb.card(base - fwd * 0.25, fwd, unit(np.cross(fwd, right)), sz * 0.8, sz * 0.9)
        z += rng.uniform(0.42, 0.62)
    # Inner-crown fill: sprays hugging the trunk so the crown isn't see-through.
    for zz in np.arange(crown0 + 0.5, H - 1.0, 0.7):
        p = point_on(pts, zz)
        a = rng.uniform(0, 2 * np.pi)
        w = 1.2 + 1.6 * (H - zz) / (H - crown0)
        for j in range(2):
            r = np.array([np.cos(a + j * np.pi / 2), np.sin(a + j * np.pi / 2), 0])
            mb.card(p - [0, 0, 0.7], unit([0, 0, 1] + rng.normal(0, 0.1, 3)), r, w, 1.5)
    top = point_on(pts, H - 0.2)
    for j in range(3):
        a = j * np.pi / 3
        mb.card(top - [0, 0, 0.6], [0, 0, 1], [np.cos(a), np.sin(a), 0], 0.7, 1.4)
    return mb.build(name, mats)


def _limb(mb, p0, d, L, r0, rng, depth, tips, droop=0.0, wob=0.12):
    n = 5 if depth == 0 else 4
    pts = wobbly_path(p0, d, L, n, rng, wob, droop)
    mb.tube(pts, np.linspace(r0, r0 * 0.35, n), ring=7 if depth == 0 else 5)
    if depth >= 1:   # foliage along the branch too, not only at the tips
        tips.append((pts[n // 2], unit(pts[-1] - pts[0])))
    if depth >= 2:
        tips.append((pts[-1], unit(pts[-1] - pts[0])))
        return
    nkids = rng.integers(3, 5) if depth == 0 else rng.integers(2, 4)
    for _ in range(nkids):
        t = rng.uniform(0.35, 0.95)
        i = min(int(t * (n - 1)), n - 2)
        p = pts[i] + (pts[i + 1] - pts[i]) * (t * (n - 1) - i)
        az = np.arctan2(d[1], d[0]) + rng.uniform(-1.2, 1.2)
        el = np.radians(rng.uniform(5, 45))
        _limb(mb, p, dir_from(az, el), L * rng.uniform(0.4, 0.6), r0 * 0.45, rng, depth + 1, tips, droop, wob)
    tips.append((pts[-1], unit(pts[-1] - pts[0])))


def make_oak(name, rng, mats):
    mb = MeshBuilder()
    H = rng.uniform(11, 16)
    r0 = rng.uniform(0.35, 0.5)
    zs = H * rng.uniform(0.25, 0.35)
    pts, _ = trunk(mb, zs + 1.0, r0, rng, flare=0.9, lean=0.04, n=8, ring=12, top_r=r0 * 0.6)
    add_roots(mb, r0, rng, 6, r0 * 4.5, 0.6)
    tips = []
    top = pts[-2]
    nl = rng.integers(4, 7)
    for j in range(nl):
        az = 2 * np.pi * j / nl + rng.uniform(-0.4, 0.4)
        el = np.radians(rng.uniform(35, 60))
        _limb(mb, top, dir_from(az, el), (H - zs) * rng.uniform(0.55, 0.7), r0 * 0.55, rng, 0, tips)
    centre = np.array([0, 0, zs + (H - zs) * 0.55])
    for p, d in tips:
        out = unit(p - centre + rng.normal(0, 0.3, 3))
        for c in range(4):
            up = unit(d + out + rng.normal(0, 0.4, 3))
            right = unit(np.cross(up, rng.normal(0, 1, 3)))
            sz = rng.uniform(1.8, 2.6)
            mb.card(p - up * sz * 0.35, up, right, sz, sz)
    return mb.build(name, mats)


def make_birch(name, rng, mats):
    mb = MeshBuilder()
    H = rng.uniform(12, 18)
    stems = 1 if rng.random() < 0.6 else rng.integers(2, 4)
    for s in range(stems):
        r0 = rng.uniform(0.13, 0.2) / (1 + 0.25 * (stems - 1))
        off = rng.normal(0, 0.25, 2) if stems > 1 else np.zeros(2)
        lean = rng.normal(0, 0.06, 2) if stems > 1 else rng.normal(0, 0.02, 2)
        h = H * rng.uniform(0.85, 1.0)
        z = np.linspace(-0.3, h, 12)
        pts = np.column_stack([off[0] + lean[0] * z, off[1] + lean[1] * z, z])
        r = np.maximum(r0 * (1 - np.clip(z, 0, None) / h) ** 0.8, 0.02) * (1 + 0.5 * np.exp(-np.clip(z, 0, None) / 0.4))
        mb.tube(pts, r, ring=8)
        crown0 = h * rng.uniform(0.35, 0.5)
        tips = []
        for zz in np.arange(crown0, h - 0.5, rng.uniform(0.32, 0.45)):
            az = rng.uniform(0, 2 * np.pi)
            p = point_on(pts, zz)
            L = (h - zz) * 0.35 + 0.6
            bp = wobbly_path(p, dir_from(az, np.radians(rng.uniform(35, 55))), L, 4, rng, 0.08, droop=0.08)
            mb.tube(bp, np.linspace(0.03, 0.008, 4), ring=4, mat=2)   # dark twig bark, not white
            tips.append((bp, L))
        for bp, L in tips:
            for c in range(max(3, int(L / 0.3))):
                t = 0.3 + 0.7 * rng.random()
                i = min(int(t * 3), 2)
                base = bp[i] + (bp[i + 1] - bp[i]) * (t * 3 - i)
                hang = unit(np.array([*(bp[-1] - bp[0])[:2] * 0.3, -1.0]) + rng.normal(0, 0.25, 3))
                right = unit(np.cross(hang, rng.normal(0, 1, 3)))
                sz = rng.uniform(1.3, 1.9)
                # Texture is drawn growing upward; flip so leaves hang from the attach point.
                mb.card(base + np.array([0, 0, 0.2]), -hang, right, sz, sz, uv=(0, 1, 1, 0))
    return mb.build(name, mats)


def make_dead(name, rng, mats):
    mb = MeshBuilder()
    H = rng.uniform(7, 14)
    r0 = H * rng.uniform(0.018, 0.024)
    pts, r = trunk(mb, H, r0, rng, flare=0.5, lean=0.04, n=10, top_r=r0 * 0.35)
    add_roots(mb, r0, rng, 4, r0 * 3.5)
    for z in rng.uniform(H * 0.3, H * 0.95, rng.integers(6, 12)):
        az = rng.uniform(0, 2 * np.pi)
        L = rng.uniform(0.4, 2.2) * (1 - z / H * 0.5)
        mb.tube(wobbly_path(point_on(pts, z), dir_from(az, np.radians(rng.uniform(-10, 35))), L, 3, rng, 0.15),
                np.linspace(0.05, 0.012, 3), ring=4)
    return mb.build(name, mats)


def make_log(name, rng, mats):
    mb = MeshBuilder()
    L = rng.uniform(6, 12)
    r0 = rng.uniform(0.22, 0.38)
    x = np.linspace(-L / 2, L / 2, 10)
    pts = np.column_stack([x, rng.normal(0, 0.05, 10).cumsum() * 0.3, np.full(10, r0 * 0.75)])
    r = r0 * (1 - 0.35 * (x + L / 2) / L)
    mb.tube(pts, r, ring=10, cap=True)
    # Root plate stump at one end (upturned roots) or broken end.
    if rng.random() < 0.5:
        for k in range(7):
            a = 2 * np.pi * k / 7
            p0 = pts[0]
            mb.tube(np.array([p0, p0 + [-0.3, 0.9 * np.cos(a), 0.9 * np.sin(a) + 0.2],
                              p0 + [-0.4, 1.4 * np.cos(a), 1.4 * np.sin(a) + 0.3]]),
                    np.array([r0 * 0.35, r0 * 0.18, 0.02]), ring=5)
    for _ in range(rng.integers(3, 7)):
        t = rng.uniform(0.2, 0.9)
        p = pts[0] + (pts[-1] - pts[0]) * t
        mb.tube(np.array([p, p + dir_from(rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 1.2)) * rng.uniform(0.3, 0.8)]),
                np.array([0.06, 0.015]), ring=4)
    return mb.build(name, mats)


def make_bush(name, rng, mats):
    mb = MeshBuilder()
    H = rng.uniform(1.0, 2.0)
    for _ in range(rng.integers(7, 13)):
        az = rng.uniform(0, 2 * np.pi)
        el = np.radians(rng.uniform(50, 80))
        bp = wobbly_path([0, 0, -0.1], dir_from(az, el), H * rng.uniform(0.7, 1.1), 4, rng, 0.12)
        mb.tube(bp, np.linspace(0.02, 0.006, 4), ring=4)
        for c in range(rng.integers(2, 4)):
            p = bp[rng.integers(1, 4)]
            up = unit(bp[-1] - bp[0] + rng.normal(0, 0.5, 3))
            sz = rng.uniform(0.55, 0.85)
            mb.card(p - up * sz * 0.4, up, unit(np.cross(up, rng.normal(0, 1, 3))), sz, sz)
    return mb.build(name, mats)


def make_fern(name, rng, mats):
    mb = MeshBuilder()
    n = rng.integers(9, 15)
    for k in range(n):
        az = 2 * np.pi * k / n + rng.uniform(-0.2, 0.2)
        L = rng.uniform(0.8, 1.35)
        d0 = dir_from(az, np.radians(rng.uniform(58, 76)))
        d1 = dir_from(az, np.radians(rng.uniform(-15, 10)))
        right = np.array([-np.sin(az), np.cos(az), 0])
        mb.bent_card([0, 0, -0.03], d0, d1, right, L * 0.5, L * 0.5, L * 0.6)
    return mb.build(name, mats)


def make_grass(name, rng, mats, cards=3):
    mb = MeshBuilder()
    h = rng.uniform(0.45, 0.8)
    for k in range(cards):
        a = np.pi * k / cards + rng.uniform(-0.2, 0.2)
        right = np.array([np.cos(a), np.sin(a), 0])
        up = unit(np.array([0, 0, 1]) + rng.normal(0, 0.12, 3))
        mb.card([0, 0, -0.05], up, right, h * 0.9, h)
    return mb.build(name, mats)


SPECIES = {
    # name: (builder, bark material key, foliage material key, variants)
    "pine": (make_pine, "bark_pine", "pine_needles", 4),
    "oak": (make_oak, "bark_oak", "oak_leaves", 3),
    "birch": (make_birch, "bark_birch", "birch_leaves", 3),   # + slot 2: twig bark
    "dead": (make_dead, "bark_dead", None, 3),
    "log": (make_log, "log_moss", None, 3),
    "bush": (make_bush, "bark_oak", "bush_leaves", 3),
    "fern": (make_fern, None, "fern", 2),
    "grass": (make_grass, None, "grass", 3),
    "flowers": (make_grass, None, "grass_flowers", 2),
}
