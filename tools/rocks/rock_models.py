"""Procedural rock meshes: fractured, eroded and stratified shapes from an icosphere.

Shape recipe (all in numpy, per vertex of a unit icosphere):
  1. low-frequency 3D fBm swell -> irregular blob, not an ellipsoid
  2. planar fracture cuts (vertices beyond a random plane are projected onto it)
     -> crisp facets and distinct silhouettes per variant
  3. anisotropic scale to the rock type's proportions (metres)
  4. strata ledges (cliffs / weathered rock): radial inset stepping with height
  5. ridged + fBm detail along the radial direction, amplitude per type
  6. Laplacian rounding for water-worn rocks; flat-ish base (sits on / in the ground)
Each asset: origin at the base contact level, +Z up, box-projected UVs, hard edges
marked on fractures.
"""
import os
import sys

import bmesh
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "common")]
from blend_util import box_uvs, mark_sharp, mesh_from_numpy  # noqa: E402


class Noise3:
    """3D gradient (Perlin) noise, vectorised."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        p = rng.permutation(256)
        self.perm = np.concatenate([p, p])
        g = rng.normal(size=(256, 3))
        self.g = g / np.linalg.norm(g, axis=1, keepdims=True)

    def __call__(self, p):
        pi = np.floor(p).astype(np.int64)
        f = p - pi
        u = f * f * f * (f * (f * 6 - 15) + 10)
        out = np.zeros(len(p))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = (pi[:, 0] + dx) & 255, (pi[:, 1] + dy) & 255, (pi[:, 2] + dz) & 255
                    h = self.perm[self.perm[self.perm[ix] + iy] + iz]
                    dot = (self.g[h] * (f - (dx, dy, dz))).sum(1)
                    w = ((u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1])
                         * (u[:, 2] if dz else 1 - u[:, 2]))
                    out += w * dot
        return out

    def fbm(self, p, octaves, lac=2.07, gain=0.5):
        a, s, tot = 1.0, 1.0, 0.0
        for o in range(octaves):
            tot = tot + a * self(p * s + o * 17.31)
            a *= gain
            s *= lac
        return tot

    def ridged(self, p, octaves, lac=2.1, gain=0.55):
        a, s, tot = 1.0, 1.0, 0.0
        for o in range(octaves):
            tot = tot + a * (0.5 - np.abs(self(p * s + o * 7.7)))
            a *= gain
            s *= lac
        return tot


def icosphere(subdiv):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    V = np.array([v.co[:] for v in bm.verts])
    F = np.array([[v.index for v in f.verts] for f in bm.faces])
    bm.free()
    return V, F


def laplacian(V, F, iters, lam=0.5, mask=None):
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.concatenate([E, E[:, ::-1]])
    deg = np.bincount(E[:, 0], minlength=n).astype(float)
    for _ in range(iters):
        acc = np.zeros_like(V)
        np.add.at(acc, E[:, 0], V[E[:, 1]])
        d = acc / deg[:, None] - V
        V = V + lam * d * (1.0 if mask is None else mask[:, None])
    return V


# Per-type recipe: dims (m), cuts, cut depth range, detail amplitude, rounding, strata.
TYPES = {
    #               subdiv dims(x,y,z)          cuts  cut_d       swell detail ridge round strata
    "boulder":     (5, (2.6, 2.0, 1.6),          (4, 7), (0.45, 0.8), 0.22, 0.035, 0.02, 2, 0.0),
    "cliff":       (5, (9.0, 6.0, 2.6),          (5, 9), (0.35, 0.7), 0.18, 0.05, 0.05, 0, 0.09),
    "mountain":    (5, (3.2, 2.4, 3.0),          (6, 11), (0.3, 0.7), 0.20, 0.04, 0.045, 0, 0.03),
    "river":       (4, (1.0, 0.75, 0.5),         (2, 4), (0.6, 0.9), 0.15, 0.012, 0.0, 6, 0.0),
    "pebble":      (3, (0.16, 0.12, 0.07),       (1, 3), (0.6, 0.9), 0.15, 0.01, 0.0, 4, 0.0),
    "weathered":   (5, (2.2, 1.8, 1.5),          (3, 6), (0.45, 0.8), 0.25, 0.05, 0.03, 1, 0.06),
}
VARIANTS = {"boulder": 5, "cliff": 4, "mountain": 4, "river": 5, "pebble": 6, "weathered": 4}


def rock_mesh(kind, rng, n3):
    subdiv, dims, (c0, c1), (d0, d1), swell, detail, ridge, rounds, strata = TYPES[kind]
    V, F = icosphere(subdiv)
    dirs = V.copy()
    off = rng.uniform(-50, 50, 3)
    p = V * (1 + swell * n3.fbm(V * 1.1 + off, 3))[:, None]
    # Fracture planes: biased away from straight down so the base stays one broad face.
    for _ in range(int(rng.integers(c0, c1 + 1))):
        n = rng.normal(size=3)
        n[2] = abs(n[2]) * 0.8 if rng.random() < 0.7 else n[2] * 0.3
        n /= np.linalg.norm(n)
        d = rng.uniform(d0, d1)
        s = p @ n - d
        p -= np.maximum(s, 0)[:, None] * n
    # Proportions with per-variant variation.
    sc = np.array(dims) * rng.uniform(0.8, 1.2, 3)
    p *= sc / 2
    # Strata: stepped radial inset with height (ledges), wavy so lines are not dead level.
    if strata > 0:
        layer = sc[2] / rng.uniform(3.5, 6.0)
        zz = p[:, 2] / layer + 0.35 * n3(p * 0.6 + off)
        frac = zz - np.floor(zz)
        # Sawtooth: each bed recesses gradually upward, then the next bed juts out (a ledge).
        inset = strata * sc.min() * np.clip((frac - 0.55) / 0.45, 0, 1) ** 2
        horiz = dirs.copy()
        horiz[:, 2] = 0
        horiz /= np.maximum(np.linalg.norm(horiz, axis=1, keepdims=True), 1e-6)
        p -= horiz * inset[:, None]
    # Surface detail: ridged cracks + fine fBm along the radial direction.
    q = p / np.maximum(sc.mean() / 2, 0.05)
    amp = sc.mean() / 2
    h = detail * n3.fbm(q * 3.0 + off, 5) + ridge * n3.ridged(q * 2.2 - off, 4)
    p += dirs * (h * amp)[:, None]
    if rounds:
        p = laplacian(p, F, rounds, 0.45)
    # Base: squash the lowest 15% so the rock sits on / sinks into the ground believably.
    zmin = p[:, 2].min()
    zcut = zmin + 0.15 * (p[:, 2].max() - zmin)
    low = p[:, 2] < zcut
    p[low, 2] = zcut - (zcut - p[low, 2]) * 0.25
    p[:, 2] -= p[:, 2].min()
    p[:, :2] -= (p[:, :2].max(0) + p[:, :2].min(0)) / 2
    return p, F


def make_rock(name, kind, rng, n3, material):
    V, F = rock_mesh(kind, rng, n3)
    me = mesh_from_numpy(name, V, F, materials=[material])
    box_uvs(me, 0.5)
    mark_sharp(me, 38.0 if kind in ("cliff", "mountain") else 50.0)
    ob = bpy.data.objects.new(name, me)
    return ob
