"""Vectorised 2D gradient noise (Perlin-style) and fractal combinators.

Pure numpy, no bpy dependency, so it can be unit-tested inside or outside Blender.
Every octave is rotated and offset by a seed-derived amount to suppress the
axis-aligned lattice artefacts that make plain fBm look like a "noise carpet".
"""
import numpy as np


class GradientNoise2D:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.perm = np.tile(rng.permutation(256), 2).astype(np.int64)
        ang = rng.uniform(0.0, 2.0 * np.pi, 256)
        self.gx = np.cos(ang)
        self.gy = np.sin(ang)

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def _grad(self, ix, iy, dx, dy):
        h = self.perm[self.perm[ix] + iy]
        return self.gx[h] * dx + self.gy[h] * dy

    def __call__(self, x, y):
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = x - x0
        fy = y - y0
        ix = x0.astype(np.int64) & 255
        iy = y0.astype(np.int64) & 255
        ix1 = (ix + 1) & 255
        iy1 = (iy + 1) & 255
        n00 = self._grad(ix, iy, fx, fy)
        n10 = self._grad(ix1, iy, fx - 1.0, fy)
        n01 = self._grad(ix, iy1, fx, fy - 1.0)
        n11 = self._grad(ix1, iy1, fx - 1.0, fy - 1.0)
        u = self._fade(fx)
        v = self._fade(fy)
        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        # Raw range is about +-0.7; rescale to roughly +-1.
        return (nx0 + v * (nx1 - nx0)) * 1.42


class Fractal:
    """fBm / ridged multifractal / domain warp built on one GradientNoise2D."""

    def __init__(self, seed, max_octaves=12):
        rng = np.random.default_rng(seed + 7919)
        self.noise = GradientNoise2D(seed)
        self.rot = rng.uniform(0.0, 2.0 * np.pi, max_octaves)
        self.off = rng.uniform(-1000.0, 1000.0, (max_octaves, 2))

    def _octave(self, o, x, y, freq):
        c, s = np.cos(self.rot[o]), np.sin(self.rot[o])
        xr = (x * c - y * s) * freq + self.off[o, 0]
        yr = (x * s + y * c) * freq + self.off[o, 1]
        return self.noise(xr, yr)

    def fbm(self, x, y, base_freq, octaves, lacunarity=2.0, gain=0.5):
        total = np.zeros_like(x)
        amp, freq, norm = 1.0, base_freq, 0.0
        for o in range(octaves):
            total += amp * self._octave(o, x, y, freq)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def ridged(self, x, y, base_freq, octaves, lacunarity=2.0, gain=2.0,
               offset=1.0, h=1.0):
        """Musgrave ridged multifractal, output roughly 0..1.

        Each octave is weighted by the previous one, so fine ridges only appear
        on top of coarse ridges, giving the dendritic spur pattern of real ranges.
        """
        total = np.zeros_like(x)
        weight = np.ones_like(x)
        freq, norm = base_freq, 0.0
        for o in range(octaves):
            sig = offset - np.abs(self._octave(o, x, y, freq))
            sig = sig * sig * weight
            weight = np.clip(sig * gain, 0.0, 1.0)
            w = (freq / base_freq) ** (-h)
            total += sig * w
            norm += w
            freq *= lacunarity
        return total / norm

    def warp(self, x, y, base_freq, strength, octaves=4):
        wx = self.fbm(x + 31.7, y - 12.3, base_freq, octaves)
        wy = self.fbm(x - 54.1, y + 77.9, base_freq, octaves)
        return x + strength * wx, y + strength * wy
