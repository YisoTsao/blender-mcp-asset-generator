"""Fast iteration: generate heightfield and write a hillshade PNG (no mesh build).

blender -b --python quick_preview.py -- --seed 42 --out /path/preview.png [--erosion 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from png_io import hillshade_rgb, write_rgb8  # noqa: E402
from terrain_core import TerrainConfig, generate  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ranges", type=int, default=4)
ap.add_argument("--erosion", type=float, default=1.0)
ap.add_argument("--detail", type=float, default=1.0)
ap.add_argument("--out", required=True)
ap.add_argument("--set", nargs="*", default=[], help="field=value overrides")
a = ap.parse_args(argv)

cfg = TerrainConfig(seed=a.seed, mountain_ranges=a.ranges, erosion=a.erosion, detail=a.detail)
for kv in a.set:
    k, v = kv.split("=")
    setattr(cfg, k, type(getattr(cfg, k))(v))
r = generate(cfg)
cell = cfg.size_m / (cfg.resolution - 1)
write_rgb8(a.out, hillshade_rgb(r.heights, cell, r.river_mask))
gy, gx = np.gradient(r.heights, cell)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
print("RESULT min %.1f max %.1f mean %.1f | slope p50 %.1f p90 %.1f p99 %.1f | %s" % (
    r.heights.min(), r.heights.max(), r.heights.mean(),
    *np.percentile(slope, [50, 90, 99]),
    {k: round(v, 2) for k, v in r.timings.items()}))
