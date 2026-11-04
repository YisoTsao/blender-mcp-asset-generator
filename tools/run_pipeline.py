#!/usr/bin/env python3
"""Forest Valley build pipeline: runs each phase in Blender, in dependency order.

Each stage's fingerprint = hash(stage scripts + shared libs + stage args + upstream
fingerprint). A stage re-runs when its fingerprint changed or its outputs are
missing, so changing Phase 1 (code or parameters) automatically rebuilds every
phase downstream of it, and unchanged phases are skipped.

  python3 tools/run_pipeline.py                      # build what is stale
  python3 tools/run_pipeline.py --seed 7 --ranges 5  # new terrain -> water etc. rebuild
  python3 tools/run_pipeline.py --force water        # force one stage (and downstream)
  python3 tools/run_pipeline.py --dry-run            # show what would run
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
FINGERPRINT = ".stage_fingerprint.json"


def stages(a):
    """Ordered stage table. `deps` are upstream stage names."""
    out = a.out_root
    return [
        {
            "name": "terrain",
            "script": "tools/terrain/build_terrain.py",
            "code": ["tools/terrain"],
            "out": f"{out}/phase1_terrain",
            "args": ["--seed", str(a.seed), "--size", str(a.size), "--res", str(a.res),
                     "--height", str(a.height), "--ranges", str(a.ranges), "--detail", str(a.detail),
                     "--out", f"{out}/phase1_terrain"],
            "outputs": ["forest_valley_terrain.blend", "terrain_meta.json"],
            "deps": [],
        },
        {
            "name": "water",
            "script": "tools/water/build_water.py",
            "code": ["tools/water", "tools/terrain"],
            "out": f"{out}/phase2_water",
            "args": ["--terrain-dir", f"{out}/phase1_terrain", "--out", f"{out}/phase2_water"],
            "outputs": ["forest_valley_water.blend", "water_bodies.json"],
            "deps": ["terrain"],
        },
        {
            "name": "vegetation",
            "script": "tools/vegetation/build_vegetation.py",
            "code": ["tools/vegetation", "tools/water", "tools/terrain"],
            "out": f"{out}/phase3_vegetation",
            "args": ["--water-dir", f"{out}/phase2_water", "--out", f"{out}/phase3_vegetation"],
            "outputs": ["forest_valley_vegetation.blend", "vegetation_meta.json"],
            "deps": ["water"],
        },
        {
            "name": "rocks",
            "script": "tools/rocks/build_rocks.py",
            "code": ["tools/rocks", "tools/common", "tools/vegetation", "tools/water", "tools/terrain"],
            "out": f"{out}/phase4_rocks",
            "args": ["--veg-dir", f"{out}/phase3_vegetation", "--water-dir", f"{out}/phase2_water",
                     "--out", f"{out}/phase4_rocks"],
            "outputs": ["forest_valley_rocks.blend", "rock_meta.json"],
            "deps": ["vegetation"],
        },
        {
            "name": "environment",
            "script": "tools/environment/build_environment.py",
            "code": ["tools/environment", "tools/rocks", "tools/common", "tools/vegetation", "tools/water", "tools/terrain"],
            "out": f"{out}/phase5_environment",
            "args": ["--rocks-dir", f"{out}/phase4_rocks", "--water-dir", f"{out}/phase2_water",
                     "--out", f"{out}/phase5_environment"],
            "outputs": ["forest_valley_environment.blend", "environment_meta.json", "environment_layout.json"],
            "deps": ["rocks"],
        },
        {
            "name": "lighting",
            "script": "tools/lighting/build_lighting.py",
            "code": ["tools/lighting", "tools/rocks", "tools/common", "tools/terrain"],
            "out": f"{out}/phase6_lighting",
            "args": ["--env-dir", f"{out}/phase5_environment", "--out", f"{out}/phase6_lighting"],
            "outputs": ["forest_valley_lighting.blend", "lighting_presets.json", "lighting_meta.json"],
            "deps": ["environment"],
        },
        {
            "name": "gameplay",
            "script": "tools/gameplay/build_gameplay.py",
            "code": ["tools/gameplay", "tools/common", "tools/terrain"],
            "out": f"{out}/phase7_gameplay",
            "args": ["--scene-dir", f"{out}/phase6_lighting", "--env-dir", f"{out}/phase5_environment",
                     "--out", f"{out}/phase7_gameplay"],
            "outputs": ["forest_valley_gameplay.blend", "gameplay_metadata.json"],
            "deps": ["lighting"],
        },
        {
            "name": "export",
            "script": "tools/export/build_export.py",
            "code": ["tools/export", "tools/common", "tools/terrain"],
            "out": f"{out}/phase8_export",
            "args": ["--root", out, "--out", f"{out}/phase8_export"],
            "outputs": ["forest_valley_final.blend", "performance_report.json", "ue_export/ue_import.py"],
            "deps": ["gameplay"],
        },
    ]


def hash_code(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        full = os.path.join(ROOT, p)
        files = [full] if os.path.isfile(full) else sorted(
            os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs if f.endswith(".py"))
        for f in files:
            h.update(os.path.relpath(f, ROOT).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def fingerprint(st, upstream_fps, render):
    h = hashlib.sha256()
    h.update(hash_code(st["code"]).encode())
    h.update(json.dumps(st["args"]).encode())
    h.update(str(render).encode())
    for fp in upstream_fps:
        h.update(fp.encode())
    return h.hexdigest()[:16]


def read_fp(st):
    try:
        with open(os.path.join(ROOT, st["out"], FINGERPRINT)) as f:
            return json.load(f).get("fingerprint")
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=float, default=500.0)
    ap.add_argument("--res", type=int, default=513)
    ap.add_argument("--height", type=float, default=110.0)
    ap.add_argument("--ranges", type=int, default=4)
    ap.add_argument("--detail", type=float, default=1.0)
    ap.add_argument("--out-root", default="output")
    ap.add_argument("--only", nargs="*", help="limit to these stages (deps must already be built)")
    ap.add_argument("--force", nargs="*", default=[], help="force these stages (and downstream)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    table = stages(a)
    fps = {}
    forced = set(a.force)
    report = []
    for st in table:
        if not os.path.exists(os.path.join(ROOT, st["script"])):
            report.append((st["name"], "missing script (not implemented yet)"))
            continue
        fp = fingerprint(st, [fps[d] for d in st["deps"] if d in fps], not a.no_render)
        fps[st["name"]] = fp
        if a.only and st["name"] not in a.only:
            continue
        have = all(os.path.exists(os.path.join(ROOT, st["out"], o)) for o in st["outputs"])
        upstream_forced = any(d in forced for d in st["deps"])
        stale = (read_fp(st) != fp) or not have or st["name"] in forced or upstream_forced
        if upstream_forced or st["name"] in forced:
            forced.add(st["name"])
        if not stale:
            report.append((st["name"], f"up to date ({fp})"))
            continue
        reason = "forced" if st["name"] in forced else ("outputs missing" if not have else "inputs changed")
        cmd = [BLENDER, "-b", "--factory-startup", "--python", st["script"], "--"] + st["args"]
        if a.no_render:
            cmd.append("--no-render")
        print(f"[{st['name']}] {reason} -> running", flush=True)
        if a.dry_run:
            report.append((st["name"], f"would run ({reason})"))
            continue
        t = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        dt = time.perf_counter() - t
        ok = proc.returncode == 0 and "Traceback" not in proc.stderr + proc.stdout and all(
            os.path.exists(os.path.join(ROOT, st["out"], o)) for o in st["outputs"])
        if not ok:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
            print(f"[{st['name']}] FAILED after {dt:.1f}s\n{tail}")
            sys.exit(1)
        with open(os.path.join(ROOT, st["out"], FINGERPRINT), "w") as f:
            json.dump({"fingerprint": fp, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "seconds": round(dt, 1), "args": st["args"]}, f, indent=2)
        report.append((st["name"], f"built in {dt:.1f}s ({fp})"))
    print("\nPipeline summary:")
    for name, msg in report:
        print(f"  {name:<11} {msg}")


if __name__ == "__main__":
    main()
