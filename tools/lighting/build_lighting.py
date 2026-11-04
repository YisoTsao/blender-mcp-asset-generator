"""Phase 6 entry point: lighting & atmosphere (dynamic sky, sun, clouds + cloud shadows,
volumetric height fog, AO / GI / reflections, weather) with six presets.

Usage (headless):
  blender -b --factory-startup --python tools/lighting/build_lighting.py -- \
      --env-dir output/phase5_environment --out output/phase6_lighting

Outputs in --out:
  forest_valley_lighting.blend   scene with the atmosphere rig; Text block 'lighting_presets.py'
                                 switches presets inside Blender
  lighting_presets.json          all presets + Unreal mapping (DirectionalLight rotation, lux,
                                 colour temperature, height fog, volumetric cloud, rain)
  preview_<preset>_<shot>.png, lighting_meta.json
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "terrain")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import atmosphere as AT  # noqa: E402
import presets as PR  # noqa: E402
from blend_util import save  # noqa: E402
from build_terrain import render, setup_render  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", default="output/phase5_environment")
    ap.add_argument("--out", default="output/phase6_lighting")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--presets", nargs="*", default=list(PR.PRESETS))
    ap.add_argument("--shots", nargs="*", default=None)
    return ap.parse_args(argv)


def configure_eevee(scene):
    ee = scene.eevee
    for attr, val in (("use_raytracing", True), ("use_fast_gi", True), ("use_shadows", True),
                      ("use_volumetric_shadows", True), ("volumetric_end", 1400.0), ("volumetric_start", 0.5),
                      ("volumetric_samples", 64), ("volumetric_tile_size", "8"), ("volumetric_light_clamp", 0.0),
                      ("shadow_ray_count", 2), ("shadow_step_count", 8), ("volumetric_shadow_samples", 16),
                      ("fast_gi_distance", 6.0), ("gi_diffuse_bounces", 2)):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except (TypeError, ValueError):
                pass
    if hasattr(ee, "ray_tracing_method"):
        ee.ray_tracing_method = "SCREEN"
    if hasattr(ee, "fast_gi_method"):
        try:
            ee.fast_gi_method = "GLOBAL_ILLUMINATION"
        except TypeError:
            pass
    if hasattr(ee, "use_volume_custom_range"):
        ee.use_volume_custom_range = True


def ue_mapping(p):
    """Unreal: X = Blender X, Y = -Blender Y (left-handed), Z up. Light points along its +X."""
    el, az = math.radians(p["sun_elev"]), math.radians(p["sun_az"])
    d = np.array([-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), -math.sin(el)])  # light travel dir
    d_ue = np.array([d[0], -d[1], d[2]])
    return {"directional_light_rotation_deg": {"pitch": round(math.degrees(math.asin(d_ue[2])), 2),
                                               "yaw": round(math.degrees(math.atan2(d_ue[1], d_ue[0])), 2), "roll": 0.0},
            **p["ue"]}


def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}
    edir = os.path.abspath(a.env_dir)
    with open(os.path.join(edir, "environment_layout.json")) as f:
        layout = json.load(f)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(edir, "forest_valley_environment.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    size_m = float(terrain.dimensions.x)
    rng = np.random.default_rng(6000)

    t = time.perf_counter()
    acol = bpy.data.collections.new("Atmosphere")
    scene.collection.children.link(acol)
    old_world = scene.world
    AT.build_world(scene)
    if old_world and old_world.users == 0:
        bpy.data.worlds.remove(old_world)
    AT.build_sun(scene)
    AT.build_clouds(scene, acol, size_m)
    AT.build_fog(scene, acol, size_m)
    AT.build_rain(scene, acol, rng)
    AT.build_night_lights(scene, layout)
    hooked = []
    for m in bpy.data.materials:
        if m.name.startswith(("M_Terrain", "M_Rock", "M_Road", "M_Trail", "M_Stone", "M_Wood", "M_Sandbag",
                              "M_Concrete", "M_HESCO", "M_Log")):
            if AT.hook_wetness(m, puddles=m.name.startswith(("M_Road", "M_Terrain")),
                               darken=0.6 if m.name.startswith("M_Terrain") else 0.5):
                hooked.append(m.name)
    setup_render(scene, a.samples)
    configure_eevee(scene)
    txt = bpy.data.texts.new("lighting_presets.py")
    with open(os.path.join(HERE, "presets.py")) as f:
        txt.write(f.read())
    PR.apply(scene, "noon")
    timings["rig"] = time.perf_counter() - t
    blend = os.path.join(out, "forest_valley_lighting.blend")
    save(blend)

    shots = [("Cam_River_Valley", "valley", 1600, 900), ("Cam_Bridge", "bridge", 1280, 720)]
    rain = bpy.data.objects["ATM_Rain"]
    rendered = []
    if not a.no_render:
        t = time.perf_counter()
        for name in a.presets:
            PR.apply(scene, name)
            for cam, tag, w, h in shots:
                if a.shots is not None and tag not in a.shots:
                    continue
                ob = bpy.data.objects.get(cam)
                if not ob:
                    continue
                rain.location = ob.matrix_world.translation
                fn = os.path.join(out, f"preview_{name}_{tag}.png")
                t0 = time.perf_counter()
                render(scene, ob, fn, w, h)
                rendered.append({"preset": name, "shot": tag, "seconds": round(time.perf_counter() - t0, 1)})
        timings["render"] = time.perf_counter() - t
        PR.apply(scene, "noon")
        rain.location = (0, 0, 0)
        save(blend)

    data = {n: {**{k: v for k, v in p.items() if k != "ue"}, "unreal": ue_mapping(p)} for n, p in PR.PRESETS.items()}
    with open(os.path.join(out, "lighting_presets.json"), "w") as f:
        json.dump({"coordinate_note": "Unreal X = Blender X, Unreal Y = -Blender Y, Z up; cm in UE",
                   "presets": data}, f, indent=2)
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {"presets": list(PR.PRESETS), "wetness_hooked_materials": hooked, "renders": rendered,
            "eevee": {k: getattr(scene.eevee, k) for k in ("use_raytracing", "use_fast_gi", "volumetric_end",
                                                         "volumetric_samples", "use_volumetric_shadows")
                      if hasattr(scene.eevee, k)},
            "timings_s": {k: round(v, 3) for k, v in timings.items()}}
    with open(os.path.join(out, "lighting_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("PHASE6_STATS", json.dumps(meta, default=str)[:3000])


if __name__ == "__main__":
    main()
