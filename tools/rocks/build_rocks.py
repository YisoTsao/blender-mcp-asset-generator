"""Phase 4 entry point: rock & geology on the Phase 3 scene.

Usage (headless):
  blender -b --factory-startup --python tools/rocks/build_rocks.py -- \
      --veg-dir output/phase3_vegetation --out output/phase4_rocks [--no-render] [--set field=value ...]

Outputs in --out:
  forest_valley_rocks.blend   Phase 3 scene + rock assets + GN rock instancers + cliff/scree terrain layers
  rock_assets.glb             all rock variants (box-projected UVs)
  rock_instances.json         transforms per rock type (UE: cm, deg) + moss / wet per instance
  scree_513.png               talus / scree mask (terrain material layer)
  preview_*.png, rock_meta.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "terrain"),
                os.path.join(HERE, "..", "water"), os.path.join(HERE, "..", "vegetation")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import rock_materials as RMAT  # noqa: E402
import rock_models as RM  # noqa: E402
import rock_placement as RP  # noqa: E402
from blend_util import (instancer_data, make_instancer, point_attr, rewrite_instancer, save,  # noqa: E402
                        terrain_arrays, terrain_attr, tri_count)
from views import ViewFinder  # noqa: E402
from build_terrain import look_at_rotation, make_camera, render, setup_render  # noqa: E402
from placement import Fields, VegConfig, canopy_map  # noqa: E402
from png_io import write_gray8  # noqa: E402
from water_core import upsample2_cubic  # noqa: E402

TREES = ("pine", "spruce", "oak", "maple", "alder", "birch", "dead")
COLL = {"boulder": "boulder", "moss": "boulder", "cliff": "cliff", "mountain": "mountain",
        "weathered": "weathered", "river": "river", "pebble": "pebble"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--veg-dir", default="output/phase3_vegetation")
    ap.add_argument("--water-dir", default="output/phase2_water")
    ap.add_argument("--out", default="output/phase4_rocks")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--set", nargs="*", default=[], help="RockConfig field=value")
    ap.add_argument("--shots", nargs="*", default=None, help="render only these preview names")
    return ap.parse_args(argv)


def build_assets(scene, seed):
    root = bpy.data.collections.new("ROCK_Assets")
    scene.collection.children.link(root)
    mats = {"rock": RMAT.rock_material("M_Rock"),
            "strata": RMAT.rock_material("M_Rock_Strata", (1.0, 0.97, 0.92), strata=True),
            "river": RMAT.rock_material("M_Rock_River", (1.05, 1.0, 0.92), smooth=True)}
    mat_for = {"boulder": "rock", "cliff": "strata", "mountain": "rock", "weathered": "strata",
               "river": "river", "pebble": "river"}
    n3 = RM.Noise3(seed + 4100)
    rng = np.random.default_rng(seed + 4200)
    colls, info, dims = {}, {}, {}
    for kind, nv in RM.VARIANTS.items():
        c = bpy.data.collections.new(f"ROCK_{kind}")
        root.children.link(c)
        info[kind], dims[kind] = [], []
        for v in range(nv):
            ob = RM.make_rock(f"rock_{kind}_{v:02d}", kind, rng, n3, mats[mat_for[kind]])
            c.objects.link(ob)
            info[kind].append({"name": ob.name, "tris": tri_count(ob.data),
                               "dims_m": [round(d, 2) for d in ob.dimensions]})
            dims[kind].append(np.array(ob.dimensions))
        colls[kind] = c
    dims["moss"] = dims["boulder"]
    return root, colls, info, dims, mats


def footprint_mask(groups, dims, size_m, cell=0.5, kinds=None, factor=0.42):
    res = int(size_m / cell) + 1
    m = np.zeros((res, res), bool)
    for kind, g in groups.items():
        if kinds and kind not in kinds:
            continue
        for p, s, v in zip(g["pos"], g["scale"], g["variant"]):
            r = factor * max(dims[kind][v][:2]) * s
            c0, r0 = int((p[0] + size_m / 2) / cell), int((p[1] + size_m / 2) / cell)
            n = int(r / cell) + 1
            rs, cs = np.mgrid[max(r0 - n, 0):min(r0 + n + 1, res), max(c0 - n, 0):min(c0 + n + 1, res)]
            inside = (cs - c0) ** 2 + (rs - r0) ** 2 <= (r / cell) ** 2
            m[rs[inside], cs[inside]] = True
    return m


def lookup(mask, size_m, x, y):
    res = mask.shape[0]
    cell = size_m / (res - 1)
    c = np.clip(np.round((x + size_m / 2) / cell).astype(int), 0, res - 1)
    r = np.clip(np.round((y + size_m / 2) / cell).astype(int), 0, res - 1)
    return mask[r, c]


def lineup_scene(colls, samples):
    """Studio scene: every rock variant in rows, for silhouette / material review."""
    sc = bpy.data.scenes.new("Rock_Lineup")
    setup_render(sc, samples)
    sc.world = bpy.data.worlds.get("World")
    col = bpy.data.collections.new("Lineup")
    sc.collection.children.link(col)
    ground = bpy.data.meshes.new("LineupGround")
    ground.from_pydata([(-40, -10, 0), (40, -10, 0), (40, 40, 0), (-40, 40, 0)], [], [(0, 1, 2, 3)])
    gm = bpy.data.materials.new("M_Lineup_Ground")
    gm.use_nodes = True
    next(n for n in gm.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = (0.12, 0.11, 0.1, 1)
    ground.materials.append(gm)
    col.objects.link(bpy.data.objects.new("LineupGround", ground))
    y = 0.0
    for kind in ("cliff", "mountain", "boulder", "weathered", "river", "pebble"):
        obs = sorted(colls[kind].objects, key=lambda o: o.name)
        mag = 6.0 if kind == "pebble" else 1.0
        w = sum(o.dimensions.x * mag for o in obs) + 1.2 * (len(obs) - 1)
        x = -w / 2
        depth = max(o.dimensions.y * mag for o in obs)
        for o in obs:
            inst = bpy.data.objects.new(o.name + "_L", o.data)
            inst.scale = (mag, mag, mag)
            inst.location = (x + o.dimensions.x * mag / 2, y + depth / 2, 0)
            inst["moss"] = 0.0
            col.objects.link(inst)
            x += o.dimensions.x * mag + 1.2
        y += depth + 1.5
    sun = bpy.data.objects.new("LineupSun", bpy.data.lights.new("LineupSun", "SUN"))
    sun.data.energy = 4.0
    sun.rotation_euler = (np.radians(50), 0, np.radians(-35))
    col.objects.link(sun)
    loc = (0, -26, 16)
    cam = make_camera(sc, "Cam_Lineup", loc, look_at_rotation(loc, (0, y * 0.42, 0)), lens=32)
    return sc, cam


def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    vdir = os.path.abspath(a.veg_dir)
    wdir = os.path.abspath(a.water_dir)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}
    with open(os.path.join(wdir, "water_bodies.json")) as f:
        wmeta = json.load(f)
    seed = wmeta["config"]["seed"]
    cfg = RP.RockConfig(seed=seed)
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    rng = np.random.default_rng(seed + 4000)

    t = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(vdir, "forest_valley_vegetation.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    H, res, size_m = terrain_arrays(terrain)
    wet, drainage = terrain_attr(terrain, "wet"), terrain_attr(terrain, "drainage")
    under = terrain_attr(terrain, "underwater")
    depth = terrain_attr(terrain, "water_depth")
    F = Fields(VegConfig(seed=seed), H, size_m, wet, drainage, under > 0.5)
    trees = {}
    for sp in TREES:
        ob = bpy.data.objects.get(f"VEG_Scatter_{sp}")
        if ob:
            d = instancer_data(ob)
            trees[sp] = {"pos": d["pos"], "scale": d["scale"]}
    canopy, _ = canopy_map(F, trees)
    trunks = np.concatenate([t["pos"][:, :2] for t in trees.values()])
    river = np.array([p["location_cm"] for p in wmeta["river_spline_points"]]) / 100
    rvel = np.array([p["velocity_cm_s"] for p in wmeta["river_spline_points"]]) / 100
    stream = [c for c in wmeta["channels"] if c["kind"] == "stream"]
    spts = np.array([p["location_cm"] for p in stream[0]["spline_points"]]) / 100 if stream else np.zeros((0, 3))
    RF = RP.RockField(F, canopy, depth[::2, ::2], under[::2, ::2], river, rvel, spts)
    timings["load_fields"] = time.perf_counter() - t

    t = time.perf_counter()
    root, colls, info, dims, mats = build_assets(scene, seed)
    timings["assets"] = time.perf_counter() - t
    t = time.perf_counter()
    bpy.ops.object.select_all(action="DESELECT")
    for c in colls.values():
        for ob in c.objects:
            ob.select_set(True)
    try:
        bpy.ops.export_scene.gltf(filepath=os.path.join(out, "rock_assets.glb"), use_selection=True, export_format="GLB")
    except Exception as exc:
        print("GLB_EXPORT_FAILED", exc)
    timings["asset_export"] = time.perf_counter() - t
    scene.view_layers[0].layer_collection.children[root.name].exclude = True

    t = time.perf_counter()
    groups, scree = RP.place_all(cfg, RF, rng, dims, trunks)
    timings["placement"] = time.perf_counter() - t

    # ---- instancers
    t = time.perf_counter()
    rcol = bpy.data.collections.new("Rocks")
    scene.collection.children.link(rcol)
    counts = {}
    for kind, g in groups.items():
        if not len(g["pos"]):
            continue
        ob, mod = make_instancer(f"ROCK_Scatter_{kind}", g, colls[COLL[kind]], rcol,
                                 extra={"moss": g["moss"], "wet": g["wet"]})
        if kind == "pebble":
            mod.show_viewport = False
        counts[kind] = int(len(g["pos"]))
    timings["instancers"] = time.perf_counter() - t

    # ---- clear vegetation that would be inside rocks (local regeneration of Phase 3 output)
    t = time.perf_counter()
    big = footprint_mask(groups, dims, size_m, kinds=("cliff", "mountain", "boulder", "weathered", "moss", "river"))
    trunk_mask = footprint_mask(groups, dims, size_m, kinds=("cliff",), factor=0.36)
    removed = {}
    for ob in [o for o in bpy.data.objects if o.name.startswith("VEG_Scatter_")]:
        sp = ob.name.split("_Scatter_")[-1]
        p = instancer_data(ob)["pos"]
        m = trunk_mask if sp in TREES else big
        drop = lookup(m, size_m, p[:, 0], p[:, 1])
        if drop.any():
            removed[sp] = rewrite_instancer(ob, keep=~drop)
    timings["veg_clear"] = time.perf_counter() - t

    # ---- terrain: scree attribute + cliff rock layer
    up = np.clip(upsample2_cubic(scree), 0, 1)[:res, :res]
    point_attr(terrain.data, "scree", up)
    RMAT.add_cliff_layer(terrain.data.materials[0])
    write_gray8(os.path.join(out, f"scree_{scree.shape[0]}.png"), (scree * 255).astype(np.uint8)[::-1])

    # ---- cameras
    def gz(x, y):
        return float(RF.h(np.array([x]), np.array([y]))[0])

    def cam_to(name, x, y, h, tx, ty, th, lens=24):
        loc = (x, y, gz(x, y) + h)
        return make_camera(scene, name, loc, look_at_rotation(loc, (tx, ty, gz(tx, ty) + th)), lens=lens)

    allr = np.concatenate([groups[k]["pos"][:, :2] for k in groups if k != "pebble"])
    allrad = np.concatenate([0.5 * np.array([max(dims[k][v][:2]) for v in groups[k]["variant"]]) * groups[k]["scale"]
                             for k in groups if k != "pebble"])
    bushes = instancer_data(bpy.data.objects["VEG_Scatter_bush"])["pos"][:, :2] \
        if bpy.data.objects.get("VEG_Scatter_bush") else np.zeros((0, 2))
    VF = ViewFinder(RF.h, np.concatenate([trunks, allr, bushes]),
                    np.concatenate([np.full(len(trunks), 2.0), allrad, np.full(len(bushes), 0.9)]),
                    lambda x, y: RF.s(canopy, x, y), size_m)

    def densest(p, r):
        cnt = np.array([(np.hypot(p[:, 0] - q[0], p[:, 1] - q[1]) < r).sum() for q in p])
        return p[cnt.argmax(), :2]

    def place(name, view, lens):
        (loc, tgt) = view
        return make_camera(scene, name, loc, look_at_rotation(loc, tgt), lens=lens)

    view, _ = VF.find(densest(groups["cliff"]["pos"], 25), 4.0, (45, 65, 85), (10, 22, 35))
    place("Cam_Cliffs", view, 32)
    view, _ = VF.find(densest(groups["boulder"]["pos"], 12), 0.8, (12, 18, 24), (2.0, 4.0, 7.0), look_down=True)
    place("Cam_Talus", view, 24)
    if len(groups.get("moss", {"pos": []})["pos"]):
        view, _ = VF.find(densest(groups["moss"]["pos"], 10), 0.3, (5, 7, 9), (1.6,))
        place("Cam_Moss_Rocks", view, 24)
    save(os.path.join(out, "forest_valley_rocks.blend"))

    if not a.no_render:
        t = time.perf_counter()
        setup_render(scene, a.samples)
        shots = [("Cam_River_Valley", "preview_valley_oblique", 1920, 1080),
                 ("Cam_Oblique", "preview_map_oblique", 1920, 1080),
                 ("Cam_Cliffs", "preview_cliffs", 1600, 900),
                 ("Cam_Talus", "preview_talus_eye", 1600, 900),
                 ("Cam_Moss_Rocks", "preview_moss_rocks", 1600, 900),
                 ("Cam_Rapid_Eye", "preview_river_rocks", 1600, 900),
                 ("Cam_Waterfall", "preview_waterfall", 1600, 900),
                 ("Cam_Forest_Eye", "preview_forest_eye", 1600, 900)]
        for cam, fname, w, h in shots:
            if a.shots is not None and fname not in a.shots:
                continue
            ob = bpy.data.objects.get(cam)
            if ob:
                render(scene, ob, os.path.join(out, fname + ".png"), w, h)
        ls, lcam = lineup_scene(colls, a.samples)
        if a.shots is not None and "preview_rock_lineup" not in a.shots:
            ls.render.resolution_percentage = 10
        ls.camera = lcam
        ls.render.resolution_x, ls.render.resolution_y = 1920, 1080
        ls.render.filepath = os.path.join(out, "preview_rock_lineup.png")
        bpy.ops.render.render(write_still=True, scene=ls.name)
        bpy.data.scenes.remove(ls)
        timings["render"] = time.perf_counter() - t
        save(os.path.join(out, "forest_valley_rocks.blend"))

    # ---- exports
    inst = {}
    for kind, g in groups.items():
        rows = np.column_stack([g["pos"] * 100, np.degrees(g["rot"]), g["scale"], g["variant"], g["moss"], g["wet"]])
        inst[kind] = {"assets": [x["name"] for x in info[COLL[kind]]],
                      "columns": ["x_cm", "y_cm", "z_cm", "rot_x_deg", "rot_y_deg", "rot_z_deg", "scale", "variant",
                                  "moss", "wet"],
                      "instances": np.round(rows, 3).tolist()}
    with open(os.path.join(out, "rock_instances.json"), "w") as f:
        json.dump(inst, f)

    # ---- validation
    checks = {}
    for kind in ("boulder", "moss", "weathered"):
        g = groups.get(kind)
        if g is not None and len(g["pos"]):
            sl = RF.s(F.slope, g["pos"][:, 0], g["pos"][:, 1])
            checks[f"{kind}_max_slope_deg"] = round(float(sl.max()), 1)
    tp = np.concatenate([g["pos"] for g in groups.values() if len(g["pos"])])
    checks["rocks_outside_map"] = int((np.abs(tp[:, :2]) > size_m / 2).sum())
    # Every rock sunk: origin (base) below ground.
    base_gap = tp[:, 2] - RF.h(tp[:, 0], tp[:, 1])
    checks["rock_bases_above_ground"] = int((base_gap > 0.05).sum())
    trunk_in = 0
    for kind in ("boulder", "mountain", "weathered", "moss", "river"):
        g = groups.get(kind)
        if g is None:
            continue
        for p, s, v in zip(g["pos"], g["scale"], g["variant"]):
            r = 0.35 * max(dims[kind][v][:2]) * s
            trunk_in += int((np.hypot(trunks[:, 0] - p[0], trunks[:, 1] - p[1]) < r).sum())
    checks["trunks_inside_rocks"] = trunk_in
    tris = {k: int(np.mean([x["tris"] for x in v])) for k, v in info.items()}
    stats = {"instances": counts, "instances_total": int(sum(counts.values())),
             "asset_variants": {k: len(v) for k, v in info.items()}, "asset_tris_mean": tris,
             "rendered_triangles_approx": int(sum(tris[COLL[k]] * n for k, n in counts.items())),
             "vegetation_removed_inside_rocks": removed, "scree_area_pct": round(float((scree > 0.3).mean() * 100), 1),
             "checks": checks}
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {"config": RP.config_dict(cfg), "stats": stats, "assets": info,
            "timings_s": {k: round(v, 3) for k, v in timings.items()}}
    with open(os.path.join(out, "rock_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("PHASE4_STATS", json.dumps({"stats": stats, "timings_s": meta["timings_s"]}))


if __name__ == "__main__":
    main()
