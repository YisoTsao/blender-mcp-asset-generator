"""Phase 3 entry point: temperate-forest vegetation on the Phase 2 scene.

Usage (headless):
  blender -b --factory-startup --python tools/vegetation/build_vegetation.py -- \
      --water-dir output/phase2_water --out output/phase3_vegetation [--no-render]

Outputs in --out:
  forest_valley_vegetation.blend   Phase 2 scene + plant assets + GN scatter instancers
  vegetation_assets.glb            all plant asset variants (for engine import / replacement)
  vegetation_instances.json        trees, logs, bushes, ferns: transforms (UE units: cm, deg)
  density_<layer>_<N>.png          grass / flowers / fern / bush / canopy / litter / moss maps
  forest_zones_<N>.png             forest stand per cell (0 = water, 1.. = PL.ZONE_NAMES order)
  textures/T_*.png                 generated foliage textures
  preview_*.png, vegetation_meta.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "terrain"), os.path.join(HERE, "..", "water")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import placement as PL  # noqa: E402
import plant_models as PM  # noqa: E402
import textures as TX  # noqa: E402
import veg_materials as VM  # noqa: E402
from build_terrain import look_at_rotation, make_camera, render, setup_render  # noqa: E402
from build_water import configure_raytracing, sky_world  # noqa: E402
from png_io import write_gray8  # noqa: E402
from water_core import bilinear, upsample2_cubic  # noqa: E402

RENDER_ONLY = ("grass", "flowers", "fern")      # too many instances for a responsive viewport


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--water-dir", default="output/phase2_water")
    ap.add_argument("--out", default="output/phase3_vegetation")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--set", nargs="*", default=[], help="VegConfig field=value")
    return ap.parse_args(argv)


def read_attr(me, name, res):
    v = np.empty(len(me.vertices), np.float32)
    me.attributes[name].data.foreach_get("value", v)
    return v.reshape(res, res).astype(np.float64)


def point_attr(me, name, values, kind="FLOAT"):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name=name, type=kind, domain="POINT")
    key = {"FLOAT_VECTOR": "vector", "INT": "value"}.get(kind, "value")
    a.data.foreach_set(key, np.asarray(values, np.int32 if kind == "INT" else np.float32).ravel())


# --------------------------------------------------------------------------- assets

def build_assets(scene, mats, rng):
    root = bpy.data.collections.new("VEG_Assets")
    scene.collection.children.link(root)
    colls, info = {}, {}
    for sp, (fn, bark_k, fol_k, nvar) in PM.SPECIES.items():
        c = bpy.data.collections.new(f"VEG_{sp}")
        root.children.link(c)
        m0 = mats[bark_k] if bark_k else mats[fol_k]
        m1 = mats[fol_k] if fol_k else mats[bark_k]
        slots = [m0, m1, mats["bark_dead"]]          # slot 2: dark twigs (birch)
        info[sp] = []
        for v in range(nvar):
            ob = fn(f"{sp}_{v:02d}", rng, slots)
            c.objects.link(ob)
            info[sp].append({"name": ob.name, "faces": len(ob.data.polygons),
                             "dims_m": [round(d, 2) for d in ob.dimensions]})
        colls[sp] = c
    return root, colls, info


def make_instancer(name, data, src_coll, target):
    n = len(data["pos"])
    me = bpy.data.meshes.new(name)
    me.vertices.add(n)
    me.vertices.foreach_set("co", np.asarray(data["pos"], np.float32).ravel())
    point_attr(me, "rot", data["rot"], "FLOAT_VECTOR")
    point_attr(me, "scale", data["scale"])
    point_attr(me, "variant", data["variant"], "INT")
    ob = bpy.data.objects.new(name, me)
    target.objects.link(ob)
    ng = bpy.data.node_groups.new(name + "_GN", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    iop = N.new("GeometryNodeInstanceOnPoints")
    ci = N.new("GeometryNodeCollectionInfo")
    ci.transform_space = "ORIGINAL"
    ci.inputs["Collection"].default_value = src_coll
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True

    def named(attr, dtype):
        a = N.new("GeometryNodeInputNamedAttribute")
        a.data_type = dtype
        a.inputs["Name"].default_value = attr
        return a.outputs["Attribute"]

    L.new(gi.outputs[0], iop.inputs["Points"])
    L.new(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    L.new(named("variant", "INT"), iop.inputs["Instance Index"])
    try:
        e2r = N.new("FunctionNodeEulerToRotation")
        L.new(named("rot", "FLOAT_VECTOR"), e2r.inputs[0])
        L.new(e2r.outputs[0], iop.inputs["Rotation"])
    except RuntimeError:
        L.new(named("rot", "FLOAT_VECTOR"), iop.inputs["Rotation"])
    L.new(named("scale", "FLOAT"), iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs[0])
    mod = ob.modifiers.new("Scatter", "NODES")
    mod.node_group = ng
    return ob, mod


def add_ground_layers(mat):
    """Leaf litter, needle litter and moss on the terrain (never on the wet bed)."""
    nt = mat.node_tree
    N, Lk = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    prev = bsdf.inputs["Base Color"].links[0].from_socket

    def attr(name):
        a = N.new("ShaderNodeAttribute")
        a.attribute_name = name
        return a.outputs["Fac"]

    dry = N.new("ShaderNodeMath")
    dry.operation = "SUBTRACT"
    dry.inputs[0].default_value = 1.0
    Lk.new(attr("wet"), dry.inputs[1])
    # Two-scale variation: metre-scale patches + fine leaf/twig speckle for close-ups.
    n1 = N.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 0.6
    n1.inputs["Detail"].default_value = 8.0
    n2 = N.new("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 9.0
    n2.inputs["Detail"].default_value = 12.0
    n2.inputs["Roughness"].default_value = 0.7
    mixn = N.new("ShaderNodeMix")
    mixn.data_type = "FLOAT"
    mixn.inputs["Factor"].default_value = 0.5
    Lk.new(n1.outputs["Fac"], mixn.inputs[2])
    Lk.new(n2.outputs["Fac"], mixn.inputs[3])
    noise_out = mixn.outputs[0]

    def layer(base, fac_attr, strength, stops):
        r = N.new("ShaderNodeValToRGB")
        e = r.color_ramp.elements
        e[0].position, e[0].color = stops[0][0], (*stops[0][1], 1)
        e[1].position, e[1].color = stops[1][0], (*stops[1][1], 1)
        Lk.new(noise_out, r.inputs["Fac"])
        f = N.new("ShaderNodeMath")
        f.operation = "MULTIPLY"
        Lk.new(attr(fac_attr), f.inputs[0])
        Lk.new(dry.outputs[0], f.inputs[1])
        f2 = N.new("ShaderNodeMath")
        f2.operation = "MULTIPLY"
        f2.use_clamp = True
        f2.inputs[1].default_value = strength
        Lk.new(f.outputs[0], f2.inputs[0])
        m = N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        Lk.new(f2.outputs[0], m.inputs["Factor"])
        Lk.new(base, m.inputs[6])
        Lk.new(r.outputs["Color"], m.inputs[7])
        return m.outputs[2]

    c = layer(prev, "grass_cover", 0.9, [(0.3, (0.045, 0.075, 0.018)), (0.7, (0.10, 0.13, 0.035))])
    c = layer(c, "litter_leaf", 0.95, [(0.35, (0.10, 0.05, 0.018)), (0.7, (0.30, 0.16, 0.06))])
    c = layer(c, "litter_needle", 0.9, [(0.35, (0.07, 0.035, 0.015)), (0.7, (0.20, 0.10, 0.045))])
    c = layer(c, "moss", 0.8, [(0.3, (0.025, 0.05, 0.012)), (0.75, (0.07, 0.11, 0.025))])
    Lk.new(c, bsdf.inputs["Base Color"])


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    wdir = os.path.abspath(a.water_dir)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}
    with open(os.path.join(wdir, "water_bodies.json")) as f:
        wmeta = json.load(f)
    cfg = PL.VegConfig(seed=wmeta["config"]["seed"])
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    rng = np.random.default_rng(cfg.seed + 1300)

    t = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(wdir, "forest_valley_water.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    me = terrain.data
    res = int(round(np.sqrt(len(me.vertices))))
    size_m = float(terrain.dimensions.x)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    H = co.reshape(-1, 3)[:, 2].reshape(res, res).astype(np.float64)
    wet, drainage = read_attr(me, "wet", res), read_attr(me, "drainage", res)
    water = read_attr(me, "underwater", res) > 0.5
    F = PL.Fields(cfg, H, size_m, wet, drainage, water)
    timings["load_fields"] = time.perf_counter() - t

    # ---- textures, materials, assets
    t = time.perf_counter()
    tex = TX.generate_all(os.path.join(out, "textures"), cfg.seed)
    mats = VM.build_all(tex)
    root, colls, asset_info = build_assets(scene, mats, np.random.default_rng(cfg.seed + 1400))
    timings["assets"] = time.perf_counter() - t
    t = time.perf_counter()
    bpy.ops.object.select_all(action="DESELECT")
    for c in colls.values():
        for ob in c.objects:
            ob.select_set(True)
    try:
        bpy.ops.export_scene.gltf(filepath=os.path.join(out, "vegetation_assets.glb"), use_selection=True,
                                  export_format="GLB")
    except Exception as exc:
        print("GLB_EXPORT_FAILED", exc)
    timings["asset_export"] = time.perf_counter() - t
    # Keep the source assets out of the rendered scene (instancers reference them).
    scene.view_layers[0].layer_collection.children[root.name].exclude = True

    # ---- placement
    t = time.perf_counter()
    variants = {sp: v[3] for sp, v in PM.SPECIES.items()}
    trees, thash = PL.place_trees(cfg, F, rng, variants)
    log_len = [float(o.dimensions.x) for o in sorted(colls["log"].objects, key=lambda o: o.name)]
    logs = PL.place_logs(cfg, F, rng, thash, log_len)
    canopy, cov = PL.canopy_map(F, trees)
    ground = PL.place_ground(cfg, F, rng, canopy, thash, variants)
    layers = PL.terrain_layers(F, canopy, cov, (ground["grass"]["pos"][:, 0], ground["grass"]["pos"][:, 1]))
    timings["placement"] = time.perf_counter() - t

    # ---- instancers
    t = time.perf_counter()
    vcol = bpy.data.collections.new("Vegetation")
    scene.collection.children.link(vcol)
    groups = dict(trees)
    groups["log"] = logs
    groups.update(ground)
    counts = {}
    for sp, d in groups.items():
        if len(d["pos"]) == 0:
            continue
        ob, mod = make_instancer(f"VEG_Scatter_{sp}", d, colls[sp], vcol)
        if sp in RENDER_ONLY:
            mod.show_viewport = False
        counts[sp] = int(len(d["pos"]))
    timings["instancers"] = time.perf_counter() - t

    # ---- terrain ground layers
    t = time.perf_counter()
    for name, v in layers.items():
        up = np.clip(upsample2_cubic(v), 0, 1)[:res, :res]
        point_attr(me, name, up)
    add_ground_layers(me.materials[0])
    timings["ground_layers"] = time.perf_counter() - t

    # ---- maps
    res_f = F.H.shape[0]
    dmaps = {"canopy": canopy, "moss": layers["moss"], "litter_leaf": cov["leaf"], "litter_needle": cov["needle"]}
    for sp in ("grass", "flowers", "fern", "bush"):
        g = np.zeros((res_f, res_f))
        if counts.get(sp):
            p = groups[sp]["pos"]
            c = np.clip(((p[:, 0] + size_m / 2) / F.cell).astype(int), 0, res_f - 1)
            r = np.clip(((p[:, 1] + size_m / 2) / F.cell).astype(int), 0, res_f - 1)
            np.add.at(g, (r, c), 1.0)
            g = g / max(np.percentile(g[g > 0], 99), 1e-6)
        dmaps[sp] = np.clip(g, 0, 1)
    for name, m in dmaps.items():
        write_gray8(os.path.join(out, f"density_{name}_{res_f}.png"), (np.clip(m, 0, 1) * 255).astype(np.uint8)[::-1])
    write_gray8(os.path.join(out, f"forest_zones_{res_f}.png"), F.zone[::-1].copy())

    # ---- stand statistics: area share and trees per species in each stand
    zone_stats = {}
    for i, zname in enumerate(PL.ZONE_NAMES):
        mask = F.zone == i + 1
        per = {}
        for sp in PL.TREE_SPECIES:
            if sp in groups and len(groups[sp]["pos"]):
                p = groups[sp]["pos"]
                z = F.zone[np.clip(((p[:, 1] + size_m / 2) / F.cell).astype(int), 0, res_f - 1),
                           np.clip(((p[:, 0] + size_m / 2) / F.cell).astype(int), 0, res_f - 1)]
                if (n := int((z == i + 1).sum())):
                    per[sp] = n
        zone_stats[zname] = {"id": i + 1, "label": PL.ZONES[zname]["label"],
                             "area_pct": round(float(mask.mean() * 100), 1), "trees": per}

    # ---- cameras
    setup_render(scene, a.samples)
    sky_world(scene)
    configure_raytracing(scene)
    cams = {}
    ground_z = lambda x, y: float(bilinear(F.H, size_m, np.array([x]), np.array([y]))[0])   # noqa: E731

    def eye_cam(name, x, y, tx, ty, h=1.7, lens=24, look_up=1.0):
        z = ground_z(x, y) + h
        tz = ground_z(tx, ty) + h * look_up
        loc = (x, y, z)
        return make_camera(scene, name, loc, look_at_rotation(loc, (tx, ty, tz)), lens=lens)

    tree_xy = np.concatenate([groups[s]["pos"][:, :2] for s in PL.TREE_SPECIES if s in groups])

    def clear_of_trunks(x, y, r=3.0):
        return np.min(np.hypot(tree_xy[:, 0] - x, tree_xy[:, 1] - y)) > r

    def best_view(x, y, want="trees"):
        """Scan 36 headings: 'trees' = many trunks 8-45 m ahead on level-ish ground;
        'mountains' = high terrain 120-200 m ahead with open ground in front."""
        z0 = ground_z(x, y)
        best, best_a = -1e9, 0.0
        rel = tree_xy - [x, y]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        for a in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            dv = np.array([np.cos(a), np.sin(a)])
            dz = ground_z(x + dv[0] * 20, y + dv[1] * 20) - z0
            infront = (np.abs((ang - a + np.pi) % (2 * np.pi) - np.pi) < np.radians(25))
            near = (infront & (dist < 6)).sum()
            if want == "trees":
                sc = (infront & (dist > 8) & (dist < 45)).sum() - 3 * abs(dz) - 20 * near
            else:
                far = [ground_z(x + dv[0] * d, y + dv[1] * d) for d in (120, 160, 200)]
                sc = max(far) - z0 - 8 * near - 0.5 * (infront & (dist < 30)).sum()
            if sc > best:
                best, best_a = sc, a
        return best_a

    # Forest interior: dense canopy, walkable, mid-slope, not hugging a trunk.
    score = canopy * (F.slope < 14) * (F.rel > 0.1) * (F.rel < 0.45) * F.dry
    for idx in np.argsort(score.ravel())[::-1][:400]:
        r, c = divmod(int(idx), score.shape[1])
        fx, fy = F.X[r, c], F.Y[r, c]
        if clear_of_trunks(fx, fy):
            break
    a_ = best_view(fx, fy, "trees")
    cams["forest"] = eye_cam("Cam_Forest_Eye", fx, fy, fx + np.cos(a_) * 30, fy + np.sin(a_) * 30, lens=22)
    # Riverside: on the bank, looking upstream along the river.
    sp_pts = wmeta["river_spline_points"]
    i = len(sp_pts) // 3
    p0 = np.array(sp_pts[i]["location_cm"][:2]) / 100
    p1 = np.array(sp_pts[i + 6]["location_cm"][:2]) / 100
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    side = np.array([-d[1], d[0]])
    bx, by = p0 + side * (sp_pts[i]["width_cm"] / 200 + 4.0)
    cams["river"] = eye_cam("Cam_Riverside_Eye", bx, by, p1[0] + d[0] * 30, p1[1] + d[1] * 30, lens=22, look_up=0.5)
    # Meadow: open valley floor (low canopy over a 15 m radius), looking at the mountains.
    from terrain_core import gaussian_blur as _gb
    open_ = 1 - np.clip(_gb(canopy, int(15 / F.cell)) * 3, 0, 1)
    score = open_ * (F.rel < 0.15) * (F.slope < 8) * F.dry * (np.abs(F.Y) < 170) * (np.abs(F.X) < 170)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    mx, my = F.X[r, c], F.Y[r, c]
    a_ = best_view(mx, my, "mountains")
    cams["meadow"] = eye_cam("Cam_Meadow_Eye", mx, my, mx + np.cos(a_) * 60, my + np.sin(a_) * 60, lens=24, look_up=4)
    blend_path = os.path.join(out, "forest_valley_vegetation.blend")
    scene.camera = bpy.data.objects.get("Cam_River_Valley") or cams["forest"]
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)

    if not a.no_render:
        t = time.perf_counter()
        shots = [("Cam_River_Valley", "preview_valley_oblique", 1920, 1080),
                 ("Cam_Oblique", "preview_map_oblique", 1920, 1080),
                 ("Cam_Top", "preview_top", 1400, 1400),
                 ("Cam_Forest_Eye", "preview_forest_eye", 1600, 900),
                 ("Cam_Riverside_Eye", "preview_riverside_eye", 1600, 900),
                 ("Cam_Meadow_Eye", "preview_meadow_eye", 1600, 900),
                 ("Cam_Waterfall", "preview_waterfall", 1600, 900)]
        for cam, fname, w, h in shots:
            ob = bpy.data.objects.get(cam)
            if ob:
                render(scene, ob, os.path.join(out, fname + ".png"), w, h)
        timings["render"] = time.perf_counter() - t
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    if os.path.exists(blend_path + "1"):
        os.remove(blend_path + "1")

    # ---- instance export (UE: cm, degrees, Z-up; yaw about Z)
    inst = {}
    for sp in (*PL.TREE_SPECIES, "log", "bush", "fern"):
        if sp not in groups or not len(groups[sp]["pos"]):
            continue
        g = groups[sp]
        rows = np.column_stack([g["pos"] * 100, np.degrees(g["rot"]), g["scale"], g["variant"]])
        inst[sp] = {"assets": [x["name"] for x in asset_info[sp]],
                    "columns": ["x_cm", "y_cm", "z_cm", "rot_x_deg", "rot_y_deg", "rot_z_deg", "scale", "variant"],
                    "instances": np.round(rows, 2).tolist()}
    with open(os.path.join(out, "vegetation_instances.json"), "w") as f:
        json.dump(inst, f)

    # ---- validation + stats
    def min_trunk_spacing():
        pts = np.concatenate([groups[s]["pos"][:, :2] for s in PL.TREE_SPECIES if s in groups])
        best = np.inf
        cell = {}
        for i, (x, y) in enumerate(pts):
            k = (int(x // 3), int(y // 3))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for j in cell.get((k[0] + di, k[1] + dj), ()):
                        best = min(best, np.hypot(*(pts[j] - (x, y))))
            cell.setdefault(k, []).append(i)
        return float(best)

    tree_pts = np.concatenate([groups[s]["pos"] for s in PL.TREE_SPECIES if s in groups])
    sl_at = F.sample(F.slope, tree_pts[:, 0], tree_pts[:, 1])
    wet_at = F.sample(F.near_water.astype(float), tree_pts[:, 0], tree_pts[:, 1])
    faces = {sp: int(np.mean([x["faces"] for x in asset_info[sp]])) for sp in asset_info}
    tris = int(sum(2 * faces[sp] * n for sp, n in counts.items()))
    area_ha = size_m * size_m / 1e4
    stats = {
        "instances": counts,
        "trees_total": int(sum(counts.get(s, 0) for s in PL.TREE_SPECIES)),
        "trees_per_ha": round(sum(counts.get(s, 0) for s in PL.TREE_SPECIES) / area_ha, 1),
        "zones": zone_stats,
        "asset_variants": {sp: len(v) for sp, v in asset_info.items()},
        "asset_faces_mean": faces,
        "rendered_triangles_approx": tris,
        "canopy_cover_pct": round(float(canopy.mean() * 100), 1),
        "checks": {
            "min_trunk_spacing_m": round(min_trunk_spacing(), 2),
            "trees_on_slope_gt_max": int((sl_at > cfg.max_tree_slope_deg + 8 * cfg.mountain_forest + 1).sum()),
            "trees_in_or_at_water": int((wet_at > 0.5).sum()),
            "max_tree_slope_deg_found": round(float(sl_at.max()), 1),
        },
    }
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {"config": PL.config_dict(cfg), "stats": stats, "assets": asset_info,
            "zones": {"map": f"forest_zones_{res_f}.png", "values": {"water": 0, **{z: i + 1 for i, z in enumerate(PL.ZONE_NAMES)}},
                      "stands": PL.ZONES},
            "timings_s": {k: round(v, 3) for k, v in timings.items()}, "water_source": os.path.relpath(wdir, out)}
    with open(os.path.join(out, "vegetation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("PHASE3_STATS", json.dumps({"stats": stats, "timings_s": meta["timings_s"]}))


if __name__ == "__main__":
    main()
