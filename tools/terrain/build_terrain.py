"""Phase 1 entry point: generate the Forest Valley terrain inside Blender.

Usage (headless):
  blender -b --factory-startup --python tools/terrain/build_terrain.py -- \
      --seed 42 --size 500 --res 513 --height 110 --ranges 4 --detail 1.0 \
      --out output/phase1_terrain [--no-render]

Outputs in --out:
  forest_valley_terrain.blend   editable mesh "Terrain" + guide curves + cameras
  heightmap_513_16bit.png       16-bit grayscale, top = north, linear 0..65535
  heightmap_513.r16             same data, little-endian raw (Unreal import)
  preview_top.png / preview_oblique.png / preview_hillshade.png
  terrain_meta.json             config, height range, stats, timings, river + spines
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from png_io import heightmap_u16, hillshade_rgb, read_gray16, write_gray16, write_rgb8  # noqa: E402
from terrain_core import TerrainConfig, config_dict, generate  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=float, default=500.0)
    ap.add_argument("--res", type=int, default=513)
    ap.add_argument("--height", type=float, default=110.0)
    ap.add_argument("--ranges", type=int, default=4)
    ap.add_argument("--detail", type=float, default=1.0)
    ap.add_argument("--erosion", type=float, default=1.0)
    ap.add_argument("--out", default="output/phase1_terrain")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[], help="extra TerrainConfig field=value")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- scene

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def build_mesh(heights, size_m, name="Terrain"):
    res = heights.shape[0]
    half = size_m / 2.0
    lin = np.linspace(-half, half, res)
    X, Y = np.meshgrid(lin, lin)
    verts = np.column_stack([X.ravel(), Y.ravel(), heights.ravel()]).astype(np.float32)
    idx = np.arange(res * res).reshape(res, res)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    quads = np.column_stack([a, b, c, d])  # CCW seen from +Z -> normals up
    nq = quads.shape[0]

    me = bpy.data.meshes.new(name)
    me.vertices.add(res * res)
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(nq * 4)
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.add(nq)
    me.polygons.foreach_set("loop_start", (np.arange(nq) * 4).astype(np.int32))
    uv = me.uv_layers.new(name="UVMap")
    loop_uv = ((verts[quads.ravel(), :2] + half) / size_m).astype(np.float32)
    uv.data.foreach_set("uv", loop_uv.ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    return obj


def add_float_attribute(mesh, name, values):
    attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
    attr.data.foreach_set("value", values.ravel().astype(np.float32))


def terrain_material(max_h):
    """Preview material: slope/height driven (grass, forest floor, rock, valley soil).

    This is a look-dev placeholder for evaluating landform; the production PBR
    layer stack belongs to the material phase.
    """
    mat = bpy.data.materials.new("M_Terrain_Preview")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.92
    # Soil/vegetation reflects little: default specular washes the ground out at grazing angles.
    bsdf.inputs["Specular IOR Level"].default_value = 0.15

    geo = nodes.new("ShaderNodeNewGeometry")
    sep_n = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Normal"], sep_n.inputs[0])
    sep_p = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Position"], sep_p.inputs[0])

    # Height factor 0..1
    hmap = nodes.new("ShaderNodeMapRange")
    hmap.inputs["From Min"].default_value = 0.0
    hmap.inputs["From Max"].default_value = max_h
    links.new(sep_p.outputs["Z"], hmap.inputs["Value"])

    # Break up bands with noise so height zones are not contour lines.
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 0.02
    tex.inputs["Detail"].default_value = 6.0
    links.new(geo.outputs["Position"], tex.inputs["Vector"])
    jitter = nodes.new("ShaderNodeMath")
    jitter.operation = "MULTIPLY_ADD"
    jitter.inputs[1].default_value = 0.18
    links.new(tex.outputs["Fac"], jitter.inputs[0])
    links.new(hmap.outputs["Result"], jitter.inputs[2])
    hjit = nodes.new("ShaderNodeMath")
    hjit.operation = "SUBTRACT"
    hjit.inputs[1].default_value = 0.09
    links.new(jitter.outputs[0], hjit.inputs[0])

    ramp_h = nodes.new("ShaderNodeValToRGB")
    cr = ramp_h.color_ramp
    stops = [(0.0, (0.055, 0.045, 0.028)), (0.05, (0.05, 0.072, 0.022)), (0.2, (0.03, 0.055, 0.018)),
             (0.6, (0.035, 0.05, 0.02)), (0.85, (0.07, 0.07, 0.045)), (1.0, (0.12, 0.115, 0.10))]
    cr.elements[0].position, cr.elements[0].color = stops[0][0], (*stops[0][1], 1)
    cr.elements[1].position, cr.elements[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, col in stops[1:-1]:
        e = cr.elements.new(pos)
        e.color = (*col, 1)
    links.new(hjit.outputs[0], ramp_h.inputs["Fac"])

    # Slope: normal.z < ~0.78 (> ~38 deg) -> rock
    ramp_s = nodes.new("ShaderNodeValToRGB")
    ramp_s.color_ramp.elements[0].position = 0.62
    ramp_s.color_ramp.elements[0].color = (1, 1, 1, 1)
    ramp_s.color_ramp.elements[1].position = 0.74
    ramp_s.color_ramp.elements[1].color = (0, 0, 0, 1)
    links.new(sep_n.outputs["Z"], ramp_s.inputs["Fac"])

    rock_noise = nodes.new("ShaderNodeTexNoise")
    rock_noise.inputs["Scale"].default_value = 0.4
    rock_noise.inputs["Detail"].default_value = 8.0
    links.new(geo.outputs["Position"], rock_noise.inputs["Vector"])
    rock_ramp = nodes.new("ShaderNodeValToRGB")
    rock_ramp.color_ramp.elements[0].color = (0.07, 0.065, 0.06, 1)
    rock_ramp.color_ramp.elements[1].color = (0.17, 0.16, 0.145, 1)
    links.new(rock_noise.outputs["Fac"], rock_ramp.inputs["Fac"])

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    links.new(ramp_s.outputs["Color"], mix.inputs["Factor"])
    links.new(ramp_h.outputs["Color"], mix.inputs[6])
    links.new(rock_ramp.outputs["Color"], mix.inputs[7])
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    return mat


def add_curve(name, pts3, collection, bevel=0.0):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts3) - 1)
    co = np.column_stack([pts3, np.ones(len(pts3))]).astype(np.float32)
    sp.points.foreach_set("co", co.ravel())
    cu.bevel_depth = bevel
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


def sample_height(h, size_m, x, y):
    res = h.shape[0]
    c = np.clip((x + size_m / 2) / size_m * (res - 1), 0, res - 1).astype(int)
    r = np.clip((y + size_m / 2) / size_m * (res - 1), 0, res - 1).astype(int)
    return h[r, c]


def setup_lighting(scene):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.42, 0.55, 0.75, 1.0)
    bg.inputs["Strength"].default_value = 0.9
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 4.0
    sun_data.angle = np.radians(1.5)
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (np.radians(55), 0, np.radians(-140))  # low-ish from SW -> relief
    scene.collection.objects.link(sun)


def setup_render(scene, samples):
    engines = [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
        if eng in engines:
            scene.render.engine = eng
            break
    if scene.render.engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = samples
        for attr in ("use_shadows", "use_raytracing"):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, True)
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.view_transform = "AgX"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0


def make_camera(scene, name, loc, rot, ortho=None, lens=35, clip_end=3000):
    cam_data = bpy.data.cameras.new(name)
    cam_data.clip_end = clip_end
    if ortho:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho
    else:
        cam_data.lens = lens
    cam = bpy.data.objects.new(name, cam_data)
    cam.location = loc
    cam.rotation_euler = rot
    scene.collection.objects.link(cam)
    return cam


def look_at_rotation(loc, target):
    from mathutils import Vector
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y").to_euler()


def render(scene, cam, path, w, h):
    scene.camera = cam
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()

    cfg = TerrainConfig(seed=a.seed, size_m=a.size, resolution=a.res, max_height_m=a.height,
                        mountain_ranges=a.ranges, detail=a.detail, erosion=a.erosion)
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))

    result = generate(cfg)
    H = result.heights
    timings = dict(result.timings)

    # ---- heightmap export (+ round-trip check)
    t = time.perf_counter()
    u16, hmin, hmax = heightmap_u16(H)
    hm_png = os.path.join(out, f"heightmap_{cfg.resolution}_16bit.png")
    write_gray16(hm_png, u16)
    u16.astype("<u2").tofile(os.path.join(out, f"heightmap_{cfg.resolution}.r16"))
    back = read_gray16(hm_png)
    assert back.shape == (cfg.resolution, cfg.resolution) and (back == u16).all(), "heightmap round-trip failed"
    cell = cfg.size_m / (cfg.resolution - 1)
    write_rgb8(os.path.join(out, "preview_hillshade.png"), hillshade_rgb(H, cell, result.river_mask))
    timings["heightmap_export"] = time.perf_counter() - t

    # ---- Blender scene
    t = time.perf_counter()
    scene = reset_scene()
    col_terrain = bpy.data.collections.new("Terrain")
    col_guides = bpy.data.collections.new("Terrain_Guides")
    scene.collection.children.link(col_terrain)
    scene.collection.children.link(col_guides)
    obj = build_mesh(H, cfg.size_m)
    col_terrain.objects.link(obj)
    add_float_attribute(obj.data, "river_mask", result.river_mask)
    add_float_attribute(obj.data, "flow_accumulation",
                        np.log1p(result.drainage_area) / np.log1p(result.drainage_area.max()))
    obj.data.materials.append(terrain_material(float(H.max())))
    obj["seed"] = cfg.seed
    obj["heightmap_min_m"] = hmin
    obj["heightmap_max_m"] = hmax

    river = result.river_polyline.copy()
    river[:, 2] = sample_height(H, cfg.size_m, river[:, 0], river[:, 1]) + 0.3
    add_curve("Guide_River_Centerline", river, col_guides, bevel=0.6)
    for i, sp in enumerate(result.range_spines):
        z = sample_height(H, cfg.size_m, sp[:, 0], sp[:, 1]) + 2.0
        add_curve(f"Guide_Range_Spine_{i + 1}", np.column_stack([sp, z]), col_guides)
    col_guides.hide_render = True
    timings["blender_mesh"] = time.perf_counter() - t

    setup_lighting(scene)
    setup_render(scene, a.samples)
    half = cfg.size_m / 2
    cam_top = make_camera(scene, "Cam_Top", (0, 0, hmax + 400), (0, 0, 0), ortho=cfg.size_m * 1.02)
    oblique_loc = (-half * 1.45, -half * 1.75, hmax + cfg.size_m * 0.55)
    cam_obl = make_camera(scene, "Cam_Oblique", oblique_loc,
                          look_at_rotation(oblique_loc, (0.0, half * 0.05, hmax * 0.1)), lens=32)
    ground_loc = (river[-8, 0] + 10, river[-8, 1], river[-8, 2] + 1.8)
    cam_ground = make_camera(scene, "Cam_Valley_Eye", ground_loc,
                             look_at_rotation(ground_loc, (river[45, 0], river[45, 1], river[45, 2] + 25)), lens=24)
    scene.camera = cam_obl

    # ---- save .blend before renders so a render failure still leaves the scene
    blend_path = os.path.join(out, "forest_valley_terrain.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)

    if not a.no_render:
        t = time.perf_counter()
        render(scene, cam_top, os.path.join(out, "preview_top.png"), 1400, 1400)
        render(scene, cam_obl, os.path.join(out, "preview_oblique.png"), 1920, 1080)
        render(scene, cam_ground, os.path.join(out, "preview_valley_eye.png"), 1600, 900)
        timings["render"] = time.perf_counter() - t
        scene.camera = cam_obl
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)

    # ---- stats
    gy, gx = np.gradient(H, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    me = obj.data
    stats = {
        "vertices": len(me.vertices),
        "quads": len(me.polygons),
        "triangles": len(me.polygons) * 2,
        "cell_size_m": cell,
        "height_min_m": hmin,
        "height_max_m": hmax,
        "height_mean_m": float(H.mean()),
        "slope_deg_p50": float(np.percentile(slope, 50)),
        "slope_deg_p90": float(np.percentile(slope, 90)),
        "slope_deg_p99": float(np.percentile(slope, 99)),
        "area_walkable_lt35deg_pct": float((slope < 35).mean() * 100),
        "area_flat_lt10deg_pct": float((slope < 10).mean() * 100),
    }
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {
        "config": config_dict(cfg),
        "stats": stats,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "heightmap": {
            "file": os.path.basename(hm_png), "bit_depth": 16, "orientation": "row0 = north (+Y)",
            "value_0_m": hmin, "value_65535_m": hmax,
            "unreal_z_scale_percent": (hmax - hmin) / 512.0 * 100.0,
            "unreal_xy_scale_cm": cell * 100.0,
        },
        "river_centerline_xyz": np.round(river, 2).tolist(),
        "range_spines_xy": [np.round(s, 2).tolist() for s in result.range_spines],
    }
    with open(os.path.join(out, "terrain_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("PHASE1_STATS", json.dumps({"stats": stats, "timings_s": meta["timings_s"]}))


if __name__ == "__main__":
    main()
