"""Phase 2 entry point: water system on top of the Phase 1 terrain.

River (pools / riffles / rapids), lake joined to the river by a backwater channel,
and a mountain tributary with a waterfall + plunge pool + gorge. The terrain mesh is
rebuilt at 2x resolution (cubic upsample) so banks and shorelines are carved at
~0.49 m instead of ~0.98 m.

Usage (headless):
  blender -b --factory-startup --python tools/water/build_water.py -- \
      --terrain-dir output/phase1_terrain --out output/phase2_water [--no-render]

Outputs in --out:
  forest_valley_water.blend        scene: hi-res carved terrain + water meshes + splines
  water_meshes.glb                 all visual water meshes (river, lake, channels, waterfall)
  heightmap_carved_<N>_16bit.png / .r16   N = 513 (UE spec) and 1025 (mesh resolution)
  flowmap_<N>.png                  RGBA8: R,G = flow dir*0.5+0.5, B = speed/vmax, A = foam
  water_mask_<N>.png / water_depth_<N>_16bit.png / drainage_<N>.png
  water_bodies.json                spline points (UE Water System units: cm), lake, channels
  preview_*.png                    renders
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "terrain"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from build_terrain import build_mesh, look_at_rotation, make_camera, render, setup_render  # noqa: E402
from png_io import heightmap_u16, write_gray8, write_gray16, write_rgba8  # noqa: E402
from water_core import WaterConfig, bilinear, config_dict, generate_water, upsample2_cubic  # noqa: E402

FLOW_LOOP_S = 2.0      # two-phase flow-map period; the animation repeats exactly every FLOW_LOOP_S


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain-dir", default="output/phase1_terrain")
    ap.add_argument("--out", default="output/phase2_water")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[], help="WaterConfig field=value")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- meshes

def mesh_from_arrays(name, verts, faces, collection):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    nf = len(faces)
    me.loops.add(nf * 4)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", (np.arange(nf) * 4).astype(np.int32))
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob


def point_attr(me, name, values, kind="FLOAT"):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name=name, type=kind, domain="POINT")
    key = "vector" if kind == "FLOAT_VECTOR" else "value"
    a.data.foreach_set(key, np.asarray(values, np.float32).ravel())


def set_uv(me, uv_per_vertex):
    uv = me.uv_layers.new(name="UVMap")
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    uv.data.foreach_set("uv", uv_per_vertex[vi].astype(np.float32).ravel())


def build_ribbon(name, sp, H, size_m, collection, lo=0, hi=None, cross=12, overlap=1.8, z_offset=0.0):
    """Water ribbon along a spline slice [lo, hi); edges tuck under the banks."""
    hi = len(sp["s"]) if hi is None else hi
    xy, n_, hw = sp["xy"][lo:hi], sp["normal"][lo:hi], sp["half_width"][lo:hi]
    n = len(xy)
    u = np.linspace(-1, 1, cross + 1)
    reach = hw + overlap
    px = xy[:, 0, None] + n_[:, 0, None] * reach[:, None] * u[None, :]
    py = xy[:, 1, None] + n_[:, 1, None] * reach[:, None] * u[None, :]
    pz = np.repeat(sp["surface_z"][lo:hi, None], cross + 1, axis=1) + z_offset
    half = size_m / 2
    px, py = np.clip(px, -half, half), np.clip(py, -half, half)
    verts = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    idx = np.arange(n * (cross + 1)).reshape(n, cross + 1)
    faces = np.column_stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                             idx[1:, 1:].ravel(), idx[1:, :-1].ravel()])[:, ::-1]
    ob = mesh_from_arrays(name, verts, faces, collection)
    me = ob.data
    if me.polygons[0].normal.z < 0:
        me.flip_normals()
    bed = bilinear(H, size_m, px.ravel(), py.ravel())
    depth = np.clip(pz.ravel() - bed, 0, None)
    prof = np.clip(1 - np.abs(u) ** 2, 0, 1) ** 0.5
    spd = (sp["speed"][lo:hi, None] * (0.35 + 0.65 * prof[None, :])).ravel()
    flow = np.column_stack([np.repeat(sp["tangent"][lo:hi], cross + 1, axis=0) * spd[:, None],
                            np.zeros(len(spd))])
    edge = np.clip((np.abs(u) * reach[:, None] - hw[:, None] + 1.2) / 1.2, 0, 1).ravel()
    foam = np.clip(np.repeat(sp["foam"][lo:hi], cross + 1) + 0.6 * edge, 0, 1)
    point_attr(me, "water_depth", depth)
    point_attr(me, "flow", flow, "FLOAT_VECTOR")
    point_attr(me, "flow_speed", spd)
    point_attr(me, "foam", foam)
    s_along = np.repeat(sp["s"][lo:hi], cross + 1)
    set_uv(me, np.column_stack([(np.tile(u, n) + 1) / 2, s_along / 10.0]))
    return ob


def build_lake_mesh(lake, H, size_m, collection):
    """Grid of terrain cells around the lake, flat at the water level."""
    res = H.shape[0]
    cell = size_m / (res - 1)
    w = lake["water"]
    d = w.copy()
    for _ in range(2):                       # dilate so the surface tucks under the shore
        e = d.copy()
        e[1:] |= d[:-1]; e[:-1] |= d[1:]; e[:, 1:] |= d[:, :-1]; e[:, :-1] |= d[:, 1:]
        d = e
    rows, cols = np.nonzero(d[:-1, :-1] | d[1:, 1:])
    used = np.zeros((res, res), bool)
    for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
        used[rows + dr, cols + dc] = True
    vid = -np.ones((res, res), np.int64)
    vr, vc = np.nonzero(used)
    vid[vr, vc] = np.arange(len(vr))
    verts = np.column_stack([-size_m / 2 + vc * cell, -size_m / 2 + vr * cell,
                             np.full(len(vr), lake["level"])])
    faces = np.column_stack([vid[rows, cols], vid[rows, cols + 1],
                             vid[rows + 1, cols + 1], vid[rows + 1, cols]])
    ob = mesh_from_arrays("Lake_Water", verts, faces, collection)
    depth = np.clip(lake["level"] - H[vr, vc], 0, None)
    point_attr(ob.data, "water_depth", depth)
    point_attr(ob.data, "flow", np.zeros((len(vr), 3)), "FLOAT_VECTOR")
    point_attr(ob.data, "flow_speed", np.zeros(len(vr)))
    point_attr(ob.data, "foam", np.clip(1 - depth / 0.35, 0, 1) * 0.7)
    set_uv(ob.data, np.column_stack([verts[:, 0] / 10.0, verts[:, 1] / 10.0]))
    return ob


def build_waterfall(fall, collection, rows=20, cols=10):
    """Free-falling water sheet: ballistic profile from the lip into the plunge pool."""
    lip = np.array(fall["lip_xyz"])
    d = np.array(fall["direction"] + [0.0])
    nrm = np.array([-d[1], d[0], 0.0])
    Hf = fall["height_m"] + 0.4                      # end below the pool surface
    g, v0 = 9.81, 1.8
    T = np.sqrt(2 * Hf / g)
    t = np.linspace(0, 1, rows + 1)
    u = np.linspace(-1, 1, cols + 1)
    ts = t * T
    fwd = 0.05 + v0 * ts                             # leaves the lip and arcs out over the drop
    drop = 0.5 * g * ts ** 2
    # Apron rows: a short flat lead-in over the lip that hides the upper ribbon's end.
    fwd = np.concatenate([[-1.2, -0.6], fwd])
    drop = np.concatenate([[0.0, 0.0], drop])
    t = np.concatenate([[0.0, 0.0], t])
    rows = len(t) - 1
    width = (fall["width_m"] * 0.5 + 0.3) * (1 + 0.15 * t)   # channel width, spreading slightly
    P = (lip[None, None, :] + d[None, None, :] * fwd[:, None, None]
         + nrm[None, None, :] * (width[:, None] * u[None, :])[:, :, None])
    P[:, :, 2] = lip[2] + 0.08 - drop[:, None]
    verts = P.reshape(-1, 3)
    idx = np.arange((rows + 1) * (cols + 1)).reshape(rows + 1, cols + 1)
    faces = np.column_stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                             idx[1:, 1:].ravel(), idx[1:, :-1].ravel()])
    ob = mesh_from_arrays("Waterfall_Sheet", verts, faces, collection)
    set_uv(ob.data, np.column_stack([np.tile((u + 1) / 2, rows + 1), np.repeat(t, cols + 1)]))
    # Spray / mist volume over the plunge pool.
    pool = lip + d * (v0 * T)
    pool[2] = fall["pool_z"]
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0,
                                         location=(pool[0], pool[1], pool[2] + 1.2))
    mist = bpy.context.active_object
    mist.name = "Waterfall_Mist"
    mist.scale = (fall["width_m"] * 1.1, fall["width_m"] * 1.1, 2.4)
    for c in mist.users_collection:
        c.objects.unlink(mist)
    collection.objects.link(mist)
    return ob, mist, pool


def add_spline(name, sp, collection, every=5):
    """UE-style water spline: point radius carries the half width."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    spl = cu.splines.new("POLY")
    idx = np.arange(0, len(sp["s"]), every)
    if idx[-1] != len(sp["s"]) - 1:
        idx = np.append(idx, len(sp["s"]) - 1)
    spl.points.add(len(idx) - 1)
    co = np.column_stack([sp["xy"][idx], sp["surface_z"][idx], np.ones(len(idx))])
    spl.points.foreach_set("co", co.astype(np.float32).ravel())
    spl.points.foreach_set("radius", sp["half_width"][idx].astype(np.float32))
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    ob.hide_render = True
    return ob, idx


# --------------------------------------------------------------------------- materials

class NodeKit:
    def __init__(self, tree):
        self.N, self.L = tree.nodes, tree.links

    def new(self, kind, **inputs):
        n = self.N.new(kind)
        for k, v in inputs.items():
            n.inputs[k].default_value = v
        return n

    def link(self, a, b):
        self.L.new(a, b)

    def math(self, op, a=None, b=None, c=None, clamp=False):
        m = self.N.new("ShaderNodeMath")
        m.operation = op
        m.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.L.new(v, m.inputs[i])
        return m.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        m = self.N.new("ShaderNodeVectorMath")
        m.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, tuple):
                m.inputs[i].default_value = v
            else:
                self.L.new(v, m.inputs[i])
        if scale is not None:
            if isinstance(scale, (int, float)):
                m.inputs["Scale"].default_value = scale
            else:
                self.L.new(scale, m.inputs["Scale"])
        return m.outputs[0]

    def attr(self, name, out="Fac"):
        a = self.N.new("ShaderNodeAttribute")
        a.attribute_name = name
        return a.outputs[out]

    def noise(self, vec, scale, detail, rough=0.5):
        n = self.new("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        self.link(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def time(self):
        v = self.N.new("ShaderNodeValue")
        v.label = "time_s"
        fc = v.outputs[0].driver_add("default_value")
        fc.driver.expression = "frame / 24"
        return v.outputs[0]


def water_material(heightmap_path=None, size_m=500.0, res=513, hmin=0.0, hmax=1.0):
    """PBR water: refraction, depth colour, looping two-phase flow ripples, foam.

    Ripples are advected along the per-vertex flow vector with two flow-map phases
    offset by half a period and cross-faded, so the animation loops seamlessly every
    FLOW_LOOP_S seconds without the texture stretching. Water depth for the shore
    fade is computed per pixel from the carved heightmap.
    """
    mat = bpy.data.materials.new("M_Water")
    mat.use_nodes = True
    # Raytraced refraction is off: Eevee's screen-space refraction trace produced
    # streak artefacts through the transparent surface. Depth colour and shore fade
    # are computed in the shader, so the look does not depend on it.
    for a_, v in (("use_raytrace_refraction", False), ("use_screen_refraction", False)):
        if hasattr(mat, a_):
            setattr(mat, a_, v)
    K = NodeKit(mat.node_tree)
    bsdf = next(n for n in K.N if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["IOR"].default_value = 1.333

    depth_attr = K.attr("water_depth")
    foam_a = K.attr("foam")
    flow = K.attr("flow", "Vector")
    geo = K.new("ShaderNodeNewGeometry")
    pos = geo.outputs["Position"]

    shore = depth_attr
    depth = depth_attr
    if heightmap_path:
        img = bpy.data.images.load(heightmap_path, check_existing=True)
        img.colorspace_settings.is_data = True
        tex = K.N.new("ShaderNodeTexImage")
        tex.image, tex.interpolation, tex.extension = img, "Linear", "EXTEND"
        mp = K.N.new("ShaderNodeMapping")
        k = (res - 1) / res / size_m
        off = 0.5 * (res - 1) / res + 0.5 / res
        mp.inputs["Scale"].default_value = (k, k, 0)
        mp.inputs["Location"].default_value = (off, off, 0)
        K.link(pos, mp.inputs["Vector"])
        K.link(mp.outputs["Vector"], tex.inputs["Vector"])
        hz = K.math("MULTIPLY_ADD", tex.outputs["Color"], hmax - hmin, hmin)
        z = K.new("ShaderNodeSeparateXYZ")
        K.link(pos, z.inputs[0])
        shore = K.math("SUBTRACT", z.outputs["Z"], hz, clamp=True)
        depth = K.math("MAXIMUM", shore, depth_attr)

    # Depth colour.
    ramp = K.N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.30, 0.28, 0.18, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.020, 0.075, 0.060, 1)
    ramp.color_ramp.elements.new(0.35).color = (0.07, 0.13, 0.09, 1)
    K.link(K.math("DIVIDE", depth, 2.2, clamp=True), ramp.inputs["Fac"])

    # Two-phase flow: phase_a = fract(t/T), phase_b = fract(t/T + 0.5).
    t = K.time()
    tn = K.math("DIVIDE", t, FLOW_LOOP_S)
    pa = K.math("FRACT", tn)
    pb = K.math("FRACT", K.math("ADD", tn, 0.5))
    wa = K.math("SUBTRACT", 1.0, K.math("ABSOLUTE", K.math("MULTIPLY_ADD", pa, 2.0, -1.0)))
    wb = K.math("SUBTRACT", 1.0, wa)
    pos_a = K.vmath("SUBTRACT", pos, K.vmath("SCALE", flow, scale=K.math("MULTIPLY", pa, FLOW_LOOP_S)))
    pos_b = K.vmath("ADD", K.vmath("SUBTRACT", pos, K.vmath("SCALE", flow, scale=K.math("MULTIPLY", pb, FLOW_LOOP_S))),
                    (17.3, 9.1, 0.0))

    def waves(p):
        return K.math("MULTIPLY_ADD", K.noise(p, 4.5, 4.0), 0.4, K.noise(p, 0.9, 6.0, 0.55))

    def blend(fa, fb):
        return K.math("ADD", K.math("MULTIPLY", fa, wa), K.math("MULTIPLY", fb, wb))

    height = blend(waves(pos_a), waves(pos_b))
    chop = K.math("MULTIPLY_ADD", K.attr("flow_speed"), 0.25, 0.12)
    bump = K.N.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = 0.06
    K.link(chop, bump.inputs["Strength"])
    K.link(height, bump.inputs["Height"])
    K.link(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Foam: coverage grows with the foam attribute; threshold = 1 - 0.5 * foam.
    def foam_noise(p):   # fine bubbles over a coarser patch structure
        return K.math("MULTIPLY_ADD", K.noise(p, 7.0, 12.0, 0.7), 0.6, K.math("MULTIPLY", K.noise(p, 1.2, 4.0), 0.4))

    fnoise = blend(foam_noise(pos_a), foam_noise(pos_b))
    th = K.math("MULTIPLY_ADD", foam_a, -0.5, 1.0)
    fr = K.N.new("ShaderNodeMapRange")
    K.link(th, fr.inputs["From Min"])
    K.link(K.math("ADD", th, 0.12), fr.inputs["From Max"])
    K.link(fnoise, fr.inputs["Value"])
    foam = fr.outputs["Result"]
    col = K.N.new("ShaderNodeMix")
    col.data_type = "RGBA"
    K.link(foam, col.inputs["Factor"])
    K.link(ramp.outputs["Color"], col.inputs[6])
    col.inputs[7].default_value = (0.85, 0.88, 0.86, 1)
    K.link(col.outputs[2], bsdf.inputs["Base Color"])
    K.link(K.math("SUBTRACT", 1.0, foam), bsdf.inputs["Transmission Weight"])
    K.link(K.math("MULTIPLY_ADD", foam, 0.55, 0.04), bsdf.inputs["Roughness"])
    fade = K.N.new("ShaderNodeMapRange")
    fade.inputs["From Max"].default_value = 0.3
    K.link(shore, fade.inputs["Value"])
    K.link(fade.outputs["Result"], bsdf.inputs["Alpha"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return mat


def waterfall_material():
    """Falling sheet: vertical streaks scrolling down, broken alpha, aerated white."""
    mat = bpy.data.materials.new("M_Waterfall")
    mat.use_nodes = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    K = NodeKit(mat.node_tree)
    bsdf = next(n for n in K.N if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.80, 0.86, 0.88, 1)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Transmission Weight"].default_value = 0.35
    bsdf.inputs["IOR"].default_value = 1.333
    uv = K.new("ShaderNodeUVMap").outputs["UV"]
    sep = K.new("ShaderNodeSeparateXYZ")
    K.link(uv, sep.inputs[0])
    # Two-phase scroll (same scheme as the river) so the sheet loops every FLOW_LOOP_S.
    tn = K.math("DIVIDE", K.time(), FLOW_LOOP_S)
    pa = K.math("FRACT", tn)
    pb = K.math("FRACT", K.math("ADD", tn, 0.5))
    wa = K.math("SUBTRACT", 1.0, K.math("ABSOLUTE", K.math("MULTIPLY_ADD", pa, 2.0, -1.0)))

    def streaks(phase, offset):
        comb = K.new("ShaderNodeCombineXYZ")
        K.link(K.math("MULTIPLY", sep.outputs["X"], 14.0), comb.inputs["X"])
        v = K.math("MULTIPLY", sep.outputs["Y"], 1.4)
        K.link(K.math("ADD", K.math("SUBTRACT", v, K.math("MULTIPLY", phase, 2.2 * FLOW_LOOP_S)), offset),
               comb.inputs["Y"])
        return K.noise(comb.outputs["Vector"], 3.0, 8.0, 0.6)

    sa, sb = streaks(pa, 0.0), streaks(pb, 13.7)
    streak = K.math("ADD", K.math("MULTIPLY", sa, wa), K.math("MULTIPLY", sb, K.math("SUBTRACT", 1.0, wa)))
    a = K.N.new("ShaderNodeMapRange")
    a.inputs["From Min"].default_value = 0.38
    a.inputs["From Max"].default_value = 0.62
    K.link(streak, a.inputs["Value"])
    edge = K.math("SUBTRACT", 1.0, K.math("POWER", K.math("ABSOLUTE", K.math("MULTIPLY_ADD", sep.outputs["X"], 2.0, -1.0)), 1.6))
    K.link(K.math("MULTIPLY", K.math("MULTIPLY_ADD", a.outputs["Result"], 0.75, 0.25), edge), bsdf.inputs["Alpha"])
    bump = K.N.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    K.link(streak, bump.inputs["Height"])
    K.link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def mist_material():
    mat = bpy.data.materials.new("M_Waterfall_Mist")
    mat.use_nodes = True
    K = NodeKit(mat.node_tree)
    out = next(n for n in K.N if n.type == "OUTPUT_MATERIAL")
    for n in list(K.N):
        if n.type == "BSDF_PRINCIPLED":
            K.N.remove(n)
    vol = K.new("ShaderNodeVolumePrincipled")
    vol.inputs["Color"].default_value = (0.9, 0.93, 0.95, 1)
    tc = K.new("ShaderNodeTexCoord")
    # Density: radial falloff in object space x noise.
    r = K.vmath("LENGTH", tc.outputs["Object"])
    fall = K.math("SUBTRACT", 1.0, r, clamp=True)
    n = K.noise(tc.outputs["Object"], 3.0, 4.0)
    dens = K.math("MULTIPLY", K.math("MULTIPLY", K.math("MULTIPLY", fall, fall), fall), K.math("MULTIPLY", n, 0.03))
    K.link(dens, vol.inputs["Density"])
    K.link(vol.outputs[0], out.inputs["Volume"])
    return mat


def sky_world(scene):
    """Gradient sky with soft cloud breakup, so reflections carry ripple detail."""
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    N, Lk = world.node_tree.nodes, world.node_tree.links
    bg = next(n for n in N if n.type == "BACKGROUND")
    tc = N.new("ShaderNodeTexCoord")
    sep = N.new("ShaderNodeSeparateXYZ")
    Lk.new(tc.outputs["Generated"], sep.inputs[0])
    ramp = N.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.62, 0.68, 0.74, 1)
    cr.elements[1].position = 0.45
    cr.elements[1].color = (0.16, 0.30, 0.58, 1)
    e = cr.elements.new(0.08)
    e.color = (0.48, 0.58, 0.70, 1)
    Lk.new(sep.outputs["Z"], ramp.inputs["Fac"])
    cl = N.new("ShaderNodeTexNoise")
    cl.inputs["Scale"].default_value = 3.0
    cl.inputs["Detail"].default_value = 6.0
    Lk.new(tc.outputs["Generated"], cl.inputs["Vector"])
    cmask = N.new("ShaderNodeMapRange")
    cmask.inputs["From Min"].default_value = 0.52
    cmask.inputs["From Max"].default_value = 0.72
    cmask.inputs["To Max"].default_value = 0.7
    Lk.new(cl.outputs["Fac"], cmask.inputs["Value"])
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    Lk.new(cmask.outputs["Result"], mix.inputs["Factor"])
    Lk.new(ramp.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (0.85, 0.87, 0.9, 1)
    Lk.new(mix.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0


def add_wet_layer(terrain_mat):
    """Darken/saturate terrain where wet; gravel/silt tone on the river and lake bed."""
    nt = terrain_mat.node_tree
    N, Lk = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == "BSDF_PRINCIPLED")
    src = bsdf.inputs["Base Color"].links[0].from_socket
    wet = N.new("ShaderNodeAttribute")
    wet.attribute_name = "wet"
    bed = N.new("ShaderNodeAttribute")
    bed.attribute_name = "underwater"
    m1 = N.new("ShaderNodeMix")
    m1.data_type = "RGBA"
    m1.blend_type = "MULTIPLY"
    Lk.new(wet.outputs["Fac"], m1.inputs["Factor"])
    Lk.new(src, m1.inputs[6])
    m1.inputs[7].default_value = (0.55, 0.5, 0.42, 1)
    gravel = N.new("ShaderNodeTexNoise")
    gravel.inputs["Scale"].default_value = 3.0
    gravel.inputs["Detail"].default_value = 8.0
    gr = N.new("ShaderNodeValToRGB")
    gr.color_ramp.elements[0].color = (0.06, 0.05, 0.035, 1)
    gr.color_ramp.elements[1].color = (0.16, 0.14, 0.10, 1)
    Lk.new(gravel.outputs["Fac"], gr.inputs["Fac"])
    m2 = N.new("ShaderNodeMix")
    m2.data_type = "RGBA"
    Lk.new(bed.outputs["Fac"], m2.inputs["Factor"])
    Lk.new(m1.outputs[2], m2.inputs[6])
    Lk.new(gr.outputs["Color"], m2.inputs[7])
    Lk.new(m2.outputs[2], bsdf.inputs["Base Color"])
    rough = N.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.92
    rough.inputs["To Max"].default_value = 0.68
    Lk.new(wet.outputs["Fac"], rough.inputs["Value"])
    Lk.new(rough.outputs["Result"], bsdf.inputs["Roughness"])


def configure_raytracing(scene):
    """Eevee raytraced reflections (mountains in the water); refraction is off per material."""
    ee = scene.eevee
    if hasattr(ee, "use_raytracing"):
        ee.use_raytracing = True
        ee.ray_tracing_method = "SCREEN"


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    tdir = os.path.abspath(a.terrain_dir)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}

    with open(os.path.join(tdir, "terrain_meta.json")) as f:
        tmeta = json.load(f)
    tcfg = tmeta["config"]
    size_m, res0 = tcfg["size_m"], tcfg["resolution"]
    cfg = WaterConfig(seed=tcfg["seed"])
    for kv in a.set:
        k, v = kv.split("=")
        cur = getattr(cfg, k)
        setattr(cfg, k, (v.lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))

    # ---- load Phase 1 terrain (exact float heights) and upsample 2x
    t = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(tdir, "forest_valley_terrain.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    me0 = terrain.data
    nv = len(me0.vertices)
    co = np.empty(nv * 3, np.float32)
    me0.vertices.foreach_get("co", co)
    H0 = co.reshape(-1, 3)[:, 2].reshape(res0, res0).astype(np.float64)
    attrs0 = {}
    for name in ("river_mask", "flow_accumulation"):
        if name in me0.attributes:
            v = np.empty(nv, np.float32)
            me0.attributes[name].data.foreach_get("value", v)
            attrs0[name] = v.reshape(res0, res0)
    H = upsample2_cubic(H0)
    res = H.shape[0]
    timings["load_upsample"] = time.perf_counter() - t

    river_xy = np.array(tmeta["river_centerline_xyz"])[:, :2]
    result = generate_water(cfg, H, size_m, river_xy)
    timings.update({f"water_{k}": v for k, v in result.timings.items()})
    Hn, sp, lake, maps = result.heights, result.spline, result.lake, result.maps

    # ---- rebuild terrain mesh at the new resolution (same object, material, name)
    t = time.perf_counter()
    tmp = build_mesh(Hn, size_m, name="Terrain_hi")
    new_me = tmp.data
    for m in me0.materials:
        new_me.materials.append(m)
    terrain.data = new_me
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(me0)
    new_me.name = "Terrain"
    me = new_me
    for name, v in attrs0.items():
        point_attr(me, name, np.clip(upsample2_cubic(v), 0, 1))
    point_attr(me, "wet", maps["wet"])
    point_attr(me, "underwater", maps["water_mask"].astype(float))
    point_attr(me, "water_depth", maps["depth"])
    point_attr(me, "drainage", maps["drainage"])
    add_wet_layer(me.materials[0])
    terrain["resolution"] = res
    timings["terrain_rebuild"] = time.perf_counter() - t

    # ---- heightmaps (513 = UE/spec deliverable, N = mesh resolution, used by the shader)
    u16, hmin, hmax = heightmap_u16(Hn)
    hm_path = os.path.join(out, f"heightmap_carved_{res}_16bit.png")
    write_gray16(hm_path, u16)
    u16.astype("<u2").tofile(os.path.join(out, f"heightmap_carved_{res}.r16"))
    Hs = Hn[::(res - 1) // (res0 - 1), ::(res - 1) // (res0 - 1)]
    u16s, smin, smax = heightmap_u16(Hs)
    write_gray16(os.path.join(out, f"heightmap_carved_{res0}_16bit.png"), u16s)
    u16s.astype("<u2").tofile(os.path.join(out, f"heightmap_carved_{res0}.r16"))

    # ---- water objects
    t = time.perf_counter()
    col = bpy.data.collections.new("Water")
    scene.collection.children.link(col)
    wmat = water_material(hm_path, size_m, res, hmin, hmax)
    water_objs = []
    ob = build_ribbon("River_Water", sp, Hn, size_m, col)
    water_objs.append(ob)
    lake_ob = None
    if lake:
        lake_ob = build_lake_mesh(lake, Hn, size_m, col)
        water_objs.append(lake_ob)
    spline_obs = [add_spline("River_Spline", sp, col)]
    fall_objs = []
    for ch in result.channels:
        if ch["kind"] == "connector":
            # 3 cm under the lake/river surfaces so overlaps never z-fight.
            water_objs.append(build_ribbon("Lake_Connector_Water", ch, Hn, size_m, col, cross=8,
                                           overlap=1.2, z_offset=-0.03))
        else:
            i_f = ch["fall_index"]
            # Upper reach stops 1 m short of the lip (no overhang over the carved drop).
            water_objs.append(build_ribbon("Stream_Upper_Water", ch, Hn, size_m, col, 0, i_f, cross=8, overlap=1.0))
            water_objs.append(build_ribbon("Stream_Lower_Water", ch, Hn, size_m, col, i_f + 1, None, cross=8,
                                           overlap=1.0))
            if ch["fall"]["height_m"] > 0.5:
                sheet, mist, pool = build_waterfall(ch["fall"], col)
                sheet.data.materials.append(waterfall_material())
                mist.data.materials.append(mist_material())
                fall_objs = [sheet, mist]
                ch["fall"]["pool_xyz"] = [float(v) for v in pool]
        spline_obs.append(add_spline(ch["name"] + "_Spline", ch, col, every=3))
    for ob in water_objs:
        ob.data.materials.append(wmat)
    old = bpy.data.objects.get("Guide_River_Centerline")
    if old:
        bpy.data.objects.remove(old)
    timings["blender_build"] = time.perf_counter() - t

    # ---- maps (mesh resolution)
    t = time.perf_counter()
    vmax = float(max(maps["speed"].max(), 1e-3))
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(maps["speed"] > 0, maps["flow_x"] / maps["speed"], 0)
        dy = np.where(maps["speed"] > 0, maps["flow_y"] / maps["speed"], 0)
    flow = np.stack([dx * 0.5 + 0.5, dy * 0.5 + 0.5, maps["speed"] / vmax, maps["foam"]], -1)
    write_rgba8(os.path.join(out, f"flowmap_{res}.png"), (np.clip(flow, 0, 1) * 255).astype(np.uint8)[::-1])
    write_gray8(os.path.join(out, f"water_mask_{res}.png"), (maps["water_mask"] * 255).astype(np.uint8)[::-1])
    write_gray8(os.path.join(out, f"drainage_{res}.png"), (maps["drainage"] * 255).astype(np.uint8)[::-1])
    dmax = float(max(maps["depth"].max(), 1e-3))
    write_gray16(os.path.join(out, f"water_depth_{res}_16bit.png"),
                 np.round(maps["depth"] / dmax * 65535).astype(np.uint16)[::-1])
    timings["maps"] = time.perf_counter() - t

    # ---- render setup + cameras
    setup_render(scene, a.samples)
    sky_world(scene)
    configure_raytracing(scene)
    xy, sz = sp["xy"], sp["surface_z"]
    n = len(xy)
    i_r = int(np.argmax(sp["rapid"]))
    i_p = int(np.argmax(np.abs(sp["bend"]) * (np.arange(n) > n * 0.25) * (np.arange(n) < n * 0.75)))
    cams = {}
    loc = (xy[5, 0] + 40, xy[5, 1] - 70, sz[5] + 55)
    cams["valley"] = make_camera(scene, "Cam_River_Valley", loc,
                                 look_at_rotation(loc, (xy[n // 2, 0], xy[n // 2, 1], sz[n // 2])), lens=28)
    j = max(i_r - 30, 0)
    nb = sp["normal"][j] * (sp["half_width"][j] + 3.0)
    gz = bilinear(Hn, size_m, np.array([xy[j, 0] + nb[0]]), np.array([xy[j, 1] + nb[1]]))[0]
    loc = (xy[j, 0] + nb[0], xy[j, 1] + nb[1], gz + 1.75)
    cams["rapid"] = make_camera(scene, "Cam_Rapid_Eye", loc,
                                look_at_rotation(loc, (xy[i_r + 8, 0], xy[i_r + 8, 1], sz[i_r + 8])), lens=22)
    nb = -np.sign(sp["bend"][i_p]) * sp["normal"][i_p] * 38
    loc = (xy[i_p, 0] + nb[0], xy[i_p, 1] + nb[1], sz[i_p] + 16)
    cams["pool"] = make_camera(scene, "Cam_Pool_Bend", loc,
                               look_at_rotation(loc, (xy[i_p, 0], xy[i_p, 1], sz[i_p] - 1)), lens=24)
    if lake:
        cx, cy = lake["center"]
        R = cfg.lake_radius_m
        loc = (cx - R * 1.6, cy - R * 1.9, lake["level"] + R * 1.1)
        cams["lake"] = make_camera(scene, "Cam_Lake", loc, look_at_rotation(loc, (cx, cy, lake["level"])), lens=26)
    stream = next((c for c in result.channels if c["kind"] == "stream"), None)
    if stream and stream["fall"]["height_m"] > 0.5:
        f = stream["fall"]
        lip = np.array(f["lip_xyz"])
        d = np.array(f["direction"])
        side = np.array([-d[1], d[0]])
        p = lip[:2] + d * 16 + side * 5
        gz = bilinear(Hn, size_m, np.array([p[0]]), np.array([p[1]]))[0]
        loc = (p[0], p[1], max(gz + 1.7, f["pool_z"] + 2.5))
        cams["waterfall"] = make_camera(scene, "Cam_Waterfall", loc,
                                        look_at_rotation(loc, (lip[0], lip[1], (lip[2] + f["pool_z"]) / 2)), lens=20)
    cam_top = bpy.data.objects.get("Cam_Top")
    blend_path = os.path.join(out, "forest_valley_water.blend")
    scene.camera = cams["valley"]
    scene.frame_start, scene.frame_end = 1, int(FLOW_LOOP_S * 24)    # one seamless flow loop
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)

    if not a.no_render:
        t = time.perf_counter()
        for key, fname, w, h in (("valley", "preview_river_oblique", 1920, 1080),
                                 ("rapid", "preview_rapid_eye", 1600, 900),
                                 ("pool", "preview_pool_bend", 1600, 900),
                                 ("lake", "preview_lake", 1600, 900),
                                 ("waterfall", "preview_waterfall", 1600, 900)):
            if key in cams:
                render(scene, cams[key], os.path.join(out, fname + ".png"), w, h)
        if cam_top:
            render(scene, cam_top, os.path.join(out, "preview_top.png"), 1400, 1400)
        timings["render"] = time.perf_counter() - t
        scene.camera = cams["valley"]
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    bak = blend_path + "1"
    if os.path.exists(bak):
        os.remove(bak)

    # ---- GLB export of the visual water meshes
    t = time.perf_counter()
    bpy.ops.object.select_all(action="DESELECT")
    for ob in water_objs + fall_objs[:1]:
        ob.select_set(True)
    try:
        bpy.ops.export_scene.gltf(filepath=os.path.join(out, "water_meshes.glb"), use_selection=True,
                                  export_format="GLB", export_attributes=True)
    except Exception as exc:
        print("GLB_EXPORT_FAILED", exc)
    timings["glb_export"] = time.perf_counter() - t

    # ---- metadata (UE Water System uses cm)
    def spline_points(c, idx):
        return [{
            "s_m": round(float(c["s"][i]), 2),
            "location_cm": [round(float(c["xy"][i, 0]) * 100, 1), round(float(c["xy"][i, 1]) * 100, 1),
                            round(float(c["surface_z"][i]) * 100, 1)],
            "width_cm": round(float(c["half_width"][i] * 2) * 100, 1),
            "depth_cm": round(float(c["depth"][i]) * 100, 1),
            "velocity_cm_s": round(float(c["speed"][i]) * 100, 1),
        } for i in idx]

    channels_meta = []
    for ch, (_, idx) in zip(result.channels, spline_obs[1:]):
        m = {"name": ch["name"], "kind": ch["kind"], "length_m": ch["length_m"], "leak_cells": ch["leaks"],
             "spline_points": spline_points(ch, idx)}
        if ch["kind"] == "stream":
            m["waterfall"] = ch["fall"]
        if ch["kind"] == "connector":
            m["joins_river_at_s_m"] = float(sp["s"][ch["river_index"]])
        channels_meta.append(m)
    faces = {ob.name: len(ob.data.polygons) for ob in water_objs + fall_objs[:1]}
    stats = {
        "terrain_mesh_resolution": res,
        "terrain_cell_m": size_m / (res - 1),
        "terrain_faces": len(me.polygons),
        "river_length_m": sp["length_m"],
        "river_width_m_min_max": [float(sp["half_width"].min() * 2), float(sp["half_width"].max() * 2)],
        "river_depth_m_min_max": [float(sp["depth"].min()), float(sp["depth"].max())],
        "river_surface_fall_m": float(sz[0] - sz[-1]),
        "river_speed_ms_min_max": [float(sp["speed"].min()), float(sp["speed"].max())],
        "rapids_at_s_m": [round(c, 1) for c in sp["rapid_centres_s"]],
        "lake_level_m": lake["level"] if lake else None,
        "lake_area_m2": float(lake["water"].sum() * (size_m / (res - 1)) ** 2) if lake else None,
        "lake_max_depth_m": float(lake["depth"].max()) if lake else None,
        "water_faces": faces,
        "water_area_pct": float(maps["water_mask"].mean() * 100),
        "leak_checks": result.checks,
        "terrain_height_range_m": [hmin, hmax],
    }
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {
        "config": config_dict(cfg),
        "terrain_source": os.path.relpath(tdir, out),
        "stats": stats,
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
        "flowmap": {"file": f"flowmap_{res}.png", "encoding": "R,G = flow dir xy *0.5+0.5 (x east, y north); "
                    "B = speed / vmax; A = foam", "vmax_m_s": vmax, "orientation": "row0 = north",
                    "preview_loop_s": FLOW_LOOP_S},
        "heightmap_carved": {"files": [f"heightmap_carved_{res0}_16bit.png", f"heightmap_carved_{res}_16bit.png"],
                             "value_0_m": hmin, "value_65535_m": hmax,
                             f"value_range_{res0}_m": [smin, smax]},
        "water_depth_map": {"value_65535_m": dmax},
        "river_spline_points": spline_points(sp, spline_obs[0][1]),
        "lake": None if not lake else {"level_cm": round(lake["level"] * 100, 1),
                                       "center_cm": [lake["center"][0] * 100, lake["center"][1] * 100]},
        "channels": channels_meta,
    }
    with open(os.path.join(out, "water_bodies.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("PHASE2_STATS", json.dumps({"stats": stats, "timings_s": meta["timings_s"]}))


if __name__ == "__main__":
    main()
