"""Atmosphere rig: physical sky + stars, sun, cloud layer (casts cloud shadows),
height fog volume, rain streak field, night work-lights, global wetness hooks."""
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "rocks"), os.path.join(HERE, "..", "common")]
from rock_materials import NK  # noqa: E402
from blend_util import make_instancer, mesh_from_numpy  # noqa: E402


def build_world(scene):
    w = bpy.data.worlds.new("ATM_World")
    w.use_nodes = True
    scene.world = w
    k = NK(w.node_tree)
    bg = next(n for n in k.N if n.type == "BACKGROUND")
    sky = k.N.new("ShaderNodeTexSky")
    sky.name = "ATM_Sky"
    ids = [i.identifier for i in sky.bl_rna.properties["sky_type"].enum_items]
    sky.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in ids else ids[0]
    sky.sun_disc = True
    sky.altitude = 300.0
    strength = k.N.new("ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.name = "ATM_SkyStrength"
    # Stars: tiny voronoi points on the view direction, only above the horizon.
    tc = k.n("ShaderNodeTexCoord")
    vor = k.n("ShaderNodeTexVoronoi", Scale=420.0)
    vor.feature = "F1"
    k.ln(tc.outputs["Generated"], vor.inputs["Vector"])
    star = k.mrange(vor.outputs["Distance"], 0.0, 0.035, 1.0, 0.0)
    star = k.math("POWER", star, 6.0)
    twinkle = k.mrange(k.noise(tc.outputs["Generated"], 60.0, 2), 0.4, 0.75, 0.0, 1.0)
    star = k.math("MULTIPLY", star, twinkle)
    sep = k.n("ShaderNodeSeparateXYZ")
    k.ln(tc.outputs["Generated"], sep.inputs[0])
    above = k.mrange(sep.outputs["Z"], 0.0, 0.15)
    star = k.math("MULTIPLY", star, above)
    stars = k.N.new("ShaderNodeMath")
    stars.operation = "MULTIPLY"
    stars.name = "ATM_Stars"
    k.ln(star, stars.inputs[0])
    # Overcast tint of the sky dome as coverage rises (reflections / ambient match the clouds).
    covm = k.N.new("ShaderNodeMapRange")
    covm.name = "ATM_CloudsWorld"
    covm.inputs[1].default_value = 0.6
    covm.inputs[2].default_value = 1.0
    k.ln(sky.outputs["Color"], strength.inputs[0])
    lum = k.n("ShaderNodeRGBToBW")
    k.ln(sky.outputs["Color"], lum.inputs[0])
    ovc = k.n("ShaderNodeCombineColor")
    for i in range(3):
        k.ln(lum.outputs[0], ovc.inputs[i])
    mix = k.N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    k.ln(covm.outputs[0], mix.inputs["Factor"])
    k.ln(sky.outputs["Color"], mix.inputs[6])
    ov2 = k.mix(1.0, ovc.outputs[0], (0.85, 0.88, 0.95), "MULTIPLY")
    k.ln(ov2, mix.inputs[7])
    k.ln(mix.outputs[2], strength.inputs[0])
    add = k.n("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.inputs["Factor"].default_value = 1.0
    k.ln(strength.outputs[0], add.inputs[6])
    sc = k.n("ShaderNodeCombineColor")
    for i, f in enumerate((0.8, 0.85, 1.0)):
        m = k.math("MULTIPLY", stars.outputs[0], f * 0.6)
        k.ln(m, sc.inputs[i])
    k.ln(sc.outputs[0], add.inputs[7])
    k.ln(add.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    return w


def build_sun(scene):
    old = bpy.data.objects.get("Sun")
    if old:
        bpy.data.objects.remove(old)
    d = bpy.data.lights.new("ATM_Sun", "SUN")
    d.use_shadow = True
    if hasattr(d, "shadow_cascade_max_distance"):
        d.shadow_cascade_max_distance = 600.0
    ob = bpy.data.objects.new("ATM_Sun", d)
    scene.collection.objects.link(ob)
    return ob


def build_clouds(scene, coll, size_m):
    """Cloud deck at 420 m: noise coverage -> alpha. Visible from below and casts moving-
    patch cloud shadows on the terrain (transparent shadows)."""
    S = size_m * 14
    me = mesh_from_numpy("ATM_Clouds", np.array([[-S, -S, 0], [S, -S, 0], [S, S, 0], [-S, S, 0]]),
                         np.array([[0, 1, 2, 3]]), smooth=False)
    ob = bpy.data.objects.new("ATM_Clouds", me)
    ob.location = (0, 0, 420.0)
    coll.objects.link(ob)
    m = bpy.data.materials.new("M_ATM_Clouds")
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "DITHERED"
    m.use_transparent_shadow = True
    m.use_backface_culling = False
    k = NK(m.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    out = next(n for n in k.N if n.type == "OUTPUT_MATERIAL")
    pos = k.n("ShaderNodeNewGeometry").outputs["Position"]
    cov = k.N.new("ShaderNodeValue")
    cov.name = "ATM_CloudCoverage"
    dark = k.N.new("ShaderNodeValue")
    dark.name = "ATM_CloudDark"
    mp = k.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / 900.0, 1 / 600.0, 1 / 600.0)
    k.ln(pos, mp.inputs["Vector"])
    n1 = k.n("ShaderNodeTexNoise", Scale=1.0, Detail=10.0, Roughness=0.6, Distortion=0.3)
    k.ln(mp.outputs["Vector"], n1.inputs["Vector"])
    # threshold = 1 - coverage
    thr = k.math("SUBTRACT", 0.78, k.math("MULTIPLY", cov.outputs[0], 0.5))
    a = k.math("SUBTRACT", n1.outputs["Fac"], thr)
    a = k.math("MULTIPLY", a, 9.0, clamp=True)
    full = k.mrange(cov.outputs[0], 0.85, 1.0)                 # overcast: solid deck
    a = k.math("MAXIMUM", a, full)
    col = k.mix(dark.outputs[0], (0.95, 0.96, 1.0), (0.32, 0.34, 0.38))
    shade = k.mrange(n1.outputs["Fac"], 0.4, 0.8, 1.0, 0.75)
    col = k.mix(1.0, col, k.ramp(shade, [(0, (0, 0, 0)), (1, (1, 1, 1))]), "MULTIPLY")
    k.ln(col, bsdf.inputs["Base Color"])
    k.ln(a, bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 1.0
    # Clouds are lit from above and the sky: an emission floor keeps undersides from going black.
    em = k.n("ShaderNodeEmission", Strength=0.35)
    k.ln(col, em.inputs["Color"])
    add = k.n("ShaderNodeAddShader")
    k.ln(bsdf.outputs[0], add.inputs[0])
    k.ln(em.outputs[0], add.inputs[1])
    tr = k.n("ShaderNodeBsdfTransparent")
    mix = k.n("ShaderNodeMixShader")
    k.ln(a, mix.inputs["Fac"])
    k.ln(tr.outputs[0], mix.inputs[1])
    k.ln(add.outputs[0], mix.inputs[2])
    k.ln(mix.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    return ob


def build_fog(scene, coll, size_m):
    """Height fog: exponential density with altitude in a box domain over the map."""
    S, top = size_m * 6.0, 900.0
    V = np.array([[x, y, z] for z in (-15.0, top) for (x, y) in ((-S, -S), (S, -S), (S, S), (-S, S))])
    F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    me = mesh_from_numpy("ATM_Fog", V, F, smooth=False)
    ob = bpy.data.objects.new("ATM_Fog", me)
    ob.visible_shadow = False
    coll.objects.link(ob)
    m = bpy.data.materials.new("M_ATM_Fog")
    m.use_nodes = True
    k = NK(m.node_tree)
    for n in list(k.N):
        if n.type == "BSDF_PRINCIPLED":
            k.N.remove(n)
    out = next(n for n in k.N if n.type == "OUTPUT_MATERIAL")
    dens = k.N.new("ShaderNodeValue")
    dens.name = "ATM_FogDensity"
    hgt = k.N.new("ShaderNodeValue")
    hgt.name = "ATM_FogHeight"
    colv = k.N.new("ShaderNodeRGB")
    colv.name = "ATM_FogColor"
    sep = k.n("ShaderNodeSeparateXYZ")
    k.ln(k.n("ShaderNodeNewGeometry").outputs["Position"], sep.inputs[0])
    z = k.math("MAXIMUM", sep.outputs["Z"], 0.0)
    e = k.math("EXPONENT", k.math("DIVIDE", k.math("MULTIPLY", z, -1.0), hgt.outputs[0]))
    # Drifting banks.
    nb = k.mrange(k.noise(k.n("ShaderNodeNewGeometry").outputs["Position"], 0.012, 3), 0.3, 0.7, 0.55, 1.35)
    d = k.math("MULTIPLY", k.math("MULTIPLY", e, dens.outputs[0]), nb)
    d = k.math("ADD", d, k.math("MULTIPLY", dens.outputs[0], 0.08))         # thin haze aloft
    vol = k.n("ShaderNodeVolumePrincipled", Anisotropy=0.45)
    k.ln(colv.outputs[0], vol.inputs["Color"])
    k.ln(d, vol.inputs["Density"])
    k.ln(vol.outputs[0], out.inputs["Volume"])
    me.materials.append(m)
    return ob


def build_rain(scene, coll, rng, n=60000, box=(70.0, 70.0, 26.0)):
    """Rain streak cards (instanced) in a box that is moved to the active camera per shot."""
    src = bpy.data.collections.new("ATM_Rain_Drop")
    scene.collection.children.link(src)
    m = bpy.data.materials.new("M_ATM_Rain")
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "BLENDED"
    m.use_backface_culling = False
    k = NK(m.node_tree)
    out = next(n for n in k.N if n.type == "OUTPUT_MATERIAL")
    for nn in list(k.N):
        if nn.type == "BSDF_PRINCIPLED":
            k.N.remove(nn)
    uv = k.n("ShaderNodeTexCoord").outputs["UV"]
    sep = k.n("ShaderNodeSeparateXYZ")
    k.ln(uv, sep.inputs[0])
    along = k.math("SUBTRACT", 1.0, k.math("ABSOLUTE", k.math("SUBTRACT", k.math("MULTIPLY", sep.outputs["Y"], 2.0), 1.0)))
    across = k.math("SUBTRACT", 1.0, k.math("ABSOLUTE", k.math("SUBTRACT", k.math("MULTIPLY", sep.outputs["X"], 2.0), 1.0)))
    a = k.math("MULTIPLY", k.math("MULTIPLY", along, across), 0.14)
    gl = k.n("ShaderNodeBsdfGlossy", Roughness=0.1)
    em = k.n("ShaderNodeEmission", Strength=0.6)
    em.inputs["Color"].default_value = (0.8, 0.85, 0.9, 1)
    ad = k.n("ShaderNodeAddShader")
    k.ln(gl.outputs[0], ad.inputs[0])
    k.ln(em.outputs[0], ad.inputs[1])
    tr = k.n("ShaderNodeBsdfTransparent")
    mix = k.n("ShaderNodeMixShader")
    k.ln(a, mix.inputs["Fac"])
    k.ln(tr.outputs[0], mix.inputs[1])
    k.ln(ad.outputs[0], mix.inputs[2])
    k.ln(mix.outputs[0], out.inputs["Surface"])
    w, h = 0.004, 0.35
    me = mesh_from_numpy("rain_drop", np.array([[-w, 0, -h], [w, 0, -h], [w, 0, h], [-w, 0, h]]),
                         np.array([[0, 1, 2, 3]]), uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), materials=[m],
                         smooth=False)
    drop = bpy.data.objects.new("rain_drop", me)
    drop.visible_shadow = False
    src.objects.link(drop)
    scene.view_layers[0].layer_collection.children[src.name].exclude = True
    pos = np.column_stack([rng.uniform(-box[0] / 2, box[0] / 2, n), rng.uniform(-box[1] / 2, box[1] / 2, n),
                           rng.uniform(-3, box[2], n)])
    # Density falls off with distance so the near field reads as streaks and far as haze.
    r = np.hypot(pos[:, 0], pos[:, 1])
    keep = rng.random(n) < np.clip(1.4 - r / (box[0] / 2), 0.15, 1)
    pos = pos[keep]
    m_ = len(pos)
    rot = np.column_stack([rng.normal(0, 0.06, m_), rng.normal(0, 0.06, m_), rng.uniform(0, np.pi, m_)])
    ob, mod = make_instancer("ATM_Rain", {"pos": pos, "rot": rot, "scale": rng.uniform(0.6, 1.4, m_),
                                          "variant": np.zeros(m_, int)}, src, coll)
    ob.visible_shadow = False
    return ob


def build_night_lights(scene, layout):
    """Warm work-lights at HQs, bunker, farmhouse and camp (only enabled at night)."""
    coll = bpy.data.collections.new("ATM_Night_Lights")
    scene.collection.children.link(coll)
    spots = []
    for name in ("HQ_South", "HQ_North"):
        s = layout["sites"][name]
        for dx, dy in ((-6, -6), (6, 6), (6, -6), (-6, 6)):
            spots.append((s["x"] + dx, s["y"] + dy, s["z"] + 3.2, 180.0, (1.0, 0.72, 0.42)))
        spots.append((s["x"], s["y"], s["z"] + 9.0, 400.0, (1.0, 0.85, 0.6)))
    for name, pw in (("Bunker", 60.0), ("Lumber_Camp", 120.0), ("Bridge", 90.0)):
        s = layout["sites"][name]
        spots.append((s["x"], s["y"], s["z"] + 2.6, pw, (1.0, 0.68, 0.38)))
    for i, (x, y, z, pw, c) in enumerate(spots):
        d = bpy.data.lights.new(f"ATM_WorkLight_{i:02d}", "POINT")
        d.energy = pw * 10
        d.color = c
        d.shadow_soft_size = 0.15
        ob = bpy.data.objects.new(d.name, d)
        ob.location = (x, y, z)
        coll.objects.link(ob)
    return coll


def hook_wetness(mat, darken=0.55, rough_drop=0.6, puddles=False):
    """Insert global wetness (scene['wetness'] via a View Layer attribute) into a material."""
    k = NK(mat.node_tree)
    bsdf = next((n for n in k.N if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
        return False
    wa = k.N.new("ShaderNodeAttribute")
    wa.attribute_type = "VIEW_LAYER"
    wa.attribute_name = "wetness"
    wet = wa.outputs["Fac"]
    if puddles:
        # Flat spots collect water: up-facing normal x noise -> near-mirror patches.
        geo = k.n("ShaderNodeNewGeometry")
        nz = k.n("ShaderNodeSeparateXYZ")
        k.ln(geo.outputs["Normal"], nz.inputs[0])
        flat = k.mrange(nz.outputs["Z"], 0.985, 0.998)
        pn = k.mrange(k.noise(geo.outputs["Position"], 0.35, 3), 0.55, 0.62)
        pud = k.math("MULTIPLY", k.math("MULTIPLY", flat, pn), wet)
    src = bsdf.inputs["Base Color"].links[0].from_socket
    col = k.mix(k.math("MULTIPLY", wet, 1.0), src, (darken, darken, darken * 0.97), "MULTIPLY")
    k.ln(col, bsdf.inputs["Base Color"])
    rin = bsdf.inputs["Roughness"]
    if rin.is_linked:
        rsrc = rin.links[0].from_socket
    else:
        v = k.N.new("ShaderNodeValue")
        v.outputs[0].default_value = rin.default_value
        rsrc = v.outputs[0]
    r = k.math("MULTIPLY", rsrc, k.math("SUBTRACT", 1.0, k.math("MULTIPLY", wet, rough_drop)))
    if puddles:
        r = k.math("MULTIPLY", r, k.math("SUBTRACT", 1.0, k.math("MULTIPLY", pud, 0.95)))
    k.ln(r, rin)
    return True
