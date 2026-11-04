"""Procedural PBR materials for man-made structures and roads (object-space textures;
UVs are also generated for export)."""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "rocks")]
from rock_materials import NK  # noqa: E402


def _base(name):
    """Texture space: the box-projected UVs (metres), so 2D patterns (bricks, mesh, weave,
    planks) lie on every face; object Z is still used for ground grime."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    k = NK(m.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    tc = k.n("ShaderNodeTexCoord")
    k.obj = tc.outputs["Object"]
    vec = tc.outputs["UV"]
    return m, k, bsdf, vec


def _bump(k, bsdf, h, strength, dist=0.02):
    b = k.n("ShaderNodeBump", Strength=strength, Distance=dist)
    k.ln(h, b.inputs["Height"])
    k.ln(b.outputs["Normal"], bsdf.inputs["Normal"])


def _dirt_grime(k, col, vec, amount=0.5):
    """Darker grime near the ground (object z) + blotches."""
    sep = k.n("ShaderNodeSeparateXYZ")
    k.ln(getattr(k, "obj", vec), sep.inputs[0])
    low = k.mrange(sep.outputs["Z"], 0.0, 1.2, amount, 0.0)
    blot = k.mrange(k.noise(vec, 1.2, 4), 0.45, 0.7, 0.0, amount * 0.6)
    f = k.math("ADD", low, blot, clamp=True)
    return k.mix(f, col, (0.05, 0.04, 0.03))


def wood(name, c0, c1, weathered=0.0, plank_scale=1.0):
    m, k, bsdf, vec = _base(name)
    mp = k.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 * plank_scale, 12.0, 12.0)
    k.ln(vec, mp.inputs["Vector"])
    grain = k.n("ShaderNodeTexWave", Scale=3.0, Distortion=8.0, Detail=4.0)
    grain.wave_type = "RINGS"
    k.ln(mp.outputs["Vector"], grain.inputs["Vector"])
    n = k.noise(vec, 3.0, 8)
    t = k.math("MULTIPLY_ADD", grain.outputs["Fac"], 0.5)
    k.ln(k.math("MULTIPLY", n, 0.5), t.node.inputs[2])
    col = k.ramp(t, [(0.2, c0), (0.8, c1)])
    if weathered:
        grey = k.mrange(k.noise(vec, 0.8, 5), 0.35, 0.65, 0.0, weathered)
        col = k.mix(grey, col, (0.16, 0.15, 0.14))
    col = _dirt_grime(k, col, vec, 0.35)
    k.ln(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    _bump(k, bsdf, grain.outputs["Fac"], 0.25, 0.01)
    return m


def masonry(name, stone0, stone1, mortar, scale=1.0, mossy=0.0):
    m, k, bsdf, vec = _base(name)
    br = k.n("ShaderNodeTexBrick", Scale=1.6 / scale, **{"Mortar Size": 0.035, "Mortar Smooth": 0.3, "Bias": 0.0,
                                                         "Brick Width": 0.7, "Row Height": 0.3})
    br.offset = 0.5
    br.squash = 1.0
    # Warp so courses are irregular rubble-stone, not machine bricks.
    wn = k.n("ShaderNodeTexNoise", Scale=0.7, Detail=2.0)
    k.ln(vec, wn.inputs["Vector"])
    warp = k.n("ShaderNodeMix")
    warp.data_type = "VECTOR"
    warp.inputs["Factor"].default_value = 0.22
    k.ln(vec, warp.inputs[4])
    k.ln(wn.outputs["Color"], warp.inputs[5])
    k.ln(warp.outputs[1], br.inputs["Vector"])
    tone = k.noise(vec, 4.0, 6)
    stones = k.ramp(tone, [(0.3, stone0), (0.75, stone1)])
    k.ln(stones, br.inputs["Color1"])
    k.ln(k.mix(0.5, stones, (0.03, 0.028, 0.024)), br.inputs["Color2"])
    br.inputs["Mortar Smooth"].default_value = 0.6
    br.inputs["Mortar"].default_value = (*mortar, 1)
    col = br.outputs["Color"]
    if mossy:
        geo = k.n("ShaderNodeNewGeometry")
        nz = k.n("ShaderNodeSeparateXYZ")
        k.ln(geo.outputs["Normal"], nz.inputs[0])
        up = k.mrange(nz.outputs["Z"], 0.4, 0.9, 0.0, mossy)
        sep = k.n("ShaderNodeSeparateXYZ")
        k.ln(k.obj, sep.inputs[0])
        low = k.mrange(sep.outputs["Z"], 0.0, 1.5, mossy * 0.7, 0.0)
        f = k.math("MULTIPLY", k.math("ADD", up, low, clamp=True), k.mrange(k.noise(vec, 2.0, 6), 0.4, 0.6))
        col = k.mix(f, col, (0.03, 0.06, 0.015))
    col = _dirt_grime(k, col, vec, 0.3)
    k.ln(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    h = k.math("MULTIPLY", br.outputs["Fac"], -1.0)
    h = k.math("ADD", h, k.math("MULTIPLY", tone, 0.3))
    _bump(k, bsdf, h, 0.7, 0.03)
    return m


def concrete(name):
    m, k, bsdf, vec = _base(name)
    n = k.noise(vec, 2.0, 10, 0.6)
    col = k.ramp(n, [(0.3, (0.17, 0.165, 0.15)), (0.7, (0.30, 0.29, 0.27))])
    mp = k.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (4.0, 4.0, 0.4)
    k.ln(vec, mp.inputs["Vector"])
    streak = k.mrange(k.noise(mp.outputs["Vector"], 1.0, 4), 0.45, 0.7, 0.0, 0.6)
    col = k.mix(streak, col, (0.07, 0.065, 0.055))
    col = _dirt_grime(k, col, vec, 0.45)
    k.ln(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.92
    _bump(k, bsdf, k.noise(vec, 25.0, 4), 0.15, 0.005)
    return m


def fabric(name, c0, c1, weave=60.0):
    m, k, bsdf, vec = _base(name)
    w = k.n("ShaderNodeTexWave", Scale=weave, Distortion=0.5)
    k.ln(vec, w.inputs["Vector"])
    n = k.noise(vec, 3.0, 6)
    col = k.ramp(n, [(0.3, c0), (0.7, c1)])
    col = _dirt_grime(k, col, vec, 0.4)
    k.ln(col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Sheen Weight"].default_value = 0.3
    _bump(k, bsdf, w.outputs["Fac"], 0.15, 0.003)
    return m


def hesco(name):
    """Geotextile bag (beige) with the welded steel mesh grid in front."""
    m, k, bsdf, vec = _base(name)
    n = k.noise(vec, 2.5, 8)
    bag = k.ramp(n, [(0.3, (0.22, 0.19, 0.13)), (0.7, (0.36, 0.32, 0.23))])
    grid = k.n("ShaderNodeTexBrick", Scale=1.0, **{"Mortar Size": 0.06, "Brick Width": 0.076, "Row Height": 0.076})
    grid.offset = 0.0
    k.ln(vec, grid.inputs["Vector"])
    wire = k.math("SUBTRACT", 1.0, grid.outputs["Fac"])
    col = k.mix(grid.outputs["Fac"], bag, (0.12, 0.12, 0.11))
    col = _dirt_grime(k, col, vec, 0.5)
    k.ln(col, bsdf.inputs["Base Color"])
    k.ln(k.mrange(grid.outputs["Fac"], 0, 1, 0.95, 0.45), bsdf.inputs["Roughness"])
    k.ln(k.mrange(grid.outputs["Fac"], 0, 1, 0.0, 0.8), bsdf.inputs["Metallic"])
    _bump(k, bsdf, wire, 0.4, 0.01)
    return m


def metal(name, paint=None, rust=0.6):
    m, k, bsdf, vec = _base(name)
    rn = k.noise(vec, 3.0, 10, 0.65)
    rust_c = k.ramp(rn, [(0.3, (0.10, 0.04, 0.015)), (0.7, (0.26, 0.11, 0.04))])
    if paint:
        f = k.mrange(k.noise(vec, 1.5, 8), 0.5 - rust * 0.15, 0.55, 1.0, 0.0)
        col = k.mix(f, rust_c, paint)
        k.ln(k.mrange(f, 0, 1, 0.85, 0.55), bsdf.inputs["Roughness"])
    else:
        col = rust_c
        bsdf.inputs["Roughness"].default_value = 0.85
        # Corrugation.
        w = k.n("ShaderNodeTexWave", Scale=10.0)
        w.wave_profile = "SIN"
        k.ln(vec, w.inputs["Vector"])
        _bump(k, bsdf, w.outputs["Fac"], 0.5, 0.03)
    bsdf.inputs["Metallic"].default_value = 0.35
    k.ln(col, bsdf.inputs["Base Color"])
    return m


def road(name, trail=False):
    """Dirt road: compacted wheel ruts (UV u across the road), grassy crown strip,
    gravel; edges fade out (dithered alpha) to blend into the terrain."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "DITHERED"
    k = NK(m.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    tc = k.n("ShaderNodeTexCoord")
    uvsep = k.n("ShaderNodeSeparateXYZ")
    k.ln(tc.outputs["UV"], uvsep.inputs[0])
    u = uvsep.outputs["X"]                           # 0..1 across
    pos = k.n("ShaderNodeNewGeometry").outputs["Position"]
    dirt = k.ramp(k.noise(pos, 1.2, 8), [(0.3, (0.075, 0.058, 0.04)), (0.75, (0.15, 0.12, 0.085))])
    grav = k.mrange(k.noise(pos, 14.0, 4), 0.55, 0.62, 0.0, 0.5)
    col = k.mix(grav, dirt, (0.19, 0.18, 0.16))
    centre = k.math("ABSOLUTE", k.math("SUBTRACT", u, 0.5))
    edge_fade = k.mrange(centre, 0.36, 0.5, 1.0, 0.0)
    if not trail:
        # Ruts at u ~ 0.3 / 0.7: darker, smoother, slightly lower (bump).
        rut = k.math("ABSOLUTE", k.math("SUBTRACT", centre, 0.2))
        rutf = k.mrange(rut, 0.0, 0.07, 1.0, 0.0)
        col = k.mix(k.math("MULTIPLY", rutf, 0.5), col, (0.05, 0.04, 0.03))
        crown = k.mrange(centre, 0.04, 0.1, 1.0, 0.0)
        cn = k.mrange(k.noise(pos, 3.0, 6), 0.4, 0.65, 0.0, 1.0)
        col = k.mix(k.math("MULTIPLY", crown, cn), col, (0.06, 0.08, 0.025))
        k.ln(k.mrange(rutf, 0, 1, 0.9, 0.7), bsdf.inputs["Roughness"])
        _bump(k, bsdf, k.math("ADD", k.math("MULTIPLY", rutf, -1.0), k.math("MULTIPLY", k.noise(pos, 6.0, 6), 0.4)),
              0.5, 0.05)
    else:
        bsdf.inputs["Roughness"].default_value = 0.92
    # Ragged edge: noise eats into the fade.
    rag = k.mrange(k.noise(pos, 2.5, 4), 0.3, 0.7, -0.35, 0.35)
    alpha = k.math("ADD", edge_fade, rag, clamp=True)
    alpha = k.mrange(alpha, 0.35, 0.65)
    k.ln(col, bsdf.inputs["Base Color"])
    k.ln(alpha, bsdf.inputs["Alpha"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    return m


def build_all():
    bark = bpy.data.materials.get("M_Bark_pine") or wood("M_Bark_fallback", (0.04, 0.03, 0.02), (0.1, 0.07, 0.05))
    return {
        "wood": wood("M_Wood_Planks", (0.06, 0.04, 0.025), (0.17, 0.12, 0.075), weathered=0.4),
        "wood_dark": wood("M_Wood_Dark", (0.03, 0.022, 0.015), (0.09, 0.065, 0.042), weathered=0.2),
        "wood_old": wood("M_Wood_Old", (0.05, 0.042, 0.034), (0.13, 0.115, 0.095), weathered=0.6),
        "wood_crate": wood("M_Wood_Crate", (0.05, 0.055, 0.03), (0.12, 0.13, 0.07), weathered=0.1),
        "wood_cut": wood("M_Wood_Cut", (0.25, 0.17, 0.09), (0.45, 0.33, 0.2), plank_scale=0.1),
        "log": wood("M_Log_Peeled", (0.05, 0.035, 0.022), (0.15, 0.105, 0.065), weathered=0.35, plank_scale=0.4),
        "bark": bark,
        "stone": masonry("M_Stone_Abutment", (0.05, 0.047, 0.042), (0.13, 0.12, 0.105), (0.07, 0.066, 0.058), 1.5, 0.5),
        "stone_wall": masonry("M_Stone_Farmhouse", (0.07, 0.058, 0.045), (0.17, 0.145, 0.115), (0.12, 0.11, 0.095), 1.5, 0.35),
        "stone_old": masonry("M_Stone_Ruin", (0.045, 0.044, 0.04), (0.13, 0.125, 0.11), (0.06, 0.058, 0.05), 1.8, 0.9),
        "plaster": concrete("M_Plaster_Old"),
        "concrete": concrete("M_Concrete"),
        "sandbag": fabric("M_Sandbag", (0.09, 0.075, 0.048), (0.17, 0.145, 0.095), 120.0),
        "canvas": fabric("M_Canvas_Olive", (0.04, 0.045, 0.02), (0.09, 0.095, 0.05), 200.0),
        "hesco": hesco("M_HESCO"),
        "metal_rust": metal("M_Metal_Corrugated_Rust"),
        "metal_paint": metal("M_Metal_Olive", paint=(0.05, 0.06, 0.03, 1), rust=0.5),
        "roof_slate": masonry("M_Roof_Slate", (0.03, 0.032, 0.035), (0.08, 0.085, 0.09), (0.02, 0.02, 0.02), 0.5),
        "road": road("M_Road_Dirt"),
        "trail": road("M_Trail_Dirt", trail=True),
    }
