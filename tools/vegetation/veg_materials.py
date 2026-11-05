"""Vegetation materials: barks (UV in metres) and alpha-card foliage.

Barks use CC0 photo scans from Poly Haven when the image is present in
assets/textures/bark (see SOURCE.md there); otherwise a procedural fallback.
Image barks store their physical tile size in the material's "uv_tile_m" property;
MeshBuilder divides the tube UVs (metres) by it, so the texture also maps correctly
after glTF export (no mapping node needed).
"""
import os

import bpy

BARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "textures", "bark")
# kind: (image file, physical size in metres (u around the trunk, v along it), procedural fallback)
BARK_IMAGES = {
    "pine": ("pine_bark.jpg", (2.0, 2.0), "pine"),
    "spruce": ("knotted_pine_bark.jpg", (1.7, 1.7), "pine"),
    "oak": ("jolcham_oak_bark_01.jpg", (1.0, 2.0), "oak"),
    "maple": ("trident_maple_bark.jpg", (1.8, 1.8), "oak"),
    "alder": ("japanese_hackberry_bark.jpg", (1.9, 1.9), "dead"),
    "dead": ("tree_bark_03.jpg", (1.0, 1.0), "dead"),
}


class K:
    def __init__(self, mat):
        self.N, self.L = mat.node_tree.nodes, mat.node_tree.links
        self.bsdf = next(n for n in self.N if n.type == "BSDF_PRINCIPLED")
        self.out = next(n for n in self.N if n.type == "OUTPUT_MATERIAL")

    def node(self, kind, **inputs):
        n = self.N.new(kind)
        for k, v in inputs.items():
            n.inputs[k].default_value = v
        return n

    def link(self, a, b):
        self.L.new(a, b)

    def ramp(self, fac, stops):
        r = self.N.new("ShaderNodeValToRGB")
        e = r.color_ramp.elements
        e[0].position, e[0].color = stops[0][0], (*stops[0][1], 1)
        e[1].position, e[1].color = stops[-1][0], (*stops[-1][1], 1)
        for pos, col in stops[1:-1]:
            e.new(pos).color = (*col, 1)
        self.link(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def uv_scaled(self, sx, sy):
        uv = self.node("ShaderNodeTexCoord").outputs["UV"]
        m = self.node("ShaderNodeMapping")
        m.inputs["Scale"].default_value = (sx, sy, 1)
        self.link(uv, m.inputs["Vector"])
        return m.outputs["Vector"]

    def bump(self, height, strength, dist=0.02):
        b = self.node("ShaderNodeBump", Strength=strength, Distance=dist)
        self.link(height, b.inputs["Height"])
        self.link(b.outputs["Normal"], self.bsdf.inputs["Normal"])


def _new(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m


def image_bark(kind):
    file, tile, fallback = BARK_IMAGES[kind]
    path = os.path.join(BARK_DIR, file)
    if not os.path.exists(path):
        print(f"BARK_IMAGE_MISSING {path} -> procedural {fallback}")
        return bark(fallback, name=kind)
    m = _new(f"M_Bark_{kind}")
    k = K(m)
    k.bsdf.inputs["Roughness"].default_value = 0.9
    tex = k.node("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path, check_existing=True)
    tex.extension = "REPEAT"
    k.link(k.node("ShaderNodeTexCoord").outputs["UV"], tex.inputs["Vector"])
    k.link(tex.outputs["Color"], k.bsdf.inputs["Base Color"])
    k.bump(tex.outputs["Color"], 0.6)
    m["uv_tile_m"] = list(tile)
    return m


def bark(kind, name=None):
    m = _new(f"M_Bark_{name or kind}")
    k = K(m)
    k.bsdf.inputs["Roughness"].default_value = 0.9
    if kind == "pine":
        # Scaly plates: voronoi cells stretched along the trunk, reddish-brown.
        v = k.node("ShaderNodeTexVoronoi", Scale=3.5)
        k.link(k.uv_scaled(1.6, 0.6), v.inputs["Vector"])
        n = k.node("ShaderNodeTexNoise", Scale=6.0, Detail=8.0)
        k.link(k.uv_scaled(1, 1), n.inputs["Vector"])
        mix = k.node("ShaderNodeMath", **{})
        mix.operation = "MULTIPLY_ADD"
        mix.inputs[1].default_value = 0.6
        k.link(v.outputs["Distance"], mix.inputs[0])
        k.link(n.outputs["Fac"], mix.inputs[2])
        col = k.ramp(mix.outputs[0], [(0.25, (0.025, 0.016, 0.010)), (0.55, (0.075, 0.042, 0.024)), (0.9, (0.15, 0.10, 0.07))])
        k.link(col, k.bsdf.inputs["Base Color"])
        k.bump(mix.outputs[0], 0.8)
    elif kind == "oak" or kind == "dead":
        # Deep vertical furrows: noise stretched along the trunk.
        n = k.node("ShaderNodeTexNoise", Scale=5.0, Detail=10.0, Roughness=0.6)
        k.link(k.uv_scaled(3.0, 0.35), n.inputs["Vector"])
        # Ridges: voronoi edge distance (blocky bark plates) modulated by noise.
        w = k.node("ShaderNodeTexVoronoi", Scale=2.5, Randomness=1.0)
        w.feature = "DISTANCE_TO_EDGE"
        k.link(k.uv_scaled(4.5, 0.22), w.inputs["Vector"])      # long vertical plates
        ridge = k.node("ShaderNodeMapRange", **{"From Min": 0.0, "From Max": 0.25})
        k.link(w.outputs["Distance"], ridge.inputs["Value"])
        h = k.node("ShaderNodeMath")
        h.operation = "MULTIPLY_ADD"
        h.inputs[1].default_value = 0.45
        k.link(n.outputs["Fac"], h.inputs[0])
        k.link(ridge.outputs["Result"], h.inputs[2])
        h2 = k.node("ShaderNodeMath")
        h2.operation = "MULTIPLY"
        h2.inputs[1].default_value = 0.7
        k.link(h.outputs[0], h2.inputs[0])
        h = h2
        if kind == "oak":
            stops = [(0.05, (0.012, 0.010, 0.008)), (0.35, (0.035, 0.029, 0.022)), (0.8, (0.075, 0.062, 0.048))]
        else:
            stops = [(0.05, (0.08, 0.075, 0.07)), (0.4, (0.25, 0.23, 0.21)), (0.9, (0.45, 0.43, 0.40))]
        k.link(k.ramp(h.outputs[0], stops), k.bsdf.inputs["Base Color"])
        k.bump(h.outputs[0], 1.0)
    elif kind == "birch":
        # White bark with dark horizontal lenticels and a dark rough base.
        n = k.node("ShaderNodeTexNoise", Scale=4.0, Detail=6.0)
        k.link(k.uv_scaled(0.6, 9.0), n.inputs["Vector"])
        marks = k.ramp(n.outputs["Fac"], [(0.62, (0.80, 0.78, 0.72)), (0.68, (0.05, 0.045, 0.04))])
        tc = k.node("ShaderNodeTexCoord")
        sep = k.node("ShaderNodeSeparateXYZ")
        k.link(tc.outputs["Object"], sep.inputs[0])
        base = k.node("ShaderNodeMapRange", **{"From Min": 0.3, "From Max": 1.4, "To Min": 1.0, "To Max": 0.0})
        k.link(sep.outputs["Z"], base.inputs["Value"])
        mix = k.node("ShaderNodeMix")
        mix.data_type = "RGBA"
        k.link(base.outputs["Result"], mix.inputs["Factor"])
        k.link(marks, mix.inputs[6])
        mix.inputs[7].default_value = (0.07, 0.06, 0.05, 1)
        k.link(mix.outputs[2], k.bsdf.inputs["Base Color"])
        k.bump(n.outputs["Fac"], 0.3)
    elif kind == "log_moss":
        n = k.node("ShaderNodeTexNoise", Scale=5.0, Detail=10.0, Roughness=0.6)
        k.link(k.uv_scaled(3.0, 0.35), n.inputs["Vector"])
        wood = k.ramp(n.outputs["Fac"], [(0.2, (0.05, 0.04, 0.03)), (0.8, (0.20, 0.17, 0.13))])
        geo = k.node("ShaderNodeNewGeometry")
        sep = k.node("ShaderNodeSeparateXYZ")
        k.link(geo.outputs["Normal"], sep.inputs[0])
        mn = k.node("ShaderNodeTexNoise", Scale=2.0, Detail=6.0)
        k.link(k.uv_scaled(1, 1), mn.inputs["Vector"])
        mm = k.node("ShaderNodeMath")
        mm.operation = "MULTIPLY_ADD"
        mm.inputs[1].default_value = 0.6
        k.link(mn.outputs["Fac"], mm.inputs[0])
        k.link(sep.outputs["Z"], mm.inputs[2])
        moss_f = k.node("ShaderNodeMapRange", **{"From Min": 0.7, "From Max": 0.95, "To Max": 0.85})
        k.link(mm.outputs[0], moss_f.inputs["Value"])
        mix = k.node("ShaderNodeMix")
        mix.data_type = "RGBA"
        k.link(moss_f.outputs["Result"], mix.inputs["Factor"])
        k.link(wood, mix.inputs[6])
        mix.inputs[7].default_value = (0.045, 0.075, 0.02, 1)
        k.link(mix.outputs[2], k.bsdf.inputs["Base Color"])
        k.bump(n.outputs["Fac"], 0.8)
    return m


def foliage(name, image_path, translucency=0.35, hue_var=0.06):
    """Alpha card material: texture colour, per-instance tint variation, translucency."""
    m = _new(f"M_Foliage_{name}")
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "DITHERED"
    m.use_backface_culling = False
    if hasattr(m, "use_transparent_shadow"):
        m.use_transparent_shadow = True
    k = K(m)
    img = bpy.data.images.load(image_path, check_existing=True)
    tex = k.node("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "CLIP"
    k.link(k.node("ShaderNodeTexCoord").outputs["UV"], tex.inputs["Vector"])
    # Per-instance colour variation (Object Info random -> hue/value shift).
    info = k.node("ShaderNodeObjectInfo")
    hsv = k.node("ShaderNodeHueSaturation")
    k.link(tex.outputs["Color"], hsv.inputs["Color"])
    hue = k.node("ShaderNodeMapRange", **{"To Min": 0.5 - hue_var, "To Max": 0.5 + hue_var})
    k.link(info.outputs["Random"], hue.inputs["Value"])
    k.link(hue.outputs["Result"], hsv.inputs["Hue"])
    val = k.node("ShaderNodeMapRange", **{"To Min": 0.8, "To Max": 1.15})
    k.link(info.outputs["Random"], val.inputs["Value"])
    k.link(val.outputs["Result"], hsv.inputs["Value"])
    k.link(hsv.outputs["Color"], k.bsdf.inputs["Base Color"])
    k.link(tex.outputs["Alpha"], k.bsdf.inputs["Alpha"])
    k.bsdf.inputs["Roughness"].default_value = 0.6
    # Translucency: mix in a Translucent BSDF lit from behind.
    tr = k.node("ShaderNodeBsdfTranslucent")
    k.link(hsv.outputs["Color"], tr.inputs["Color"])
    tr_alpha = k.node("ShaderNodeBsdfTransparent")
    mix = k.node("ShaderNodeMixShader", Fac=translucency)
    k.link(k.bsdf.outputs[0], mix.inputs[1])
    k.link(tr.outputs[0], mix.inputs[2])
    # Alpha must also cut the translucent lobe.
    cut = k.node("ShaderNodeMixShader")
    k.link(tex.outputs["Alpha"], cut.inputs["Fac"])
    k.link(tr_alpha.outputs[0], cut.inputs[1])
    k.link(mix.outputs[0], cut.inputs[2])
    k.link(cut.outputs[0], k.out.inputs["Surface"])
    return m


def build_all(tex_paths):
    mats = {f"bark_{b}": image_bark(b) for b in BARK_IMAGES}
    mats["bark_birch"] = bark("birch")
    mats["log_moss"] = bark("log_moss")
    for name, path in tex_paths.items():
        mats[name] = foliage(name, path, translucency=0.25 if name.endswith("needles") else 0.4,
                             hue_var=0.03 if name.endswith("needles") else 0.06)
    return mats
