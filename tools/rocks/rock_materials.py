"""PBR rock shading (procedural, object/world space so no UV seams in Blender;
UVs are still generated for engine export).

Layers: granite-like base with large tone variation and fine grain, dark fracture
lines (voronoi edges), pale lichen spots, moss on up-facing surfaces driven by a
per-instance `moss` attribute (habitat moisture x shade), wet darkening for river
rocks (`wet` instance attribute), cavity AO. The same colour network is reused as
a cliff layer on steep terrain so rock meshes and bare slopes match.
"""
import bpy


class NK:
    def __init__(self, nt):
        self.nt, self.N, self.L = nt, nt.nodes, nt.links

    def n(self, kind, **inputs):
        node = self.N.new(kind)
        for k, v in inputs.items():
            node.inputs[k].default_value = v
        return node

    def ln(self, a, b):
        self.L.new(a, b)

    def math(self, op, a, b=None, clamp=False):
        m = self.N.new("ShaderNodeMath")
        m.operation = op
        m.use_clamp = clamp
        for i, x in enumerate((a, b)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = x
            else:
                self.ln(x, m.inputs[i])
        return m.outputs[0]

    def mrange(self, v, a, b, c=0.0, d=1.0):
        m = self.n("ShaderNodeMapRange", **{"From Min": a, "From Max": b, "To Min": c, "To Max": d})
        self.ln(v, m.inputs["Value"])
        return m.outputs["Result"]

    def ramp(self, fac, stops):
        r = self.N.new("ShaderNodeValToRGB")
        e = r.color_ramp.elements
        e[0].position, e[0].color = stops[0][0], (*stops[0][1], 1)
        e[1].position, e[1].color = stops[-1][0], (*stops[-1][1], 1)
        for pos, col in stops[1:-1]:
            e.new(pos).color = (*col, 1)
        self.ln(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def mix(self, fac, a, b, blend="MIX"):
        m = self.N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        m.blend_type = blend
        for sock, x in ((m.inputs["Factor"], fac), (m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(x, (int, float)):
                sock.default_value = x
            elif isinstance(x, tuple):
                sock.default_value = (*x, 1) if len(x) == 3 else x
            else:
                self.ln(x, sock)
        return m.outputs[2]

    def attr(self, name, kind="GEOMETRY"):
        a = self.N.new("ShaderNodeAttribute")
        a.attribute_name = name
        a.attribute_type = kind
        return a.outputs["Fac"]

    def noise(self, vec, scale, detail=6.0, rough=0.55):
        t = self.n("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        self.ln(vec, t.inputs["Vector"])
        return t.outputs["Fac"]


def rock_color(k, vec, scale=1.0, tint=(1.0, 1.0, 1.0), strata_z=None):
    """Returns (colour, height) sockets of the shared rock surface network."""
    big = k.noise(vec, 0.3 / scale, 5)
    grain = k.noise(vec, 9.0 / scale, 14, 0.62)
    t = k.math("MULTIPLY_ADD", big, 0.8)
    k.ln(k.math("MULTIPLY", grain, 0.25), t.node.inputs[2])
    base = k.ramp(t, [(0.3, (0.034 * tint[0], 0.032 * tint[1], 0.029 * tint[2])),
                      (0.55, (0.085 * tint[0], 0.079 * tint[1], 0.069 * tint[2])),
                      (0.8, (0.15 * tint[0], 0.142 * tint[1], 0.127 * tint[2]))])
    # Vertical run-off streaks (stretched noise along Z).
    mp = k.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.5 / scale, 2.5 / scale, 0.25 / scale)
    k.ln(vec, mp.inputs["Vector"])
    streak = k.mrange(k.noise(mp.outputs["Vector"], 1.0, 4), 0.35, 0.65, 0.7, 1.05)
    base = k.mix(1.0, base, k.ramp(streak, [(0.0, (0, 0, 0)), (1.0, (1, 1, 1))]), "MULTIPLY")
    if strata_z is not None:
        # Sedimentary banding along Z, warped so beds are not dead level.
        w = k.n("ShaderNodeTexWave", Scale=strata_z, Distortion=4.0, Detail=3.0)
        w.wave_type = "BANDS"
        w.bands_direction = "Z"
        k.ln(vec, w.inputs["Vector"])
        band = k.mrange(w.outputs["Fac"], 0.0, 1.0, 0.8, 1.12)
        base = k.mix(1.0, base, k.ramp(band, [(0.0, (0, 0, 0)), (1.0, (1, 1, 1))]), "MULTIPLY")
    # Sparse fractures: thin iso-lines of a distorted noise, masked to a few regions.
    fn = k.n("ShaderNodeTexNoise", Scale=0.9 / scale, Detail=3.0, Distortion=0.6)
    k.ln(vec, fn.inputs["Vector"])
    line = k.math("ABSOLUTE", k.math("SUBTRACT", fn.outputs["Fac"], 0.5))
    line = k.mrange(line, 0.0, 0.018, 1.0, 0.0)
    mask = k.mrange(k.noise(vec, 0.5 / scale, 2), 0.45, 0.6)
    crack = k.math("MULTIPLY", line, mask)
    base = k.mix(k.math("MULTIPLY", crack, 0.75), base, (0.02, 0.018, 0.016))
    # Lichen: pale grey-green spots.
    lich = k.mrange(k.noise(vec, 4.0 / scale, 4, 0.5), 0.64, 0.70, 0.0, 0.35)
    base = k.mix(lich, base, (0.13, 0.135, 0.10))
    height = k.math("MULTIPLY_ADD", grain, 0.35)
    k.ln(k.math("MULTIPLY", crack, -0.6), height.node.inputs[2])
    height = k.math("ADD", height, k.math("MULTIPLY", big, 0.5))
    return base, height


def rock_material(name="M_Rock", tint=(1.0, 1.0, 1.0), smooth=False, strata=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    k = NK(m.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    tc = k.n("ShaderNodeTexCoord")
    vec = tc.outputs["Object"]
    col, height = rock_color(k, vec, 0.5 if smooth else 1.0, tint, strata_z=0.45 if strata else None)
    geo = k.n("ShaderNodeNewGeometry")
    nz = k.n("ShaderNodeSeparateXYZ")
    k.ln(geo.outputs["Normal"], nz.inputs[0])
    # Moss: up-facing x instance moisture, broken by noise (grows from crevices outward).
    up = k.mrange(nz.outputs["Z"], 0.15, 0.75)
    mo = k.math("MULTIPLY", up, k.attr("moss", "INSTANCER"))
    brk = k.mrange(k.noise(vec, 1.6, 6), 0.35, 0.6, 0.0, 1.6)
    mo = k.math("MULTIPLY", mo, brk)
    mo = k.math("MULTIPLY", mo, 1.6, clamp=True)
    moss_col = k.ramp(k.noise(vec, 11.0, 8), [(0.3, (0.022, 0.042, 0.010)), (0.7, (0.075, 0.11, 0.028))])
    col = k.mix(mo, col, moss_col)
    # Wet (river rocks): darker and glossier.
    wet = k.attr("wet", "INSTANCER")
    col = k.mix(k.math("MULTIPLY", wet, 0.8), col, (0.5, 0.48, 0.45), "MULTIPLY")
    ao = k.n("ShaderNodeAmbientOcclusion", Distance=0.6)
    k.ln(col, ao.inputs["Color"])
    aof = k.mrange(ao.outputs["AO"], 0.0, 1.0, 0.55, 1.0)
    col = k.mix(1.0, col, k.ramp(aof, [(0.0, (0, 0, 0)), (1.0, (1, 1, 1))]), "MULTIPLY")
    k.ln(col, bsdf.inputs["Base Color"])
    rough = k.mrange(wet, 0.0, 1.0, 0.62 if smooth else 0.86, 0.32)
    rough = k.math("ADD", rough, k.math("MULTIPLY", mo, 0.12), clamp=True)
    k.ln(rough, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    b = k.n("ShaderNodeBump", Strength=0.25 if smooth else 0.6, Distance=0.03)
    k.ln(height, b.inputs["Height"])
    k.ln(b.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def add_cliff_layer(terrain_mat):
    """Steep terrain shows the same rock (world space, strata) + scree at cliff bases."""
    k = NK(terrain_mat.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    prev = bsdf.inputs["Base Color"].links[0].from_socket
    geo = k.n("ShaderNodeNewGeometry")
    nz = k.n("ShaderNodeSeparateXYZ")
    k.ln(geo.outputs["Normal"], nz.inputs[0])
    col, height = rock_color(k, geo.outputs["Position"], 1.6, (1.0, 0.98, 0.94), strata_z=0.22)
    steep = k.mrange(nz.outputs["Z"], 0.80, 0.70)          # ~37 deg -> ~46 deg
    dry = k.math("SUBTRACT", 1.0, k.attr("wet"))
    steep = k.math("MULTIPLY", steep, dry)
    c = k.mix(steep, prev, col)
    # Scree / talus gravel where rockfall collects.
    scree = k.attr("scree")
    grav = k.ramp(k.noise(geo.outputs["Position"], 2.5, 10, 0.7),
                  [(0.3, (0.07, 0.066, 0.058)), (0.7, (0.19, 0.18, 0.16))])
    sc = k.math("MULTIPLY", scree, k.mrange(k.noise(geo.outputs["Position"], 0.5, 4), 0.35, 0.6), clamp=True)
    c = k.mix(sc, c, grav)
    k.ln(c, bsdf.inputs["Base Color"])
    b = k.n("ShaderNodeBump", Strength=0.35, Distance=0.05)
    k.ln(k.math("MULTIPLY", height, steep), b.inputs["Height"])
    k.ln(b.outputs["Normal"], bsdf.inputs["Normal"])
