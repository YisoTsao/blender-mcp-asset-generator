"""Shared bpy helpers for phases 4+: numpy mesh building, point-cloud instancers,
terrain read/write and instance filtering (local regeneration)."""
import bpy
import numpy as np


# --------------------------------------------------------------------------- meshes

def mesh_from_numpy(name, verts, faces, uvs=None, mat_idx=None, materials=(), smooth=True):
    """faces: (n, k) int array (all faces k-gons) or list of index lists.
    uvs: per-loop (n_loops, 2) in face order, optional."""
    verts = np.asarray(verts, np.float32)
    if isinstance(faces, np.ndarray):
        sizes = np.full(len(faces), faces.shape[1], np.int32)
        flat = faces.astype(np.int32).ravel()
    else:
        sizes = np.array([len(f) for f in faces], np.int32)
        flat = np.concatenate([np.asarray(f, np.int32) for f in faces]) if len(faces) else np.zeros(0, np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(flat))
    me.loops.foreach_set("vertex_index", flat)
    me.polygons.add(len(sizes))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me.polygons.foreach_set("loop_start", starts)
    if mat_idx is not None:
        me.polygons.foreach_set("material_index", np.asarray(mat_idx, np.int32))
    if uvs is not None:
        uv = me.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", np.asarray(uvs, np.float32).ravel())
    me.update(calc_edges=True)
    me.validate(clean_customdata=False)
    for m in materials:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    return me


def box_uvs(me, scale=1.0):
    """Per-loop box (tri-planar) projection in metres: robust UVs for engine export."""
    nl = len(me.loops)
    vi = np.empty(nl, np.int32)
    me.loops.foreach_get("vertex_index", vi)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    pn = np.empty(len(me.polygons) * 3, np.float32)
    me.polygons.foreach_get("normal", pn)
    pn = np.abs(pn.reshape(-1, 3))
    tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", tot)
    ln = np.repeat(pn, tot, axis=0)
    axis = ln.argmax(1)
    p = co[vi] * scale
    uv = np.where(axis[:, None] == 0, p[:, [1, 2]], np.where(axis[:, None] == 1, p[:, [0, 2]], p[:, [0, 1]]))
    lay = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    lay.data.foreach_set("uv", uv.astype(np.float32).ravel())


def mark_sharp(me, angle_deg=35.0):
    """Flat-shade hard edges (fractures) but keep smooth shading elsewhere."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    lim = np.radians(angle_deg)
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > lim:
            e.smooth = False
    bm.to_mesh(me)
    bm.free()


def tri_count(me):
    tot = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", tot)
    return int((tot - 2).sum())


# --------------------------------------------------------------------------- attributes

def point_attr(me, name, values, kind="FLOAT"):
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name=name, type=kind, domain="POINT")
    key = {"FLOAT_VECTOR": "vector", "INT": "value"}.get(kind, "value")
    a.data.foreach_set(key, np.asarray(values, np.int32 if kind == "INT" else np.float32).ravel())


def read_point_attr(me, name):
    a = me.attributes[name]
    n = len(a.data)
    if a.data_type == "FLOAT_VECTOR":
        v = np.empty(n * 3, np.float32)
        a.data.foreach_get("vector", v)
        return v.reshape(n, 3)
    v = np.empty(n, np.int32 if a.data_type == "INT" else np.float32)
    a.data.foreach_get("value", v)
    return v


# --------------------------------------------------------------------------- terrain

def terrain_arrays(ob):
    me = ob.data
    n = len(me.vertices)
    res = int(round(np.sqrt(n)))
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    H = co.reshape(-1, 3)[:, 2].reshape(res, res).astype(np.float64)
    return H, res, float(ob.dimensions.x)


def terrain_attr(ob, name):
    me = ob.data
    res = int(round(np.sqrt(len(me.vertices))))
    return read_point_attr(me, name).reshape(res, res).astype(np.float64)


def set_terrain_heights(ob, H):
    me = ob.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co[:, 2] = H.ravel()
    me.vertices.foreach_set("co", co.ravel())
    me.update()


# --------------------------------------------------------------------------- instancers

def make_instancer(name, data, src_coll, target, extra=None, realize=False):
    """Point cloud (pos/rot/scale/variant + extra float attrs) instancing `src_coll` children
    via Geometry Nodes. Extra attributes propagate to the instances and can be read in
    shaders with an Attribute node of type INSTANCER."""
    n = len(data["pos"])
    me = bpy.data.meshes.new(name)
    me.vertices.add(n)
    me.vertices.foreach_set("co", np.asarray(data["pos"], np.float32).ravel())
    point_attr(me, "rot", data["rot"], "FLOAT_VECTOR")
    point_attr(me, "scale", data["scale"])
    point_attr(me, "variant", data["variant"], "INT")
    for k, v in (extra or {}).items():
        point_attr(me, k, v, "FLOAT_VECTOR" if np.ndim(v) == 2 else "FLOAT")
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
    e2r = N.new("FunctionNodeEulerToRotation")
    L.new(named("rot", "FLOAT_VECTOR"), e2r.inputs[0])
    L.new(e2r.outputs[0], iop.inputs["Rotation"])
    L.new(named("scale", "FLOAT"), iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs[0])
    mod = ob.modifiers.new("Scatter", "NODES")
    mod.node_group = ng
    return ob, mod


def instancer_data(ob):
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    out = {"pos": co.reshape(n, 3).astype(np.float64)}
    for a in me.attributes:
        if a.name in ("position",) or a.name.startswith(".") or a.domain != "POINT":
            continue
        out[a.name] = read_point_attr(me, a.name)
    return out


def rewrite_instancer(ob, keep=None, dz=None):
    """Drop points (keep mask) and/or move them vertically; keeps all attributes."""
    d = instancer_data(ob)
    n = len(d["pos"])
    keep = np.ones(n, bool) if keep is None else np.asarray(keep, bool)
    if dz is not None:
        d["pos"][:, 2] += dz
    old = ob.data
    me = bpy.data.meshes.new(old.name)
    m = int(keep.sum())
    me.vertices.add(m)
    me.vertices.foreach_set("co", d["pos"][keep].astype(np.float32).ravel())
    for k, v in d.items():
        if k == "pos":
            continue
        kind = "FLOAT_VECTOR" if v.ndim == 2 else ("INT" if v.dtype.kind == "i" else "FLOAT")
        point_attr(me, k, v[keep], kind)
    ob.data = me
    bpy.data.meshes.remove(old)
    me.name = ob.name
    return int(n - m)


def bilinear(H, size_m, x, y):
    res = H.shape[0]
    fx = np.clip((np.asarray(x) + size_m / 2) / size_m * (res - 1), 0, res - 1.001)
    fy = np.clip((np.asarray(y) + size_m / 2) / size_m * (res - 1), 0, res - 1.001)
    c, r = fx.astype(int), fy.astype(int)
    tx, ty = fx - c, fy - r
    return (H[r, c] * (1 - tx) * (1 - ty) + H[r, c + 1] * tx * (1 - ty)
            + H[r + 1, c] * (1 - tx) * ty + H[r + 1, c + 1] * tx * ty)


def all_instancers(prefixes=("VEG_Scatter_", "ROCK_Scatter_")):
    return [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith(prefixes)]


def snap_instancers_to_terrain(dH, size_m, prefixes=("VEG_Scatter_", "ROCK_Scatter_")):
    """After terrain edits: move every instance by the terrain height change under it."""
    for ob in all_instancers(prefixes):
        p = instancer_data(ob)["pos"]
        if len(p):
            dz = bilinear(dH, size_m, p[:, 0], p[:, 1])
            if np.abs(dz).max() > 1e-4:
                rewrite_instancer(ob, dz=dz)


def remove_instances_in(mask_fn, prefixes=("VEG_Scatter_", "ROCK_Scatter_"), margins=None):
    """mask_fn(x, y, margin) -> bool array of points to remove. margins: {name_suffix: m}."""
    removed = {}
    for ob in all_instancers(prefixes):
        p = instancer_data(ob)["pos"]
        if not len(p):
            continue
        sp = ob.name.split("_Scatter_")[-1]
        drop = mask_fn(p[:, 0], p[:, 1], (margins or {}).get(sp, 0.0))
        if drop.any():
            removed[ob.name] = rewrite_instancer(ob, keep=~drop)
    return removed


def save(path):
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    import os
    if os.path.exists(path + "1"):
        os.remove(path + "1")
