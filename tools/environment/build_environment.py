"""Phase 5 entry point: roads, trails, bridge, outposts, ruins and other structures.

Usage (headless):
  blender -b --factory-startup --python tools/environment/build_environment.py -- \
      --rocks-dir output/phase4_rocks --water-dir output/phase2_water --out output/phase5_environment

The Phase 4 scene is regenerated locally: terrain is graded (roads, pads, trench),
every vegetation / rock instance is re-snapped to the new ground, and instances inside
roads or structure footprints are removed (cleared trees in the lumber camp become stumps).

Outputs in --out:
  forest_valley_environment.blend
  environment_layout.json     sites, road polylines, structures (world transforms, colliders, cover points)
  structures.glb              all structure meshes (world placed)
  heightmap_graded_1025_16bit.png / .r16, heightmap_graded_513_16bit.png / .r16
  preview_*.png, environment_meta.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "terrain"),
                os.path.join(HERE, "..", "water"), os.path.join(HERE, "..", "vegetation"), os.path.join(HERE, "..", "rocks")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import env_materials as EM  # noqa: E402
import layout as LY  # noqa: E402
import structures as ST  # noqa: E402
from blend_util import (bilinear, instancer_data, make_instancer, mark_sharp, mesh_from_numpy, point_attr,  # noqa: E402
                        rewrite_instancer, save, set_terrain_heights, terrain_arrays, terrain_attr, tri_count, box_uvs)
from build_terrain import look_at_rotation, make_camera, render, setup_render  # noqa: E402
from kit import Geo, rot_z  # noqa: E402
from placement import Fields, VegConfig, canopy_map  # noqa: E402
from png_io import heightmap_u16, write_gray16  # noqa: E402
from rock_materials import NK  # noqa: E402
from terrain_core import gaussian_blur, smoothstep  # noqa: E402
from views import ViewFinder  # noqa: E402
from water_core import project_to_polyline, upsample2_cubic  # noqa: E402

TREES = ("pine", "spruce", "oak", "maple", "alder", "birch", "dead")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocks-dir", default="output/phase4_rocks")
    ap.add_argument("--water-dir", default="output/phase2_water")
    ap.add_argument("--out", default="output/phase5_environment")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--shots", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="EnvConfig field=value")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- helpers

def geo_object(name, g, mats, coll, loc=(0, 0, 0), yaw=0.0):
    V, F, M = g.arrays()
    me = mesh_from_numpy(name, V, F, mat_idx=M, materials=[mats[s] for s in g.slots], smooth=True)
    mark_sharp(me, 30.0)
    box_uvs(me, 1.0)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = (0, 0, yaw)
    coll.objects.link(ob)
    return ob


def to_world(points, loc, yaw):
    R = rot_z(yaw)
    return [list(np.round(R @ np.asarray(p[:3], float) + np.asarray(loc, float), 3)) for p in points]


def ribbon(name, xy, half_w, Hfun, mat, coll, margin=0.7, lift=0.035, n_across=9):
    """Road surface strip draped on the graded terrain; UV u = 0..1 across, v = metres along."""
    d = np.gradient(xy, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    W = half_w + margin
    t = np.linspace(-1, 1, n_across)
    P = xy[:, None, :] + nrm[:, None, :] * (t[None, :, None] * W)
    z = Hfun(P[..., 0].ravel(), P[..., 1].ravel()).reshape(P.shape[:2]) + lift
    V = np.concatenate([P, z[..., None]], -1).reshape(-1, 3)
    n = len(xy)
    idx = np.arange(n * n_across).reshape(n, n_across)
    F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    U = np.broadcast_to((t + 1) / 2, (n, n_across))
    Vv = np.broadcast_to(s[:, None] / (2 * W), (n, n_across))
    uv_v = np.stack([U, Vv], -1).reshape(-1, 2)
    uvs = uv_v[F.ravel()]
    me = mesh_from_numpy(name, V, F, uvs=uvs, materials=[mat])
    ob = bpy.data.objects.new(name, me)
    ob.visible_shadow = False
    coll.objects.link(ob)
    return ob


def stamp_mask(res, size_m, discs):
    """discs: iterable of (x, y, r) -> bool mask on a res x res grid."""
    m = np.zeros((res, res), bool)
    cell = size_m / (res - 1)
    for x, y, r in discs:
        c0, r0 = int(round((x + size_m / 2) / cell)), int(round((y + size_m / 2) / cell))
        n = int(r / cell) + 1
        rs, cs = np.mgrid[max(r0 - n, 0):min(r0 + n + 1, res), max(c0 - n, 0):min(c0 + n + 1, res)]
        inside = ((cs - c0) * cell) ** 2 + ((rs - r0) * cell) ** 2 <= r * r
        m[rs[inside], cs[inside]] = True
    return m


def lookup(mask, size_m, x, y):
    res = mask.shape[0]
    cell = size_m / (res - 1)
    c = np.clip(np.round((x + size_m / 2) / cell).astype(int), 0, res - 1)
    r = np.clip(np.round((y + size_m / 2) / cell).astype(int), 0, res - 1)
    return mask[r, c]


def add_worn_layer(terrain_mat):
    """Compacted bare soil around roads, pads and camps (attribute 'worn')."""
    k = NK(terrain_mat.node_tree)
    bsdf = next(n for n in k.N if n.type == "BSDF_PRINCIPLED")
    prev = bsdf.inputs["Base Color"].links[0].from_socket
    pos = k.n("ShaderNodeNewGeometry").outputs["Position"]
    soil = k.ramp(k.noise(pos, 0.9, 8), [(0.3, (0.06, 0.047, 0.032)), (0.75, (0.13, 0.105, 0.075))])
    f = k.math("MULTIPLY", k.attr("worn"), k.mrange(k.noise(pos, 0.35, 4), 0.3, 0.6, 0.6, 1.0), clamp=True)
    k.ln(k.mix(f, prev, soil), bsdf.inputs["Base Color"])


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    timings = {}
    with open(os.path.join(os.path.abspath(a.water_dir), "water_bodies.json")) as f:
        wmeta = json.load(f)
    seed = wmeta["config"]["seed"]
    cfg = LY.EnvConfig(seed=seed)
    for kv in a.set:
        k, v = kv.split("=")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    rng = np.random.default_rng(seed + 5000)

    t = time.perf_counter()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(os.path.abspath(a.rocks_dir), "forest_valley_rocks.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    H0, res, size_m = terrain_arrays(terrain)
    under = terrain_attr(terrain, "underwater") > 0.5
    wet = terrain_attr(terrain, "wet")
    F = Fields(VegConfig(seed=seed), H0, size_m, wet, terrain_attr(terrain, "drainage"), under)
    trees = {sp: instancer_data(bpy.data.objects[f"VEG_Scatter_{sp}"]) for sp in TREES
             if bpy.data.objects.get(f"VEG_Scatter_{sp}")}
    canopy, _ = canopy_map(F, trees)
    canopy_full = np.clip(upsample2_cubic(canopy), 0, 1)[:res, :res]
    rp = wmeta["river_spline_points"]
    river = np.array([p["location_cm"] for p in rp]) / 100
    rs = np.array([p["s_m"] for p in rp])
    rw = np.array([p["width_cm"] for p in rp]) / 100
    L = LY.Layout(cfg, H0.copy(), size_m, under, canopy_full, river, rs, rw, wmeta["stats"]["rapids_at_s_m"])
    timings["load"] = time.perf_counter() - t

    # ---- sites
    t = time.perf_counter()
    sites = L.choose_sites(rng)
    b = sites["Bridge"]
    bdir = np.array([np.cos(b.yaw), np.sin(b.yaw)])
    span = b.extra["span"]
    ab = [np.array([b.x, b.y]) + s * bdir * (span / 2 + 2.0) for s in (-1, 1)]
    deck_z = max(b.extra["water_z"] + cfg.bridge_clearance_m + 0.4, *(float(L.h(*p)) + 0.15 for p in ab))
    # Pads (building platforms).
    pad_r = {"HQ_South": cfg.hq_radius_m, "HQ_North": cfg.hq_radius_m, "Farmhouse": 13, "Ruins": 13,
             "Lumber_Camp": 12, "Bunker": 5}
    for name, r in pad_r.items():
        s = sites[name]
        s.z = L.pad(s.x, s.y, r * 0.85, blend=7.0)
    timings["sites"] = time.perf_counter() - t

    # ---- roads / trails
    t = time.perf_counter()

    def gate(s, local=(1.0, 0.0)):
        R = rot_z(s.yaw)[:2, :2]
        return np.array([s.x, s.y]) + R @ np.asarray(local)

    hq_s, hq_n = sites["HQ_South"], sites["HQ_North"]
    g_s = gate(hq_s, (cfg.hq_radius_m * 0.72 + 6.0, 0))
    g_n = gate(hq_n, (cfg.hq_radius_m * 0.72 + 6.0, 0))
    def bank(p):
        return np.sign(project_to_polyline(np.array([p[0]]), np.array([p[1]]), river[:, :2])[2][0])

    a_s, a_n = (ab[0], ab[1]) if bank(ab[0]) == bank(g_s) else (ab[1], ab[0])
    # Approach ramps: leave each abutment straight along the bridge axis before routing.
    out_s = a_s + (a_s - [b.x, b.y]) / np.linalg.norm(a_s - [b.x, b.y]) * 7.0
    out_n = a_n + (a_n - [b.x, b.y]) / np.linalg.norm(a_n - [b.x, b.y]) * 7.0
    avoid = [s for n, s in sites.items() if n not in ("Bridge", "Trench")]
    roads = {}
    def straight(p, q):
        n = max(int(np.linalg.norm(q - p)), 2)
        return p + (q - p) * np.linspace(0, 1, n + 1)[:, None]

    roads["Road_South"] = np.vstack([L.route(g_s, out_s, "road", avoid)[:-1], straight(out_s, a_s)])
    roads["Road_North"] = np.vstack([straight(a_n, out_n)[:-1], L.route(out_n, g_n, "road", avoid)])
    main_xy = np.vstack([roads["Road_South"], roads["Road_North"]])
    main_bank = np.sign(project_to_polyline(main_xy[:, 0], main_xy[:, 1], river[:, :2])[2])
    half = {"Road_South": cfg.road_half_width_m, "Road_North": cfg.road_half_width_m}

    def branch(name, site, kind, half_w, dist_out):
        s = sites[site]
        d = np.hypot(main_xy[:, 0] - s.x, main_xy[:, 1] - s.y) + 1e4 * (main_bank != bank((s.x, s.y)))
        start = main_xy[int(np.argmin(d))]
        v = start - [s.x, s.y]
        end = np.array([s.x, s.y]) + v / np.linalg.norm(v) * dist_out
        roads[name] = L.route(start, end, kind, [x for x in avoid if x.name != site])
        half[name] = half_w

    branch("Road_Farmhouse", "Farmhouse", "road", 2.0, pad_r["Farmhouse"] * 0.85 + 1.0)
    branch("Track_Lumber_Camp", "Lumber_Camp", "road", 1.6, pad_r["Lumber_Camp"] * 0.85 + 1.0)
    branch("Trail_Ruins", "Ruins", "trail", cfg.trail_half_width_m, pad_r["Ruins"] * 0.85 + 1.0)
    branch("Trail_Bunker", "Bunker", "trail", cfg.trail_half_width_m, pad_r["Bunker"] * 0.85 + 1.0)
    # Stream crossings (any water a road passes over, other than at the main bridge): culvert
    # bridges; the road profile is anchored to the deck height on both sides.
    crossings, ribbons = [], {}
    for n, xy in roads.items():
        wet = lookup(L.water, size_m, xy[:, 0], xy[:, 1])
        fix = {}
        segs, cur0 = [], 0
        i = 0
        while i < len(xy):
            if not wet[i]:
                i += 1
                continue
            j = i
            while j < len(xy) and wet[j]:
                j += 1
            i0, i1 = max(i - 3, 0), min(j + 2, len(xy) - 1)
            zc = max(float(L.h(*xy[i0])), float(L.h(*xy[i1]))) + 0.25
            fix[i0], fix[i1] = zc, zc
            p0, p1 = xy[i0], xy[i1]
            crossings.append({"road": n, "p0": p0, "p1": p1, "z": zc, "width": 2 * half[n] + 0.6})
            segs.append((cur0, i0 + 1))
            cur0 = i1
            i = j
        segs.append((cur0, len(xy)))
        ribbons[n] = [xy[a_:b_] for a_, b_ in segs if b_ - a_ > 2]
        if n == "Road_South":
            fix[len(xy) - 1] = deck_z
        if n == "Road_North":
            fix[0] = deck_z
        L.grade_path(n, xy, half[n], z_fix=fix)
    widths = {n: L.paths[n]["half_width"] for n in roads}
    # Trench: zig-zag line across the approach on the far bank.
    tr = sites["Trench"]
    u = np.linspace(-16, 16, 33)
    zz = 1.6 * np.sign(np.sin(u / 6.0 * np.pi)) * np.minimum(1, np.abs(np.sin(u / 6.0 * np.pi)) * 3)
    R = rot_z(tr.yaw)[:2, :2]
    trench_xy = (R @ np.vstack([u, zz])).T + [tr.x, tr.y]
    ok = ~lookup(L.water | (L.water_dist < 4), size_m, trench_xy[:, 0], trench_xy[:, 1])
    # Leave a gap where a road crosses the trench line.
    for n, xy in roads.items():
        d = np.min(np.hypot(trench_xy[:, None, 0] - xy[None, :, 0], trench_xy[:, None, 1] - xy[None, :, 1]), axis=1)
        ok &= d > widths[n] + 3.0
    # Keep the longest contiguous run (a single continuous trench segment).
    runs, cur = [], []
    for i, k in enumerate(ok):
        if k:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    trench_xy = trench_xy[max(runs, key=len)]
    trench_floor = L.trench(trench_xy, cfg.trench_depth_m, cfg.trench_half_width_m)
    L.finish()
    timings["roads"] = time.perf_counter() - t

    # ---- apply terrain edits; re-snap instances
    t = time.perf_counter()
    dH = L.H - H0
    set_terrain_heights(terrain, L.H)
    from blend_util import snap_instancers_to_terrain
    snap_instancers_to_terrain(dH, size_m)
    # Clearing masks on a 0.5 m grid.
    mres = int(size_m / 0.5) + 1
    road_discs = [(x, y, widths[n] + 0.8) for n, xy in roads.items() for (x, y) in xy[::1]]
    road_discs += [(x, y, cfg.trench_half_width_m + 1.0) for (x, y) in trench_xy]
    bridge_discs = [(b.x + bdir[0] * s, b.y + bdir[1] * s, 3.5) for s in np.linspace(-span / 2 - 2, span / 2 + 2, 20)]
    site_discs = [(sites[n].x, sites[n].y, r) for n, r in pad_r.items()]
    hard = stamp_mask(mres, size_m, road_discs + bridge_discs + site_discs)
    trees_mask = stamp_mask(mres, size_m, [(x, y, r + 2.2) for (x, y, r) in road_discs + bridge_discs + site_discs])
    camp = sites["Lumber_Camp"]
    camp_mask = stamp_mask(mres, size_m, [(camp.x, camp.y, 22.0)])
    removed, stumps_xy = {}, []
    for ob in [o for o in bpy.data.objects if o.name.startswith(("VEG_Scatter_", "ROCK_Scatter_")) and o.type == "MESH"]:
        sp = ob.name.split("_Scatter_")[-1]
        p = instancer_data(ob)["pos"]
        if not len(p):
            continue
        big = sp in TREES or sp in ("bush", "log", "cliff", "mountain", "boulder", "weathered", "moss")
        drop = lookup(trees_mask if big else hard, size_m, p[:, 0], p[:, 1])
        if sp in TREES and sp != "dead":
            cut = lookup(camp_mask, size_m, p[:, 0], p[:, 1]) & ~drop
            cut &= rng.random(len(p)) < 0.75
            stumps_xy += [tuple(q[:2]) for q in p[cut | drop & lookup(camp_mask, size_m, p[:, 0], p[:, 1])]]
            drop |= cut
        if drop.any():
            removed[ob.name] = rewrite_instancer(ob, keep=~drop)
    timings["regen_instances"] = time.perf_counter() - t

    # ---- materials & structures
    t = time.perf_counter()
    mats = EM.build_all()
    ecol = bpy.data.collections.new("Environment")
    scene.collection.children.link(ecol)
    structs = []

    def place(name, g, site, z=None, yaw=None, sub=None):
        yaw = site.yaw if yaw is None else yaw
        loc = (site.x, site.y, site.z if z is None else z)
        coll = sub or ecol
        ob = geo_object(name, g, mats, coll, loc, yaw)
        structs.append({"name": name, "site": site.name, "location_m": list(np.round(loc, 3)),
                        "yaw_deg": round(float(np.degrees(yaw)), 2), "tris": tri_count(ob.data),
                        "colliders": [{"type": k, "center_m": to_world([c], loc, yaw)[0], "size_m": list(np.round(s, 3)),
                                       "yaw_deg": round(float(np.degrees(y + yaw)), 2)} for (k, c, s, y) in g.colliders],
                        "cover": [{"pos_m": to_world([(x, y, zc)], loc, yaw)[0], "kind": kd} for (x, y, zc, kd) in g.cover]})
        return ob

    def sub(name):
        c = bpy.data.collections.new(f"ENV_{name}")
        ecol.children.link(c)
        return c

    spawns = {}
    for name in ("HQ_South", "HQ_North"):
        g, info = ST.outpost(rng, cfg.hq_radius_m)
        place(f"SM_Outpost_{name}", g, sites[name], sub=sub(name))
        spawns[name] = to_world([(x, y, 0.0) for x, y in info["spawns_local"]], (sites[name].x, sites[name].y, sites[name].z),
                                sites[name].yaw)
    gb = ST.bridge(span, 4.4, deck_z - b.extra["water_z"] + 1.0, rng)
    b.z = deck_z
    place("SM_Bridge", gb, b, z=deck_z, yaw=b.yaw, sub=sub("Bridge"))
    place("SM_Farmhouse", ST.farmhouse(rng), sites["Farmhouse"], yaw=rng.uniform(0, 2 * np.pi), sub=sub("Farmhouse"))
    place("SM_Chapel_Ruins", ST.ruins(rng), sites["Ruins"], yaw=rng.uniform(0, 2 * np.pi), sub=sub("Ruins"))
    place("SM_Lumber_Camp", ST.lumber_camp(rng), camp, yaw=rng.uniform(0, 2 * np.pi), sub=sub("Lumber_Camp"))
    place("SM_Bunker", ST.bunker(rng), sites["Bunker"], sub=sub("Bunker"))
    # Trench works in world xy relative to the trench site, then dropped onto the floor profile.
    tc = sub("Trench")
    ny = rot_z(tr.yaw)[:2, 1]
    enemy = float(np.sign((b.x - tr.x) * ny[0] + (b.y - tr.y) * ny[1]) or 1.0)
    tg = ST.trench_works(trench_xy - [tr.x, tr.y], cfg.trench_depth_m, cfg.trench_half_width_m, enemy, rng)
    V = np.vstack(tg.v)
    _, sidx, _ = project_to_polyline(V[:, 0] + tr.x, V[:, 1] + tr.y, trench_xy)
    fz = np.interp(sidx, np.arange(len(trench_floor)), trench_floor)
    off = 0
    for i, vs in enumerate(tg.v):
        tg.v[i] = vs + np.column_stack([np.zeros(len(vs)), np.zeros(len(vs)), fz[off:off + len(vs)]])
        off += len(vs)
    tr.z = 0.0
    tg.colliders = [(k, [c[0], c[1], c[2] + float(np.interp(project_to_polyline(np.array([c[0] + tr.x]), np.array([c[1] + tr.y]),
                                                                                   trench_xy)[1][0], np.arange(len(trench_floor)),
                                                                     trench_floor))], s, y) for (k, c, s, y) in tg.colliders]
    tg.cover = [(x, y, z + float(np.interp(project_to_polyline(np.array([x + tr.x]), np.array([y + tr.y]), trench_xy)[1][0],
                                         np.arange(len(trench_floor)), trench_floor)), kd) for (x, y, z, kd) in tg.cover]
    place("SM_Trench", tg, tr, z=0.0, yaw=0.0, sub=tc)
    # Tank traps across the far-bank meadow in front of the trench.
    hh = ST.hedgehog(rng)
    for i, s_ in enumerate(np.linspace(-12, 12, 5)):
        p = np.array([tr.x, tr.y]) + rot_z(tr.yaw)[:2, :2] @ np.array([s_, 9.0 * enemy])
        if lookup(L.water, size_m, np.array([p[0]]), np.array([p[1]]))[0]:
            continue
        hs = LY.Site(f"Hedgehog_{i}", p[0], p[1], 1.0, rng.uniform(0, np.pi))
        hs.z = float(L.h(*p)) - 0.05
        place(f"SM_Hedgehog_{i:02d}", hh, hs, sub=tc)
    # Roads.
    rcol = sub("Roads")
    for n, segs in ribbons.items():
        for k, xy in enumerate(segs):
            ribbon(n if k == 0 else f"{n}_{k}", xy, widths[n], L.h, mats["trail" if n.startswith("Trail") else "road"],
                   rcol, margin=0.7 if widths[n] > 1 else 0.4)
    for k, c in enumerate(crossings):
        d = c["p1"] - c["p0"]
        span_c = float(np.linalg.norm(d))
        cs = LY.Site(f"Culvert_{k}", *((c["p0"] + c["p1"]) / 2), span_c / 2, float(np.arctan2(d[1], d[0])))
        cs.z = c["z"]
        place(f"SM_Culvert_Bridge_{k:02d}", ST.bridge(max(span_c - 4.0, 2.0), c["width"], 1.2, rng), cs, sub=rcol)
    timings["structures"] = time.perf_counter() - t

    # ---- debris: stumps (camp + forest) and branch piles
    t = time.perf_counter()
    dcol = bpy.data.collections.new("ENV_Debris_Assets")
    scene.collection.children.link(dcol)
    stump_c = bpy.data.collections.new("ENV_stump")
    pile_c = bpy.data.collections.new("ENV_branch_pile")
    dcol.children.link(stump_c)
    dcol.children.link(pile_c)
    for i in range(4):
        geo_object(f"stump_{i:02d}", ST.stump(rng), mats, stump_c)
        geo_object(f"branch_pile_{i:02d}", ST.branch_pile(rng), mats, pile_c)
    scene.view_layers[0].layer_collection.children[dcol.name].exclude = True
    # Forest stumps / branch piles: in forest, off roads.
    fx, fy = rng.uniform(-size_m / 2 + 5, size_m / 2 - 5, (2, 6000))
    can = bilinear(canopy_full, size_m, fx, fy)
    sl = bilinear(L.slope, size_m, fx, fy)
    keep = (rng.random(len(fx)) < can * 0.5) & (sl < 30) & ~lookup(trees_mask, size_m, fx, fy)
    keep &= ~lookup(L.water_dist < 3, size_m, fx, fy)
    fx, fy = fx[keep], fy[keep]
    n_st = len(fx) // 2
    st_xy = np.vstack([np.array(stumps_xy).reshape(-1, 2), np.column_stack([fx[:n_st], fy[:n_st]])])
    pl_xy = np.column_stack([fx[n_st:], fy[n_st:]])
    debris = {}
    for nm, xy, coll in (("stump", st_xy, stump_c), ("branch_pile", pl_xy, pile_c)):
        if not len(xy):
            continue
        z = L.h(xy[:, 0], xy[:, 1])
        d = {"pos": np.column_stack([xy, z - 0.02]), "rot": np.column_stack([np.zeros(len(xy)), np.zeros(len(xy)),
                                                                           rng.uniform(0, 2 * np.pi, len(xy))]),
             "scale": rng.uniform(0.8, 1.25, len(xy)), "variant": rng.integers(0, 4, len(xy))}
        make_instancer(f"ENV_Scatter_{nm}", d, coll, ecol)
        debris[nm] = d
    timings["debris"] = time.perf_counter() - t

    # ---- terrain: worn-soil layer
    worn_discs = [(x, y, widths[n] + 1.6) for n, xy in roads.items() for (x, y) in xy[::2]]
    worn_discs += [(sites[n].x, sites[n].y, r * 0.9) for n, r in pad_r.items()]
    worn = stamp_mask(res, size_m, worn_discs).astype(float)
    worn = np.clip(gaussian_blur(worn, 3) * 1.2, 0, 1) * (1 - under)
    point_attr(terrain.data, "worn", worn)
    add_worn_layer(terrain.data.materials[0])
    terrain.data.update()

    # ---- heightmaps (graded)
    u16, hmin, hmax = heightmap_u16(L.H)
    write_gray16(os.path.join(out, f"heightmap_graded_{res}_16bit.png"), u16)
    u16.astype("<u2").tofile(os.path.join(out, f"heightmap_graded_{res}.r16"))
    h513 = L.H[::2, ::2]
    u16b, _, _ = heightmap_u16(h513)
    write_gray16(os.path.join(out, "heightmap_graded_513_16bit.png"), u16b)
    u16b.astype("<u2").tofile(os.path.join(out, "heightmap_graded_513.r16"))

    # ---- cameras
    t = time.perf_counter()
    obst, obr = [], []
    for sp, d in trees.items():
        obst.append(d["pos"][:, :2])
        obr.append(np.full(len(d["pos"]), 2.0))
    obst = np.concatenate(obst)
    obr = np.concatenate(obr)
    VF = ViewFinder(L.h, obst, obr, lambda x, y: bilinear(canopy_full, size_m, x, y), size_m)
    shots = []
    for name, cam, dz, dists, hts in (("HQ_South", "Cam_HQ_South", 2.0, (32, 42, 55), (6, 12, 20)),
                                      ("Bridge", "Cam_Bridge", 1.0, (22, 30, 40), (3, 6, 10)),
                                      ("Farmhouse", "Cam_Farmhouse", 2.5, (18, 24, 32), (1.7, 4, 8)),
                                      ("Ruins", "Cam_Ruins", 2.5, (20, 26, 34), (1.7, 5, 10)),
                                      ("Lumber_Camp", "Cam_Lumber_Camp", 1.5, (16, 22, 28), (1.7, 4, 8)),
                                      ("Trench", "Cam_Trench", 0.0, (8, 12, 16), (1.7, 3)),
                                      ("Bunker", "Cam_Bunker", 1.0, (10, 14, 18), (1.7, 3))):
        s = sites[name]
        view, _ = VF.find((s.x, s.y), dz, dists, hts, headings=32)
        loc, tgt = view
        if name == "Bridge":
            tgt = (b.x, b.y, deck_z + 0.5)
        make_camera(scene, cam, loc, look_at_rotation(loc, tgt), lens=26)
        shots.append((cam, f"preview_{name.lower()}", 1600, 900))
    # Road-level eye camera on the main road looking along it.
    xy = roads["Road_South"]
    i = len(xy) // 2
    p0, p1 = xy[i], xy[min(i + 25, len(xy) - 1)]
    loc = (p0[0], p0[1], float(L.h(*p0)) + 1.7)
    make_camera(scene, "Cam_Road_Eye", loc, look_at_rotation(loc, (p1[0], p1[1], float(L.h(*p1)) + 1.5)), lens=24)
    shots = [("Cam_Oblique", "preview_map_oblique", 1920, 1080), ("Cam_Top", "preview_top", 1400, 1400),
             ("Cam_Road_Eye", "preview_road_eye", 1600, 900)] + shots
    timings["cameras"] = time.perf_counter() - t
    blend = os.path.join(out, "forest_valley_environment.blend")
    scene.camera = bpy.data.objects.get("Cam_Bridge")
    save(blend)

    # ---- exports
    t = time.perf_counter()
    bpy.ops.object.select_all(action="DESELECT")
    for o in ecol.all_objects:
        if o.name.startswith("SM_") or o.name.startswith(("Road_", "Track_", "Trail_")):
            o.select_set(True)
    try:
        bpy.ops.export_scene.gltf(filepath=os.path.join(out, "structures.glb"), use_selection=True, export_format="GLB")
    except Exception as exc:
        print("GLB_EXPORT_FAILED", exc)
    for o in list(stump_c.objects) + list(pile_c.objects):
        o.select_set(False)
    timings["export"] = time.perf_counter() - t

    if not a.no_render:
        t = time.perf_counter()
        setup_render(scene, a.samples)
        for cam, fname, w, h in shots:
            if a.shots is not None and fname not in a.shots:
                continue
            ob = bpy.data.objects.get(cam)
            if ob:
                render(scene, ob, os.path.join(out, fname + ".png"), w, h)
        timings["render"] = time.perf_counter() - t
        save(blend)

    # ---- validation
    def grade(segs):
        g = 0.0
        for xy in segs:
            z = L.h(xy[:, 0], xy[:, 1])
            s = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            g = max(g, float(np.max(np.abs(np.diff(z)) / np.maximum(s, 1e-6))))
        return g

    def cross(xy, hw):
        d = np.gradient(xy, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        nrm = np.column_stack([-d[:, 1], d[:, 0]])
        za = L.h(*(xy + nrm * hw).T)
        zb = L.h(*(xy - nrm * hw).T)
        return float(np.degrees(np.arctan(np.max(np.abs(za - zb)) / (2 * hw))))

    checks = {
        "road_max_grade_pct": {n: round(grade(ribbons[n]) * 100, 1) for n in roads},
        "stream_crossings_with_bridge": len(crossings),
        "road_max_cross_slope_deg": {n: round(cross(xy, widths[n]), 1) for n, xy in roads.items()},
        "structures_in_water": int(sum(lookup(L.water, size_m, np.array([s.x]), np.array([s.y]))[0]
                                       for n, s in sites.items() if n not in ("Bridge",))),
        "bridge_deck_above_water_m": round(deck_z - b.extra["water_z"], 2),
        "pad_flatness_max_dev_m": {n: round(float(np.abs(L.H[np.hypot(L.X - sites[n].x, L.Y - sites[n].y) < r * 0.8]
                                                         - sites[n].z).max()), 3) for n, r in pad_r.items()},
        "trees_on_roads": 0,
    }
    for sp in TREES:
        ob = bpy.data.objects.get(f"VEG_Scatter_{sp}")
        if ob:
            p = instancer_data(ob)["pos"]
            checks["trees_on_roads"] += int(lookup(hard, size_m, p[:, 0], p[:, 1]).sum())
    layout = {"sites": {n: s.as_dict() for n, s in sites.items()},
              "bridge": {"deck_z_m": round(deck_z, 3), "span_m": round(span, 2), "width_m": 4.4,
                         "abutments_xy_m": [list(np.round(p, 2)) for p in ab]},
              "roads": {n: {"half_width_m": widths[n], "kind": n.split("_")[0].lower(),
                            "polyline_xyz_m": np.round(np.column_stack([xy, L.h(xy[:, 0], xy[:, 1])]), 2).tolist()}
                        for n, xy in roads.items()},
              "trench": {"polyline_xy_m": np.round(trench_xy, 2).tolist(), "floor_z_m": np.round(trench_floor, 2).tolist(),
                         "depth_m": cfg.trench_depth_m},
              "hq_spawn_hints_m": spawns, "structures": structs,
              "debris": {n: {"columns": ["x_cm", "y_cm", "z_cm", "yaw_deg", "scale", "variant"],
                             "instances": np.round(np.column_stack([d["pos"] * 100, np.degrees(d["rot"][:, 2]), d["scale"],
                                                                    d["variant"]]), 2).tolist()} for n, d in debris.items()}}
    with open(os.path.join(out, "environment_layout.json"), "w") as f:
        json.dump(layout, f)
    stats = {"sites": list(sites), "structures": {s["name"]: s["tris"] for s in structs},
             "structure_tris_total": int(sum(s["tris"] for s in structs)),
             "road_length_m": {n: round(float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))), 1)
                               for n, xy in roads.items()},
             "colliders_total": int(sum(len(s["colliders"]) for s in structs)),
             "cover_points_total": int(sum(len(s["cover"]) for s in structs)),
             "debris": {n: len(d["pos"]) for n, d in debris.items()},
             "instances_removed": removed, "terrain_edit_m": {"max_cut": round(float(-dH.min()), 2),
                                                             "max_fill": round(float(dH.max()), 2),
                                                             "area_pct": round(float((np.abs(dH) > 0.02).mean() * 100), 2)},
             "checks": checks}
    timings["total_wall"] = time.perf_counter() - t_all
    meta = {"config": LY.config_dict(cfg), "stats": stats, "timings_s": {k: round(v, 3) for k, v in timings.items()},
            "heightmap_graded": {"file": f"heightmap_graded_{res}_16bit.png", "value_0_m": hmin, "value_65535_m": hmax}}
    with open(os.path.join(out, "environment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("PHASE5_SITES", json.dumps(layout["sites"]))
    print("PHASE5_STATS", json.dumps({"stats": stats, "timings_s": meta["timings_s"]}))


if __name__ == "__main__":
    main()
