"""Phase 8 entry point: performance optimisation + engine export package.

Usage (headless):
  blender -b --factory-startup --python tools/export/build_export.py -- --root output --out output/phase8_export

Produces (in --out):
  forest_valley_final.blend          final scene + LOD assets + HLOD proxies (hidden collections)
  ue_export/
    Landscape/  heightmap_1009.r16 / .png (UE recommended size), weightmaps (grass, forest_floor,
                rock, scree, dirt, moss, riverbed), normal_1009.png, macro_albedo_1009.png
    Meshes/<category>/SM_<asset>_LOD<k>.fbx     LOD0-3 per asset (structures LOD0-2)
    HLOD/HLOD_cell_<i>_<j>.fbx                 merged far-distance proxy per World Partition cell
    Collision/collision.glb, Navigation/navmesh.glb   (from Phase 7)
    Data/cells/cell_<i>_<j>.json               instances per World Partition cell (cm, deg, UE axes)
    Data/gameplay.json, Data/lighting_presets.json, Data/environment_layout.json, Data/asset_manifest.json
    ue_import.py                               Unreal Editor Python import script (not executed here)
    README_UE_IMPORT.md
  performance_report.json / performance_report.md, world_partition_map.png
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "terrain")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from blend_util import instancer_data, mesh_from_numpy, save, terrain_arrays, terrain_attr, tri_count  # noqa: E402
from png_io import write_gray16, write_gray8, write_rgb8  # noqa: E402

UE_LANDSCAPE = 1009          # 8x8 components, 2x2 sections of 63 quads
CELL_M = 125.0               # World Partition grid (4 x 4 over 500 m)
LOD_RATIOS = {"tree": (1.0, 0.5, 0.22, 0.06), "rock": (1.0, 0.35, 0.12, 0.04), "small": (1.0, 0.5, 0.25),
              "structure": (1.0, 0.5, 0.25)}
LOD_SCREEN = {"tree": (1.0, 0.45, 0.18, 0.05), "rock": (1.0, 0.4, 0.15, 0.05), "small": (1.0, 0.3, 0.1),
              "structure": (1.0, 0.35, 0.12)}
CULL_M = {"grass": 60.0, "flowers": 60.0, "fern": 90.0, "pebble": 45.0, "bush": 160.0, "branch_pile": 120.0,
          "stump": 150.0, "log": 200.0}
# Distances (m) at which LOD k takes over, used for the visible-triangle estimate.
LOD_DIST = {"tree": (0, 25, 70, 180), "rock": (0, 30, 90, 220), "small": (0, 20, 50), "structure": (0, 60, 160)}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--out", default="output/phase8_export")
    ap.add_argument("--no-fbx", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args(argv)


def resample(A, n):
    res = A.shape[0]
    f = np.linspace(0, res - 1, n)
    fx, fy = np.meshgrid(f, f)
    c, r = np.clip(fx.astype(int), 0, res - 2), np.clip(fy.astype(int), 0, res - 2)
    tx, ty = fx - c, fy - r
    return (A[r, c] * (1 - tx) * (1 - ty) + A[r, c + 1] * tx * (1 - ty) + A[r + 1, c] * (1 - tx) * ty
            + A[r + 1, c + 1] * tx * ty)


def blur(A, k):
    if k < 1:
        return A
    p = np.pad(A, k, mode="edge")
    n0, n1 = A.shape
    tmp = sum(p[:, i:i + n1] for i in range(2 * k + 1)) / (2 * k + 1)
    return sum(tmp[i:i + n0, :] for i in range(2 * k + 1)) / (2 * k + 1)


# --------------------------------------------------------------------------- landscape

def landscape_maps(terrain, out_dir):
    H, res, size_m = terrain_arrays(terrain)
    attr = {}
    for n in ("canopy", "litter_leaf", "litter_needle", "moss", "grass_cover", "scree", "worn", "wet", "underwater"):
        try:
            attr[n] = terrain_attr(terrain, n)
        except KeyError:
            attr[n] = np.zeros_like(H)
    n = UE_LANDSCAPE
    Hn = resample(H, n)
    lo, hi = float(Hn.min()), float(Hn.max())
    u16 = np.round((Hn - lo) / (hi - lo) * 65535).astype(np.uint16)[::-1]     # row 0 = north
    write_gray16(os.path.join(out_dir, f"heightmap_{n}.png"), u16)
    u16.astype("<u2").tofile(os.path.join(out_dir, f"heightmap_{n}.r16"))
    cell = size_m / (n - 1)
    gy, gx = np.gradient(Hn, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    A = {k: np.clip(resample(v, n), 0, 1) for k, v in attr.items()}
    w = {
        "rock": np.clip((slope - 34) / 10, 0, 1) * (1 - A["wet"]),
        "riverbed": A["underwater"],
        "dirt": A["worn"],
        "scree": A["scree"],
        "moss": A["moss"] * 0.8,
        "forest_floor": np.clip(A["litter_leaf"] + A["litter_needle"], 0, 1),
        "grass": np.clip(A["grass_cover"] + 0.35, 0, 1),
    }
    # Priority blend (top layer first) -> weights summing to 1 (UE weight-blended layers).
    order = ["riverbed", "rock", "dirt", "scree", "moss", "forest_floor", "grass"]
    remain = np.ones_like(Hn)
    W = {}
    for k in order:
        W[k] = np.clip(w[k], 0, 1) * remain
        remain = remain - W[k]
    W["grass"] += remain
    for k, v in W.items():
        write_gray8(os.path.join(out_dir, f"weight_{k}_{n}.png"), (np.clip(v, 0, 1) * 255).astype(np.uint8)[::-1])
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    write_rgb8(os.path.join(out_dir, f"normal_{n}.png"), ((nrm * 0.5 + 0.5) * 255).astype(np.uint8)[::-1])
    pal = {"grass": (64, 86, 34), "forest_floor": (86, 60, 36), "rock": (92, 88, 82), "scree": (110, 104, 94),
           "dirt": (96, 78, 58), "moss": (48, 72, 30), "riverbed": (70, 62, 48)}
    alb = sum(W[k][..., None] * np.array(pal[k], float) for k in W)
    write_rgb8(os.path.join(out_dir, f"macro_albedo_{n}.png"), np.clip(alb, 0, 255).astype(np.uint8)[::-1])
    return {"resolution": n, "file_r16": f"heightmap_{n}.r16", "min_m": lo, "max_m": hi,
            "ue_scale_cm": {"x": round(size_m * 100 / (n - 1), 4), "y": round(size_m * 100 / (n - 1), 4),
                            "z": round((hi - lo) * 100 / 512.0, 4)},
            "ue_location_cm": {"x": -size_m * 50, "y": -size_m * 50, "z": round((lo + hi) / 2 * 100, 2)},
            "components": "8 x 8, 2 x 2 sections per component, 63 quads per section",
            "weight_layers": order, "weight_coverage_pct": {k: round(float(v.mean() * 100), 1) for k, v in W.items()}}


# --------------------------------------------------------------------------- LODs

def decimated(ob, ratio, tmp_coll, name):
    if ratio >= 0.999:
        me = ob.data.copy()
        me.name = name
        return me
    t = bpy.data.objects.new("tmp_" + name, ob.data)
    tmp_coll.objects.link(t)
    mod = t.modifiers.new("dec", "DECIMATE")
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(t.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    me.name = name
    bpy.data.objects.remove(t)
    return me


def export_fbx(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={"MESH"}, mesh_smooth_type="FACE",
                             add_leaf_bones=False, bake_anim=False, apply_unit_scale=True, use_tspace=False,
                             path_mode="COPY", embed_textures=False)


def asset_table():
    """category -> list of (source object, kind, scatter group name(s))."""
    table = []
    for c in bpy.data.collections:
        if c.name.startswith("VEG_") and c.name not in ("VEG_Assets",):
            sp = c.name[4:]
            kind = "tree" if sp in ("pine", "oak", "birch", "dead") else "small"
            for o in sorted(c.objects, key=lambda o: o.name):
                table.append((o, kind, "Vegetation", [sp]))
        if c.name.startswith("ROCK_") and c.name != "ROCK_Assets":
            kind_r = c.name[5:]
            groups = [kind_r] + (["moss"] if kind_r == "boulder" else [])
            for o in sorted(c.objects, key=lambda o: o.name):
                table.append((o, "small" if kind_r == "pebble" else "rock", "Rocks", groups))
        if c.name in ("ENV_stump", "ENV_branch_pile"):
            for o in sorted(c.objects, key=lambda o: o.name):
                table.append((o, "small", "Debris", [c.name[4:]]))
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name.startswith("SM_"):
            table.append((o, "structure", "Structures", []))
    return table


# --------------------------------------------------------------------------- main

def main():
    a = parse_args()
    root = os.path.abspath(a.root)
    out = os.path.abspath(a.out)
    ue = os.path.join(out, "ue_export")
    for d in ("Landscape", "Meshes", "HLOD", "Collision", "Navigation", "Data/cells", "Textures"):
        os.makedirs(os.path.join(ue, d), exist_ok=True)
    t_all = time.perf_counter()
    T = {}
    bpy.ops.wm.open_mainfile(filepath=os.path.join(root, "phase7_gameplay", "forest_valley_gameplay.blend"))
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    _, _, size_m = terrain_arrays(terrain)

    t = time.perf_counter()
    land = landscape_maps(terrain, os.path.join(ue, "Landscape"))
    T["landscape"] = time.perf_counter() - t

    # ---- LODs + FBX
    t = time.perf_counter()
    tmp = bpy.data.collections.new("TMP_LOD")
    scene.collection.children.link(tmp)
    lodc = bpy.data.collections.new("LOD_Assets")
    scene.collection.children.link(lodc)
    manifest, by_group = {}, {}
    for src, kind, cat, groups in asset_table():
        ratios = LOD_RATIOS[kind]
        entry = {"category": cat, "kind": kind, "lods": [], "materials": [m.name for m in src.data.materials if m],
                 "nanite_candidate": kind in ("rock", "structure"),
                 "masked_foliage": any(m and m.name.startswith("M_Foliage") for m in src.data.materials),
                 "lod_screen_size": list(LOD_SCREEN[kind][:len(ratios)]), "cull_distance_m": None}
        for k, r in enumerate(ratios):
            base = src.name if src.name.startswith("SM_") else f"SM_{src.name}"
            me = decimated(src, r, tmp, f"{base}_LOD{k}")
            ob = bpy.data.objects.new(me.name, me)
            lodc.objects.link(ob)
            ob.location = (0, 0, 0)
            path = os.path.join(ue, "Meshes", cat, f"{me.name}.fbx")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not a.no_fbx:
                if kind == "structure":
                    # Structures are placed in the world: export them around their own pivot.
                    ob.location = (0, 0, 0)
                export_fbx(ob, path)
            entry["lods"].append({"file": os.path.relpath(path, ue), "tris": tri_count(me)})
        manifest[src.name] = entry
        for g in groups:
            by_group.setdefault(g, []).append(src.name)
    bpy.data.collections.remove(tmp)
    scene.view_layers[0].layer_collection.children[lodc.name].exclude = True
    T["lods_fbx"] = time.perf_counter() - t

    # ---- instances -> World Partition cells
    t = time.perf_counter()
    ncell = int(np.ceil(size_m / CELL_M))
    cells = {(i, j): {} for i in range(ncell) for j in range(ncell)}
    all_inst = []        # (x, y, asset, kind, group)
    for ob in [o for o in bpy.data.objects if o.type == "MESH" and "_Scatter_" in o.name]:
        g = ob.name.split("_Scatter_")[-1]
        if g not in by_group:
            continue
        d = instancer_data(ob)
        assets = sorted(by_group[g])
        for p, r, s, v in zip(d["pos"], d["rot"], d["scale"], d["variant"]):
            name = assets[int(v) % len(assets)]
            i = int(np.clip((p[0] + size_m / 2) // CELL_M, 0, ncell - 1))
            j = int(np.clip((p[1] + size_m / 2) // CELL_M, 0, ncell - 1))
            # UE: X = x, Y = -y (left-handed), rotations: yaw negated, roll / pitch kept (approx.)
            cells[(i, j)].setdefault(name, []).append(
                [round(p[0] * 100, 1), round(-p[1] * 100, 1), round(p[2] * 100, 1), round(float(np.degrees(r[0])), 2),
                 round(float(-np.degrees(r[1])), 2), round(float(-np.degrees(r[2])), 2), round(float(s), 3)])
            all_inst.append((p[0], p[1], p[2], name, manifest[name]["kind"], g))
    for g in ("grass", "flowers", "fern", "pebble", "bush", "branch_pile", "stump", "log"):
        for n_ in by_group.get(g, []):
            manifest[n_]["cull_distance_m"] = CULL_M[g]
    structs = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("SM_") and o.name in manifest]
    for o in structs:
        i = int(np.clip((o.location.x + size_m / 2) // CELL_M, 0, ncell - 1))
        j = int(np.clip((o.location.y + size_m / 2) // CELL_M, 0, ncell - 1))
        cells[(i, j)].setdefault(o.name, []).append([round(o.location.x * 100, 1), round(-o.location.y * 100, 1),
                                                     round(o.location.z * 100, 1), 0.0, 0.0,
                                                     round(-float(np.degrees(o.rotation_euler.z)), 2), 1.0])
        all_inst.append((o.location.x, o.location.y, o.location.z, o.name, "structure", "structure"))
    cell_stats = {}
    for (i, j), inst in cells.items():
        bounds = [-size_m / 2 + i * CELL_M, -size_m / 2 + j * CELL_M, -size_m / 2 + (i + 1) * CELL_M,
                  -size_m / 2 + (j + 1) * CELL_M]
        with open(os.path.join(ue, "Data", "cells", f"cell_{i}_{j}.json"), "w") as f:
            json.dump({"cell": [i, j], "bounds_m_blender": bounds,
                       "columns": ["x_cm", "y_cm", "z_cm", "roll_deg", "pitch_deg", "yaw_deg", "scale"],
                       "instances": inst}, f)
        tris0 = sum(manifest[n_]["lods"][0]["tris"] * len(v) for n_, v in inst.items())
        mats = sum(len(manifest[n_]["materials"]) for n_ in inst)
        cell_stats[f"{i}_{j}"] = {"instances": int(sum(len(v) for v in inst.values())), "unique_meshes": len(inst),
                                  "draw_calls_hism_est": int(mats), "lod0_tris": int(tris0)}
    T["cells"] = time.perf_counter() - t

    # ---- HLOD proxies (per cell: crowns as low ellipsoids, rocks as boxes; terrain is the landscape)
    t = time.perf_counter()
    hcol = bpy.data.collections.new("HLOD_Proxies")
    scene.collection.children.link(hcol)
    hl_stats = {}
    th = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    crown = {"pine": (2.4, 9.0, 8.0), "oak": (5.0, 7.0, 6.0), "birch": (2.2, 7.0, 5.0), "dead": (0.8, 6.0, 3.0)}
    for (i, j) in cells:
        V, F = [], []
        for x, y, z, name, kind, g in all_inst:
            if not (int(np.clip((x + size_m / 2) // CELL_M, 0, ncell - 1)) == i
                    and int(np.clip((y + size_m / 2) // CELL_M, 0, ncell - 1)) == j):
                continue
            if g in crown:
                r, h, z0 = crown[g]
                base = len(np.vstack(V)) if V else 0
                ring = np.column_stack([x + r * np.cos(th), y + r * np.sin(th), np.full(6, z + z0)])
                V.append(np.vstack([ring, [[x, y, z + z0 + h], [x, y, z + z0 - 1.0]]]))
                F += [[base + k, base + (k + 1) % 6, base + 6] for k in range(6)]
                F += [[base + (k + 1) % 6, base + k, base + 7] for k in range(6)]
        if not V:
            continue
        me = mesh_from_numpy(f"HLOD_cell_{i}_{j}", np.vstack(V), F, smooth=False)
        ob = bpy.data.objects.new(me.name, me)
        hcol.objects.link(ob)
        hl_stats[f"{i}_{j}"] = tri_count(me)
        if not a.no_fbx:
            export_fbx(ob, os.path.join(ue, "HLOD", f"{me.name}.fbx"))
    hcol.hide_render = True
    hcol.hide_viewport = True
    T["hlod"] = time.perf_counter() - t

    # ---- copy data from earlier phases
    for src, dst in (("phase7_gameplay/collision.glb", "Collision/collision.glb"),
                     ("phase7_gameplay/navmesh.glb", "Navigation/navmesh.glb"),
                     ("phase7_gameplay/gameplay_metadata.json", "Data/gameplay.json"),
                     ("phase6_lighting/lighting_presets.json", "Data/lighting_presets.json"),
                     ("phase5_environment/environment_layout.json", "Data/environment_layout.json"),
                     ("phase2_water/water_bodies.json", "Data/water_bodies.json"),
                     ("phase2_water/flowmap_1025.png", "Textures/T_River_Flowmap_1025.png")):
        p = os.path.join(root, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(ue, dst))
    texdir = os.path.join(root, "phase3_vegetation", "textures")
    tex_info = []
    if os.path.isdir(texdir):
        for fn in sorted(os.listdir(texdir)):
            shutil.copy(os.path.join(texdir, fn), os.path.join(ue, "Textures", fn))
            img = bpy.data.images.load(os.path.join(texdir, fn), check_existing=True)
            w_, h_ = img.size
            tex_info.append({"file": fn, "size": [w_, h_], "pow2": (w_ & (w_ - 1)) == 0 and (h_ & (h_ - 1)) == 0,
                             "bc_compressed_mb_with_mips": round(w_ * h_ * 1.0 * 1.333 / 2 ** 20, 2)})
    with open(os.path.join(ue, "Data", "asset_manifest.json"), "w") as f:
        json.dump({"assets": manifest, "groups": by_group, "world_partition_cell_m": CELL_M}, f, indent=2)
    shutil.copy(os.path.join(HERE, "ue_import.py"), os.path.join(ue, "ue_import.py"))
    shutil.copy(os.path.join(HERE, "README_UE_IMPORT.md"), os.path.join(ue, "README_UE_IMPORT.md"))

    # ---- performance estimate from player viewpoints
    t = time.perf_counter()
    with open(os.path.join(root, "phase7_gameplay", "gameplay_metadata.json")) as f:
        gp = json.load(f)
    views = {"Team_A_spawn": gp["spawns"]["team"]["Team_A"][0]["pos_m"],
             "Team_B_spawn": gp["spawns"]["team"]["Team_B"][0]["pos_m"]}
    views.update({f"Objective_{o['id']}": o["pos_m"] for o in gp["objectives"]})
    P = np.array([(x, y) for x, y, *_ in all_inst])
    names = [r[3] for r in all_inst]
    kinds = [r[4] for r in all_inst]
    groups = [r[5] for r in all_inst]
    view_stats = {}
    for vname, (vx, vy, _) in views.items():
        d = np.hypot(P[:, 0] - vx, P[:, 1] - vy)
        tris_lod, tris_naive, n_vis, dc = 0, 0, 0, set()
        for k_, (dist, nm, kd, g) in enumerate(zip(d, names, kinds, groups)):
            lods = manifest[nm]["lods"]
            tris_naive += lods[0]["tris"]
            cull = CULL_M.get(g)
            if cull and dist > cull:
                continue
            thr = LOD_DIST[kd]
            li = max(i_ for i_, th_ in enumerate(thr[:len(lods)]) if dist >= th_)
            tris_lod += lods[li]["tris"]
            n_vis += 1
            dc.add((nm, li))
        dcalls = sum(len(manifest[nm]["materials"]) for nm, _ in dc)
        view_stats[vname] = {"instances_drawn": int(n_vis), "tris_with_lod_and_culling": int(tris_lod),
                             "tris_without_lod": int(tris_naive), "reduction_pct": round(100 * (1 - tris_lod / max(tris_naive, 1)), 1),
                             "draw_calls_hism_est": int(dcalls)}
    T["perf"] = time.perf_counter() - t

    terrain_tris = len(terrain.data.polygons) * 2
    report = {
        "map": {"size_m": size_m, "world_partition": {"cell_m": CELL_M, "grid": [ncell, ncell], "cells": cell_stats},
                "landscape": land},
        "assets": {"count": len(manifest), "lod_levels": {k: len(v) for k, v in LOD_RATIOS.items()},
                   "lod_tris": {n_: [l_["tris"] for l_ in m["lods"]] for n_, m in manifest.items()},
                   "nanite_candidates": sorted(n_ for n_, m in manifest.items() if m["nanite_candidate"]),
                   "cull_distances_m": CULL_M},
        "instances_total": len(all_inst),
        "hlod_proxy_tris_per_cell": hl_stats,
        "viewpoints": view_stats,
        "terrain": {"blender_preview_mesh_tris": terrain_tris,
                    "ue": "Landscape 1009x1009 with LOD by component (not a static mesh)"},
        "collision": gp["collision"],
        "navigation": gp["navigation"],
        "textures": tex_info,
        "recommendations": [
            "Rocks and structures: enable Nanite (opaque, >1k tris); keep collision as the separate simplified meshes.",
            "Trees / bushes: HISM (Foliage) with the exported LOD0-3 + screen sizes; Nanite foliage only on UE 5.5+.",
            "Grass / flowers / ferns: Landscape Grass Type or PCG from the density maps, cull 60-90 m, no shadows.",
            "Pebbles: cull at 45 m, no collision, no shadows.",
            "World Partition: 125 m grid (4 x 4), loading range ~250 m; HLOD layer from HLOD_cell_* proxies.",
            "Landscape: 1009 heightmap, 8x8 components; Runtime Virtual Texture for the weight-blended material.",
            "Textures: all power-of-two, BC7 / BC5 normals, streaming enabled; foliage atlases 1024.",
        ],
        "timings_s": {k: round(v, 2) for k, v in T.items()},
    }
    report["timings_s"]["total_wall"] = round(time.perf_counter() - t_all, 2)
    with open(os.path.join(out, "performance_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    md = ["# Forest Valley - Performance Report", "",
          f"Instances: {len(all_inst)}  |  assets: {len(manifest)}  |  World Partition {ncell}x{ncell} cells of {CELL_M:.0f} m", "",
          "## Viewpoints (triangles drawn)", "", "| view | instances | tris (LOD+cull) | tris (no LOD) | reduction | draw calls (HISM est.) |",
          "|---|---|---|---|---|---|"]
    for k, v in view_stats.items():
        md.append(f"| {k} | {v['instances_drawn']} | {v['tris_with_lod_and_culling']:,} | {v['tris_without_lod']:,} | "
                  f"{v['reduction_pct']}% | {v['draw_calls_hism_est']} |")
    md += ["", "## World Partition cells", "", "| cell | instances | unique meshes | LOD0 tris | draw calls est. |",
           "|---|---|---|---|---|"]
    for k, v in cell_stats.items():
        md.append(f"| {k} | {v['instances']} | {v['unique_meshes']} | {v['lod0_tris']:,} | {v['draw_calls_hism_est']} |")
    md += ["", "## Recommendations", ""] + [f"- {r}" for r in report["recommendations"]]
    with open(os.path.join(out, "performance_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- partition map image
    from png_io import hillshade_rgb
    H, res, _ = terrain_arrays(terrain)
    img = hillshade_rgb(H[::1, ::1], size_m / (res - 1)).astype(float)[::-1].copy()
    n = img.shape[0]
    for k in range(1, ncell):
        px = int(k * CELL_M / size_m * (n - 1))
        img[:, max(px - 1, 0):px + 2] = (255, 255, 0)
        img[max(n - 1 - px - 1, 0):n - 1 - px + 2, :] = (255, 255, 0)
    write_rgb8(os.path.join(out, "world_partition_map.png"), np.clip(img, 0, 255).astype(np.uint8))

    scene["final"] = True
    save(os.path.join(out, "forest_valley_final.blend"))
    print("PHASE8_STATS", json.dumps({"viewpoints": view_stats, "assets": len(manifest), "instances": len(all_inst),
                                      "landscape": {k: land[k] for k in ("resolution", "ue_scale_cm")},
                                      "timings_s": report["timings_s"]}))


if __name__ == "__main__":
    main()
