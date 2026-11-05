"""Phase 7 entry point: FPS gameplay integration.

Usage (headless):
  blender -b --factory-startup --python tools/gameplay/build_gameplay.py -- \
      --scene-dir output/phase6_lighting --env-dir output/phase5_environment --out output/phase7_gameplay

Adds to the scene (hidden from renders):
  Gameplay   spawn / objective / pickup / cover markers (empties)
  Collision  separate simplified collision: terrain (2 m), structure boxes (UCX_), tree capsules, rock hulls
  NavMesh    walkable surface (agent radius 0.4 m, slope <= 40 deg, wade depth <= 0.7 m)
Outputs:
  gameplay_metadata.json   spawns, objectives, pickups, cover, combat / kill zones, routes, validation
  navmesh.glb, collision.glb, tactical_map.png, exposure_map.png, forest_valley_gameplay.blend
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "common"), os.path.join(HERE, "..", "terrain")]

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import gameplay_core as GC  # noqa: E402
from blend_util import instancer_data, mesh_from_numpy, save, terrain_arrays, terrain_attr  # noqa: E402
from png_io import hillshade_rgb, write_rgb8  # noqa: E402

TREE_H = {"pine": 18.0, "spruce": 21.0, "oak": 13.0, "maple": 15.0, "alder": 12.5, "birch": 12.0, "dead": 10.0}
TRUNK_R = {"pine": 0.3, "spruce": 0.3, "oak": 0.55, "maple": 0.42, "alder": 0.24, "birch": 0.22, "dead": 0.25}
CROWN_R = {"pine": 1.6, "spruce": 1.6, "oak": 3.5, "maple": 3.0, "alder": 1.8, "birch": 1.6, "dead": 0.0}
SINK = {"cliff": 0.6, "mountain": 0.35, "boulder": 0.25, "weathered": 0.42, "moss": 0.32, "river": 0.38, "pebble": 0.35}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="output/phase6_lighting")
    ap.add_argument("--env-dir", default="output/phase5_environment")
    ap.add_argument("--out", default="output/phase7_gameplay")
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args(argv)


def empty(coll, name, loc, kind="PLAIN_AXES", size=1.0, yaw=0.0, props=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = kind
    ob.empty_display_size = size
    ob.location = loc
    ob.rotation_euler = (0, 0, yaw)
    for k, v in (props or {}).items():
        ob[k] = v
    coll.objects.link(ob)
    return ob


def grid_mesh(name, G, mask, step, coll):
    """Quad mesh of the masked cells of grid G (every `step` cells)."""
    H = G.H[::step, ::step]
    m = mask[::step, ::step]
    n0, n1 = H.shape
    X, Y = G.X[::step, ::step], G.Y[::step, ::step]
    keep = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
    idx = np.arange(n0 * n1).reshape(n0, n1)
    F = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1)[keep]
    used = np.unique(F)
    remap = -np.ones(n0 * n1, int)
    remap[used] = np.arange(len(used))
    V = np.column_stack([X.ravel(), Y.ravel(), H.ravel()])[used]
    me = mesh_from_numpy(name, V, remap[F], smooth=False)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob, len(F)


def boxes_mesh(name, boxes, coll):
    V, F = [], []
    base = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) / 2
    BF = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    for i, (c, s, yaw) in enumerate(boxes):
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        V.append((base * s) @ R.T + c)
        F.append(BF + 8 * i)
    me = mesh_from_numpy(name, np.vstack(V), np.vstack(F), smooth=False)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def cylinders_mesh(name, cyl, coll, seg=8):
    V, F = [], []
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for i, (x, y, z0, r, h) in enumerate(cyl):
        ring = np.column_stack([x + r * np.cos(th), y + r * np.sin(th)])
        V.append(np.vstack([np.column_stack([ring, np.full(seg, z0)]), np.column_stack([ring, np.full(seg, z0 + h)])]))
        o = 2 * seg * i
        F += [[o + j, o + (j + 1) % seg, o + seg + (j + 1) % seg, o + seg + j] for j in range(seg)]
        F += [[o + j for j in range(seg)][::-1], [o + seg + j for j in range(seg)]]
    me = mesh_from_numpy(name, np.vstack(V), F, smooth=False)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def disc_px(img, size_m, x, y, r_px, col, alpha=1.0, ring=False):
    n = img.shape[0]
    c = int((x + size_m / 2) / size_m * (n - 1))
    r = int((1 - (y + size_m / 2) / size_m) * (n - 1))
    rs, cs = np.mgrid[max(r - r_px - 2, 0):min(r + r_px + 3, n), max(c - r_px - 2, 0):min(c + r_px + 3, n)]
    d = np.hypot(rs - r, cs - c)
    m = (np.abs(d - r_px) < 1.6) if ring else (d <= r_px)
    img[rs[m], cs[m]] = img[rs[m], cs[m]] * (1 - alpha) + np.array(col) * alpha


def main():
    a = parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    t_all = time.perf_counter()
    T = {}
    cfg = GC.GameConfig()
    rng = np.random.default_rng(cfg.seed + 7000)
    with open(os.path.join(os.path.abspath(a.env_dir), "environment_layout.json")) as f:
        lay = json.load(f)
    sdir = os.path.abspath(a.scene_dir)
    src = os.path.join(sdir, "forest_valley_lighting.blend")
    if not os.path.exists(src):
        src = os.path.join(os.path.abspath(a.env_dir), "forest_valley_environment.blend")
    bpy.ops.wm.open_mainfile(filepath=src)
    scene = bpy.context.scene
    terrain = bpy.data.objects["Terrain"]
    H, res, size_m = terrain_arrays(terrain)
    depth_full = terrain_attr(terrain, "water_depth") * terrain_attr(terrain, "underwater")
    t = time.perf_counter()
    G = GC.Grid(H, size_m, cfg.nav_cell_m)
    depth = GC._bilinear_idx(depth_full, (G.X + size_m / 2) / size_m * (res - 1), (G.Y + size_m / 2) / size_m * (res - 1))

    # ---- obstacles
    discs, crowns, cyl = [], [], []
    for sp in TREE_H:
        ob = bpy.data.objects.get(f"VEG_Scatter_{sp}")
        if not ob:
            continue
        d = instancer_data(ob)
        for p, s in zip(d["pos"], d["scale"]):
            discs.append((p[0], p[1], TRUNK_R[sp] * s, TREE_H[sp] * s))
            cyl.append((p[0], p[1], p[2], TRUNK_R[sp] * s, 4.0))
            if CROWN_R[sp]:
                crowns.append((p[0], p[1], CROWN_R[sp] * s * 0.7, TREE_H[sp] * s))
    rock_dims = {}
    for c in bpy.data.collections:
        if c.name.startswith("ROCK_") and c.name != "ROCK_Assets" and len(c.objects):
            rock_dims[c.name[5:]] = np.array([o.dimensions[:] for o in sorted(c.objects, key=lambda o: o.name)])
    rock_dims["moss"] = rock_dims.get("boulder")
    rock_hulls = []
    rock_cover = []
    for ob in [o for o in bpy.data.objects if o.name.startswith("ROCK_Scatter_")]:
        kind = ob.name[len("ROCK_Scatter_"):]
        if kind == "pebble" or rock_dims.get(kind) is None:
            continue
        d = instancer_data(ob)
        dims = rock_dims[kind]
        for p, s, v, r in zip(d["pos"], d["scale"], d["variant"], d["rot"]):
            dm = dims[int(v) % len(dims)] * s
            rad = 0.42 * max(dm[0], dm[1])
            h = dm[2] * (1 - SINK[kind])
            discs.append((p[0], p[1], rad, h))
            rock_hulls.append((p.copy() + (0, 0, dm[2] * 0.5 - SINK[kind] * dm[2]), dm * 0.85, float(r[2])))
            if h > 0.8 and kind != "cliff":
                rock_cover.append((p[0], p[1], rad, h))
    logs = bpy.data.objects.get("VEG_Scatter_log")
    if logs:
        for p in instancer_data(logs)["pos"]:
            discs.append((p[0], p[1], 0.45, 0.6))
    boxes, deck_boxes, struct_boxes = [], [], []
    for s in lay["structures"]:
        for i, c in enumerate(s["colliders"]):
            yaw = np.radians(c["yaw_deg"])
            if c["type"] == "box":
                cx, cy, cz = c["center_m"]
                sx, sy, sz = c["size_m"]
                boxes.append((cx, cy, cz - sz / 2, sx, sy, sz, yaw))
                struct_boxes.append((np.array(c["center_m"]), np.array(c["size_m"]), yaw))
                if s["name"].startswith(("SM_Bridge", "SM_Culvert")) and i == 0:
                    deck_boxes.append((cx, cy, sx + 1.0, sy, yaw))
            else:
                cx, cy, cz = c["center_m"]
                r, L = c["size_m"]
                discs.append((cx, cy, r, L))
                cyl.append((cx, cy, cz - L / 2, r, L))
    blocked, top = GC.raster_obstacles(G, discs, boxes)
    override = np.zeros_like(blocked)
    for cx, cy, sx, sy, yaw in deck_boxes:
        dx, dy = G.X - cx, G.Y - cy
        u = dx * np.cos(yaw) + dy * np.sin(yaw)
        v = -dx * np.sin(yaw) + dy * np.cos(yaw)
        override |= (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2 - 0.3)
    nav, slope, nav_stats = GC.nav_grid(cfg, G, depth, blocked, override)
    # Deck cells: heights for the nav mesh are the deck, not the riverbed.
    deck_z = lay["bridge"]["deck_z_m"]
    G.H = np.where(override & (depth > 0), np.maximum(G.H, deck_z), G.H)
    T["nav"] = time.perf_counter() - t

    # ---- cover points
    t = time.perf_counter()
    cover = []
    for s in lay["structures"]:
        for c in s["cover"]:
            cover.append({"pos_m": c["pos_m"], "kind": c["kind"], "source": s["name"]})
    for x, y, rad, h in rock_cover:
        for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False) + rng.uniform(0, 1.5):
            px, py = x + np.cos(ang) * (rad + 0.7), y + np.sin(ang) * (rad + 0.7)
            r_, c_ = G.rc(px, py)
            if nav[r_, c_]:
                cover.append({"pos_m": [round(px, 2), round(py, 2), round(float(G.H[r_, c_]), 2)],
                              "kind": "full" if h > 1.35 else "crouch", "source": "rock",
                              "facing_deg": round(float(np.degrees(ang)), 1)})
    # Thin to >= 2 m spacing.
    kept, pts = [], []
    for c in cover:
        p = np.array(c["pos_m"][:2])
        if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < 2.0:
            continue
        kept.append(c)
        pts.append(p)
    cover = kept
    cover_xy = np.array([c["pos_m"][:2] for c in cover])
    T["cover"] = time.perf_counter() - t

    # ---- visibility / exposure (4 m grid)
    t = time.perf_counter()
    V = GC.Grid(H, size_m, cfg.vis_cell_m)
    blk_v = np.zeros_like(V.H)
    for x, y, r, h in crowns + [d for d in discs if d[2] > 0.8]:
        rr, cc = V.rc(x, y)
        blk_v[rr, cc] = max(blk_v[rr, cc], h)
    sb, stop = GC.raster_obstacles(V, [], boxes)
    blk_v = np.maximum(blk_v, stop)
    nav_v = nav[::int(cfg.vis_cell_m / cfg.nav_cell_m), ::int(cfg.vis_cell_m / cfg.nav_cell_m)][:V.n, :V.n]
    towers = [(c["pos_m"][0], c["pos_m"][1], c["pos_m"][2] + 1.7) for c in cover if c["kind"] == "tower"]
    exp, vant = GC.exposure_map(cfg, V, blk_v, nav_v, rng, towers)
    exp_n = np.clip(exp / max(np.percentile(exp[nav_v], 98), 1), 0, 1)      # robust to a few extreme spots
    T["visibility"] = time.perf_counter() - t

    # ---- spawns, objectives
    t = time.perf_counter()
    sites = lay["sites"]
    centre = np.array([np.mean([sites[n]["x"] for n in sites]), np.mean([sites[n]["y"] for n in sites])])
    spawns = {}
    for team, hq in (("Team_A", "HQ_South"), ("Team_B", "HQ_North")):
        pts = []
        for x, y, _ in lay["hq_spawn_hints_m"][hq]:
            rc = GC.nearest_nav(nav, G, x, y)
            if rc is None:
                continue
            px, py = G.X[rc], G.Y[rc]
            yaw = float(np.arctan2(centre[1] - py, centre[0] - px))
            pts.append({"pos_m": [round(px, 2), round(py, 2), round(float(G.H[rc]), 2)], "yaw_deg": round(np.degrees(yaw), 1)})
        spawns[team] = pts
    objectives = []
    for letter, name in zip("ABCD", ("Bridge", "Farmhouse", "Ruins", "Lumber_Camp")):
        s = sites[name]
        rc = GC.nearest_nav(nav, G, s["x"], s["y"], 40)
        # Engagement range: median open sight distance around the objective.
        eye = (G.X[rc], G.Y[rc], float(G.H[rc]) + cfg.eye_h_m)
        sight = []
        for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            dist = 250.0
            for d in np.arange(5, 250, 5.0):
                q = (eye[0] + np.cos(ang) * d, eye[1] + np.sin(ang) * d)
                zq = float(V.sample(V.H + blk_v, q[0], q[1]))
                if zq > eye[2] + 0.5 or abs(q[0]) > size_m / 2 or abs(q[1]) > size_m / 2:
                    dist = d
                    break
            sight.append(dist)
        med = float(np.median(sight))
        objectives.append({"id": letter, "name": name, "type": "capture_point",
                           "pos_m": [round(float(G.X[rc]), 2), round(float(G.Y[rc]), 2), round(float(G.H[rc]), 2)],
                           "capture_radius_m": round(min(s["radius"] + 6, 20.0), 1),
                           "combat_zone": {"radius_m": 55.0, "median_sightline_m": med,
                                           "engagement": "CQB" if med < 30 else ("medium" if med < 90 else "long")}})
    # FFA spawns: low exposure, away from objectives, farthest-point sampling.
    cand = np.argwhere(nav[::4, ::4]) * 4
    cx, cy = G.X[cand[:, 0], cand[:, 1]], G.Y[cand[:, 0], cand[:, 1]]
    e_c = V.sample(exp_n, cx, cy)
    d_obj = np.min([np.hypot(cx - o["pos_m"][0], cy - o["pos_m"][1]) for o in objectives], axis=0)
    good = (e_c < 0.3) & (d_obj > 35) & (np.abs(cx) < size_m / 2 - 15) & (np.abs(cy) < size_m / 2 - 15)
    gx, gy = cx[good], cy[good]
    ffa = []
    if len(gx):
        i = int(rng.integers(len(gx)))
        chosen = [i]
        dmin = np.hypot(gx - gx[i], gy - gy[i])
        while len(chosen) < cfg.ffa_spawns:
            j = int(dmin.argmax())
            if dmin[j] < cfg.ffa_spawn_min_sep_m:
                break
            chosen.append(j)
            dmin = np.minimum(dmin, np.hypot(gx - gx[j], gy - gy[j]))
        for j in chosen:
            rc = G.rc(gx[j], gy[j])
            ffa.append({"pos_m": [round(float(gx[j]), 2), round(float(gy[j]), 2), round(float(G.H[rc]), 2)],
                        "yaw_deg": round(float(np.degrees(np.arctan2(centre[1] - gy[j], centre[0] - gx[j]))), 1)})
    T["spawns_objectives"] = time.perf_counter() - t

    # ---- routes (2 m grid) HQ -> objectives; validation of reachability
    t = time.perf_counter()
    nav2 = nav[::2, ::2]
    cost2 = 1 + 2.0 * V.sample(exp_n, G.X[::2, ::2], G.Y[::2, ::2]) + slope[::2, ::2] / 20

    def rc2(x, y):
        r, c = G.rc(x, y)
        return (int(r) // 2, int(c) // 2)

    routes, route_mask = [], np.zeros_like(nav2)
    for team, hq in (("Team_A", "HQ_South"), ("Team_B", "HQ_North")):
        s0 = spawns[team][0]["pos_m"] if spawns[team] else [sites[hq]["x"], sites[hq]["y"]]
        for o in objectives:
            a_, b_ = rc2(*s0[:2]), rc2(*o["pos_m"][:2])
            a_ = a_ if nav2[a_] else tuple(np.argwhere(nav2)[np.argmin(np.sum((np.argwhere(nav2) - a_) ** 2, 1))])
            b_ = b_ if nav2[b_] else tuple(np.argwhere(nav2)[np.argmin(np.sum((np.argwhere(nav2) - b_) ** 2, 1))])
            path = GC.shortest_path(nav2, cost2, a_, b_)
            if path is None:
                routes.append({"team": team, "to": o["id"], "reachable": False})
                continue
            P = np.array(path)
            xy = np.column_stack([G.X[::2, ::2][P[:, 0], P[:, 1]], G.Y[::2, ::2][P[:, 0], P[:, 1]]])
            L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
            route_mask[P[:, 0], P[:, 1]] = True
            routes.append({"team": team, "to": o["id"], "reachable": True, "length_m": round(L, 1),
                           "run_time_s": round(L / cfg.run_speed_ms, 1),
                           "polyline_xy_m": np.round(xy[::5], 1).tolist()})
    T["routes"] = time.perf_counter() - t

    # ---- pickups
    def snap_near(x, y, rad, prefer=None):
        for _ in range(60):
            ang, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, rad)
            px, py = x + np.cos(ang) * d, y + np.sin(ang) * d
            r_, c_ = G.rc(px, py)
            if nav[r_, c_] and slope[r_, c_] < 20:
                return [round(float(px), 2), round(float(py), 2), round(float(G.H[r_, c_]) + 0.05, 2)]
        rc = GC.nearest_nav(nav, G, x, y)
        return [round(float(G.X[rc]), 2), round(float(G.Y[rc]), 2), round(float(G.H[rc]), 2)]

    pickups = []
    for o in objectives:
        x, y = o["pos_m"][:2]
        near_cover = [c["pos_m"] for c in cover if np.hypot(c["pos_m"][0] - x, c["pos_m"][1] - y) < 25]
        for kind, n in (("weapon", 2), ("ammo", 2), ("health", 1)):
            for _ in range(n):
                base = near_cover[int(rng.integers(len(near_cover)))] if near_cover else [x, y]
                pickups.append({"type": kind, "objective": o["id"], "pos_m": snap_near(base[0], base[1], 3.0)})
    for hq in ("HQ_South", "HQ_North"):
        s = sites[hq]
        for kind, n in (("weapon", 1), ("ammo", 2), ("health", 2)):
            for _ in range(n):
                pickups.append({"type": kind, "objective": hq, "pos_m": snap_near(s["x"], s["y"], 10.0)})
    for r in routes:
        if r.get("reachable") and len(r["polyline_xy_m"]) > 4:
            P = r["polyline_xy_m"]
            for f, kind in ((0.45, "ammo"), (0.6, "health")):
                q = P[int(len(P) * f)]
                pickups.append({"type": kind, "objective": f"route_{r['team']}_{r['to']}", "pos_m": snap_near(q[0], q[1], 4.0)})

    # ---- kill zones: exposed, little cover, on likely routes
    t = time.perf_counter()
    cov_density = np.zeros_like(exp_n)
    if len(cover_xy):
        for (x, y) in cover_xy:
            rr, cc = V.rc(x, y)
            cov_density[rr, cc] += 1
    from_terrain = cov_density
    k = int(10 / cfg.vis_cell_m)
    pad = np.pad(from_terrain, k)
    cd = sum(pad[k + i:k + i + exp_n.shape[0], k + j:k + j + exp_n.shape[1]] for i in range(-k, k + 1) for j in range(-k, k + 1))
    cd = cd / (np.pi * 100) * 100                                             # per 100 m2
    rm = np.zeros_like(nav_v)
    rr_, cc_ = np.nonzero(route_mask)
    rm[np.clip(rr_ // 2, 0, V.n - 1), np.clip(cc_ // 2, 0, V.n - 1)] = True
    rm = GC.dilate(rm, 3)
    kz = nav_v & (exp_n > cfg.kill_zone_exposure) & (cd < cfg.kill_zone_max_cover_per_100m2) & rm
    lab, n = GC.label_components(kz)
    kill_zones = []
    for i in range(1, n + 1):
        m = lab == i
        area = float(m.sum() * cfg.vis_cell_m ** 2)
        if area < 400:
            continue
        xs, ys = V.X[m], V.Y[m]
        kill_zones.append({"centroid_m": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)], "area_m2": area,
                           "bbox_m": [round(float(xs.min()), 1), round(float(ys.min()), 1), round(float(xs.max()), 1),
                                      round(float(ys.max()), 1)],
                           "mean_exposure": round(float(exp_n[m].mean()), 2)})
    T["kill_zones"] = time.perf_counter() - t

    # ---- validation
    def on_nav(p):
        r_, c_ = G.rc(p[0], p[1])
        return bool(nav[max(r_ - 1, 0):r_ + 2, max(c_ - 1, 0):c_ + 2].any())

    allpts = ([s["pos_m"] for t_ in spawns.values() for s in t_] + [f["pos_m"] for f in ffa]
              + [o["pos_m"] for o in objectives] + [p["pos_m"] for p in pickups])
    vis_ab = 0
    for sa in spawns["Team_A"]:
        for sb in spawns["Team_B"]:
            p = (sa["pos_m"][0], sa["pos_m"][1], sa["pos_m"][2] + cfg.eye_h_m)
            q = (sb["pos_m"][0], sb["pos_m"][1], sb["pos_m"][2] + cfg.eye_h_m)
            vis_ab += GC.visible(V, blk_v, p, q)
    checks = {"points_off_navmesh": int(sum(not on_nav(p) for p in allpts)),
              "objectives_unreachable": int(sum(not r["reachable"] for r in routes)),
              "team_spawn_pairs_with_line_of_sight": int(vis_ab),
              "navigable_area_pct": round(float(nav.mean() * 100), 1),
              "route_run_time_s_range": [min(r["run_time_s"] for r in routes if r.get("reachable")),
                                         max(r["run_time_s"] for r in routes if r.get("reachable"))]}

    # ---- Blender: markers, collision, navmesh
    t = time.perf_counter()
    gcol = bpy.data.collections.new("Gameplay")
    scene.collection.children.link(gcol)
    for team, pts in spawns.items():
        for i, s in enumerate(pts):
            empty(gcol, f"GP_Spawn_{team}_{i:02d}", s["pos_m"], "SINGLE_ARROW", 1.5, np.radians(s["yaw_deg"]) - np.pi / 2,
                  {"team": team})
    for i, s in enumerate(ffa):
        empty(gcol, f"GP_Spawn_FFA_{i:02d}", s["pos_m"], "SINGLE_ARROW", 1.5, np.radians(s["yaw_deg"]) - np.pi / 2)
    for o in objectives:
        empty(gcol, f"GP_Objective_{o['id']}_{o['name']}", o["pos_m"], "CIRCLE", o["capture_radius_m"], 0.0,
              {"engagement": o["combat_zone"]["engagement"]}).rotation_euler = (np.pi / 2, 0, 0)
    for i, p in enumerate(pickups):
        empty(gcol, f"GP_Pickup_{p['type']}_{i:03d}", p["pos_m"], "CUBE", 0.3, 0.0, {"type": p["type"]})
    ccol = bpy.data.collections.new("GP_Cover")
    gcol.children.link(ccol)
    for i, c in enumerate(cover):
        empty(ccol, f"GP_Cover_{c['kind']}_{i:04d}", c["pos_m"], "SPHERE", 0.25)
    gcol.hide_render = True
    col_c = bpy.data.collections.new("Collision")
    scene.collection.children.link(col_c)
    terrain_col, tquads = grid_mesh("COL_Terrain", GC.Grid(H, size_m, 2.0), np.ones((251, 251), bool), 1, col_c)
    boxes_mesh("UCX_Structures", struct_boxes, col_c)
    cylinders_mesh("COL_Tree_Trunks", [c for c in cyl], col_c)
    boxes_mesh("UCX_Rocks", rock_hulls, col_c)
    col_c.hide_render = True
    navcol = bpy.data.collections.new("NavMesh")
    scene.collection.children.link(navcol)
    nav_ob, nquads = grid_mesh("NAV_Walkable", G, nav, 2, navcol)
    nav_ob.location.z = 0.05
    navcol.hide_render = True
    for c in (col_c, navcol, ccol):
        c.hide_viewport = True
    for c_name, fname in (("Collision", "collision.glb"), ("NavMesh", "navmesh.glb")):
        bpy.ops.object.select_all(action="DESELECT")
        coll = bpy.data.collections[c_name]
        coll.hide_viewport = False
        for o in coll.objects:
            o.select_set(True)
        try:
            bpy.ops.export_scene.gltf(filepath=os.path.join(out, fname), use_selection=True, export_format="GLB")
        except Exception as exc:
            print("GLB_EXPORT_FAILED", exc)
        coll.hide_viewport = True
    T["blender"] = time.perf_counter() - t
    save(os.path.join(out, "forest_valley_gameplay.blend"))

    # ---- maps
    n = 1001
    Hm = GC._bilinear_idx(H, *np.meshgrid(np.linspace(0, res - 1, n), np.linspace(0, res - 1, n)))
    img = hillshade_rgb(Hm, size_m / (n - 1)).astype(float)[::-1].copy()          # row 0 = north
    navm = GC._bilinear_idx(nav.astype(float), *np.meshgrid(np.linspace(0, G.n - 1, n), np.linspace(0, G.n - 1, n)))[::-1]
    img = img * (0.55 + 0.45 * navm[..., None])
    wat = GC._bilinear_idx((depth > 0.05).astype(float), *np.meshgrid(np.linspace(0, G.n - 1, n), np.linspace(0, G.n - 1, n)))[::-1]
    img = img * (1 - 0.7 * wat[..., None]) + np.array([40, 90, 160]) * 0.7 * wat[..., None]
    for r in lay["roads"].values():
        for x, y, _ in r["polyline_xyz_m"][::2]:
            disc_px(img, size_m, x, y, max(int(r["half_width_m"] * 2), 1), (170, 140, 100), 0.7)
    kz_up = GC._bilinear_idx(kz.astype(float) * (lab > 0), *np.meshgrid(np.linspace(0, V.n - 1, n), np.linspace(0, V.n - 1, n)))[::-1]
    img = img * (1 - 0.45 * kz_up[..., None]) + np.array([220, 30, 30]) * 0.45 * kz_up[..., None]
    for r in routes:
        if r.get("reachable"):
            for x, y in r["polyline_xy_m"]:
                disc_px(img, size_m, x, y, 1, (80, 160, 255) if r["team"] == "Team_A" else (255, 120, 60), 0.8)
    for c in cover:
        disc_px(img, size_m, c["pos_m"][0], c["pos_m"][1], 1, (235, 235, 235), 0.8)
    for o in objectives:
        disc_px(img, size_m, o["pos_m"][0], o["pos_m"][1], int(o["capture_radius_m"] * 2), (255, 220, 0), 1.0, ring=True)
        disc_px(img, size_m, o["pos_m"][0], o["pos_m"][1], int(o["combat_zone"]["radius_m"] * 2), (255, 220, 0), 0.6, ring=True)
    for p in pickups:
        col = {"weapon": (255, 140, 0), "ammo": (255, 255, 90), "health": (60, 230, 90)}[p["type"]]
        disc_px(img, size_m, p["pos_m"][0], p["pos_m"][1], 3, col)
    for team, pts in spawns.items():
        for s in pts:
            disc_px(img, size_m, s["pos_m"][0], s["pos_m"][1], 4, (40, 110, 255) if team == "Team_A" else (255, 50, 40))
    for s in ffa:
        disc_px(img, size_m, s["pos_m"][0], s["pos_m"][1], 3, (200, 80, 255))
    write_rgb8(os.path.join(out, "tactical_map.png"), np.clip(img, 0, 255).astype(np.uint8))
    heat = (np.clip(exp_n, 0, 1) * 255).astype(np.uint8)[::-1]
    write_rgb8(os.path.join(out, "exposure_map.png"),
               np.stack([heat, (255 - heat) // 3, 255 - heat], -1).repeat(4, 0).repeat(4, 1))

    meta = {"config": GC.config_dict(cfg), "coordinate_note": "metres, Blender axes (UE: X=X, Y=-Y, cm x100)",
            "spawns": {"team": spawns, "free_for_all": ffa}, "objectives": objectives, "pickups": pickups,
            "cover_points": cover, "kill_zones": kill_zones, "routes": routes,
            "navigation": {**nav_stats, "cell_m": cfg.nav_cell_m, "agent_radius_m": cfg.agent_radius_m,
                           "max_slope_deg": cfg.max_walk_slope_deg, "max_wade_depth_m": cfg.max_wade_depth_m,
                           "navmesh_quads_2m": int(nquads)},
            "collision": {"terrain_quads_2m": int(tquads), "structure_boxes": len(struct_boxes),
                          "tree_capsules": len(cyl), "rock_hulls": len(rock_hulls),
                          "note": "visual and collision geometry are separate objects / files"},
            "checks": checks, "timings_s": {k: round(v, 2) for k, v in T.items()}}
    meta["timings_s"]["total_wall"] = round(time.perf_counter() - t_all, 2)
    with open(os.path.join(out, "gameplay_metadata.json"), "w") as f:
        json.dump(meta, f)
    summary = {"team_spawns": {k: len(v) for k, v in spawns.items()}, "ffa_spawns": len(ffa),
               "objectives": [(o["id"], o["name"], o["combat_zone"]["engagement"], o["combat_zone"]["median_sightline_m"])
                              for o in objectives],
               "pickups": {k: sum(p["type"] == k for p in pickups) for k in ("weapon", "ammo", "health")},
               "cover_points": len(cover), "kill_zones": len(kill_zones), "navigation": meta["navigation"],
               "checks": checks, "timings_s": meta["timings_s"]}
    with open(os.path.join(out, "gameplay_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("PHASE7_STATS", json.dumps(summary))


if __name__ == "__main__":
    main()
