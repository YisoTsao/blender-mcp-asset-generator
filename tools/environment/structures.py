"""Structure builders (numpy Geo): military outpost, bridge, abandoned farmhouse,
stone ruins, lumber camp, bunker, trench works, tank traps, natural debris.

All builders work in a local frame (origin = site centre at ground / pad level,
+X = site 'forward' direction) and return a Geo with collision primitives and
cover points recorded alongside the visual geometry.
"""
import numpy as np

from kit import Geo, rot_z


# --------------------------------------------------------------------------- small pieces

def sandbag_wall(g, p0, p1, layers, rng, z0=0.0, bag=(0.58, 0.32, 0.15)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    yaw = np.arctan2(d[1], d[0])
    R = rot_z(yaw)
    n = max(int(L / (bag[0] * 0.95)), 1)
    for k in range(layers):
        off = 0.5 if k % 2 else 0.0
        for i in range(n - (1 if k % 2 else 0)):
            u = (i + 0.5 + off) * L / n
            c = np.array([u, rng.normal(0, 0.02), z0 + bag[2] * (k + 0.5) * 0.92])
            g.box(R @ c + np.array([*p0, 0]), (bag[0] * rng.uniform(0.93, 1.03), bag[1], bag[2]), "sandbag",
                  yaw=yaw + rng.normal(0, 0.05), collide=False, taper=0.18)
    h = z0 + layers * bag[2] * 0.92
    mid = (p0 + p1) / 2
    g.colliders.append(("box", [mid[0], mid[1], h / 2], [L, bag[1] * 1.1, h], float(yaw)))
    for t in np.linspace(0.15, 0.85, max(int(L / 2.5), 1)):
        q = p0 + d * t
        g.cover.append((q[0], q[1], h, "crouch" if h < 1.2 else "full"))
    return h


def hesco_line(g, p0, p1, h=1.35, s=1.05):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    yaw = np.arctan2(d[1], d[0])
    n = max(int(round(L / s)), 1)
    for i in range(n):
        c = p0 + d * (i + 0.5) / n
        g.box((c[0], c[1], h / 2), (L / n - 0.03, s, h), "hesco", yaw=yaw, collide=False)
    mid = (p0 + p1) / 2
    g.colliders.append(("box", [mid[0], mid[1], h / 2], [L, s, h], float(yaw)))
    for t in np.linspace(0.1, 0.9, max(int(L / 3), 1)):
        q = p0 + d * t
        g.cover.append((q[0], q[1], h, "full"))


def crate(g, c, rng, yaw=0.0, size=(1.2, 0.8, 0.8)):
    g.box((c[0], c[1], c[2] + size[2] / 2), size, "wood_crate", yaw=yaw)
    g.cover.append((c[0], c[1], c[2] + size[2], "crouch"))


def drum(g, c, r=0.29, h=0.88):
    g.cyl((c[0], c[1], c[2]), (c[0], c[1], c[2] + h), r, "metal_paint", seg=12, collide=True)


def watchtower(g, c, rng, h=7.0, half=1.6):
    cx, cy, cz = c
    legs = [(sx * half, sy * half) for sx in (-1, 1) for sy in (-1, 1)]
    for (x, y) in legs:
        g.beam((cx + x * 1.12, cy + y * 1.12, cz - 0.4), (cx + x, cy + y, cz + h + 2.3), 0.24, 0.24, "wood_dark",
               collide=False)
        g.colliders.append(("box", [cx + x, cy + y, cz + h / 2], [0.3, 0.3, h + 2.0], 0.0))
    # X bracing on all four sides, two bays.
    for z0, z1 in ((0.3, h / 2), (h / 2, h - 0.1)):
        for (a, b) in ((0, 1), (1, 3), (3, 2), (2, 0)):
            (xa, ya), (xb, yb) = legs[a], legs[b]
            g.beam((cx + xa, cy + ya, cz + z0), (cx + xb, cy + yb, cz + z1), 0.12, 0.08, "wood_dark")
            g.beam((cx + xb, cy + yb, cz + z0), (cx + xa, cy + ya, cz + z1), 0.12, 0.08, "wood_dark")
    # Platform and parapet.
    g.box((cx, cy, cz + h), (2 * half + 0.6, 2 * half + 0.6, 0.15), "wood")
    for k in range(4):
        a = k * np.pi / 2
        R = rot_z(a)
        p0 = R @ np.array([half + 0.25, -half - 0.25, 0])
        p1 = R @ np.array([half + 0.25, half + 0.25, 0])
        sandbag_wall(g, (cx + p0[0], cy + p0[1]), (cx + p1[0], cy + p1[1]), 4, rng, z0=cz + h + 0.08)
    # Roof.
    g.box((cx, cy, cz + h + 2.4), (2 * half + 1.0, 2 * half + 1.0, 0.08), "metal_rust", collide=False)
    # Ladder.
    lx = cx - half - 0.6
    for s in (-0.25, 0.25):
        g.beam((lx - 0.9, cy + s, cz), (lx, cy + s, cz + h), 0.06, 0.08, "wood")
    for t in np.linspace(0.05, 0.95, 22):
        p = np.array([lx - 0.9 * (1 - t), cy, cz + h * t])
        g.beam(p + (0, -0.25, 0), p + (0, 0.25, 0), 0.05, 0.05, "wood")
    g.cover.append((cx, cy, cz + h + 0.7, "tower"))


def tent(g, c, yaw, L=5.5, W=4.0, H=2.6):
    g.prism(c, L, W, H, "canvas", yaw=yaw)
    R = rot_z(yaw)
    for s in (-1, 1):
        p = R @ np.array([s * (L / 2 + 0.05), 0, 0]) + np.asarray(c)
        g.cyl(p, p + (0, 0, H + 0.2), 0.05, "wood_dark", seg=6)
    g.colliders.append(("box", [c[0], c[1], c[2] + H / 2], [L, W, H], float(yaw)))


# --------------------------------------------------------------------------- sites

def outpost(rng, radius=20.0):
    """Team HQ: HESCO perimeter with a gate on +X, watchtower, tents, sandbagged
    positions, command post, crates and fuel drums."""
    g = Geo()
    s = radius * 0.72
    gate = 3.2
    # Perimeter (gate gap on +X side).
    hesco_line(g, (-s, -s), (s, -s))
    hesco_line(g, (-s, s), (s, s))
    hesco_line(g, (-s, -s), (-s, s))
    hesco_line(g, (s, -s), (s, -gate))
    hesco_line(g, (s, gate), (s, s))
    # Gate chicane of sandbags outside the gate.
    sandbag_wall(g, (s + 3.0, -gate - 1.5), (s + 3.0, -0.6), 6, rng)
    sandbag_wall(g, (s + 3.0, 0.6), (s + 3.0, gate + 1.5), 6, rng)
    watchtower(g, (-s + 2.6, s - 2.6, 0.0), rng)
    tent(g, (-s * 0.35, -s * 0.55, 0), 0.0)
    tent(g, (-s * 0.35, -s * 0.05, 0), 0.0)
    # Command post: sandbag U with timber roof.
    cp = np.array([s * 0.45, s * 0.55])
    sandbag_wall(g, cp + (-2.5, -2.0), cp + (2.5, -2.0), 9, rng)
    sandbag_wall(g, cp + (-2.5, -2.0), cp + (-2.5, 2.0), 9, rng)
    sandbag_wall(g, cp + (2.5, -2.0), cp + (2.5, 2.0), 9, rng)
    for x in np.linspace(-2.4, 2.4, 9):
        g.beam((cp[0] + x, cp[1] - 2.3, 1.3), (cp[0] + x, cp[1] + 2.3, 1.3), 0.2, 0.18, "wood_dark")
    # Fighting positions at the perimeter corners (inside).
    for (x, y) in ((s - 3, -s + 3), (s - 3, s - 3), (-s + 3, -s + 3)):
        a = np.arctan2(y, x)
        R = rot_z(a + np.pi / 2)
        p0 = np.array([x, y]) + R[:2, :2] @ np.array([-1.6, 0])
        p1 = np.array([x, y]) + R[:2, :2] @ np.array([1.6, 0])
        sandbag_wall(g, p0, p1, 6, rng)
    # Supplies.
    for i in range(7):
        c = (s * 0.1 + (i % 3) * 1.3, -s * 0.6 + (i // 3) * 0.9, 0)
        crate(g, c, rng, yaw=rng.normal(0, 0.05))
    crate(g, (s * 0.1 + 0.6, -s * 0.6 + 0.45, 0.8), rng, size=(1.0, 0.6, 0.5))
    for i in range(6):
        drum(g, (-s * 0.75 + (i % 3) * 0.65, s * 0.15 + (i // 3) * 0.65, 0))
    # Flag pole.
    g.cyl((0.5, 0.5, 0), (0.5, 0.5, 8.0), 0.06, "metal_paint", seg=8)
    spawns = [(x, y) for x in np.linspace(-s * 0.6, s * 0.6, 4) for y in (-s * 0.25, s * 0.25)]
    return g, {"spawns_local": spawns, "gate_local": (s + 4.5, 0.0)}


def bridge(span, width, depth_below_deck, rng):
    """Timber trestle bridge along local X, deck top at z=0, stone abutments at both ends."""
    g = Geo()
    L = span + 4.0
    # Deck planks across the width.
    n = int(L / 0.32)
    for i in range(n):
        x = -L / 2 + (i + 0.5) * L / n
        g.box((x, rng.normal(0, 0.02), -0.06), (L / n - 0.025, width + rng.uniform(-0.05, 0.1), 0.12), "wood",
              collide=False)
    g.colliders.append(("box", [0, 0, -0.06], [L, width, 0.12], 0.0))
    # Stringers.
    for y in np.linspace(-width / 2 + 0.3, width / 2 - 0.3, 4):
        g.cyl((-L / 2, y, -0.32), (L / 2, y, -0.32), 0.2, "log", seg=10)
    # Pile bents every ~4.5 m.
    nb = max(int(span / 4.5), 2)
    for x in np.linspace(-span / 2 + 2.0, span / 2 - 2.0, nb):
        g.beam((x, -width / 2 - 0.3, -0.62), (x, width / 2 + 0.3, -0.62), 0.32, 0.3, "wood_dark")
        for y in (-width / 2 + 0.2, -width / 6, width / 6, width / 2 - 0.2):
            g.cyl((x, y, -depth_below_deck - 1.5), (x, y, -0.6), 0.17, "log", seg=10, collide=True)
        # Cross bracing.
        g.beam((x, -width / 2 + 0.2, -depth_below_deck + 0.3), (x, width / 2 - 0.2, -1.0), 0.12, 0.1, "wood_dark")
    # Railings.
    for y in (-width / 2 + 0.08, width / 2 - 0.08):
        for x in np.arange(-L / 2 + 0.3, L / 2, 1.6):
            g.box((x, y, 0.55), (0.12, 0.12, 1.1), "wood_dark", collide=False)
        g.beam((-L / 2 + 0.3, y, 1.05), (L / 2 - 0.3, y, 1.05), 0.1, 0.12, "wood")
        g.beam((-L / 2 + 0.3, y, 0.55), (L / 2 - 0.3, y, 0.55), 0.08, 0.1, "wood")
        g.colliders.append(("box", [0, y, 0.55], [L, 0.15, 1.1], 0.0))
    # Stone abutments.
    for sx in (-1, 1):
        x0 = sx * (L / 2 - 1.0)
        for k in range(4):
            z = -0.3 - k * 0.6
            for j in range(4):
                y = -width / 2 - 0.5 + (j + 0.5 + 0.5 * (k % 2)) * (width + 1.0) / 4
                if abs(y) > width / 2 + 0.5:
                    continue
                g.box((x0 + rng.normal(0, 0.04), y, z), (2.2, (width + 1.0) / 4 - 0.04, 0.58), "stone",
                      yaw=rng.normal(0, 0.02), collide=False)
        g.colliders.append(("box", [x0, 0, -1.2], [2.2, width + 1, 2.4], 0.0))
    return g


def farmhouse(rng):
    """Abandoned two-storey stone farmhouse (collapsed NE corner, half the roof gone),
    with a ruined timber barn and broken fences."""
    g = Geo()
    L, W, H, t = 11.0, 7.5, 5.8, 0.55
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    front = [(2.0, 3.1, 0, 2.3), (4.4, 5.6, 0.9, 2.2), (7.0, 8.2, 0.9, 2.2),
             (1.4, 2.5, 3.3, 4.6), (4.4, 5.6, 3.3, 4.6), (7.6, 8.8, 3.3, 4.6)]
    side = [(2.5, 3.6, 0.9, 2.2), (2.5, 3.6, 3.3, 4.6)]
    g.wall((x0, y0), (x1, y0), H, t, "stone_wall", openings=front, ruin=0.12, rng=rng)
    g.wall((x1, y1), (x0, y1), H, t, "stone_wall", openings=[(4.0, 5.2, 0, 2.3), (7.0, 8.2, 3.3, 4.6)], ruin=0.75,
           rng=rng)
    gable = H + 2.6
    tops = g.wall((x0, y1), (x0, y0), H, t, "stone_wall", openings=side, ruin=0.0, rng=rng)
    del tops
    # Gable triangle on the west end (intact) as stacked courses.
    for k in range(6):
        z = H + k * 0.43
        hw = (W / 2) * (1 - k / 6)
        g.box((x0, 0, z + 0.21), (t, 2 * hw, 0.43), "stone_wall", collide=False)
    g.wall((x1, y0), (x1, y1), H, t, "stone_wall", openings=side, ruin=0.85, rng=rng)
    # Interior partition + upper floor joists (some fallen).
    g.wall((-1.0, y0 + t / 2), (-1.0, y1 - t / 2), 2.9, 0.3, "plaster", openings=[(3.0, 4.0, 0, 2.1)], rng=rng)
    for x in np.arange(x0 + 0.6, x1 - 0.5, 0.7):
        if x > 2.0 and rng.random() < 0.6:
            # Fallen joist: one end down on the floor.
            g.beam((x, y0 + 0.4, 0.15), (x + rng.normal(0, 0.4), y1 - 0.6, 2.4), 0.16, 0.22, "wood_old")
            continue
        g.beam((x, y0 + 0.3, 3.0), (x, y1 - 0.3, 3.0), 0.16, 0.22, "wood_old")
    # Roof: west half survives (rafters + some panels), east half gone.
    for x in np.arange(x0 - 0.2, 0.5, 0.75):
        for s in (-1, 1):
            g.beam((x, s * (W / 2 + 0.3), H - 0.1), (x, 0, gable), 0.12, 0.18, "wood_old")
    g.beam((x0 - 0.3, 0, gable), (0.6, 0, gable - 0.15), 0.2, 0.2, "wood_old")
    rg = Geo()
    rg.roof((x0 / 2 - 0.4, 0, H), L / 2 - 0.6, W, gable - H, 0.08, "roof_slate", overhang=0.35, missing=0.35, rng=rng)
    g.merge(rg)
    # Chimney.
    g.box((x0 + 1.2, y1 - 0.6, (gable + 1.2) / 2), (0.8, 0.8, gable + 1.2), "stone_wall")
    # Rubble heaps at the collapsed corner and inside.
    for _ in range(140):
        u = rng.random()
        c = np.array([x1 - 1.5 + rng.normal(0, 1.6) * u, y1 - 1.2 + rng.normal(0, 1.4) * u, 0])
        c[2] = max(0.0, 0.9 * np.exp(-((c[0] - x1 + 1.5) ** 2 + (c[1] - y1 + 1.2) ** 2) / 4.0)) * rng.random()
        s = rng.uniform(0.15, 0.45, 3)
        g.box(c + (0, 0, s[2] / 2), s, "stone_wall", yaw=rng.uniform(0, np.pi), collide=False)
    g.colliders.append(("box", [x1 - 1.5, y1 - 1.2, 0.45], [3.0, 2.6, 0.9], 0.0))
    # Barn ruin (timber frame, partial planking, sagging roof) east of the house.
    bx, by = x1 + 7.5, 1.0
    for x in (-3, 0, 3):
        for y in (-2.5, 2.5):
            g.beam((bx + x, by + y, 0), (bx + x + rng.normal(0, 0.05), by + y, 3.4), 0.2, 0.2, "wood_old", collide=True)
    for y in (-2.5, 2.5):
        g.beam((bx - 3, by + y, 3.4), (bx + 3, by + y, 3.3), 0.2, 0.22, "wood_old")
    for x in np.arange(-3, 3.01, 0.22):
        if rng.random() < 0.7:
            g.box((bx + x, by - 2.6, 1.3), (0.2, 0.04, 2.6 * rng.uniform(0.6, 1.0)), "wood_old", collide=False)
    g.colliders.append(("box", [bx, by - 2.6, 1.3], [6.0, 0.1, 2.6], 0.0))
    for i, x in enumerate(np.arange(-3.3, 3.31, 0.9)):
        if i % 3 == 1:
            continue
        sag = 0.5 * np.exp(-x * x / 2)
        g.box((bx + x, by, 3.55 - sag), (0.88, 5.8, 0.04), "metal_rust", R=rot_z(0) @ _tilt(0.15 * np.sign(x)),
              collide=False)
    # Broken fence along the yard.
    for i, x in enumerate(np.arange(-10, 10.1, 2.0)):
        if rng.random() < 0.25:
            continue
        p = (x, -W / 2 - 6.0, 0)
        g.beam(p, (x + rng.normal(0, 0.08), -W / 2 - 6.0, 1.2), 0.12, 0.12, "wood_old")
        if rng.random() < 0.7:
            g.beam((x, -W / 2 - 6.0, 0.9), (x + 2.0, -W / 2 - 6.0, 0.9 - rng.uniform(0, 0.6)), 0.06, 0.12, "wood_old")
    g.cover += [(x0 - 0.4, y, 1.2, "full") for y in (-2, 2)] + [(x1 + 0.4, 0, 1.2, "full")]
    return g


def _tilt(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ruins(rng):
    """Old stone chapel ruin: broken nave walls with tall windows, a round tower stump,
    column bases, fallen blocks, a low perimeter wall."""
    g = Geo()
    L, W, H, t = 16.0, 8.5, 6.5, 0.85
    x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
    win = [(u, u + 0.9, 2.0, 4.8) for u in (2.5, 6.0, 9.5, 13.0)]
    g.wall((x0, y0), (x1, y0), H, t, "stone_old", openings=win, ruin=0.6, rng=rng, block=0.6)
    g.wall((x1, y1), (x0, y1), H, t, "stone_old", openings=win, ruin=0.7, rng=rng, block=0.6)
    g.wall((x0, y1), (x0, y0), H + 1.5, t, "stone_old", openings=[(3.4, 5.1, 0, 3.2)], ruin=0.35, rng=rng, block=0.6)
    g.wall((x1, y0), (x1, -1.0), H, t, "stone_old", ruin=0.5, rng=rng, block=0.6)
    g.wall((x1, 1.2), (x1, y1), H, t, "stone_old", ruin=0.8, rng=rng, block=0.6)
    # Round tower stump at the NE corner.
    tc = np.array([x1 + 2.2, y1 + 1.6])
    seg = 14
    for i in range(seg):
        a0, a1 = 2 * np.pi * i / seg, 2 * np.pi * (i + 1) / seg
        if i in (3, 4):
            continue                                        # breach
        p0 = tc + 2.8 * np.array([np.cos(a0), np.sin(a0)])
        p1 = tc + 2.8 * np.array([np.cos(a1), np.sin(a1)])
        h = 8.5 * (0.45 + 0.55 * abs(np.sin(i * 0.7 + 1))) * rng.uniform(0.8, 1.0)
        g.wall(p0, p1, h, 0.9, "stone_old", ruin=0.2, rng=rng, block=0.5, collide=False)
    g.colliders.append(("capsule", [tc[0], tc[1], 3.5], [3.2, 7.0], 0.0))
    # Column bases down the nave.
    for x in np.linspace(x0 + 3, x1 - 3, 4):
        for y in (-1.8, 1.8):
            hcol = rng.uniform(0.6, 3.5)
            g.box((x, y, 0.2), (0.9, 0.9, 0.4), "stone_old")
            g.cyl((x, y, 0.4), (x, y, 0.4 + hcol), 0.32, "stone_old", seg=12, collide=True)
            if hcol > 1.0:
                g.cover.append((x, y, min(hcol, 1.8), "full"))
    # Fallen blocks and column drums.
    for _ in range(70):
        c = np.array([rng.uniform(x0 - 5, x1 + 5), rng.uniform(y0 - 5, y1 + 5)])
        if x0 < c[0] < x1 and y0 < c[1] < y1 and rng.random() < 0.6:
            continue
        s = rng.uniform(0.3, 0.8, 3) * (1, 1, 0.7)
        g.box((c[0], c[1], s[2] / 2 - 0.05), s, "stone_old", yaw=rng.uniform(0, np.pi), collide=False)
    for _ in range(5):
        c = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), 0.3])
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a), 0]) * rng.uniform(0.8, 1.6)
        g.cyl(c - d, c + d, 0.32, "stone_old", seg=12, collide=True)
        g.cover.append((c[0], c[1], 0.6, "crouch"))
    # Low dry-stone boundary wall (ellipse, gaps).
    for i in range(24):
        if i % 6 == 0:
            continue
        a0, a1 = 2 * np.pi * i / 24, 2 * np.pi * (i + 1) / 24
        p0 = np.array([np.cos(a0) * (L / 2 + 6), np.sin(a0) * (W / 2 + 6)])
        p1 = np.array([np.cos(a1) * (L / 2 + 6), np.sin(a1) * (W / 2 + 6)])
        g.wall(p0, p1, 1.0, 0.6, "stone_old", ruin=0.4, rng=rng, block=0.5)
        m = (p0 + p1) / 2
        g.cover.append((m[0], m[1], 0.9, "crouch"))
    g.cover += [(x0 - 0.6, 0, 3, "full"), (x1 + 0.6, -2.5, 3, "full")]
    return g


def log_cabin(g, c, yaw, rng, L=6.0, W=4.6, H=2.6):
    R = rot_z(yaw)
    c = np.asarray(c, float)
    r = 0.15
    n = int(H / (2 * r * 0.9))
    walls = [((-L / 2, -W / 2), (L / 2, -W / 2), [(1.0, 2.0, 0, 2.0), (3.6, 4.6, 0.9, 1.8)]),
             ((L / 2, W / 2), (-L / 2, W / 2), [(2.3, 3.3, 0.9, 1.8)]),
             ((-L / 2, W / 2), (-L / 2, -W / 2), []),
             ((L / 2, -W / 2), (L / 2, W / 2), [(1.8, 2.8, 0.9, 1.8)])]
    for k in range(n):
        z = r + k * 2 * r * 0.9
        for wi, (p0, p1, ops) in enumerate(walls):
            if (wi % 2) != (k % 2):
                zz = z + r * 0.9
            else:
                zz = z
            p0, p1 = np.array(p0, float), np.array(p1, float)
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            ext0, ext1 = p0 - d * 0.35, p1 + d * 0.35
            Lw = np.linalg.norm(ext1 - ext0)
            cuts = [(0.0, Lw)]
            for (u0, u1, z0, z1) in ops:
                if z0 <= zz <= z1:
                    o0, o1 = u0 + 0.35, u1 + 0.35            # opening in extended-log coordinates
                    cuts = [(a, b) for (a, b) in sum([[(a, min(b, o0)), (max(a, o1), b)] for (a, b) in cuts], [])
                            if b - a > 0.4]
            for (a, b) in cuts:
                q0 = np.array([*(ext0 + d * a), zz])
                q1 = np.array([*(ext0 + d * b), zz])
                g.cyl(R @ q0 + c, R @ q1 + c, r * rng.uniform(0.95, 1.05), "log", seg=12, caps=True)
    # Gable ends + roof.
    gz = n * 2 * r * 0.9 + r
    for s in (-1, 1):
        for k in range(5):
            hw = (W / 2 + 0.2) * (1 - k / 5)
            q0 = np.array([s * L / 2, -hw, gz + k * 0.3])
            q1 = np.array([s * L / 2, hw, gz + k * 0.3])
            g.cyl(R @ q0 + c, R @ q1 + c, r, "log", seg=12)
    rg = Geo()
    rg.roof((0, 0, gz), L + 0.4, W, 1.5, 0.1, "wood_old", overhang=0.5)
    g.merge(rg, yaw=yaw, offset=c)
    g.colliders.append(("box", list(c + (0, 0, H / 2)), [L, W, H], float(yaw)))
    for (x, y) in ((L / 2 + 0.5, 0), (-L / 2 - 0.5, 0), (0, W / 2 + 0.5), (0, -W / 2 - 0.5)):
        p = R @ np.array([x, y, 0]) + c
        g.cover.append((p[0], p[1], 2.2, "full"))


def log_pile(g, c, yaw, rng, length=4.0, rows=4):
    R = rot_z(yaw)
    c = np.asarray(c, float)
    r = 0.2
    for k in range(rows):
        for i in range(rows - k):
            y = (i - (rows - k - 1) / 2) * 2 * r * 1.02
            z = r + k * 2 * r * 0.87
            q0 = np.array([-length / 2 + rng.normal(0, 0.1), y, z])
            q1 = np.array([length / 2 + rng.normal(0, 0.1), y, z])
            g.cyl(R @ q0 + c, R @ q1 + c, r * rng.uniform(0.85, 1.1), "log", seg=8)
    h = rows * 2 * r * 0.87
    g.colliders.append(("box", list(c + (0, 0, h / 2)), [length, rows * 2 * r, h], float(yaw)))
    g.cover.append((c[0], c[1], h, "crouch" if h < 1.2 else "full"))


def lumber_camp(rng):
    g = Geo()
    log_cabin(g, (-5.5, 3.5, 0), 0.1, rng)
    log_cabin(g, (5.0, 5.5, 0), -0.2, rng, L=5.0, W=4.0)
    # Open sawmill shed.
    sx, sy = 1.0, -5.5
    for x in (-3.5, 0, 3.5):
        for y in (-2.0, 2.0):
            g.beam((sx + x, sy + y, 0), (sx + x, sy + y, 3.0 if y < 0 else 3.6), 0.22, 0.22, "wood_old", collide=True)
    rg = Geo()
    for x in np.arange(-4.0, 4.01, 0.9):
        rg.box((x, 0, 3.3), (0.88, 5.0, 0.04), "metal_rust", R=rot_z(0) @ _tilt(0) @ _tilt_x(-0.12), collide=False)
    g.merge(rg, offset=(sx, sy, 0))
    g.beam((sx - 2.5, sy, 0.8), (sx + 2.5, sy, 0.8), 0.6, 0.12, "wood")         # saw bench
    g.cyl((sx, sy - 0.4, 1.0), (sx, sy + 0.4, 1.0), 0.45, "metal_paint", seg=16)  # saw blade housing
    log_pile(g, (-6.0, -6.0, 0), 0.05, rng, 4.5, 4)
    log_pile(g, (8.0, -3.0, 0), 1.4, rng, 4.0, 3)
    # Plank stack.
    for k in range(6):
        g.box((5.5, -8.0, 0.1 + k * 0.06), (3.0, 1.0, 0.05), "wood", collide=False)
    g.colliders.append(("box", [5.5, -8.0, 0.2], [3.0, 1.0, 0.4], 0.0))
    return g


def _tilt_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def bunker(rng):
    """Concrete pillbox: embrasure facing +X, rear entrance behind a blast wall."""
    g = Geo()
    L, W, H, t = 5.5, 4.5, 2.3, 0.6
    g.wall((L / 2, -W / 2), (L / 2, W / 2), H, t, "concrete", openings=[(0.9, W - 0.9, 1.15, 1.55)], rng=rng, block=0.6)
    g.wall((-L / 2, W / 2), (-L / 2, -W / 2), H, t, "concrete", openings=[(1.6, 2.6, 0, 1.95)], rng=rng, block=0.6)
    g.wall((-L / 2, -W / 2), (L / 2, -W / 2), H, t, "concrete", rng=rng, block=0.6)
    g.wall((L / 2, W / 2), (-L / 2, W / 2), H, t, "concrete", rng=rng, block=0.6)
    g.box((0, 0, H + 0.25), (L + 0.8, W + 0.8, 0.5), "concrete")
    g.box((-L / 2 - 1.6, 0, 1.0), (0.5, 2.6, 2.0), "concrete")                  # blast wall
    for y in (-W / 2 - 0.2, W / 2 + 0.2):
        sandbag_wall(g, (-L / 2, y), (L / 2, y), 2, rng, z0=H + 0.5)
    g.cover += [(L / 2 - 0.5, 0, 1.15, "embrasure"), (-L / 2 - 1.2, 0, 2.0, "full")]
    return g


def trench_works(xy_local, depth, half_w, enemy_side, rng):
    """Revetment planks on both walls, duckboards, sandbag parapet + wire on the enemy side.
    xy_local: (n, 2) polyline in local coords; z values handled by caller (relative to floor)."""
    g = Geo()
    d = np.gradient(xy_local, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    for i in range(0, len(xy_local) - 1, 1):
        p, q = xy_local[i], xy_local[i + 1]
        for s in (-1, 1):
            a = p + nrm[i] * s * half_w
            b = q + nrm[i + 1] * s * half_w
            m = (a + b) / 2
            seg = np.linalg.norm(b - a)
            yaw = np.arctan2(b[1] - a[1], b[0] - a[0])
            g.box((m[0], m[1], depth / 2), (seg + 0.02, 0.05, depth), "wood_dark", yaw=yaw, collide=False)
            if i % 2 == 0:
                g.box((a[0], a[1], depth / 2 + 0.1), (0.1, 0.1, depth + 0.2), "wood_dark", collide=False)
        m = (p + q) / 2
        yaw = np.arctan2(q[1] - p[1], q[0] - p[0])
        g.box((m[0], m[1], 0.05), (np.linalg.norm(q - p), 2 * half_w - 0.15, 0.06), "wood_old", yaw=yaw, collide=False)
    # Parapet on the enemy side (z relative to floor -> at ground level = depth).
    side = enemy_side
    pts = xy_local + nrm * side * (half_w + 0.45)
    for i in range(0, len(pts) - 2, 2):
        sandbag_wall(g, pts[i], pts[i + 2], 3, rng, z0=depth - 0.05)
    # Barbed wire coils further out.
    wpts = xy_local + nrm * side * (half_w + 5.0)
    for i in range(0, len(wpts) - 1):
        a, b = wpts[i], wpts[i + 1]
        seg = np.linalg.norm(b - a)
        dd = (b - a) / max(seg, 1e-6)
        nn = np.array([-dd[1], dd[0]])
        k = 14
        prev = None
        for j in range(k + 1):
            tt = j / k
            ang = tt * 2 * np.pi * 3
            pt = np.array([*(a + dd * seg * tt + nn * 0.4 * np.cos(ang)), depth + 0.45 + 0.4 * np.sin(ang)])
            if prev is not None:
                g.beam(prev, pt, 0.015, 0.015, "metal_rust")
            prev = pt
        if i % 3 == 0:
            g.beam((*a, depth - 0.2), (*a, depth + 1.0), 0.05, 0.05, "metal_rust")
    for p in pts[::4]:
        g.cover.append((p[0], p[1], depth + 0.4, "trench"))
    return g


def hedgehog(rng):
    g = Geo()
    for a, b in (((-0.9, 0, 0), (0.9, 0, 1.4)), ((0, -0.9, 0), (0, 0.9, 1.4)), ((0.9, 0.9, 0), (-0.9, -0.9, 1.4))):
        g.beam(a, b, 0.16, 0.16, "metal_rust", collide=False)
    g.colliders.append(("box", [0, 0, 0.7], [1.8, 1.8, 1.4], 0.0))
    g.cover.append((0, 0, 1.0, "crouch"))
    return g


def stump(rng, r=None):
    g = Geo()
    r = r or rng.uniform(0.2, 0.42)
    h = rng.uniform(0.3, 0.7)
    g.cyl((0, 0, -0.2), (0, 0, h), r, "bark", seg=10, r1=r * 0.9, caps=False)
    # Cut top (lighter wood) slightly slanted.
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    top = np.column_stack([np.cos(th) * r * 0.9, np.sin(th) * r * 0.9, h + 0.06 * np.cos(th)])
    g.add(top, [list(range(10))], "wood_cut")
    return g


def branch_pile(rng):
    g = Geo()
    for _ in range(12):
        a = rng.uniform(0, np.pi)
        L = rng.uniform(1.0, 2.6)
        c = rng.normal(0, 0.4, 2)
        d = np.array([np.cos(a), np.sin(a)]) * L / 2
        z = rng.uniform(0.05, 0.45)
        g.cyl((*(c - d), z), (*(c + d), z + rng.normal(0, 0.15)), rng.uniform(0.03, 0.07), "bark", seg=5,
              caps=False)
    return g
