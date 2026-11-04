# Forest Valley – Unreal Engine 5 import

Coordinates: Blender metres → Unreal centimetres; **UE X = Blender X, UE Y = −Blender Y**, Z up.
All instance files under `Data/cells/` are already converted (cm, UE axes, roll/pitch/yaw in degrees).

## 1. Landscape
Landscape mode → Import from file → `Landscape/heightmap_1009.r16` (or `.png`).
- Section size 63 quads, 2×2 sections per component, 8×8 components (1009 × 1009).
- Scale and location: use `ue_scale_cm` / `ue_location_cm` from `performance_report.json → map.landscape`.
- Layers (weight-blended): `riverbed, rock, dirt, scree, moss, forest_floor, grass`
  → import `Landscape/weight_<layer>_1009.png` into each layer info.
- `normal_1009.png` and `macro_albedo_1009.png` are for the far-distance / RVT macro layer.

## 2. Meshes, instances, gameplay, lighting
Enable the *Python Editor Script Plugin*, then run `ue_import.py` (Tools → Execute Python Script).
It imports `Meshes/**/SM_*_LOD0.fbx` with LOD1–3, enables Nanite on rocks / structures,
spawns one HISM actor per World Partition cell (125 m grid), PlayerStarts (tagged Team_A / Team_B),
TargetPoints for objectives / pickups, and the lighting rig of the `PRESET` in the script.
The script was generated, not run: there is no Unreal Engine on the build machine.

## 3. Collision / navigation
- Visual and collision geometry are separate: `Collision/collision.glb`
  (COL_Terrain 2 m, UCX_Structures boxes, COL_Tree_Trunks, UCX_Rocks). For UE, keep the landscape's own
  collision and use the UCX boxes as simple collision for the structure meshes.
- `Navigation/navmesh.glb` is the analysed walkable surface (reference / validation). Place a
  NavMeshBoundsVolume over the map and build the Recast navmesh in UE (agent radius 40 cm, max slope 40°).

## 4. Grass / ground cover
Grass, flowers and ferns are exported as instances in the cell files, too. For production, drive them with
Landscape Grass Types or PCG from `phase3_vegetation/density_*.png` (cull 60–90 m, no shadows).
