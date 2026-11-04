# Forest Valley - Performance Report

Instances: 209122  |  assets: 76  |  World Partition 4x4 cells of 125 m

## Viewpoints (triangles drawn)

| view | instances | tris (LOD+cull) | tris (no LOD) | reduction | draw calls (HISM est.) |
|---|---|---|---|---|---|
| Team_A_spawn | 20346 | 1,655,028 | 15,749,152 | 89.5% | 347 |
| Team_B_spawn | 17945 | 1,568,950 | 15,749,152 | 90.0% | 348 |
| Objective_A | 17335 | 1,692,796 | 15,749,152 | 89.3% | 369 |
| Objective_B | 21600 | 1,856,083 | 15,749,152 | 88.2% | 326 |
| Objective_C | 18733 | 1,791,407 | 15,749,152 | 88.6% | 369 |
| Objective_D | 18347 | 1,744,738 | 15,749,152 | 88.9% | 371 |

## World Partition cells

| cell | instances | unique meshes | LOD0 tris | draw calls est. |
|---|---|---|---|---|
| 0_0 | 1537 | 39 | 1,502,404 | 81 |
| 0_1 | 13488 | 45 | 1,284,916 | 101 |
| 0_2 | 13340 | 46 | 996,774 | 102 |
| 0_3 | 797 | 29 | 1,340,308 | 58 |
| 1_0 | 16874 | 52 | 606,228 | 104 |
| 1_1 | 24885 | 47 | 528,060 | 101 |
| 1_2 | 21479 | 51 | 635,534 | 105 |
| 1_3 | 18484 | 52 | 530,014 | 113 |
| 2_0 | 19658 | 52 | 539,676 | 113 |
| 2_1 | 19184 | 52 | 1,011,132 | 109 |
| 2_2 | 20452 | 58 | 464,782 | 118 |
| 2_3 | 23246 | 50 | 619,916 | 105 |
| 3_0 | 3393 | 40 | 1,222,184 | 85 |
| 3_1 | 1536 | 48 | 1,548,768 | 93 |
| 3_2 | 8658 | 43 | 1,311,638 | 96 |
| 3_3 | 2111 | 39 | 1,606,818 | 84 |

## Recommendations

- Rocks and structures: enable Nanite (opaque, >1k tris); keep collision as the separate simplified meshes.
- Trees / bushes: HISM (Foliage) with the exported LOD0-3 + screen sizes; Nanite foliage only on UE 5.5+.
- Grass / flowers / ferns: Landscape Grass Type or PCG from the density maps, cull 60-90 m, no shadows.
- Pebbles: cull at 45 m, no collision, no shadows.
- World Partition: 125 m grid (4 x 4), loading range ~250 m; HLOD layer from HLOD_cell_* proxies.
- Landscape: 1009 heightmap, 8x8 components; Runtime Virtual Texture for the weight-blended material.
- Textures: all power-of-two, BC7 / BC5 normals, streaming enabled; foliage atlases 1024.
