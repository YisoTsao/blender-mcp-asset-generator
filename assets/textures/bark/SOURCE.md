# Bark texture sources

All files are diffuse (albedo) maps from [Poly Haven](https://polyhaven.com), licensed **CC0 1.0** (public domain:
commercial use, modification and redistribution allowed, no attribution required — recorded here for traceability).
Downloaded at 1K (`*_diff_1k.jpg`) and downscaled to 512 px (oak: 512 × 1024), JPEG quality 85.

| File | Poly Haven asset | Author | Physical size | Used for |
|---|---|---|---|---|
| `pine_bark.jpg` | https://polyhaven.com/a/pine_bark | Dimitrios Savva | 2.0 × 2.0 m | pine |
| `knotted_pine_bark.jpg` | https://polyhaven.com/a/knotted_pine_bark | Dimitrios Savva | 1.7 × 1.7 m | spruce |
| `jolcham_oak_bark_01.jpg` | https://polyhaven.com/a/jolcham_oak_bark_01 | Charlotte Baglioni | 1.0 × 2.0 m | oak, bush stems |
| `trident_maple_bark.jpg` | https://polyhaven.com/a/trident_maple_bark | Charlotte Baglioni | 1.8 × 1.8 m | maple |
| `japanese_hackberry_bark.jpg` | https://polyhaven.com/a/japanese_hackberry_bark | Charlotte Baglioni | 1.9 × 1.9 m | alder |
| `tree_bark_03.jpg` | https://polyhaven.com/a/tree_bark_03 | Rob Tuytel | 1.0 × 1.0 m | dead trees, birch twigs |
| `bark_brown_01.jpg` | https://polyhaven.com/a/bark_brown_01 | Rob Tuytel | 1.0 × 1.0 m | (spare; logs keep the procedural moss bark) |

Mapping: `tools/vegetation/veg_materials.py` (`BARK_IMAGES`) stores the physical size in the material's
`uv_tile_m`; `plant_models.MeshBuilder.build` divides the tube UVs (metres) by it, so the textures tile at
real-world scale in Blender and after glTF export. If a file is missing, the procedural bark is used instead.
Birch keeps its procedural white bark (no suitable CC0 scan).
