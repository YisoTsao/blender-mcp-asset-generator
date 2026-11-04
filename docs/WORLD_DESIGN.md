# Realistic FPS World Environment — World Design Specification

## 1. Project Vision

建立一套高擬真 3D FPS 射擊遊戲地圖生成系統。

核心目標是透過 Claude Code、Blender Python API 及 Unreal Engine 5，建立具有 AAA Game Environment 視覺品質的自然環境。

不採用 Low-poly、卡通或簡化幾何風格。

整體追求 Photorealistic、Physically Based Rendering（PBR）及真實自然地貌。

## 2. Technical Stack

### Environment Generation
- Blender
- Blender Python API (bpy)
- Geometry Nodes
- Procedural Mesh Generation
- Heightmap Generation
- Hydraulic Erosion
- Thermal Erosion
- Python Noise Algorithms

### Game Engine
- Unreal Engine 5
- Landscape System
- Nanite
- Lumen
- PCG Framework
- Water System
- Niagara
- Material Editor
- Collision System
- Navigation Mesh

### Asset Formats
- FBX：模型資產交換
- GLB：通用場景與資產交換
- EXR / PNG：Heightmap、Normal、Albedo 等貼圖
- JSON：地圖結構及遊戲邏輯
- BLEND：Blender 原始專案

## 3. World Specifications

第一張地圖：

- Map Name: Forest Valley
- World Size: 1000m × 1000m
- Maximum Elevation Difference: 200m
- Art Style: Photorealistic
- Terrain Type: Mountain Forest Valley
- Environment: Temperate Forest
- Weather: Clear / Cloudy / Foggy / Rain
- Lighting: Dynamic Daylight
- Unit: Meter

地圖必須具備自然的高低差，不允許使用單純平面地板加上零散山丘的方式假裝自然地形。

## 4. Terrain Generation

建立獨立 Terrain Generator。

### Terrain Features
- Large Mountain Ranges
- Rolling Hills
- Valleys
- Rocky Cliffs
- Natural Slopes
- River Basins
- Flat Combat Areas
- Natural Terrain Boundaries

### Generation Algorithms
- Perlin Noise
- Simplex Noise
- Hydraulic Erosion
- Thermal Erosion
- Multi-scale Terrain Noise
- Heightmap Generation

### Requirements
- 山脈輪廓自然且具有尺度變化。
- 山谷必須符合自然地形走向。
- 山坡必須有合理的坡度。
- 河流區域必須預留合理的地勢。
- 地形必須避免明顯的重複 Noise 圖案。
- 支援固定 Seed。
- 支援局部重新生成。
- 支援不同地形解析度。

## 5. Water System

建立獨立 Water Generation System。

### River
- River Spline
- River Width Variation
- River Depth
- River Bed
- River Banks
- Water Flow Direction
- Water Current Speed
- Shallow Water
- Deep Water
- Rapids

### Lake
- Natural Lake Shore
- Variable Water Depth
- Underwater Terrain
- Water Surface

### Waterfalls
- Waterfall Mesh
- Niagara Water Particles
- Foam
- Splash Effects

### Water Rendering
- Physically Based Water Material
- Animated Normal Maps
- Flow Maps
- Refraction
- Reflection
- Depth-based Color
- Shoreline Foam

重要：水體的視覺效果與物理流體模擬必須分開設計。一般河流使用即時 Shader 和 Flow Map，不應全部依賴預先烘焙的 Fluid Simulation。

## 6. Vegetation System

建立自然森林生態系統。

### Trees
- Pine Trees
- Oak Trees
- Birch Trees
- Dead Trees
- Fallen Trees

### Ground Vegetation
- Grass
- Ferns
- Bushes
- Moss
- Roots
- Fallen Leaves
- Small Plants

### Placement Rules
- 根據地形高度分配植被。
- 根據坡度控制樹木生成。
- 根據濕度分配不同植物。
- 河流附近增加濕地植被。
- 高海拔區域改變樹種。
- 避免樹木均勻排列。
- 避免樹木互相穿插。
- 避免植被出現在懸崖或不合理位置。

使用 PCG、Instance Placement 或 Blender Geometry Nodes。

## 7. Rock and Geological System

建立岩石與地質生成系統。

### Rock Types
- Large Boulders
- Rocky Cliffs
- Mountain Rocks
- River Rocks
- Small Pebbles
- Weathered Rocks
- Moss-covered Rocks

### Requirements
- 岩石尺寸具有自然變化。
- 不同岩石必須具有不同輪廓。
- 河床岩石應具有合理分布。
- 山坡岩石應符合重力及地形邏輯。
- 支援 PBR 材質及苔蘚覆蓋。

## 8. Environment System

場景包含：

- Dirt Roads
- Forest Trails
- Bridges
- Wooden Structures
- Military Outposts
- Abandoned Buildings
- Defensive Structures
- Ruins
- Natural Debris

所有建築與人造物件都必須符合場景世界觀。

## 9. Lighting and Atmosphere

建立完整環境光照系統：

- Dynamic Sky
- Directional Sunlight
- Sky Atmosphere
- Volumetric Fog
- Ambient Occlusion
- Global Illumination
- Reflection
- Cloud Shadows
- Weather Effects

支援：
- Morning
- Noon
- Sunset
- Night
- Foggy Weather
- Rainy Weather

## 10. FPS Gameplay Integration

地圖必須能夠作為真正的 FPS 遊戲關卡使用。

包含：

- Player Spawn Points
- Team Spawn Points
- Weapon Spawn Points
- Ammo Spawn Points
- Health Points
- Cover Locations
- Combat Zones
- Objective Locations
- Kill Zones
- Collision Mesh
- Navigation Mesh

視覺模型與碰撞模型必須分離。

地圖必須保留合理的戰鬥空間、視線距離、掩體及移動路線。

## 11. Performance Requirements

不能只追求畫質而忽略遊戲效能。

必須考慮：

- Nanite 適用資產
- LOD / HLOD
- Instanced Static Mesh
- PCG Culling
- Distance-based Vegetation
- Texture Streaming
- Collision Simplification
- World Partition
- Landscape Streaming
- Draw Call Optimization

大型地圖必須採取分區管理，而不是把全部資產合併成單一巨大 Mesh。

## 12. Deliverables

每張地圖應能產生：

1. Blender Source Project (.blend)
2. Terrain Heightmap
3. Terrain Material Maps
4. FBX / GLB Assets
5. Unreal Engine Landscape
6. Unreal Engine Map Scene
7. Gameplay Metadata JSON
8. Preview Images
9. Performance Report

## 13. Development Principles

- 優先採用真實地形生成演算法。
- 不使用大量隨機物件堆疊取代自然場景設計。
- 所有生成參數必須可配置。
- 支援 Seed 重現。
- 支援局部重新生成。
- 支援模組化資產替換。
- 每個生成模組都必須能獨立測試。
- 每個階段必須實際產生 3D 場景並檢查。
- 不可只以 Python 程式碼存在作為完成標準。
- 必須同時檢查視覺品質、幾何品質及遊戲執行效能。