# 天空方塊島：守護之心｜技術架構與開發實作規格

> **Technical Architecture / Data Contract / Claude Code Implementation Plan**  
> 版本：**v2.0**｜日期：2026-11-01｜與 [`01_GAME_DESIGN_DOCUMENT.md`](01_GAME_DESIGN_DOCUMENT.md) 配套  
> 優先權：GDD 的**遊戲行為規則**為唯一真相；此文件定義資料與實作契約  
> 前提：一人開發、Web Demo 優先、無後端、可延伸 PC / Steam

## 0. 與舊版架構的非相容性變更

原先 `Cell[x,z] + elevation` 的單層 16×16 地圖不足以描述多個懸浮平台和橋梁。**不得直接把舊版 16×16 改為 32×32 就當成升級完成。**

- 地圖改為 `width=32, depth=32`，支援不規則、非填滿的地表集合；未佈置的格子是 Void。
- 使用 `SurfaceCell.id` 作為導航節點唯一識別，支援 `platformId`、`elevation 0..6`；未來可處理同 X/Z 不同高度的多層表面。
- 連接不同平台使用明確的 `Connector.chain`（橋／階梯／坡道），不能直接以距離近判斷可走。
- 3×3 Core **不可直接踩入**，改為通往 `core.approachNodeIds` 的多目標路徑搜尋。
- 一座塔占 2×2，同平台同高度；放置、升級、移除由資料層更新 4 格 footprint。
- 塔射擊必須通過**3D 距離 + 地圖邏輯 LOS**，不得仰賴 UI 的 Three.js Raycaster 決定傷害。
- 地形與路障只在 PREPARE 修改；關鍵橋／橋端保護區不在 MVP 編輯範圍。

## 1. 推薦技術棧與架構原則

| 層 | 技術 | 約束 |
|---|---|---|
| 建置 | Vite + React + TypeScript strict | Web 可靜態部署；避免不必要 SSR |
| 場景 | Three.js + React Three Fiber + Drei | 視覺渲染、相機、輸入拾取 |
| UI | React + Tailwind + Zustand | 低頻 HUD；不要每個敵人每 tick 觸發 React render |
| 模擬核心 | 純 TypeScript + 固定 tick | 不依賴 React / Three.js / DOM，接受 Commands，輸出 State + Events |
| 規則／Schema | Zod + 明確 TypeScript 型別 | 關卡、存檔、資產 manifest 驗證 |
| 尋路 | 自行實作的 A* over NavigationGraph | 連接節點、路障、雙入口、多目標到核心 |
| 存檔 | IndexedDB + schemaVersion | 只在安全時機存完整戰局 |
| 測試 | Vitest + Playwright | 邏輯回歸 + 完整遊戲流程 |
| 建模 | 程式幾何 / Blockbench / Blender | GLB 模型與遊戲占位資料分離 |
| AI 資產 | Meshy（選用） | 僅開發時批次產生；玩家 Runtime 不呼叫 Meshy API |
| PC 發行 | Electron 或其他包裝路線（後評估） | 渲染、資源、Steam 相容性需獨立測試 |

### 依賴方向（不可逆）

```text
User Input / React HUD
        │  GameCommand
        ▼
Command Controller (preview / validate / commit)
        │
        ├── Level Domain Data / Rules (Pure TS)
        │           │
        │           └── NavigationGraph + A* + LOS
        ▼
Game Simulation (20 TPS, deterministic)
        │
        ├── GameSnapshot ─────► R3F Renderer / Interpolation
        │                            └── VFX / Sounds / HUD
        │
        └── GameEvent ────────► UI Feedback / SFX / Analytics
        │
        └── Save Serializer ──► IndexedDB
```

**不得**在 Simulation 中 import Three.js、React、Zustand 或存取 Canvas；場景 Mesh 並不是唯一狀態來源。

## 2. 專案目錄

```text
skybound-defenders/
├── docs/
│   ├── 01_GAME_DESIGN_DOCUMENT.md
│   ├── 02_TECHNICAL_ARCHITECTURE.md
│   ├── ART_BIBLE.md
│   └── DECISIONS.md
├── public/assets/
│   ├── models/                 # GLB
│   ├── audio/
│   └── textures/
├── src/
│   ├── app/                    # bootstrap / lifecycle
│   ├── domain/
│   │   ├── types.ts            # strict domain contracts
│   │   ├── schemas.ts          # Zod runtime validation
│   │   └── definitions.ts      # tower/enemy/terrain balance
│   ├── levels/
│   │   ├── l1-01.definition.ts
│   │   ├── l1-01.generator.ts
│   │   └── level-validator.ts
│   ├── navigation/
│   │   ├── build-graph.ts
│   │   ├── astar.ts
│   │   ├── connectivity.ts
│   │   └── line-of-sight.ts
│   ├── commands/
│   │   ├── preview.ts
│   │   ├── validate.ts
│   │   └── execute.ts
│   ├── simulation/
│   │   ├── engine.ts
│   │   ├── fixed-loop.ts
│   │   ├── enemy.ts
│   │   ├── towers.ts
│   │   ├── combat.ts
│   │   ├── waves.ts
│   │   └── economy.ts
│   ├── scene/
│   │   ├── terrain/
│   │   ├── connectors/
│   │   ├── towers/
│   │   ├── enemies/
│   │   ├── camera/
│   │   └── effects/
│   ├── ui/
│   ├── persistence/
│   ├── state/                   # UI store + sim subscription adapter
│   └── assets/
├── scripts/
│   ├── build-level.ts
│   └── inspect-glb.ts
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

先建立單一 App，不需要預先抽出多遊戲 monorepo；等可玩的 Demo 穩定後才考慮將 Voxel Core 抽包。

## 3. 地圖與資料契約

### 3.1 核心型別（建議 TypeScript）

```ts
export type CellId = string;
export type PlatformId = string;
export type Elevation = 0 | 1 | 2 | 3 | 4 | 5 | 6;
export type GridPos = { x: number; z: number };

export type SurfaceKind = 'ground' | 'bridgeDeck' | 'stairDeck' | 'rampDeck';
export type TerrainKind = 'grass' | 'stone' | 'wood' | 'water';
export type ConnectorKind = 'BRIDGE' | 'STAIR' | 'RAMP';

export interface SurfaceCell {
  id: CellId;                  // stable, unique across all surfaces
  platformId: PlatformId;     // ground platform or connector platform ID
  pos: GridPos;               // x/z, within width/depth
  elevation: Elevation;       // top surface height index
  kind: SurfaceKind;
  terrain: TerrainKind;
  editable: boolean;          // only ordinary ground cells
  buildable: boolean;         // tower footprint candidate
  immutable: boolean;         // portals/core/protected zones
  tags?: string[];
}

export interface PlatformDef {
  id: PlatformId;
  name: string;
  cellIds: CellId[];
}

export interface ConnectorDef {
  id: string;
  kind: ConnectorKind;
  /** Includes ground endpoint A, intermediate deck nodes, endpoint B. */
  chain: CellId[];
  bidirectional: true;        // MVP all connectors allow two-way traversal
  immutable: true;            // bridge collapse/destruction is post-MVP
}

export interface PortalDef { id: string; cellId: CellId }

export interface CoreDef {
  footprintCellIds: CellId[]; // exactly 9 cells = 3x3
  approachNodeIds: CellId[];  // valid ground endpoints adjacent to core
  maxHp: number;              // MVP 20
}

export interface LevelDef {
  schemaVersion: 2;
  id: string;
  name: string;
  width: number;              // MVP 32
  depth: number;              // MVP 32
  heightStepWorld: 0.5;
  cells: SurfaceCell[];       // sparse set, no implicit floor in void
  platforms: PlatformDef[];
  connectors: ConnectorDef[];
  portals: PortalDef[];      // MVP 2, extensible 3+
  core: CoreDef;
  initialGold: number;       // 220
  maxShortestPathCost: number;// MVP 160
  waves: WaveDef[];
}

export type TowerKind = 'arrow' | 'frost' | 'cannon';
export type EnemyKind = 'slime' | 'runner' | 'golem';

export interface TowerDef {
  kind: TowerKind;
  buildCost: number;
  baseDamage: number;
  baseRangeWorld: number;
  cooldownSeconds: number;
  damageType: 'physical' | 'magic';
  muzzleOffsetWorld: number;  // configured attack origin above platform surface
  blastRadiusWorld?: number;
  slow?: { ratio: number; durationSeconds: number };
}

export interface EnemyDef {
  kind: EnemyKind;
  hp: number;
  speedCellsPerSecond: number;
  coreDamage: number;
  goldReward: number;
  physicalResistance: number; // 0.2 for golem
  targetOffsetWorld: number;  // configured target center above surface
}

export interface SpawnEntry {
  enemyKind: EnemyKind;
  count: number;
  portalIds: string[];       // deterministic distribution rule
  intervalTicks: number;
}
export interface WaveDef {
  id: string;
  entries: SpawnEntry[];
  clearGold: number;
}
```

`SurfaceCell` 不使用 `[x + z * width]` 當唯一 ID，避免未來同一 X/Z 上方與下方各有通道時資料衝突。**MVP 限制單一 X/Z 至多一個可遊玩表面（橋與普通地面皆算）**，讓 3D 選取與地形渲染先穩定；使用 `cellId` 是提前為真正多層重疊預留結構，**不是已實作重疊表面支援**。

### 3.2 局內狀態與建築占地

```ts
export type GamePhase =
  | 'MENU' | 'PREPARE' | 'COMBAT' | 'WAVE_RESULT' | 'VICTORY' | 'DEFEAT';

export interface TowerInstance {
  id: string;
  kind: TowerKind;
  anchorCellId: CellId;         // top-left (min x/z)
  footprintCellIds: CellId[];  // exactly 4, same elevation/platform
  level: 1 | 2 | 3;
  cooldownTicks: number;
  investedGold: number;
  targeting: 'FIRST';
}

export interface WallInstance { id: string; cellId: CellId; cost: 15 }

export interface EnemyInstance {
  id: string;
  kind: EnemyKind;
  portalId: string;
  hp: number;
  state: 'moving' | 'dead' | 'reachedCore';
  pathNodeIds: CellId[];
  edgeIndex: number;            // segment currently moving across
  segmentProgress: number;     // [0,1], render interpolation source
  slowUntilTick: number;
  spawnTick: number;
}

export interface GameState {
  schemaVersion: 2;
  levelId: string;
  phase: GamePhase;
  tick: number;
  rngSeed: number;
  gold: number;
  coreHp: number;
  currentWaveIndex: number;     // 0-based
  spawnCursor: number;
  waveTick: number;
  awardedWaveIds: string[];
  editedElevations: Record<CellId, Elevation>;
  walls: WallInstance[];
  towers: TowerInstance[];
  enemies: EnemyInstance[];
  nextEntityId: number;
}

export type GameCommand =
  | { type: 'SET_ELEVATION'; cellId: CellId; next: Elevation }
  | { type: 'PLACE_WALL'; cellId: CellId }
  | { type: 'REMOVE_WALL'; wallId: string }
  | { type: 'BUILD_TOWER'; kind: TowerKind; anchorCellId: CellId }
  | { type: 'UPGRADE_TOWER'; towerId: string }
  | { type: 'SELL_TOWER'; towerId: string }
  | { type: 'START_WAVE' };

export type CommandResult =
  | { ok: true; state: GameState; events: GameEvent[] }
  | { ok: false; state: GameState; code: RejectionCode; message: string };

export type RejectionCode =
  | 'INVALID_PHASE' | 'OUT_OF_BOUNDS' | 'NOT_EDITABLE'
  | 'OCCUPIED' | 'INVALID_FOOTPRINT' | 'INVALID_ELEVATION'
  | 'PROTECTED_CONNECTOR' | 'INSUFFICIENT_GOLD'
  | 'NO_ROUTE_FROM_PORTAL' | 'ROUTE_TOO_LONG'
  | 'NOT_FOUND' | 'INVALID_LEVEL_DATA';
```

**資料來源規範**：`LevelDef` 是不可變基底；GameState 只儲存修改差異及實體，從 `LevelDef + GameState` 派生 `DerivedMap`。不允許場景 Renderer 修改 `LevelDef` 或直接修改狀態物件。

### 3.3 Level Schema 驗證（啟動前必跑）

1. `width, depth` 是合法整數，所有 x/z 在盤面內。
2. `cellId` 全部唯一，引用存在；MVP 不允許同一 x/z 出現兩個表面節點。
3. 各平台 cellIds 不重複，地格的 platformId 與索引一致。
4. Connector 的 `chain` 長度 ≥2；首尾是地面端點，中間是 deck；每步 x/z 四方向相鄰、高度差 ≤1，無失效節點；兩個端點屬於不同平台。
5. 普通地形與 Connector 的隱含邊不得互相穿越；所有 Connector 橋面不建塔、不編輯。
6. Portals 為 2 個以上且互不重複，位於合法可行走地格；不可建塔、不可編輯。
7. Core footprint 恰好 9 個節點，排列成**同平台、同高度**的 3×3 正方形；Approach 至少 2 格，位於核心周邊正交相鄰的普通地格，且不與 footprint 重疊。
8. 初始每個 Portal 到任一 Approach 的最短合法路線存在，成本 ≤ `maxShortestPathCost`。
9. `editable`、`buildable`、`immutable` 組合合理；Connector 端點與保護區禁止修改。
10. 所有 `WaveDef` 敵人、數量、Portal ID、clearGold 都符合 Schema。

## 4. NavigationGraph：真正的多平台尋路

### 4.1 圖形資料

```ts
interface NavEdge { to: CellId; cost: 1; via?: string }
interface NavigationGraph {
  neighbors: Map<CellId, NavEdge[]>;
  nodes: Map<CellId, SurfaceCell>;
}
```

`buildNavigationGraph(level, state)` 由**已驗證的 Domain Data**產生可走圖：

1. 讀入有效 `SurfaceCell`；Void、水、Core footprint、Tower/WALL footprint 不可進入。
2. **普通同平台邊**：只在兩個 `kind='ground'` 節點相鄰（Manhattan=1）、高度差 ≤1 時建立雙向邊。
3. **跨平台邊**：僅透過 `Connector.chain` 提供的連續節點建立雙向邊，必要時保留 `via=connectorId`，以便動畫判定。
4. `bridgeDeck/stairDeck/rampDeck` **不能自行與旁邊其他地格建立普通邊**，避免從橋上橫移到懸崖。
5. 初版 `cost=1` 表示走一步；所有 Connector 都有實際中間節點，無瞬間傳送。
6. 使用穩定排序的 `cellId`／鄰邊順序，確保多條等長路徑時產生相同結果。

### 4.2 A* 的輸入與輸出

```ts
function findPathToCore(
  graph: NavigationGraph,
  startNodeId: CellId,
  goalNodeIds: readonly CellId[],
  maxCost: number,
): { nodeIds: CellId[]; cost: number } | null;
```

- 以 `abs(dx - goal.x) + abs(dz - goal.z)` 對多個 Approach 的最小值作 heuristic。因 Connector 也是實體相鄰單步，其成本下界成立；若日後加入電梯／傳送邊，應改為安全的 `h=0`（Dijkstra）或重設 admissible heuristic。
- 遇到 Core Approach 之一即完成，採 deterministic tie-breaking（例如 f、g、cellId）。
- 不以 Three.js `Raycaster` 或 Mesh 距離做通行判斷。
- 放置／拆除物件、升降地形時重新建立／更新圖與所有入口路徑。
- COMBAT 開始前為當前 wave 所有入口生成路線快照；戰鬥中不修改地形，敵人依其入口快照走完整路線。
- 若 Spawn 分布跨入口，仍要使用與預覽相同的導航算法，不得讓怪物走另一條藏在 Mesh 裡的近路。

### 4.3 兩段橋的 L1-01 拓樸範例

```text
P_LOW elevation 0             P_MID elevation 2            P_HIGH elevation 4
Portal A                       Portal B                         Core 3x3
  (3,14)                          (14,20)                          approach
     └── (9,14,0) ─ (10,14,1) ─ (11,14,2) ─ (12,14,2)
                                      C_AB: RAMP / BRIDGE

                              (20,13,2) ─ (21,13,3) ─ (22,13,4) ─ (23,13,4)
                                                     C_BC: RAMP / BRIDGE
```

> 上述是**拓樸連接示意**，不是照比例完整地圖；遊戲內兩橋透過其端點接入各自平台的普通行走邊。

**重要細節**：A* 的節點類型 `bridgeDeck` 仍需有碰撞與動畫用的世界座標，但只有 Connector chain 才能通行；橋梁側邊的護欄、橋柱與純視覺懸空岩石不參與圖構建。

## 5. 2×2 塔、3×3 核心與保護區

### 5.1 Footprint 生成

`resolveTowerFootprint(anchorId)` 必須：

- 找到 Anchor 的 `(x,z,platformId,elevation)`；尋找 `(x,z)、(x+1,z)、(x,z+1)、(x+1,z+1)` 的四個**普通地格**。
- 四格全部存在、同一 `platformId/elevation`、`buildable=true`、不占用、非 Portal/Core/Connector/Protected。
- 若有任一失敗，回傳 `INVALID_FOOTPRINT`；不能自動將塔擺放到距離最近的可用位置。
- 塔射擊中心 = footprint 四格平面平均中心；實際 `muzzleOffsetWorld` 由 tower definition 提供。

### 5.2 核心與 Approach

- Core 3×3 九格占用並永久保留；即使位於平台高地也不可建塔或升降。
- 至少 2 個指定 Approach；以 4 鄰接規則驗證它們與 Core footprint 邊界的關係。
- Route 的目標是 `approachNodeIds`；敵人走到 Approach → `reachedCore`，核心扣血，怪物離場。
- Approach 是**保護格**且不可放塔、牆或升降，防止 Core 3×3 與路線驗證間出現死角。

## 6. 地形編輯：預覽／驗證／提交（原子性）

```text
Input (hover / click)
  └─► create GameCommand
       ├─► phase == PREPARE ?
       ├─► target id / footprint valid ?
       ├─► not immutable, not protected, enough gold ?
       ├─► copy-on-write tentative GameState
       ├─► apply command to tentative state
       ├─► derive terrain + occupancy
       ├─► validate connector invariants / footprint
       ├─► build NavigationGraph
       ├─► check EACH portal to ANY core approach
       ├─► check route cost <= 160
       └─► success -> commit & charge; failure -> return unchanged original state
```

### 6.1 成本／限制

| 指令 | 金額 | 額外檢查 |
|---|---:|---|
| `SET_ELEVATION` +1 | −12 | 只改 editable ordinary ground；<=6 |
| `SET_ELEVATION` −1 | −8 | 只改 editable ordinary ground；>=0 |
| `PLACE_WALL` | −15 | 只占 1×1 ordinary ground；不可堵死任何 Portal |
| `REMOVE_WALL` | +7 | 存在該路障並有拆除權限 |
| `BUILD_TOWER` | −70/85/105 | 同高同平台 2×2 footprint；所有 Portal 可達 |
| `UPGRADE_TOWER` | Lv1→Lv2：−ceil(base×0.7)；Lv2→Lv3：−ceil(base×1.1) | 等級不能 >3；不改 footprint |
| `SELL_TOWER` | +floor(invested×0.65) | 釋放四格；重算通行圖 |

`SET_ELEVATION` 只接受 next 與現況高度差恰好為 1；不能一次跳高多層繞過經濟成本。**所有命令失敗時金額、格子、塔、路徑、事件序列保持原樣**。

### 6.2 必須維持的不變量

- `gold >= 0`，Core HP 範圍 `[0,maxHp]`。
- 任一 Portal 在目前配置下都能抵達至少一個 Core Approach。
- Connector 的節點與高度固定、合法，不受編輯影響。
- 多塔與路障不能重疊；一座塔必為四格合法 footprint。
- 只有 PREPARE 可以改戰場；COMBAT 中禁止任何建造／路線修改。
- 路線預覽與 Wave 開始使用**同一套圖與搜尋規則**。
- 所有修改只能以 `executeCommand` 推動，不允許 Scene UI 直接扣金或改地圖。

### 6.3 針對多入口的完整檢查偽碼

```ts
function validateRoutes(level: LevelDef, draft: GameState): ValidationResult {
  const graph = buildNavigationGraph(level, draft);
  for (const portal of level.portals) {
    const route = findPathToCore(
      graph, portal.cellId, level.core.approachNodeIds,
      level.maxShortestPathCost,
    );
    if (!route) {
      return { ok: false, code: 'NO_ROUTE_FROM_PORTAL', portalId: portal.id };
    }
    if (route.cost > level.maxShortestPathCost) {
      return { ok: false, code: 'ROUTE_TOO_LONG', portalId: portal.id };
    }
  }
  return { ok: true };
}
```

**實作注意**：若 `findPathToCore` 在超過上限時直接回 `null`，應區分「無路」與「有路但超上限」，例如先求最短路再比對成本，或回傳明確的狀態值；不可讓使用者看到錯誤原因不準確。

## 7. 攻擊範圍與視線 LOS

### 7.1 規則計算

```ts
const heightStepWorld = 0.5;
const towerElevationWorld = towerElevation * heightStepWorld;
const enemyElevationWorld = enemyElevation * heightStepWorld;

const towerOrigin = {
  x: footprintCenter.x,
  y: towerElevationWorld + towerDef.muzzleOffsetWorld,
  z: footprintCenter.z,
};
const enemyTarget = {
  x: enemyInterpolatedPos.x,
  y: enemyElevationWorld + enemyDef.targetOffsetWorld,
  z: enemyInterpolatedPos.z,
};
const highGroundBonus = 0.3 * Math.min(
  3, Math.max(0, towerElevation - enemyElevation),
);
const effectiveRange = towerDef.baseRangeWorld + levelBonus + highGroundBonus;
const inRange = distance3D(towerOrigin, enemyTarget) <= effectiveRange;
const canShoot = inRange && hasLogicalLineOfSight(towerOrigin, enemyTarget, derivedMap);
```

- `hasLogicalLineOfSight` 以**資料層**產生的遮擋 AABB／體素範圍作線段相交檢查；不得使用每一幀 UI Raycast 作傷害依據。
- 遮擋類型：實心岩壁、超過射線高度的實心地形、標記 `opaque` 的路障；**不含**草、雲、橋欄杆、單純裝飾物。
- 起點所在的塔基及目標所在格子的遮擋要明確排除，以免自我遮擋；其他塔是否遮擋暫定**否**，避免規則過於複雜。
- LOS 結果可於建造預覽與選塔後可視化。計算次數只需在攻擊索敵時做，視線預覽可用節流更新。
- 修改高度後遮擋資料須更新；**不應只更新導航圖而忘記 LOS 遮擋**。

### 7.2 投射物與傷害

- 射擊命中與傷害在該模擬 tick 內結算；視覺箭／砲彈可用 0.1～0.3 秒動畫追上，不用碰撞 Mesh 決定傷害。
- 炮塔對半徑 1.05 world units 內目標一次爆炸，對同一目標只傷害一次。
- `damage = round(baseDamage × levelMultiplier × (1 - physicalResistance))`，物理抗性只影響物理傷害。
- 冰晶塔減速 30% 持續 1.8 秒，不疊加，只更新 `slowUntilTick`。
- `FIRST` 以敵人沿路徑剩餘行走距離（格長與 segmentProgress）排序，而非到核心直線距離。

## 8. 固定時間模擬與戰鬥狀態

### 8.1 20 TPS Game Loop

```ts
const STEP_SECONDS = 0.05; // 20 TPS

function advanceOneTick(state: GameState, level: LevelDef): TickResult {
  // 1. Spawn queued enemies based on fixed waveTick.
  // 2. Advance enemy movement along the frozen route nodes.
  // 3. Resolve enemy reaching protected Core Approach.
  // 4. Update tower cooldown and targeting (3D range + LOS).
  // 5. Resolve damage/slow/death/gold rewards exactly once.
  // 6. Emit immutable GameEvents for renderer / audio.
  // 7. Determine wave clear, victory, defeat.
  // 8. Increment simulation tick and return next state.
  throw new Error('Implement in simulation module');
}
```

- `requestAnimationFrame` 只推進 accumulator 和渲染內插，不能直接決定傷害。
- 幀率下降需限制 catch-up ticks，避免一次凍結長時間；要有降速策略而非亂跳模擬狀態。
- `PAUSED` 不增加 tick；2× 速度需讓模擬推進兩倍 ticks，不能雙倍傷害。
- 不使用 `Math.random` 控制關鍵戰鬥；同關卡 seed＋命令序列＋tick 數，結果應可重現。
- 所有 spawn／die／reach-core 是互斥且只能結算一次。Core HP≤0 必須進入 DEFEAT，不應再領取勝利獎勵。
- 不同敵人走同橋時可暫時允許重疊（MVP 不需避碰），但視覺以小量位置偏移改善辨識；路徑本體仍固定。

### 8.2 Snapshot 與表現

- Sim 每 tick 提供資料快照／事件；Renderer 透過前後快照內插位置，避免 20FPS 動畫感。
- React/Zustand 訂閱低頻狀態，例如 Gold、波次、Core HP；不為 100 隻怪各建一套每幀 React state。
- 視覺特效、鏡頭震動、音效不回饋 Simulation，且不進存檔。

## 9. Scene、地形渲染、鏡頭與效能

### 9.1 島嶼建模策略

- **資料**：每個 SurfaceCell 是頂面節點，整個盤面是 sparse set，不把虛空當草地。
- **外觀**：對每個平台補出地基、側面、懸崖與岩層。側面可用重複方塊，但**不是**每個地下體素都要有獨立 React Mesh。
- **渲染**：相同材質使用 `InstancedMesh`，或按 Platform/Chunk 合併外露面；只繪製可見表面，隱藏內部面。
- **Connector**：木橋／斜橋／階梯作獨立 prefab，對應 chain 節點與固定碰撞盒。橋梁視覺不能飄離實際可走節點。
- **更新**：地形編輯只更新影響的實例矩陣／Chunk、路徑 Overlay、LOS Collider，不要整張島全量重建。

### 9.2 Raycast 與格子拾取

- 介面使用 Three.js `Raycaster` 選取地格或塔；InstancedMesh 命中須能映射 `instanceId → cellId`。
- 優先只選可操作的**頂面**；點到懸崖側面時顯示可選格的清楚規則，不要隨機跳到另一層。
- 畫面座標轉 Domain NodeId 後才送 Command Controller，不能直接修改 GLB Mesh。
- 如果地圖後期支援同一 X/Z 上下重疊表面，選取必須使用**命中 Mesh 的 cellId**，不能只用 `x/z` 查表。

### 9.3 畫面

- Orthographic Camera、俯視 35°～45°；Q/E 90° 旋轉、縮放、適量平移。
- 柔和陰影、環境光、少量 AO；天空背景乾淨；不以重度景深或 Bloom 遮住戰術路線。
- 相機轉向時維持核心與重要 Connector 可辨識；可選擇隱藏擋視線的前景樹木。
- 先對測試用桌面設備定目標 60 FPS；記錄 FPS、Draw Calls、Triangles、資產大小與載入時間；沒有實測不得宣稱全裝置 60FPS。

## 10. GLB / Meshy 低成本資產流程

### 10.1 Asset Registry

```ts
interface AssetRecord {
  id: string;
  path: string;                // /assets/models/xxx.glb
  category: 'terrain' | 'bridge' | 'tower' | 'enemy' | 'decoration';
  source: 'procedural' | 'blender' | 'blockbench' | 'meshy' | 'other';
  license: string;
  sourceUrl?: string;
  reviewed: boolean;
  sizeWorld: { x: number; y: number; z: number };
  origin: 'bottom-center';
  triangles?: number;
  materials?: number;
  textures?: number;
}
```

- 固定 Y-up，+Y 高度、每格 1 世界單位、單位格高度 0.5；模型底部原點與邏輯 `Footprint` 對齊。
- 重要資產：三塔、三怪、橋梁、核心；地形與樹石用程序幾何最划算。
- Meshy 批次任務要控制 Prompt、風格、色票、比例、輸出、授權與來源；每個新模型都經 `inspect-glb` 檢查尺寸、面數、貼圖、GLB 是否可載入。
- 不把 Meshy API Key 交給遊戲前端，也不在玩家遊玩時生成模型。
- 同一種塔用不同顏色、旗幟與附加方塊表現 Lv2/Lv3，避免重複建模。

## 11. 存檔與相容性

- IndexedDB 主鍵：`skybound.v2.save.<slot>`；`schemaVersion=2` 與 `levelId` 必填。
- 存檔包含 GameState（地形修改、塔、牆、Gold、Core HP、已完成波次、獎勵紀錄），不包含 Three.js Mesh 或 Texture。
- **安全存檔點**：PREPARE / WAVE_RESULT；COMBAT 中顯示「本階段結束後自動儲存」即可。
- 讀檔時依 Zod 驗證，重新載入 LevelDef 再 deriveMap，不接受任意外部 JSON 直接覆蓋內部資料。
- 舊 v1 存檔（單層 16×16）**不保證升級相容**；開發版可提示重開新關或寫一次性 migration；不得靜默轉換造成錯誤。
- 已領取清波獎勵需用 `awardedWaveIds` 防重；讀檔後不可重複發金幣。

## 12. 測試矩陣（必須寫在實作前或同步完成）

### 12.1 Schema / Level Validator

| 測試 | 預期 |
|---|---|
| 32×32 有 Void 與不規則平台 | 合法，Void 不產生導航節點 |
| Connector 中段缺失 | 不合法 |
| Connector 兩端高度差過大，鏈節點無逐級過渡 | 不合法 |
| 橋面意外與鄰接平台自動相連 | 必須不發生 |
| 任一 Portal 初始無法到 Core Approach | 不合法 |
| Core 不是 3×3／有重複格 | 不合法 |
| Approach 距 Core 遠、不是邊界鄰格 | 不合法 |
| 同 X/Z 重疊 SurfaceCell（MVP） | 拒絕，提示不支援重疊表面 |
| 連接端點在可編輯區 | 不合法 |

### 12.2 Pathfinding / Edit Command

| 測試 | 預期 |
|---|---|
| 低→中→高跨兩橋 | 只能循 Connector chain，不穿越斷崖 |
| 任意地格相差 2 級 | 無普通地面邊 |
| 高台塔 2×2 其中一格在不同高度 | `INVALID_FOOTPRINT` |
| 2×2 塔跨平台、跨橋或接觸虛空 | `INVALID_FOOTPRINT` |
| 堵住 Portal A、B 任一入口 | 拒絕，Gold 不變、狀態不變 |
| 在可編輯區升降導致無路 | 拒絕，地形與 LOS 資料不更新 |
| Connector 桥口／橋面修改 | `PROTECTED_CONNECTOR` |
| 改路後新路超過 160 cost | `ROUTE_TOO_LONG` |
| 放路障合法且繞路 | 路線 Overlay 與 Simulation 路徑一致 |
| 升降兩次試圖從 0 直接到 2 | 必須分兩指令與兩次付費 |

### 12.3 LOS / Combat

| 測試 | 預期 |
|---|---|
| 3D 直線距離超出射程 | 不可射擊 |
| 高台塔有射程加成，但岩壁遮擋 | 仍不可射擊 |
| 草、橋欄杆遮擋模型 | 不影響邏輯射擊 |
| 升地改變 LOS | 下次預覽／攻擊判定會更新 |
| 炮擊多隻怪、傷害結算 | 每個敵人每發至多受一次傷 |
| 冰晶塔重複減速 | 不疊加，只刷新時間 |
| 怪物到 Approach | 扣 Core HP、不得給 Gold |
| 怪物在到達前被殺 | 給 Gold、不得扣 Core HP |
| 同 Seed/Commands/Ticks | 結果完全一致 |

### 12.4 E2E

1. 開始 L1-01：畫面出現三塊高低平台、兩條橋、兩入口與 3×3 核心。
2. 放置 2×2 箭塔；看到四格占位、Gold 扣除與實際射程預覽。
3. 嘗試把橋口封死；顯示明確錯誤且金額不變。
4. 改變可編輯格高度；怪物路線與塔 LOS 即時更新。
5. 開始 Wave 1，史萊姆沿兩段橋／階梯抵達高台，攻擊與核心受損正常。
6. 完成 Wave 5 → Victory；重新開始一局結果重置。
7. 在 PREPARE 存檔後重新整理，塔、牆、高度、金幣與已完成波次均一致。

## 13. 效能目標與風險控制

- MVP 只有 32×32 的水平**容量上限**，實際 SurfaceCell 因 Void 可能少於 1,024；多層外觀地基不等於邏輯節點數量。
- 路徑在 PREPARE 編輯時重算；COMBAT 路線凍結，避免每隻怪物每幀做 A*。
- 當大量怪物行走同一路線時，重用入口路徑；仍保留怪物自己的 `pathProgress`。
- 若 Tile 提示、LOS 預覽昂貴，進行節流，且 UI 運算不影響固定 tick 模擬。
- 建立 debug overlay：可行走節點、Connector edges、各入口路線、障礙 footprint、LOS、FPS、Draw Calls。
- 不依賴動態完整 3D NavMesh；若日後新增飛行與真正垂直多層表面，再評估架構升級。

## 14. Claude Code 執行方式與分階段 Prompt

### 14.1 Master Prompt（第一次請先使用）

```text
你是本專案的 Lead Game Developer、Technical Artist 與 QA Engineer。

請先完整閱讀 docs/01_GAME_DESIGN_DOCUMENT.md 與
 docs/02_TECHNICAL_ARCHITECTURE.md。

本版核心已定案：
- 32x32 sparse multi-platform voxel map，elevation 0~6
- 3 個主要高度平台、2 條明確的 Connector chain
- 2 個 Portal、3x3 Core + approach nodes
- 2x2 Tower Footprint
- 準備期才可升降地形、放路障與建塔
- 多入口 A* 路徑驗證與原子性提交
- 射擊必須 3D 距離 + 邏輯 Line-of-Sight
- Pure TypeScript 20TPS simulation 與 React/R3F 分離

請先讀取 Repo、提出與現有架構的差異與變更清單，
建立 docs/DECISIONS.md 記錄任何設計問題。
不能默默套用舊的 16x16 單層規則，也不能一次實作全部功能。
每個 Phase 完成後都執行 Typecheck、Lint、Unit Tests，
並回報檔案變更、驗收結果、未完成項目。
現在只執行 Phase 0。
```

### 14.2 Phase 0 — Domain / Schema / Level Validator

```text
請只實作 Phase 0：
1. 建立 Vite + React + TS strict、Vitest、Playwright、Zod。
2. 實作 LevelDef/SurfaceCell/Connector/Core/TowerInstance/Command 型別。
3. 實作 Zod schema 與 Level Validator。
4. 建立 32x32 L1-01 三平台、兩連接件的資料產生器。
5. 測試所有 Connector chain、3x3 Core、Approach、2x2 Footprint。
先不做戰鬥、UI、美術。驗收：所有關卡不變量測試通過。
```

### 14.3 Phase 1 — 多層 Voxel 場景與選取

```text
請只實作 Phase 1：
1. 用 R3F 顯示三塊不同高度的漂浮平台，實際有崖壁與虛空。
2. 用程式化幾何／InstancedMesh 畫地形。
3. 增加兩條與 Connector chain 對齊的木橋／斜坡／階梯。
4. 顯示 3x3 Core、兩個 Portal、2x2 塔位預覽。
5. Orthographic Camera、Q/E 90度旋轉、縮放、格子拾取。
6. 做 Debug Overlay：cellId、elevation、platformId。
不得把視覺 Mesh 當成導航資料。
驗收：點擊任一橋面都能取得正確 cellId，場景觀感接近美術參考。
```

### 14.4 Phase 2 — 多平台 A*、編輯與射程視線

```text
請只實作 Phase 2：
1. buildNavigationGraph：普通同平台邊 + Connector chain 邊。
2. 多入口到 Core Approach 的 deterministic A*。
3. 計算 2x2 塔 footprint、牆與可編輯地形。
4. preview/execute command 原子性檢查、Gold 扣費。
5. 兩個入口的路線 Overlay；封路顯示錯誤且不扣錢。
6. 建立 3D 攻擊距離與邏輯 LOS、預覽遮擋。
7. 寫 Pathfinding/LOS/EditCommand 完整單元測試。
不要加入怪物戰鬥。
驗收：地形真能改路、橋不能亂跨、LOS 會受崖壁影響。
```

### 14.5 Phase 3 — 第一個完整波次

```text
請只實作 Phase 3：
1. 建立 Pure TS 20TPS Simulation。
2. 史萊姆沿入口路線與 Connector chain 移動。
3. 箭塔依 3D Range + LOS 自動索敵與射擊。
4. 完成 Core Approach 抵達結算、擊殺 Gold、Wave 1。
5. 渲染層內插位置與播放視覺投射物動畫。
6. 測試死亡/抵達核心互斥、固定 Seed 可重現。
驗收：PREPARE → COMBAT → WAVE_RESULT 完整可玩。
```

### 14.6 Phase 4 — 內容完整化

```text
請只實作 Phase 4：
1. 增加冰晶塔、砲塔、疾風獸、石殼巨人。
2. 加入塔升到 Lv3、出售、五波敵人與清波獎勵。
3. 核心血量、勝敗、星級、可重玩。
4. IndexedDB 安全存檔／載入（schemaVersion=2）。
5. 補足所有整合測試與 E2E。
不得加入多人與 Roguelite 選卡。
驗收：完整五波可玩、經濟和 LOS 判定一致、獎勵不重複。
```

### 14.7 Phase 5 — 美術打磨與對外 Demo

```text
請只實作 Phase 5：
1. 統一 Voxel ART_BIBLE，替換過於粗糙的占位模型。
2. 補齊塔與怪物動畫、柔和陰影、地形編輯與命中特效。
3. 加入 HUD、教程、錯誤提示、設定、音效與鏡頭遮擋處理。
4. Profile FPS/Draw Calls/模型容量、優化大量重複方塊。
5. 手動試玩 + Playwright E2E + 回歸測試。
6. 產出 Demo 發行檢查表與資產授權清單。
驗收：陌生玩家能完成第一局，沒有穿崖／封路／死檔問題。
```

## 15. 技術驗收總表

- [ ] 使用 32×32 **稀疏多平台**資料，不默認每個格都有土地。
- [ ] 三平台與 2 條 Connector 有**實際節點鏈**、不可瞬移跨空隙。
- [ ] 高度 0–6 可配置，世界 Y 高度映射一致。
- [ ] 核心 3×3、Approach 目標、2×2 塔的占地與驗證一致。
- [ ] 任一入口在所有合法編輯後仍可到核心、長度上限規則有效。
- [ ] 只在 PREPARE 編輯；非法指令不改 GameState，不扣 Gold。
- [ ] 3D 射程和 Logic LOS 正確；高地與岩壁會真實影響塔。
- [ ] 戰鬥固定 20TPS、可測試、與 R3F 分離。
- [ ] 五波完整、獎勵防重、存檔可復原。
- [ ] 模型來源清楚，玩家遊戲時不呼叫 Meshy。
- [ ] Typecheck、Lint、Vitest 與 Playwright 通過。

## 16. 開發前可查閱的官方文件

技術選型將隨套件版本變動，以下僅為後續查證入口：

- Three.js InstancedMesh：https://threejs.org/docs/#api/en/objects/InstancedMesh
- Three.js Raycaster：https://threejs.org/docs/#api/en/core/Raycaster
- React Three Fiber：https://r3f.docs.pmnd.rs/
- Zod：https://zod.dev/
- Vitest：https://vitest.dev/
- Playwright：https://playwright.dev/
- glTF Transform：https://gltf-transform.dev/
- Meshy API：https://docs.meshy.ai/

---

> **最後提醒**：先驗證 NavigationGraph 與合法改路，再開始大量製作 3D 模型。這款遊戲的獨特性來自**跨平台的地形策略**，不是模型數量。
