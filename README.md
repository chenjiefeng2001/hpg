# hpg — WebGPU Rendering Runtime

> 一个抽象中心为 **GPU 工作（Draw Work）** 而非 3D 对象的 WebGPU 渲染运行时。
> 生态位：**位于原生 WebGPU 之上，Three.js 之类的 SceneGraph 之下**。

## 设计军规（不可协商）

1. **不替用户做场景决策** — 没有 Scene / Node / Object3D。上层(ECS / SceneGraph / React 状态)
   只需把每帧要画的东西摊平成 `RenderItem[]` 提交，渲染器负责产出最优的 GPU Command。
2. **渐进增强而非强制先进** — 画 1 个物体和 1 万个物体是**同一条代码路径**、**同一份 Shader**。
   单物体 = `instanceCount = 1` 的实例化绘制；相邻同状态绘制自动合并为 Instanced Draw。
3. **500 对象准则** — 任何优化不得让 500 个物体的简单场景变慢。热路径零堆分配。

## 渲染意图包（最小工作单元）

```ts
interface RenderItem {
  geometry: Geometry;        // 顶点/索引数据切片（静态显存，注册一次）
  pipeline: PipelineHandle;  // 管线（状态排序的主键）
  bindGroup?: BindGroupHandle;
  transforms?: Float32Array; // 单个或 N 个 4x4 矩阵（列主序）；缺省 = 单位矩阵*1
  instanceCount?: number;    // = transforms.length / 16，缺省自动推导
  instanceData?: Float32Array; // 每实例跨步扁平数据（如颜色）
  bounding?: BoundingSphere; // 声明后才有资格进入后续 GPU 剔除
  depth?: number;            // 覆盖排序深度
}
```

WGSL 侧永远面向同一套实例语义编写——Direct / Instanced /（未来的）Indirect 对用户完全无感：

```wgsl
struct InstanceData { modelMatrix: mat4x4<f32>, color: vec4<f32> };
@group(1) @binding(0) var<storage, read> inst : array<InstanceData>;
```

## 数据流

```
submit(items) ─▶ validate ─▶ spatialKey(xyz→sortDepth)
            ─▶ packKey(pass|pipeline|bindGroup|depth) ─▶ counting-sort (O(n), 稳定)
            ─▶ collect: 相邻同 geometry+pipeline 合并为 Batch（实例重排 remap）
            ─▶ RingBuffer: 单块瞬态显存，顺序写入，帧尾归位（零 GC）
            ─▶ ExecutionBackend: bindGroup(偏移切片) + drawIndexed(..., instanceCount)
```

## Phase 状态

- **Phase 1 ✅** 核心数据模型：`RenderItem` 契约、数学库（列主序右手系）、
  Geometry 动态竞技场（共享显存 + free-list）、单帧环形缓冲池、录制的渲染命令 IR。
- **Phase 2 ✅** 渲染核心：64 位排序键编码、三路计数排序、自动合批器、
  Pipeline/BindGroup 内容哈希缓存、统一实例存储模型、执行后端（Direct/Instanced 统一）、
  `Renderer.submit()` 全链路、统计 HUD、可运行 Demo。
- **Phase 3 ⏳（门槛已预留）** `bounding` 字段 + Compute Culling + `drawIndexedIndirect`，
  须证明 500 对象零负优化后方可进 Core。

## 工具链

| 命令 | 作用 |
|---|---|
| `npm run dev` | 启动 Demo（Vite） |
| `npm test` | Vitest 单元 + 集成测试（纯逻辑，无需 GPU） |
| `npm run typecheck` | TS strict 全量检查 |
| `npm run build` | Demo 产物构建 + 库 d.ts 输出 |

Node ≥ 18 / 浏览器需支持 WebGPU（Chrome 113+ / Edge 113+）。
