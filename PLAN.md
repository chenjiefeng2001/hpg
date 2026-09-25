# hpg 下一步工作规划（修正版）

> 本规划基于架构评审反馈修正。核心调整：
> 1. 去除未经验证的性能声明
> 2. 将自动合批定位为优化手段而非语义要求
> 3. 聚焦架构正确性而非"比 Three.js 快"
> 4. 承认当前已知风险

> 本文件是阶段执行记录，不是当前 API 的唯一来源。当前 package contract、CI 触发条件、
> GitHub Release 行为和消费者限制以 `README.md`、`package.json`、`.github/workflows/` 与
> `scripts/verify-package.mjs` 为准。历史阶段的数字、文件清单和性能结果仅表示当时的测量快照。

---

## 架构原则（新增）

```
RenderItem
    │
    ├── incompatible → Direct Draw（逐对象提交）
    │
    └── compatible
          │
          ▼
       Instance Batch（自动合批）

Batching is an optimization, not a semantic requirement.
```

合批失败不应成为渲染错误的来源。Direct Draw 路径与 Instanced Draw 路径必须产生相同的视觉结果。

---

## Sprint 0 — 阻塞项 已完成

| # | 问题 | 文件 | 修复方案 | 状态 |
|---|------|------|----------|------|
| 0.1 | **Demo 缺失** | `index.html`, `demo/main.ts` | 创建 5 实例自动合批 Demo | 完成 |
| 0.2 | **Polyfill 位掩码错误** | `test/renderer.test.ts`, `test/ringbuffer.test.ts` | `STORAGE: 0x04` → `0x80` | 完成 |
| 0.3 | **Depth 纹理不 resize** | `src/core/renderer.ts` | 每帧检查 canvas 尺寸，变化时重建 depth texture | 完成 |
| 0.4 | **`exports` 只指向 `.ts` 源码** | `package.json` | 添加 `types` + `import` 双入口 | 完成 |
| 0.5 | **tsconfig 不一致** | `tsconfig.build.json` | 统一 strict flags，移除 `noUncheckedIndexedAccess` | 完成 |

---

## Sprint 1 — 架构正确性 已完成

### 1A. GeometryArena 生命周期管理 完成

| 任务 | 说明 | 状态 |
|------|------|------|
| WeakMap 元数据追踪 | `GeometryMeta` 记录 vertex/index 切片归属 | 完成 |
| `destroyGeometry()` | 归还切片到 free-list，合并相邻 block | 完成 |
| 池链设计 | `_vertexPools` / `_indexPools` 保留旧池，旧 Geometry 引用不失效 | 完成 |
| `dispose()` | 销毁所有池，释放 GPU 内存 | 完成 |
| `nextPow2()` 溢出保护 | `> 0x40000000` 时返回原值 | 完成 |

### 1B. Renderer 生命周期管理 完成

| 任务 | 说明 | 状态 |
|------|------|------|
| `dispose()` 方法 | 销毁 depthTexture、arena、ring、清理 globalBGs | 完成 |
| 防止 double-dispose | `_disposed` 标志位 | 完成 |

### 1C. 热路径堆分配修复 完成

| 位置 | 修复 | 状态 |
|------|------|------|
| `orderScratch` | `number[]` + `push()` → `Int32Array` + 索引赋值 | 完成 |
| `Set<number>` | 每帧 `new Set()` → 类字段 `_pipelineSet` + `.clear()` | 完成 |

### 附加修复

| 修复 | 说明 |
|------|------|
| `RingBuffer.dispose()` | 新增方法，销毁底层 GPUBuffer |
| `index.ts` 导出 | 补全 `VS_INSTANCED`、`FS_COLOR` shader 变量导出 |

---

## Sprint 2 — 测试覆盖 已完成

### 2A. 缺失模块测试 完成

| 测试文件 | 覆盖内容 | 用例数 |
|----------|----------|--------|
| `test/geometry.test.ts` **新增** | createGeometry (有/无索引)、分配对齐、池扩容、free-list 复用、dispose、池链 | 11 |
| `test/executor.test.ts` **新增** | 单批执行、pipeline 切换、geometry 切换、空批、stats 累加 | 6 |
| `test/math.test.ts` **补充** | rotationQuat (单位/180°X/180°Y/正交性/边界保留)、fromTranslationRotationScale (平移/缩放/组合) | +8 |

### 2B. 现有测试完善 完成

| 测试文件 | 补充内容 |
|----------|----------|
| `test/renderer.test.ts` **重写** | instanceData 上传验证、多实例数据量、custom transforms instanceCount、depth 排序、dispose |
| `test/ringbuffer.test.ts` **补充** | write() 方法、srcOffset/srcLength slice、endFrame 复位、allocations 递增、dispose 双调用 |

### 2C. 测试基础设施 完成

| 任务 | 说明 | 状态 |
|------|------|------|
| 提取 polyfill 到 `test/setup.ts` | GPUBufferUsage / GPUTextureUsage / GPUShaderStage / GPUCullMode 等 | 完成 |
| 修正 polyfill 位掩码 | STORAGE: 0x80, UNIFORM: 0x100, VERTEX: 0x200 | 完成 |
| vitest `setupFiles` 配置 | 各测试文件移除内联 polyfill | 完成 |
| vitest `coverage` 配置 | provider 已在配置中声明；provider package 与 coverage script 尚未纳入当前 package contract | 未完成 |
| `fake-gpu.ts` 增强 | buffer.destroy(), texture.destroy(), pipeline.getBindGroupLayout() | 完成 |

### 测试覆盖统计

```
Test Files:  8 passed (8)
Tests:      66 passed (66)
```

---

## Sprint 3 — 功能完善 已完成

### 3A. 数学库扩展 完成

| 函数 | 用途 | 测试 |
|------|------|------|
| `orthographic()` | 2D/UI 渲染必需 | 完成 NDC 映射、对称范围、边界 |
| `invert()` | 射线拾取、法线变换 | 完成 单位/旋转/平移/缩放/奇异/互逆验证 |
| `translation()` / `scaling()` | TRS 分解操作 | 完成 对角线/零元素 |
| `rotationX()` / `rotationZ()` | 多轴旋转支持 | 完成 θ=0/π/π/2 正交性 |

### 3B. Shader 变体 完成

| 变体 | 用途 | 测试 |
|------|------|------|
| `VS_FLAT` | 无实例颜色，使用 uniform 颜色 | 完成 struct/uniform/无 instances |
| `FS_DEPTH_ONLY` | 深度预通道 | 完成 输出零颜色 |

### 3C. 多顶点 Buffer 支持 已完成

该阶段的早期记录曾暂缓多顶点 Buffer 支持；后续 Phase 7A 已完成 `Geometry.vertexBuffers[]`、多 slot
绑定以及 renderer / executor 路径适配。此处保留为历史决策记录。

### 3D. Pipeline 哈希碰撞防护 完成

| 修改 | 说明 |
|------|------|
| `PipelineCache.getOrCreate()` | 改用 canonical string 作 Map key，消除哈希碰撞 |
| 新增 `test/pipeline-cache.test.ts` | 去重验证、canonical 确定性、fnv1a64 一致性 |
| 新增 `test/shaders.test.ts` | VS_INSTANCED / FS_COLOR / VS_FLAT / FS_DEPTH_ONLY 结构验证 |

### 测试覆盖统计

```
Test Files:  9 passed (9)
Tests:      96 passed (96)
```

---

## Sprint 4 — 打磨 已完成

### 4A. 代码清理 完成

| 项 | 说明 | 状态 |
|----|------|------|
| 移除 `commands.ts` 中未使用的 `CommandEncoderState` | 死代码 | 完成 |
| 移除 `geometry.ts:272` 死变量 `size` | Sprint 1 重写时已清理 | 完成 |
| `nextPow2()` 添加 `> 2^30` 溢出保护 | Sprint 1 已添加 | 完成 |
| `index.ts` 导出 shader 变体 | Sprint 3 已导出 `VS_FLAT` / `FS_DEPTH_ONLY` | 完成 |

### 4B. 包管理完善 完成

| 项 | 说明 | 状态 |
|----|------|------|
| `package.json` 添加 `files` | `["dist/lib", "LICENSE", "CHANGELOG.md", "README.md"]` | 完成 |
| `package.json` 添加 `peerDependencies: @webgpu/types` | `>=0.1.0` (optional) | 完成 |
| `package.json` 添加 `engines: >=18` | Node 版本约束 | 完成 |
| `package.json` 添加 `prepack: npm run build` | 确保 `npm pack` / `npm publish` 前构建 | 完成 |
| ESLint + Prettier | 留待后续迭代（需项目级配置决策） | 未执行 |

### 4C. 文档 完成

| 项 | 说明 | 状态 |
|----|------|------|
| README Phase 状态更新 | Sprint 1-3 完成状态 | 完成 |
| README 已知修复记录 | 按 Sprint 整理修复表 | 完成 |
| README 工具链补充 | coverage provider 配置已存在，但 `@vitest/coverage-v8` 与 coverage script 尚未纳入当前 package contract | 未完成 |

### 历史统计

以下数字记录 Sprint 4 当时的阶段快照，不代表当前仓库规模。当前基线以 README 和最新测试结果为准：
`19` 个测试文件、`230` 个测试，以及当前 package boundary 验证。

---

## Phase 5 — GPU Culling + Indirect Drawing 已完成

### 5A. GPU 视锥剔除 Compute Shader 完成

| 组件 | 说明 | 文件 |
|------|------|------|
| `CS_FRUSTUM_CULL` | WGSL Compute Shader：Gribb-Hartmann 裁剪面提取 + sphere-plane 相交测试 | `src/shaders/culling.ts` |
| `CullingPipeline` | 封装 compute pipeline + buffers（spheres/visibility/uniforms） | `src/core/culling.ts` |
| `CullUniforms` | VP 矩阵 (64B) + sphereCount (4B) + geometryCount (4B) + pad (8B) = 80B uniform | `src/shaders/culling.ts` |

### 5B. Indirect Draw 路径 完成

| 组件 | 说明 | 文件 |
|------|------|------|
| `ExecutionBackend.runIndirect()` | `drawIndexedIndirect` 路径，从 indirect buffer 读取绘制参数 | `src/core/executor.ts` |
| `DrawIndexedIndirectArgs` | 20 字节间接绘制参数结构（WebGPU 规范） | `src/core/executor.ts` |
| `Renderer.submitCulled(vpMatrix)` | 完整 GPU Culling + Indirect Draw 流程 | `src/core/renderer.ts` |

### 5C. 数据流

```
submitCulled(items, vpMatrix)
  │
  ├─ 1. 收集 bounding spheres → N × vec4 (GPUBuffer)
  │
  ├─ 2. Compute Shader 视锥剔除 → visibility buffer (N × u32)
  │     workgroup_size(64), dispatch(ceil(N/64))
  │
  ├─ 3. CPU 回读 visibility → 构建 indirect draw args
  │     每 batch 一个 DrawIndexedIndirectArgs (20B)
  │
  └─ 4. render pass → runIndirect() → drawIndexedIndirect(...)
```

### 5D. Benchmark Demo 完成

| 项 | 说明 |
|----|------|
| `demo/phase5.ts` | 500 对象网格场景，按 C 键切换 CPU/GPU 路径 |
| `phase5.html` | 独立入口页面 |
| 对比指标 | drawCalls / instances / batches / itemsDrawn |

### 测试覆盖统计

```
Test Files: 10 passed (10)
Tests:     105 passed (105)
Build:     vite build (26.6KB) + tsc declarations
```

---

## Phase 6 — GPU-driven Correctness + Benchmark 已完成

### 6A. Correctness First 完成

| 问题 | 修正 |
|------|------|
| `submitCulled()` 的 `instanceCount` 未按 visibility 过滤 | Compute shader `atomicAdd` 直接填充 indirect args |
| `indexCount` 被误用为原子计数器 | 恢复为 CPU 预写 `indexCount` |
| 多 geometry 间接绘制参数混乱 | 每 geometry 独立 indirect args + 独立 draw call |

### 6B. 全 GPU-driven Execution 完成

| 组件 | 变更 |
|------|------|
| `CS_FRUSTUM_CULL` | 当前 compute bind group 共 7 bindings；`DrawArgs.instanceCount` 由 atomic 填充，其余字段由 CPU 预写 |
| `CullingPipeline.cull()` | 当前 API: `cull(vpMatrix, spheres, geometryIds, geometryCount, drawArgsTemplate?, timestamps?)` → 返回 draw args、compaction buffers、`slotBases` 与 `maxSlotBytes` |
| `Renderer.submitCulled()` | 按 geometry 分组 + 直写 instance buffer（跳过 batcher）+ pre-fill draw args + indirect draw |
| CPU readback | **完全移除** — 零 `getMappedRange()` |

数据流：

```
submitCulled(items, vpMatrix)
  │
  ├─ CPU: 按 geometry 分组 + 收集 bounding spheres
  │       直接写入 instance buffer（跳过 batcher）
  │       预写 draw args (indexCount/firstIndex/baseVertex/firstInstance)
  │
  ├─ GPU: Compute Shader 视锥剔除
  │       → 原子递增 drawArgs[geoIdx].instanceCount
  │       CPU 不读取任何结果
  │
  └─ GPU: render pass → drawIndexedIndirect(drawArgsBuffer)
```

### 6C. Benchmark 完成

| Case | Objects | Visible% | Path |
|------|--------:|--------:|------|
| A    |     500 |     100% | Direct |
| B    |     500 |      50% | GPU Culling |
| C    |     500 |      10% | GPU Culling |
| D    |   5,000 |      10% | GPU Culling |
| E    |  50,000 |      10% | GPU Culling |
| F    |  50,000 |       1% | GPU Culling |

测量指标：CPU submit 时间 (median/20) / draw calls / instances / batches / itemsDrawn

```bash
npm run bench   # 浏览器自动打开 benchmark/index.html
```

### 测试覆盖统计

```
Test Files: 10 passed (10)
Tests:     112 passed (112)
Build:     vite build (28.0KB) + tsc declarations
```

### Compaction Correctness Fix 完成

Phase 6 原始实现的 `instanceCount` 正确，但 instance 数据未被 compaction。
vertex shader 用 `instance_index` 直接读 instance buffer，绘制前 N 个而非可见 N 个。

修正：
- `CS_FRUSTUM_CULL` → 当前 compute bind group 共 7 bindings，包含 `compactedIndices` 与 `compactionCounters`
- `CullingPipeline.cull()` → 返回 draw args、compaction buffers、`slotBases` 与 `maxSlotBytes`
- `VS_INSTANCED_COMPACTION` → `instances[compactedIndices[instanceIdx]]`
- `Renderer.submitCulled()` → 创建 2-binding compaction bind group

### Phase 6.5 — Measurement: CPU Submission Crossover 完成

**测试环境**: RTX 3070 Laptop GPU / D3D11 / Edge 154 / Windows 10

**数据** (CPU submit time only, 50 samples + 10 warm-up):

```text
Instances │   Direct │  Batcher │ GPU Culling
──────────┼──────────┼──────────┼────────────
      500 │  0.30 ms │  0.10 ms │    0.20 ms
     1000 │  0.50 ms │  0.20 ms │    0.20 ms
     5000 │ 16.30 ms │  0.60 ms │    0.40 ms
    50000 │163.60 ms │ 10.20 ms │    3.90 ms
   100000 │      —   │      —   │    7.60 ms
   500000 │      —   │      —   │   32.90 ms
```

**结论** (严谨表述):

1. **CPU Submission Crossover**: GPU Culling 的 CPU submit time 在 1K–5K instances 之间从"不优于 Batcher"变为"优于 Batcher"。50K 时 GPU Culling CPU submit ≈ 38% Batcher。

2. **GPU Culling CPU 成本对 visibility ratio 不敏感**:
   - 50K / 100% → 3.9 ms
   - 50K / 50%  → 3.9 ms
   - 50K / 1%   → 3.8 ms
   - 50K / 0%   → 3.6 ms

   符合当前 O(N) candidate culling 模型（GPU 对全部 N 个 candidate 执行 culling，不因可见数 M 减少而降低 CPU/upload 成本）。

3. **500K 的瓶颈已从 rasterization 转移到 candidate preparation/upload**。下一阶段关注方向：candidate preparation / buffer upload 优化，而非 culling shader 本身。

4. **注意**: 以上是 CPU submit time；该组历史记录没有把 GPU execution time 纳入同一结论。
   不能将 "CPU Submission Crossover" 等同于 "true break-even"。

### 成本模型

```text
                    Direct       Batcher       GPU Culling
──────────────────────────────────────────────────────────
CPU preparation      O(N)         O(N)           O(N)
Draw submission      O(N)         O(B)           O(G)
GPU culling           —             —             O(N)
Rasterization        O(N)         O(N)           O(M)

N = candidate instances
B = batches
G = indirect draw groups
M = visible instances
```

GPU Culling 的价值: 将 GPU rasterization 从 O(N) 变为 O(M)。当 M << N 时收益超过 O(N) culling + upload 开销。

### Phase 7A — API Stabilization 完成

Phase 7A 聚焦 Renderer / GeometryArena 的公开 API 改进：

1. **`GeometryArena.stats()`**: 暴露池化显存使用统计（pool 数量、容量、已用、碎片、活跃几何体数）。
2. **`Renderer.create(desc)`**: 工厂方法，单对象描述符创建 Renderer，简化调用。
3. **`Renderer.geometryArena`**: getter 暴露内部 GeometryArena 实例（供 `stats()` / `createGeometry` / `destroyGeometry`）。
4. **多顶点 buffer API 形状**: `Geometry.vertexBuffers: BufferSlice[]` 保留多 slot 形状；当前 `GeometryArena` 明确只支持一个 vertex-step layout，多 slot 会注册期拒绝。

### Phase 7B — GPU Timestamp Query 完成

新增 `TimestampQuery` 工具类，支持 GPU 时间戳查询：

```typescript
const tq = new TimestampQuery(device, 2);
const renderPass = encoder.beginRenderPass({
  colorAttachments,
  timestampWrites: tq.timestampWrites(0, 1),
});
// ... draw calls ...
renderPass.end();
tq.resolve(encoder);
device.queue.submit([encoder.finish()]);
const timestamps = await tq.readback(device);  // GPU 纳秒时间戳
const gpuMs = (timestamps[1] - timestamps[0]) / 1e6;
```

集成点：
- `submitDirect(items, timestamps?)`: 可选 GPU 时间戳。
- `submitCulled(items, vpMatrix, timestamps?)`: 可选 GPU 时间戳（当前只覆盖 render pass，不包含 culling compute）。
- `CullingPipeline.cull(vpMatrix, spheres, geometryIds, geometryCount, drawArgsTemplate?, timestamps?)`: 当前 API 同时支持 draw-args 模板与可选 compute pass 时间戳。
- Benchmark: Direct / GPU Culling 路径自动采集 render-pass GPU 时间（ns → ms），输出 GPU render ms (median) + p95。

时间戳通过标准 WebGPU pass descriptor 的 `timestampWrites` 注入；不再调用非标准的 pass 方法。

## Benchmark 规范

任何性能声明必须基于实测数据。建议的 benchmark 框架：

```text
Hardware:
CPU:
GPU:
Browser:
WebGPU implementation:

Test scene:
Objects:
Triangles:
Pipelines:
Textures:

Results:
                    Three.js       hpg
------------------------------------------------
CPU frame time
GPU frame time
Draw calls
Pipeline switches
Buffer upload
Allocated bytes
```

在没有实测数据之前，不应在文档中出现具体数字。

---

## Phase 11 — 渲染链路正确性修复 已完成

真实模型（benchmark/assets 的 Khronos 语料）加载后不显示/显示错误的根因与修复：

| # | 根因 | 修复 |
|---|------|------|
| 1 | `importGltfAsset()` 的 bounds 用未变换的局部顶点坐标 + viewer 写死 near/far `0.1/100` → Duck/Fox 整模被远平面裁掉（放射到整个屏幕空白） | bounds 改为世界空间 AABB（含节点变换/全局缩放）；near/far 随模型半径自适应；`ImportOptions.scale` 连平移一起缩 |
| 2 | GPU Culled 路径结构性不可用：① `registerPipeline` 追加的 group=1 布局（1 entry）与 `submitCulled` 的 compaction bind group（2 entries）不兼容 → 校验失败 ② 按 `pipeline.id` 分组 → 共享管线时只画第一个 mesh ③ 顶点着色器未走 compaction ④ 视锥平面提取把列当行用 → 视锥内物体全被剔除 ⑤ compaction 映射方向/索引空间不一致（VS 按 slot 读、shader 按全局 index 写） | ① `PipelineDesc.compaction` + 独立 group=1 布局 ② 按 (geometry, pipeline) 分组 ③ ④ 重写平面提取并新增 CPU 参考 `extractFrustumPlanes()` ⑤ slot 区按 geometry 划分 + `geometryBases` 表 + 绑定 2 个 dynamic offset |
| 3 | ring 扩容后 `instanceBindGroup` 悬空（单帧实例 >1 MiB 即整帧无输出）；实例区只做 16 字节对齐（动态偏移要求 256） | `ensureInstanceBindGroup()` 检测 buffer 身份重建；`RingBuffer.alloc(bytes, RING_ALIGN)` |
| 4 | 材质链路断开（`sceneToRenderItems` 丢弃 materials，实例颜色恒为白） | `instanceData` 写入 `baseColorFactor`（含长度不足保护，避免 NaN） |
| 5 | 真实模型解析：非对齐 UINT 索引崩溃、归一化整数属性乱码、外部 URI/Draco 静默出错、非三角形拓扑产生错误几何 | accessor 读取重写（componentType / byteStride / 对齐 / 复制回退）+ 明确报错 + 跳过并告警 |
| 6 | 同一 mesh 被多个节点引用时几何体重复上传 → 永不合批 | 导入器按 primitive 缓存 `Geometry` |

测试：`test/render-chain.test.ts`（17）、`test/gltf-robustness.test.ts`（10，含真实语料）、executor +3、culling 补充；总计 167 通过。

---

## Phase 12 — 渲染器审计（Audit）已完成

用 `test/tmp-*.test.ts` 探针逐项验证渲染链路，发现并修复：

| # | 发现 | 修复 |
|---|------|------|
| 1 | 帧内混用不同 `bytesPerInstance` 的管线时，实例数据按「帧内最大跨步」统一排布 → 跨步较小的管线读到错位数据 | `Batcher.collect()` 改为**按批所属管线**的跨步落位（批起始仍 256 对齐） |
| 2 | 相机深度排序用几何体局部中心，未应用实例矩阵 → 多实例场景排序无效 | `resolveDepth01()` 用实例矩阵把局部包围盒中心变换到世界空间，多实例取最近者 |
| 3 | `submitCulled()` 逐帧创建 compaction bind group / buffer | 缓存 compaction bind group，跨帧复用（`render-chain.test.ts` 覆盖） |
| 4 | **GeometryArena 跨池复用写错池**：free-list 只记 offset，不记所属池 → 复用旧池空闲块时写进最新池的同名偏移，覆盖存活几何体 | free-list 记录 `pool`，分配返回 `{ pool, offset }`，写入/切片落在该块真正的池；合并限定同池。回归测试见 `geometry.test.ts` |

审计探针（`test/tmp-audit*.test.ts`）为临时脚手架 —— 已于 **Phase 15.5 Finalize** 删除（零断言、只有 `console.log`）。

---

## Phase 13 — 真实浏览器验证（Browser-verified）已完成

「加载模型不显示」的最后两个根因。fake GPU 测试全绿但页面全黑，因为两者都只在真实 WebGPU
下才会暴露（前者是绑定校验错误，后者是队列顺序）。

### 验证方法

用 CDP 驱动本机 Chrome（headless=new + WebGPU）打开 `demo/glb-viewer.html?asset=<path>`，
收集 `pushErrorScope('validation')` 错误 + canvas 像素统计（litPixels / avgLuma）。
这补上了此前「只能靠肉眼在浏览器里看」的空白 —— 不再依赖 fake GPU 的乐观假设。

### 修复

| # | 根因 | 症状 | 修复 |
|---|------|------|------|
| 1 | 实例 bind group 绑定**整块** ring buffer（size = capacity），而 WebGPU 要求 `offset + dynamicOffset + bindingSize ≤ bufferSize` | 帧内第 2 个批的 dynamic offset(256) 越界 → render pass 失效 → 整帧全黑。单 mesh 模型正常，多 mesh 模型必黑 | 绑定尺寸显式声明为本帧最大批的实例字节数；ring / compactedIndices 预留等量余量 |
| 2 | draw args 的 CPU 预写在 `cull()`（已提交 compute pass）**之后**入队 | `writeBuffer` 覆盖原子填充的 `instanceCount` → 0 实例被绘制。**无校验错误**，静默全黑 | CPU 字段作为模板随 `cull(..., drawArgsTemplate)` 在 dispatch 前写入 |

### 防回归

- `test/fake-gpu.ts` 现在按 Dawn 规则校验 dynamic offset 范围（越界写入 `gpuErrors`），
  并记录按入队顺序的 GPU 操作日志（`ops`）
- `test/render-chain.test.ts` 新增：多批/多 item/多 geometry 的绑定范围断言；
  draw args 预写必须早于 dispatch 的断言
- 绑定尺寸 = 本帧最大批也被断言（既覆盖数据、又保证所有 offset 在范围内）

### 浏览器实测（Chrome 153 / Dawn / Windows）

```text
模型                        mesh   Direct(lit/avgLuma)   GPU Culled
──────────────────────────────────────────────────────────────────
DamagedHelmet                  1      1476 / 133.1        1476 / 133.1
WaterBottle                    1      1308 / 106.5        1308 / 106.5
CesiumMilkTruck                5      1991 / 150.8        1991 / 150.8
BrainStem                     59       434 /  58.5         434 /  58.5
Lantern                        3       531 /  70.0         531 /  70.0
```

两条路径亮度网格近似一致。

---

## 风险登记

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| GeometryArena free-list 实现复杂 | 可能引入内存管理 bug | 先实现简单版本，用测试覆盖边界情况 |
| 池链设计导致引用追踪开销 | 可能影响热路径性能 | 用 WeakRef 或引用计数，避免遍历 |
| 自动合批条件不断扩展 | Batcher 可能吞噬整个 Renderer | 明确合批边界，Direct Draw 路径必须保持简洁 |
| Phase 3 GPU Culling 性能不确定 | 可能无法满足 500 对象零负优化 | 先在 demo 中验证，通过后再进 Core |

---

## 执行顺序

```
Sprint 0 (阻塞项)  ──→ Demo + 修复 + depth resize + exports
Sprint 1 (架构正确性) ──→ Geometry 生命周期 + Renderer dispose + 零分配
Sprint 2 (测试覆盖)  ──→ geometry.test + executor.test + polyfill 统一
Sprint 3 (功能完善)  ──→ 数学库 + shader 变体 + 多顶点 buffer + 哈希防护
Sprint 4 (打磨)     ──→ 代码清理 + 包管理 + 文档 + benchmark 框架
Phase 5  (GPU Culling) ──→ CS_FRUSTUM_CULL + CullingPipeline + submitCulled + runIndirect
Phase 6  (Full GPU-driven) ──→ 7-binding compute + compaction mapping + VS_INSTANCED_COMPACTION
Phase 6.5 (Measurement) ──→ 3-path benchmark + CPU Submission Crossover 分析
Phase 7A (API)       ──→ GeometryArena.stats() + Renderer.create() + 多顶点 buffer
Phase 7B (Timestamp) ──→ GPU timestamp query + benchmark GPU timing
Phase 8A (Boundary)  ──→ Vite library mode + source maps + tarball boundary
Phase 8B (Docs)      ──→ README Quick Start + API Reference + Architecture
Phase 8C (Release)   ──→ LICENSE (MIT) + CHANGELOG.md + CI gate
Phase 9  (Release)   ──→ v0.1.0 release record; current tag workflow creates GitHub Release
Phase 10A (glTF)     ──→ GLB parser (POSITION/NORMAL/UV/TANGENT + PBR + hierarchy)
Phase 10B (Importer) ──→ glTF → hpg adapter (interleave + UV flip + sceneToRenderItems)
Phase 10C (Tests)    ──→ 6 glTF tests, 122/122 total
Phase 10D (Demo)     ──→ Browser GLB viewer (orbit camera, Direct/Culled toggle, file input)
Phase 10E (Benchmark)──→ Node parse benchmark 完成 | Browser full-pipeline benchmark 完成
Phase 11  (Validate) ──→ Real Blender GLB validation + architecture freeze
Phase 11* (Correctness)─→ 世界空间 bounds + 自适应近远平面 + 正确视锥平面/compaction + ring 生命周期 + 材质 + 解析健壮性
```

---

## Phase 10E — Real-Workload Benchmark 完成

### 10E-1: Node Parse Benchmark 完成

合成 GLB 模型（Blender-like 特征：hierarchy, varying mesh sizes, multiple materials）。

```text
Model                         GLB Size   Meshes     Verts   Indices   Mats   Parse(ms)   p95(ms)
───────────────────────────────────────────────────────────────────────────────────────────────────
XS   (10 meshes, ~10K verts)    279 KB      10    10,837     9,486     3       0.28      0.49
S    (50 meshes, ~50K verts)    1.3 MB      50    49,625    45,543     5       1.02      1.47
M    (200 meshes, ~500K verts)  11.9 MB    200   474,142   483,114    10       8.36     13.01
L    (500 meshes, ~2M verts)    49.3 MB    500  1,995,789 1,731,243   15      33.86     37.01
XL   (1000 meshes, ~5M verts)  122.9 MB   1000  4,964,886 4,574,274   20      75.86     84.82
```

**结论**: Parse 线性 ~0.62 ms/MB。Material count 对 parse 无显著影响。50 MB GLB ≈ 34ms。

### 10E-2: Browser Full-Pipeline Benchmark 完成

`benchmark/glb-bench.html` — 在 WebGPU 浏览器中运行。

```text
Asset Statistics
Model        GLB Size  Meshes  Verts      Indices   Mats  Items  Parse    Import
Small (50)   1283 KB   50      49,625     45,543    5     50     5.3 ms   2.6 ms
Medium (200) 12176 KB  200     474,142    483,114   10    200    25.4 ms  11.5 ms
Large (500)  50456 KB  500     1,995,789  1,731,243 15    500    96.5 ms  54.1 ms

CPU Submit Time (median)
Model        Direct  GPU Culled  Ratio
Small        0.10 ms 0.10 ms     1.00x
Medium       0.10 ms 0.10 ms     1.00x
Large        0.30 ms 0.10 ms     3.00x

Time Breakdown
Small  — Total CPU:  8.0 ms  Parse 66% · Import 32% · Submit 1%
Medium — Total CPU: 37.0 ms  Parse 69% · Import 31% · Submit 0%
Large  — Total CPU: 150.9 ms Parse 64% · Import 36% · Submit 0%
```

**10E 结论**:
- 该次观察中 CPU submit 未显示瓶颈；该结论只适用于记录的硬件、浏览器和场景
- Parse 和 Import 在该组样本中呈线性趋势
- GPU execution 的测量能力与结果应以当前 benchmark 运行记录为准
- 性能结论不构成跨硬件、跨 workload 的架构承诺

---

## Phase 14 — Real-Asset Compatibility（真实 Blender 资产兼容）已完成

**目标**: 确认 hpg 能不能稳定吃真实导出的 GLB，而不是只吃自造的 synthetic GLB。

**原则**: 先测量、后实现。审计只报告不修代码；修的是「静默错误」，feature 缺失按命中量排序再做。

### 14A — 真实资产语料 完成

`benchmark/assets/models` 下 23 个真实 GLB，覆盖 7 类导出器：

```text
light   4  （几千～几万 tri，单 mesh）
medium  4  （DamagedHelmet / Lantern / WaterBottle，单～3 mesh）
heavy   5  （BrainStem 59 prim / CesiumMan / Fox，含蒙皮 + 动画）
feature 11 （ClearCoat / Transmission / AlphaBlend / MultiUV / Morph / 贴图编码 …）
```

合计 ~171 primitives、~1.6M indices、148 materials、74 textures、4 个带动画、3 个带蒙皮。

### 14B — Real Feature Audit 完成

`npm run audit`（`benchmark/asset-compat.ts` + `asset-compat-cli.ts`）：独立读 glTF JSON，
逐模型输出 feature 使用表 + 支持矩阵（supported / ignored / broken + 症状），并与 `parseGltf` 结果对照。

实测结论（详见 CHANGELOG）：

| 状态 | feature | 命中 |
|---|---|---|
| **broken**（静默错误数据） | 无 —— 无 sparse accessor、无压缩网格、23/23 解析成功 | 0 |
| **wrong**（可见错误输出） | `baseColorTexture`、`alphaMode=MASK/BLEND`、蒙皮（`JOINTS_0`/`WEIGHTS_0`） | 22 / 6 / 3 模型 |
| **lossy** | `TANGENT`（导入时丢弃）、`TEXCOORD_1`、morph、动画、`KHR_materials_*` 扩展 | 8 / 2 / 1 / 4 / 8 模型 |

第一个真实阻塞点因此明确：**材质/贴图路径**（22/23 模型有 `baseColorTexture`），其次是蒙皮。

### 14C — 诚实降级：未实现 feature 不得静默 完成

真实资产的问题不是「加载失败」，而是「加载成功但结果不对」。因此先把所有未实现 feature
变成结构化信息：`GltfAsset.warnings`（按 feature 去重）+ console 提示（全局去重一次）+
`demo/glb-viewer` 的警告面板。

**仍未实现（下一个真阻塞点）**: `baseColorTexture` / PBR 贴图采样、`alphaMode` MASK/BLEND、蒙皮与动画。

### 14D — Real Assets as Regression Corpus 完成

`test/asset-corpus.test.ts`：23 个真实模型的 golden 几何规模 + 全链路
（parse → import → `submit` / `submitCulled`）+ 此前只在真实 Chrome 暴露的两类 GPU-only bug 不回归。

`test/gltf-compat-warnings.test.ts`：提示文案台账 + 「审计发现的 feature 必须出现在 warnings」。

### 14E — Browser Parity 完成（承接 Phase 13）

Direct 与 GPU Culled 亮度网格近似一致（Chrome 153 / Dawn / Windows）已由 Phase 13 建立；
14D 在 Node + fake GPU 侧复现同类不变量（无校验错误、indirect draw 数、预写顺序）。

### 下一步（按证据排序）

1. ~~**Material/Texture 路径**（22/23 模型）~~ → **Phase 15A 已完成**（`baseColorTexture` + sRGB + sampler +
   per-material bind group group=2 + 可注入 ImageDecoder；`alphaMode=MASK` 已实现）。BLEND 仍待做。
2. ~~`TANGENT` 保留到 canonical layout~~ → **Phase 15B 已完成**（stride 48，缺失填默认值；不实现 normal mapping）。
3. **蒙皮 + 动画**（3 模型）：JOINTS/WEIGHTS 上传 + skin matrix palette；这一步才会真正改变 runtime 的数据布局，放最后。

---

## Phase 15 — Material / Texture 最小闭环 + TANGENT 保留 已完成

目标很窄：**不做 Material System**，只把实测出来的第一个用户可见错误
（`baseColorTexture` 被 22/23 模型使用而 runtime 不采样）修掉。

```
glTF baseColorTexture
  → image decode（ImageDecoder，可注入）
  → GPUTexture（rgba8unorm-srgb）
  → sampler
  → material bind group（固定 group=2）
  → RenderItem.bindGroup → fragment textureSample()
```

| 子项 | 内容 | 状态 |
|---|---|---|
| 15A-1 | `parseGltf` 解析 images/textures/samplers + `AssetMaterial.baseColorTexture`/`alphaMode`；`MaterialStore`（sRGB、白色 fallback、解码去重）；`registerPipeline` 固定 group 槽位（0 global / 1 instance / 2 material）；`FS_MATERIAL` + `VS_INSTANCED_MATERIAL(_COMPACTION)`；`sceneToRenderItems(..., materials?)`；OPAQUE | 完成 |
| 15A-2 | `alphaMode=MASK`（alphaCutoff + discard） | 完成 |
| 15B | TANGENT 保留进 canonical layout（stride 48，缺失填 `(1,0,0,1)`），**不实现 normal mapping** | 完成 |
| 15C | 回归：`test/material-path.test.ts` + `test/execution-grouping.test.ts` + 审计台账更新 | 完成 |
| 15D | 执行分组 regression lock（同 geometry/pipeline、不同 material ⇒ 独立执行组，Direct + Culled） | 完成 |
| 15E | 真实 Chrome 逐模型验证（23 GLB × Direct/GPU Culled 贴图一致性） | 完成 |
| 15F | `alphaMode=BLEND` 需求判断（只评估，不实现） | 完成（判断：暂不实现） |
| 15A-3 | （即 15F 的结论）BLEND 实现：透明排序 + 混合状态 + depth 策略 | 等真实需求 |

### 验收不变量（已固化进测试）

- 材质路径存在时实例颜色为白（baseColorFactor 只在 material uniform 中，不重复相乘）
- 贴图格式恒为 `rgba8unorm-srgb`（glTF base color 是 color data）
- 同一 image 只解码一次；解码失败退回白色并记录 `stats.skipped`
- `RenderItem.bindGroup` 被绑定到 **group 2**（`registerPipeline([global, material]) → [global, instance, material]`）
- canonical 顶点布局 stride 48 且 4 个 location 全声明；tangent 缺失时填默认值

### 15E — 真实 Chrome 验证结果 完成

`npm run verify:browser`（`benchmark/browser-material-check.mjs`，无第三方依赖：
Node 内置 WebSocket + CDP 驱动本机 Chrome `headless=new --enable-unsafe-webgpu`，
Vite dev server 作为子进程）。逐模型加载 `demo/glb-viewer.html?harness=1&mode=…&asset=…`，
采集材质统计 + WebGPU validation error + canvas 像素签名（litPixels / avgLuma / 8×8 亮度网格）。

Chrome 153 / Dawn / Windows：

```text
模型                              items  mats  tex  withMat  litPixels  avgLuma  err  Direct/Culled
feature/AlphaBlendModeTest           9     6    6     9     392794    23.42     0   gridΔ0  litΔ0
feature/ClearCoatTest               27    19    1    27     360347    14.00     0   gridΔ0  litΔ0
feature/TransmissionTest            22    14    7    22     393131    29.01     0   gridΔ0  litΔ0
feature/TextureEncodingTest         14    14    5    14     387568    18.61     0   gridΔ0  litΔ0
heavy/BrainStem                     59    59    0    59     395418    17.91     0   gridΔ0.1 litΔ0
light/BoxTextured                    1     1    1     1     395920    37.54     0   gridΔ0  litΔ0
medium/DamagedHelmet                 1     1    1     1     380148    19.18     0   gridΔ0  litΔ0
…（共 23/23）
```

结论：**23/23 模型无 validation error、无贴图跳过、Direct 与 GPU Culled 亮度网格近似一致**。
其中 `ClearCoatTest`（19 材质 / 1 贴图）与 `TextureEncodingTest`（14 材质 / 5 贴图）验证了
**同一 image 只解码一次**与**多材质不串图**；`BrainStem`（59 材质 / 0 贴图）验证了
白色 fallback 路径。

### 15F — `alphaMode=BLEND` 需求判断 完成（结论：暂不实现）

实测只有 2 个 **Khronos feature-test** 模型命中 BLEND，且各自只有 1 个材质：

| 模型 | 材质数 | BLEND 材质 |
|---|---:|---|
| `feature/AlphaBlendModeTest.glb` | 6 | `MatBlend`（textured, doubleSided） |
| `feature/ClearCoatTest.glb` | 19 | `Partial_Coating`（textured） |

没有生产资产命中。而 BLEND 的代价不是 `discard`，而是
`transparent classification → sorting → depthWrite/depthTest → blend state → 可能的独立 render phase`，
属于 render execution semantics 的改动。因此**保持 warning、不实现**，
并把边界固化为测试（`test/asset-corpus.test.ts` 的 Phase 15F 用例）——
一旦出现第 3 个（尤其非 `feature/` 的生产资产），测试失败以强制一次显式决策。

---

## Phase 15.5 — 项目状态审计：Execution Boundary 到底还缺什么？

> 触发原因：roadmap 判断出现漂移 —— 讨论从「hpg 的 execution runtime 是否成立」
> 滑向「hpg 对真实 glTF 的兼容性够不够」。本节把边界重新锁回最初基线，
> 并用已有证据回答一个问题：**执行边界还缺什么是「已经被证据证明」的？**

### 重新锁定的基线

hpg 是 **Render Work → GPU Execution** 的 runtime，**不是 rendering engine**。

```text
Scene / ECS / Asset / Material      ← 上层负责（不在 hpg 边界内）
          ↓
      RenderItem[]                   ← hpg 的唯一输入
          ↓
   hpg: sort → batch → bind → execute
          ↓
        WebGPU
```

GLB / Material / Texture 的作用是**给 hpg 喂越来越真实的 Render Work**，
用来验证 execution runtime 与 GPU correctness —— 不是把 hpg 做成 glTF engine。

判据因此不是「真实 GLB 用了什么 glTF feature」，而是：

> **这个能力是不是 execution boundary 真正缺失的？**

并且 **`问号 ≠ TODO`**：未被证据证明的未知项（skin / animation / transparency / 生产负载）
保持未知，不因「看起来应该有」而提前实现。

### 证据清单（全部可复现）

| 证据 | 产物 | 状态 |
|---|---|---|
| RenderItem execution / batching / sorting / GeometryArena / indirect / dynamic offset | `renderer` `geometry` `batcher` `executor` 测试 | 完成 |
| GPU culling + compaction + indirect（零 readback） | `test/culling.test.ts` | 完成 |
| 真实 GPU-only bug 防回归（绑定尺寸 / offset 越界、draw args 写入顺序） | `test/render-chain.test.ts` + `test/fake-gpu.ts`（Dawn 规则校验 + ops 顺序） | 完成 |
| 真实 GLB → RenderItems（23 语料，几何 golden） | `test/asset-corpus.test.ts` | 完成 |
| 真实 GLB 兼容性台账（feature → supported/ignored/wrong） | `npm run audit` | 完成 |
| 材质/贴图路径 + 执行分组隔离 | `test/material-path.test.ts`、`test/execution-grouping.test.ts` | 完成 |
| 真实 Chrome / WebGPU 端到端（Direct vs Culled 亮度网格近似一致） | `npm run verify:browser` | 完成 |
| 执行边界契约（global binding 切片 / 实例记录布局） | `test/pipeline-descriptor.test.ts` | 完成 本轮补齐 |

### 结论：execution boundary 没有「已被证据证明」的功能性缺失

23/23 真实资产、当前 hardening 279 个回归、Direct/Culled 亮度网格近似一致、零 validation error ——
**目前没有任何证据**说明主链（RenderItem → submission → batching → indirect → culling）
缺能力或需要重新设计。

唯一的证据链空白是「长时运行 / 生产负载」这类**未知**。按 `问号 ≠ TODO`，
它们不是 roadmap 项，而是在真实 workload 出现时才需要测量的东西。

### 本轮真正的产出：4 处「声明了但被静默忽略」的边界契约

不是新功能，而是执行边界上已经存在、却名不符实的契约
（同一类问题：**不支持必须明确报错，而不是悄悄渲染错**）：

| # | 契约 | 原行为 | 现行为 |
|---|---|---|---|
| 1 | `GlobalBinding.byteOffset` / `byteLength` | 被忽略，永远绑整块 buffer（同一 buffer 的多个 uniform 切片会读到同一份数据） | 真正进入 `createBindGroup` |
| 2 | 管线缓存键的 `globalBindings` | 只含 `binding:buffer.size` → 同一 buffer 的不同切片塌成同一管线 / 同一 bind group | 键含 offset / size |
| 3 | `PipelineDesc.modelMatrixOffset` | 被忽略，mat4 永远写在记录开头 → 着色器按声明偏移读取会拿到错位数据 | `submit` / `submitDirect` / `submitCulled` 三条写入路径都按偏移落位，预留区显式清零 |
| 4 | `sceneToRenderItems(scene, pipeline, globalBindings?)` | 第 3 个参数从未被使用（静默无效） | 移除；`materials` 上移为第 3 个参数 |

`modelMatrixOffset` 的非法值现在在**注册期**报错（16 字节对齐、记录装得下），
而不是让着色器读到错位矩阵。

### 明确不做（等证据）

```text
Skin / Animation      ← 3/23 模型命中，但会改变 runtime 数据模型；等真实 workload 推动
alphaMode=BLEND       ← 2 个 Khronos feature-test 模型、无生产命中（见 15F）
其他 PBR 贴图         ← normal / mr / occlusion / emissive，保持 warning
SceneGraph / Material System / RenderGraph / AssetManager
                      ← 不在 hpg 边界内，不因「看起来应该有」而加入
```

**Phase 16 不由本轮指定。** 下一步应当由新的真实 workload / 测量证据决定，
而不是由「还剩哪些 glTF feature 没支持」决定。

### 架构冻结原则（仍然有效）

不提前加入 IBL / Clearcoat / Transmission / Sheen / Anisotropy；GPU timestamp profiling 仅作为
instrumentation capability，不作为功能开发前置条件。

---

## Phase 15.5 Finalize — Baseline Freeze 已完成

> 触发原因：主链已被证据证明成立之后，**不再为「让 roadmap 看起来完整」而继续开发**。
> 本轮不加 feature、不做 integration spike，只把当前状态固化成一个可长期维护的 checkpoint。
> 冻结版本：**`0.2.0`**（见 CHANGELOG）。

### 1. 冻结的验证结果（全部可复现）

| Gate | 命令 | 结果 |
| --- | --- | --- |
| 类型检查 | `npm run typecheck` | 0 error |
| 全量回归 | `npm test` | 19 files / 230 passed（冻结时历史记录；当前 hardening 为 279） |
| 真实资产审计 | `npm run audit` | parse ok 23/23、BROKEN（静默错误数据）= 0 |
| 真实 Chrome + WebGPU | `npm run verify:browser` | 23/23：validation error = 0、贴图跳过 = 0、Direct/Culled 亮度网格近似一致 |
| 库构建 | `npm run build` | `dist/lib/index.js`、`.d.ts` 与 sourcemap 产出；大小是一次工具链快照 |
| Demo 构建 | `npm run build:demo` | 多页面入口全部产出（index / phase5 / glb-viewer / benchmark / glb-bench） |

上表是冻结时的历史验证记录。当前 hardening 重新执行了 279 个 Node 测试和 23/23 Chrome gate；
GitHub Actions 自动执行可复现的 Node、资产、构建、打包和消费者检查，release/publish 依赖 browser gate。
Release workflow 创建 GitHub Release，不执行 npm registry publish。

```text
Execution chain: verified
Real GLB corpus: 23/23
Browser validation: 23/23
Direct/Culled parity: verified
GPU validation errors: 0
Known unsupported features: explicit warnings
Known contract violations: 0
```

**测试数 236 → 230 的说明**：删除了 3 个只有 `console.log`、零断言的临时审计探针
（`test/tmp-audit.test.ts` / `tmp-audit2.test.ts` / `tmp-dbg.test.ts`，Phase 12 脚手架，共 6 个伪用例）。
它们观察的不变量已由 `test/render-chain.test.ts` / `test/geometry.test.ts` 的断言覆盖 ——
这是**清理伪用例**，不是减少测试覆盖。

### 2. UNKNOWN 正式登记（不是 TODO）

以下项目**没有**被证据证明是 execution boundary 的缺失，因此**没有 Phase 编号**，也不进入 roadmap：

| UNKNOWN | 为什么现在不做 |
| --- | --- |
| Skin / Animation | 3/23 命中，但会改变 runtime 数据模型；等真实 workload 推动 |
| `alphaMode=BLEND` | 2 个 Khronos feature-test 模型、无生产命中（见 15F）；牵动 render execution semantics |
| 其他 PBR 贴图 | normal / metallic-roughness / occlusion / emissive —— 保持显式 warning |
| 长时运行 / resource pressure | 无测量数据 |
| 生产规模 workload | 无测量数据 |
| 目标硬件 GPU performance baseline | 只有单机单点数据（RTX 3070 Laptop / D3D11 / Edge154） |
| 真实项目接入验证 | 无外部消费者 |

### 3. Public API 检查（结论：不改）

| 检查项 | 结论 |
| --- | --- |
| 为测试暴露的内部 API | 未发现。`uniformBindGroupLayout` 被 demo / benchmark 使用；`renderer.instanceBuffer` 已在注释与 README 中标注「仅调试 / 统计，不要跨帧持有」 |
| 参数命名一致性 | 一致（`vsCode` / `fsCode`、`vertexLayouts`、`bindGroupLayouts`、`bytesPerInstance`、`modelMatrixOffset`） |
| 默认行为是否明确 | 明确（`bytesPerInstance` 默认 80 且注释要求等于 WGSL `sizeof(InstanceData)`；`depthFormat` 默认 `depth24plus`；不传 `opts.camera` 时保持提交顺序） |
| unsupported 行为是否明确 | 明确 —— 未实现 feature 走 `asset.warnings`；非法布局 / 深度格式不符在 `registerPipeline()` 阶段抛错 |
| 生命周期 / ownership 歧义 | 无。README「Ownership」表 + `dispose()` / `destroyGeometry()` 双调用幂等 |
| README 与实际行为一致性 | 当前 README 已记录 package contract、CI / Release 边界和已知 consumer 限制 |

唯一被记录但**保留**的设计重复：`new Renderer(device, context, format, opts?)` 与
`Renderer.create(desc)` 两条构造入口并存。不改 —— 改动是纯 breaking change，
且文档只承诺 `Renderer.create`。

### 4. 下一份证据从哪里来

benchmark 与 23 模型语料**保留但不主动扩大**。以后只有三类事情能重新打开开发：

```text
真实 workload → 发现 correctness 问题       → 修复
真实 workload → 发现 performance bottleneck → measurement → optimization
真实 workload → 发现 Render Work 无法表达    → architecture discussion
```

> **下一次开发动作由下一份 workload evidence 触发，不由 roadmap 的「完整性」触发。**

### 5. Clean consumer verification（package boundary）已完成

在项目外的全新空目录里做真实消费者验证：`npm pack` → 安装 tarball → `import` → `tsc`。

| 检查 | 结果 |
| --- | --- |
| tarball 内容 | 24 files；`src` `test` `demo` `benchmark` 零泄漏；具体大小与 hash 由 CI artifact 生成，不作为文档 contract |
| 裸 Node ESM `import`（**零 WebGPU 全局对象**） | OK |
| import 阶段读取的 WebGPU 全局对象 | **0 个**（修复前 2 个） |
| runtime 路径（Node + 最小 device 桩） | 7/7：43 exports、GLB `parseGltf`/`flattenScene`、frustum、`GeometryArena` |
| tsc：bundler + `types: ["@webgpu/types"]`，skipLibCheck **false** | **0 error** |
| tsc：bundler + `types: []`，skipLibCheck false | 80 error，其中 **67 个来自包内 `.d.ts`**（TS2304） |
| tsc：bundler + `types: []`，skipLibCheck true | 13 error（只剩消费方自己的）；包内类型静默退化为 `any` |
| tsc：node16 / nodenext | 24 error（17 个来自包内 `.d.ts`，TS2834）→ **F3 KNOWN** |

本轮由该验证发现并修复的问题：

**F1（已修）import 期崩溃。** `src/core/texture.ts` 在模块顶层求值
`GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST`，导致任何非 WebGPU 环境
（Node / jsdom / SSR / 构建期资产管线）只要 `import` 包入口就 `ReferenceError`——
即使只用到 `parseGltf`；也与 `package.json` 的 `sideEffects: false` 矛盾。
改为调用期惰性求值（`textureUsage()`）。
> `GeometryArena.createGeometry()` 等**调用期**读取 `GPUBufferUsage` 属预期行为 ——
> 那些调用本来就需要 WebGPU 设备；问题只在「仅仅是 import 就抛错」。

**F2（已文档化）声明文件不自带 `@webgpu/types`。** `@webgpu/types` 是 *optional* peer，
npm 不会自动安装；`tsc` 在 declaration emit 时丢弃了 `/// <reference types>` 指令。
消费方必须自行 `npm i -D @webgpu/types` 并在 tsconfig `types` 中声明（README 已补，
含「`skipLibCheck: true` 会静默把类型退化成 `any`」的警告）。不改 package.json：
把它设成必装 peer 对纯 JS 消费方是多余负担。

**F3（KNOWN，未修）`moduleResolution: node16/nodenext` 下类型不可解析。** 产出的 `.d.ts` 使用
无扩展名相对路径（`export * from './types'`），Node16/NodeNext 要求 `./types.js` → 17 × TS2834，
消费方的 `Mat4` / `RenderItem` / `RenderStats` 等类型全部消失（runtime 无碍：bundle 是单文件）。
修法是给 `src` 的 46 处相对 import 补 `.js` 扩展名。**现在不修**：没有消费者证据表明有人用
node16 解析（bundler 配置 0 error），按「只由证据驱动」原则登记为 KNOWN，而不是顺手改 10 个文件。
