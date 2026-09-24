# Changelog

## 0.2.0 — 技术基线冻结（Baseline Freeze）

**这是一条稳定的 `0.x` execution runtime 技术基线，不是 production-ready release。**

hpg 提供 WebGPU-first 的 **Render Work → GPU Execution** runtime：batching、sorting、
GPU culling、indirect execution、geometry management，以及经过真实 GLB 语料 / 真实 Chrome
验证的基础 textured rendering。不支持的 asset feature 会**显式暴露**（structured warnings +
注册期报错），而不是静默产生错误结果。

它**不承诺**是 production-ready 的通用 3D rendering runtime —— 缺少 workload 证据
（长时运行 / resource pressure / 目标硬件性能基线 / 真实项目接入）。

```text
0.2.0 ≠ 所有功能完成 ≠ production ready
0.2.0 =  当前 execution boundary 的稳定、可复现技术基线
```

### 冻结时的验证结果（全部可复现）

| Gate | 命令 | 结果 |
|---|---|---|
| 类型检查 | `npm run typecheck` | 0 error |
| 回归测试 | `npm test` | 19 files / 230 passed |
| 真实资产审计 | `npm run audit` | parse ok 23/23、BROKEN（静默错误数据）= 0 |
| 真实 Chrome + WebGPU | `npm run verify:browser` | 23/23：validation error = 0、贴图跳过 = 0、Direct/Culled 像素级一致 |
| 库构建 | `npm run build` | `dist/lib/index.js`、`.d.ts` 与 sourcemap 产出；大小是一次工具链快照 |
| Demo 构建 | `npm run build:demo` | 多页面入口全部产出（index / phase5 / glb-viewer / benchmark / glb-bench） |

```text
Execution chain: verified
Real GLB corpus: 23/23
Browser validation: 23/23
Direct/Culled parity: verified
GPU validation errors: 0
Known unsupported features: explicit warnings
Known contract violations: 0
```

**测试数 236 → 230**：删除 3 个只有 `console.log`、零断言的临时审计探针
（`test/tmp-audit.test.ts`、`test/tmp-audit2.test.ts`、`test/tmp-dbg.test.ts` ——
Phase 12 的脚手架，对应 6 个伪用例）。它们观察的不变量已由
`test/render-chain.test.ts` / `test/geometry.test.ts` 的断言覆盖。

### 消费者验证（clean install / consumer test）

在项目外的全新空目录里做真实消费者验证：`npm pack` → 安装 tarball → `import` → `tsc`。

```text
tarball            24 files（dist/lib + LICENSE / CHANGELOG / README + package.json）
src/test/demo 泄漏  none
裸 Node ESM import  OK（零 WebGPU 全局对象）
import 期读全局数   0（修复前 2）
runtime 路径        7/7（43 exports · parseGltf · flattenScene · frustum · GeometryArena）
tsc bundler + @webgpu/types   0 error
```

### CI and release plumbing

The repository now has three GitHub Actions workflows:

- `.github/workflows/ci.yml` runs the reproducible Node, asset, build, package-boundary, and
  external-consumer checks for pull requests and pushes to `main`.
- `.github/workflows/release.yml` runs the release checks when a `vX.Y.Z` tag is pushed, then creates
  a GitHub Release with the final npm tarball attached.
- `.github/workflows/publish-npm.yml` provides a protected manual npm publication path with an
  `NPM_TOKEN` secret, provenance, and an explicit `next` or `latest` distribution tag.
- `prepack` builds the library before `npm pack` and `npm publish` create a package archive.
- The real Chrome/WebGPU gate remains a local command (`npm run verify:browser`) because WebGPU
  availability on hosted runners is environment-dependent.
- The GitHub Release workflow does not publish to the npm registry; npm publication is deliberately
  a separate manual operation.

The six frozen verification gates remain part of the baseline record, but they are not all executed
by the hosted CI runner. The browser gate must be run separately on a WebGPU-capable machine.

**F1（已修）import 期崩溃。** `src/core/texture.ts` 曾在模块顶层求值
`GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST`，导致任何非 WebGPU 环境
（Node / jsdom / SSR / 构建期资产管线）只要 `import` 包入口就 `ReferenceError`——
即使只用到 `parseGltf` 这类纯函数；同时也与 `package.json` 声明的 `sideEffects: false` 矛盾。
改为调用期惰性求值（`textureUsage()`）。
> `GeometryArena.createGeometry()` 等**调用期**读取 `GPUBufferUsage` 属预期行为：
> 那些调用本来就需要 WebGPU 设备。

**F2（已文档化）声明文件不自带 `@webgpu/types`。** `@webgpu/types` 是 *optional* peer，npm 不会
自动安装；`tsc` 在 declaration emit 时丢弃了 `/// <reference types>`。消费方必须自行
`npm i -D @webgpu/types` 并在 tsconfig `types` 中声明（README 已写明）。
注意：`skipLibCheck: true` 会**静默**把这些类型退化成 `any`，而不是报错。

**F3（KNOWN，未修）`moduleResolution: node16/nodenext` 下类型不可解析。** 产出的 `.d.ts` 使用
无扩展名相对路径（`export * from './types'`），Node16/NodeNext 要求 `./types.js` → 17 × TS2834，
消费方的 `Mat4` / `RenderItem` / `RenderStats` 等类型全部消失（runtime 无碍：bundle 是单文件）。
修法是给 `src` 的 46 处相对 import 补 `.js` 扩展名。**现在不修**：没有消费者证据表明有人用
node16 解析（bundler 配置为 0 error），按「只由证据驱动」原则登记为 KNOWN。

### UNKNOWN（不是 TODO，没有 Phase 编号）

以下项目**没有**被证据证明是 execution boundary 的缺失，因此不进入 roadmap：

```text
UNKNOWN
- Skin / Animation
- alphaMode = BLEND
- 其他 PBR 贴图（normal / metallic-roughness / occlusion / emissive）
- 长时运行行为 / resource pressure
- 生产规模 workload
- 目标硬件 GPU performance baseline
- 真实项目接入验证
```

### 重新打开开发的三类触发条件

```text
真实 workload → correctness 问题        → 修复
真实 workload → performance bottleneck  → measurement → optimization
真实 workload → Render Work 无法表达    → architecture discussion
```

「还剩哪些 glTF feature 没支持」**不是**触发条件。

---

### 变更明细

针对「加载模型后前端页面不显示 / 显示错误」的根因修复。

### Fixed — 取景与裁剪

- `importGltfAsset()` 返回的 `bounds` 现在是**世界空间** AABB（含节点变换与全局缩放），
  之前用未变换的局部顶点坐标，导致 Duck/Fox 这类「顶点大尺度 + 节点缩放」模型取景完全错误
- `ImportOptions.scale` 现在是真正的全局缩放（连平移一起缩放）
- `demo/glb-viewer` 的 near/far 随模型尺寸自适应（原固定 `0.1/100` 会把 >100 单位的模型整个裁掉）

### Fixed — GPU 剔除（submitCulled）

- 视锥平面提取修正：`vp[i]` 在 WGSL 中是**列**，原实现按行读取并用同行的 w 分量混合 →
  实测所有视锥内物体被判为不可见。现使用正确的 Gribb-Hartmann 行列组合，
  并新增 CPU 参考实现 `extractFrustumPlanes()` / `sphereInFrustum()`
- `submitCulled()` 按 **(geometry, pipeline)** 分组，之前按 `pipeline.id` 分组，
  共享同一管线的多个 mesh 会退化成只画第一个（BrainStem 59 个 mesh → 1 个 draw call）
- `registerPipeline({ compaction: true })` 新增：group=1 使用 `[instance, compactedIndices]`
  布局。此前 culled 路径用的 bind group 与管线布局 entry 数不匹配，`setBindGroup(1, ...)`
  必然 WebGPU 校验失败
- 逐实例包围球：用 `Geometry.bounds`（局部 AABB）× 实例矩阵推导世界空间包围球；
  无包围盒信息时保守地永不剔除

### Fixed — 多 mesh 模型整帧全黑（真实浏览器复现）

两个只有在真实 WebGPU 下才会暴露的缺陷，此前在 fake GPU 测试里全绿却在页面上「什么都没画」：

1. **实例 bind group 绑定尺寸缺省 → dynamic offset 越界**
   `createBindGroup` 曾绑定整块 ring buffer（`size = capacity`），而 WebGPU 要求
   `bindingOffset + dynamicOffset + bindingSize ≤ bufferSize` —— 于是帧内第 2 个批
   （dynamic offset 256）必然越界。Chrome/Dawn 报：
   `Dynamic Offset[0] (256) is out of bounds … Did you forget to specify the binding's size?`，
   整个 render pass 失效 → 整帧全黑。单 batch 帧 offset 恒为 0，所以单 mesh 模型能显示、
   多 mesh 模型（Duck / Lantern / BrainStem / ClearCoatTest…）必黑。
   现改为显式声明绑定尺寸 = 本帧最大批的实例字节数，并给 ring / compactedIndices 预留等量余量，
   保证每个批的 offset 都落在范围内（仍是一个 bind group + 逐批 dynamic offset）。
2. **draw args 的 CPU 预写在 compute dispatch 之后入队，把原子填充的 `instanceCount` 清零**
   `queue.writeBuffer` 按入队顺序执行，`submitCulled` 在 `cull()` 之后写 `indexCount/firstIndex/...`，
   把 GPU 刚原子累加的 `instanceCount` 覆盖回 0 → `drawIndexedIndirect` 画 0 个实例。
   **无任何校验错误**，只是什么都不显示。现改为把 CPU 字段作为模板随 `cull()` 一起在 dispatch
   之前写入（`CullingPipeline.cull(..., drawArgsTemplate?)`）。

验证方式（真实 Chrome 153 / Dawn，通过 CDP 驱动）：

| 模型 | Direct litPixels / avgLuma | GPU Culled |
|------|---------------------------|------------|
| DamagedHelmet（1 mesh） | 1476 / 133.1 | 1476 / 133.1 |
| WaterBottle（1 mesh） | 1308 / 106.5 | 1308 / 106.5 |
| CesiumMilkTruck（5 items） | 1991 / 150.8 | 1991 / 150.8 |
| BrainStem（59 meshes） | 434 / 58.5 | 434 / 58.5 |
| Lantern（3 meshes） | 531 / 70.0 | 531 / 70.0 |

Direct 与 GPU Culled 两条路径现在像素级一致（架构原则：合批/剔除失败不得改变视觉结果）。

### Fixed — GeometryArena 池链

- 跨池复用旧池的空闲切片时，数据被写进了**最新池的同名偏移**，覆盖该池中存活几何体的顶点区间
  （实测：释放 pool0 中的 96B 切片后再申请同尺寸几何体，会覆盖 pool1 中 5 MiB 几何体的开头，
  且 slice 指向 pool1@0）。free-list 现在记录空闲块**所属的池**，分配结果返回 `{ pool, offset }`，
  写入与切片都落在该块真正的池上；相邻块合并也限定在同一池内
- 只有全新分配才落在最新池（并触发容量检查），池链不变式不变

### Fixed — 实例缓冲生命周期

- `RingBuffer` 扩容后重建实例 bind group：底层 `GPUBuffer` 被替换后旧 bind group 会读到废弃缓冲，
  单帧实例数据 >1 MiB（约 13k 实例）时整帧无输出
- `RingBuffer.alloc()` 支持对齐参数，实例区改用 `RING_ALIGN`(256) 对齐 ——
  作为 storage buffer 动态偏移时必须满足 `minStorageBufferOffsetAlignment`
- `ExecutionBackend` 逐批重绑 group=1 的 dynamic offset，并支持 `RenderItem.bindGroup`（group=2）

### Fixed — 资源（材质 / 解析）

- `sceneToRenderItems()` 把材质 `baseColorFactor` 写入 `instanceData`（之前材质被完全丢弃，模型恒为白色）
- glTF accessor 读取重写：支持归一化整数属性（UBYTE/SHORT）、交错 byteStride、
  非对齐 accessor（复制到对齐缓冲），修复 `MultiUVTest.glb` 的 `Uint32Array` 对齐崩溃；
  UBYTE 索引提升为 uint16；无索引 primitive 在顶点数 >65535 时生成 uint32 索引
- 明确的错误提示：Draco / meshopt 压缩、外部 buffer URI、非 GLB 容器、非三角形拓扑（跳过并告警）

### Build

- `vite.demo.config.ts` 显式声明多页面入口（此前 build 只产出根 `index.html`，
  `demo/glb-viewer.html` 等页面不在产物中）

### Demo

- `demo/glb-viewer` 新增 `?asset=<url>` 直达加载（例：`?asset=/benchmark/assets/models/medium/DamagedHelmet.glb`），
  便于复现与自动化验证
- 新增 GPU 错误浮层：`device.onuncapturederror` + 逐帧 `pushErrorScope('validation')` 的错误直接显示在 HUD 上。
  此前 WebGPU 校验错误只进控制台，页面看上去就是「模型加载了但什么都没画」

### Tests

- 新增 `test/render-chain.test.ts`（15）/ `test/gltf-robustness.test.ts`（10）＋ executor 补充 3 例
- `test/fake-gpu.ts` 记录 bind group / indirect draw / dispatch，用于断言实例缓冲与剔除路径

### Added — 真实资产兼容性审计（Phase 14）

- `npm run audit` —— `benchmark/asset-compat.ts` **独立于解析器**读原始 glTF JSON，逐模型枚举
  顶点属性 / primitive / 材质 / 贴图 / 场景 / 动画 / 扩展的使用情况，与 `parseGltf` 结果对照，
  输出「feature 使用率 → hpg 状态（supported / ignored / broken）→ 症状」矩阵。
  23 个真实 GLB（Blender / glTF-Transform / COLLADA2GLTF / 3ds Max / babylon.js 导出）实测：

  | feature | 命中模型 | 状态 |
  |---|---:|---|
  | `baseColorTexture` | 22 / 23 | ignored（材质退化为纯色） |
  | `metallicRoughness` / `normal` / `occlusion` / `emissive` 贴图 | 13 / 9 / 7 / 9 | ignored |
  | `alphaMode=MASK` / `BLEND` | 4 / 2 | ignored（按不透明渲染） |
  | `JOINTS_0` + `WEIGHTS_0` + skin | 3 | ignored（渲染绑定姿势） |
  | `TEXCOORD_1` / morph target / 动画 | 2 / 1 / 4 | ignored |
  | 压缩扩展（Draco / meshopt） | 0 | 不可读 → 明确报错 |

  结论：解析层面 23/23 无阻塞（无 sparse accessor、无压缩网格）；缺的是**材质/贴图**这条路径。

- `GltfAsset.warnings` —— 把所有「模型用到、hpg 未实现」的 feature 结构化输出（不再只是 console 日志），
  并按 feature 全局去重一次：贴图未采样、alphaMode 被当不透明、蒙皮渲染绑定姿势、morph target、
  多套 UV、动画未播放、扩展材质（`KHR_materials_*`）与必需扩展（如 `KHR_texture_transform`）。
  目的是把「加载成功但结果与作者意图不符」变成可展示、可断言的信息。
- `demo/glb-viewer` 新增黄色警告面板 + HUD 行 `Unsupported: N feature(s)`，加载真实模型时直接可见。

### Tests — 真实资产回归语料（Phase 14D/14E）

- `test/asset-corpus.test.ts`（10）：把 benchmark 语料固化成回归集 —— 逐模型跑
  `parseGltf → importGltfAsset → sceneToRenderItems → submit / submitCulled`，断言
  几何规模 golden、无 GPU 校验错误、多批 dynamic offset 真正非零、indirect draw 数 == geometry 组数、
  draw args 预写早于 dispatch、世界空间 bounds、全局缩放作用于平移分量。
  此前只会在真实 Chrome 暴露的两类 GPU-only bug（绑定尺寸/动态偏移越界、draw args 覆盖原子计数）
  现在在 Node 下就会失败。
- `test/gltf-compat-warnings.test.ts`（4）：锁定未实现 feature 提示的文案类别，
  并交叉校验「审计（独立读 JSON）发现的 feature → 解析器必须报出对应 warning」。

测试总数 194 → 208。

### Added — Material / Texture 路径（Phase 15A）+ TANGENT 保留（15B）

真实语料 22/23 个模型使用 `baseColorTexture`，而此前材质只使用 `baseColorFactor`（纯色）——
这是**第一个用户可见的资产兼容阻塞点**。Phase 15A 只打通最小闭环，不引入完整 Material System：

```
GLB image（编码字节）
  → ImageDecoder（可注入）
  → GPUTexture（sRGB） + GPUSampler
  → material bind group（固定 group=2）
  → RenderItem.bindGroup → fragment textureSample()
```

- `parseGltf()` 新增解析 `images` / `textures` / `samplers`：`AssetMaterial.baseColorTexture`
  （image + sampler + uv 通道）、`alphaMode`、`alphaCutoff`，以及 `GltfAsset.images`（只保留编码字节）。
  解析器**不解码**，保持环境无关（Node 无 `createImageBitmap`）。
- `ImageDecoder` 接口 + `createBrowserImageDecoder()`（Blob → createImageBitmap → OffscreenCanvas
  → RGBA8，`colorSpaceConversion:'none'` 保留原始 sRGB 值）。
- `MaterialStore`：解码并上传每张 base color 贴图（`rgba8unorm-srgb`，glTF base color 是 color data），
  逐材质生成 bind group —— binding 0 texture / 1 sampler / 2 material uniform（32B）。
  无贴图材质绑定 1×1 白色 fallback，着色器无需分支；同一 image 只解码一次；解码失败退回白色并记录
  `stats.skipped`。
- `registerPipeline()` 的 bind group 槽位固定化：**group 0 = 全局，group 1 = 实例，group 2+ = 调用方
  额外布局**（材质）。传入 `[global, material]` 得到 `[global, instance, material]`，
  与 executor 的 1=instance / 2=material 录制一致（此前实例布局被追加到末尾，材质无法占用 group=2）。
- 新增着色器 `MATERIAL_LAYOUT_TEMPLATE` / `VS_INSTANCED_MATERIAL` /
  `VS_INSTANCED_MATERIAL_COMPACTION` / `FS_MATERIAL`：采样 baseColorTexture，
  `alphaMode == 1 (MASK)` 时按 `alphaCutoff` discard。BLEND 仍走 warning（需要透明排序与混合状态）。
- `sceneToRenderItems(scene, pipeline, materials?)` 第 3 个参数为材质提供者；
  附上材质时实例颜色写白色（baseColorFactor 由 material uniform 提供，避免乘两次）。
  （原先第 3 个参数 `globalBindings` 从未被使用 —— 全局 uniform 由 `PipelineDesc.globalBindings`
  在注册时声明、渲染器按管线注入 group=0；这个静默无效参数已移除。）
- **TANGENT 不再丢失**：canonical 顶点布局扩展为 stride 48（pos3 + norm3 + uv2 + tan4），
  缺失时填默认 `(1,0,0,1)`。布局恒定便于后续 normal mapping，但本阶段**不实现** normal map 采样。

审计更新：`TANGENT` → supported；`material.baseColorTexture` / `material.alphaMode=MASK`
从 WRONG 台账移出（已解析 / 已实现），`baseColorTexture` 标为「需外部 ImageDecoder」。

测试总数 208 → 229（新增 `test/material-path.test.ts` 13 例、`test/execution-grouping.test.ts` 5 例、shaders 3 例）。

### Added — 执行分组 regression lock（渲染正确性）

材质路径引入了一个与「dynamic offset 越界 / draw args 覆盖原子计数」同级的
**GPU 资源状态被错误复用**风险：分组键若只用 `(geometry, pipeline)`，
共享同一 geometry 但材质不同的实例会共用第一个材质的 `group=2` → 串贴图，
且**没有任何 validation error**。

- `Renderer.submitCulled()` 的间接绘制分组键加入 `bindGroup`（一个 geometry 组只能绑一个 group=2）。
- `Renderer.submitDirect()` 逐 item 重绑 `group=2`（此前完全不绑定）。
- `test/execution-grouping.test.ts` 固化不变量：
  `same geometry + same pipeline + different bindGroup ⇒ 独立执行组`，
  覆盖 Direct / GPU Culled / 多 geometry × 多 material / 非零 dynamic offset，
  且同 material 仍能正常合批。

### Added — 真实 Chrome 材质验证（Phase 15E）

`npm run verify:browser`（`benchmark/browser-material-check.mjs`）—— 无第三方依赖：
Node 内置 `WebSocket` + CDP 驱动本机 Chrome（`headless=new --enable-unsafe-webgpu`），
自起 Vite dev server 作为子进程，逐模型加载 `demo/glb-viewer.html?harness=1&mode=…&asset=…`，
采集材质/贴图统计、WebGPU validation error 与 canvas 像素签名（litPixels / avgLuma / 8×8 亮度网格），
并对比 Direct 与 GPU Culled。

Chrome 153 / Dawn / Windows 实测 **23/23 模型**：无 validation error、无贴图跳过、
Direct 与 GPU Culled 像素级一致（`gridΔ ≤ 0.1`、`litΔ = 0`）。
其中 `ClearCoatTest`（19 材质 / 1 贴图）与 `TextureEncodingTest`（14 材质 / 5 贴图）
验证了同一 image 只解码一次与多材质不串图；`BrainStem`（59 材质 / 0 贴图）验证了白色 fallback。

### Fixed — 4 处「声明了但被静默忽略」的执行边界契约（Phase 15.5 审计）

状态审计（见 PLAN.md「Phase 15.5」）确认主链没有已被证据证明的功能性缺失，
但发现 4 处公开契约名不符实 —— 属于同一类问题：**不支持必须明确报错，而不是悄悄渲染错**。

| # | 契约 | 原行为 | 现行为 |
|---|---|---|---|
| 1 | `GlobalBinding.byteOffset` / `byteLength` | 被忽略，永远绑整块 buffer（同一 buffer 的多个 uniform 切片会读到同一份数据） | 真正进入 `createBindGroup` |
| 2 | 管线缓存键的 `globalBindings` | 只含 `binding:buffer.size` → 同一 buffer 的不同切片塌成同一管线 / 同一 bind group | 键含 offset / size |
| 3 | `PipelineDesc.modelMatrixOffset` | 被忽略，mat4 永远写在记录开头 → 着色器按声明偏移读取会拿到错位数据 | `submit` / `submitDirect` / `submitCulled` 三条写入路径都按偏移落位，预留区显式清零 |
| 4 | `sceneToRenderItems(scene, pipeline, globalBindings?)` | 第 3 个参数从未被使用（静默无效） | 移除；`materials` 上移为第 3 个参数 |

`PipelineDesc.modelMatrixOffset` 的非法值（非 16 的倍数 / 记录装不下 mat4）
现在在 `registerPipeline()` 阶段就报错。

新增 `test/pipeline-descriptor.test.ts`（7 例）钉死 1 与 3 的行为。

### Decision — `alphaMode=BLEND` 暂不实现（Phase 15F）

实测只有 2 个 Khronos feature-test 模型命中，且各自只有 1 个材质
（`AlphaBlendModeTest` 的 `MatBlend`、`ClearCoatTest` 的 `Partial_Coating`），无生产资产命中。
BLEND 牵动 `transparent classification → sorting → depthWrite/depthTest → blend state`，
可能演变成独立 render phase，属于 render execution semantics 的改动 ——
保持 warning，并把边界固化为测试（`test/asset-corpus.test.ts` Phase 15F 用例）。

---

## 0.1.0

Initial release.

### Core

- `Renderer` — GPU work scheduler: `RenderItem[]` → sorted, batched, bound GPU commands
- `Renderer.create(desc)` — factory method with descriptor object
- `Renderer.submit(items)` — auto-batching path (instanced draw)
- `Renderer.submitCulled(items, vpMatrix)` — GPU frustum culling + indirect draw
- `Renderer.geometryArena` — geometry pool manager
- `Renderer.dispose()` — release all GPU resources

### Geometry

- `GeometryArena` — pooled vertex/index memory with free-list recycling
- `GeometryArena.createGeometry()` — upload vertices + indices
- `GeometryArena.destroyGeometry()` — free arena allocation
- `GeometryArena.stats()` — pool usage metrics (capacity, fragmentation, live count)
- Multi vertex buffer support (`Geometry.vertexBuffers[]`)

### GPU-Driven Pipeline

- Compute shader frustum culling (Gribb-Hartmann plane extraction)
- Atomic compaction mapping (`compactedIndices[originalIdx] = slot`)
- `drawIndexedIndirect` with GPU-filled instance counts
- Zero readback — CPU never reads GPU results

### Instrumentation

- `TimestampQuery` — GPU timestamp query (write → resolve → readback)
- Optional GPU timing on `submitDirect` and `submitCulled`

### Internals

- 47-bit spatial key encoding (layer | pipeline | bindGroup | depth)
- Counting sort (O(n), stable)
- Ring buffer transient memory (per-frame sequential allocation)
- Pipeline cache with content hashing (FNV-1a)
- Instance buffer direct write (no Float32Array-to-GPU copy)

### Requirements

- Node ≥ 18
- WebGPU-capable browser (Chrome 113+ / Edge 113+)
- `@webgpu/types` (optional peer dependency)
