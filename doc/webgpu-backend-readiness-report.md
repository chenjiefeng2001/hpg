# WebGPU 后端成熟度与生产可用性分析报告

## 1. 评估范围

本报告评估当前 `hpg` 仓库距离一个真实可用的 WebGPU rendering backend 的差距。

本报告的代码行号以阶段一修复前的基线为主；阶段一新增代码后，部分行号会向后移动，文件和符号名称保持不变。

评估依据：

- 当前源代码和公开 API
- 19 个测试文件、282 个测试（本轮 hardening 新增回归覆盖）
- 23 个真实 GLB 资产回归
- Chrome/Dawn 本地浏览器验证记录
- GitHub Actions 的 clean-checkout 门禁
- 最终 npm tarball 的消费者安装、import 和 TypeScript 检查

Phase 名称和 roadmap 完成数量不作为完成度依据。结论只基于当前实现和已有证据。

## 2. 当前状态结论

当前项目不是概念原型。它已经具备真实的 WebGPU 执行链：

```text
RenderItem[]
  -> 排序
  -> 合批
  -> 实例数据组装
  -> RingBuffer
  -> WebGPU render pass
  -> queue.submit
```

GPU culling 路径也已经存在：

```text
RenderItem[]
  -> 世界空间包围球
  -> compute frustum culling
  -> atomic compaction
  -> indirect draw
```

当前最准确的定位是：

> 一个真实可执行的 WebGPU Render Work -> GPU Execution 技术基线，窄域 opaque/MASK 场景接近 MVP；尚未达到通用 production rendering backend 门槛。

如果目标限定为以下范围，当前已经接近可用 MVP：

- Chrome WebGPU
- 单 canvas
- 静态 triangle-list
- opaque 或 MASK
- 单 color target
- 一个额外材质 bind group
- 短时间运行

如果目标包含长期运行、跨设备可靠性、完整输入契约和通用资源管理，目前仍未达到生产可用标准。

## 3. 工程成熟度评分

评分为工程判断，不是成功概率。

| 维度 | 评分 | 当前判断 |
|---|---:|---|
| 执行链 | 3.5 / 5 | Direct、Batched、GPU culling 和 indirect draw 均有实际实现 |
| GPU 正确性 | 3.0 / 5 | 23 个资产在 Chrome 中验证了主要 happy path，但 fake GPU 不执行 WGSL |
| 资源与内存 | 2.5 / 5 | Arena、RingBuffer、free-list 和扩容处理存在，缺少长期压力证据 |
| 输入与输出契约 | 2.5 / 5 | RenderItem 链路清楚，复杂输入和多 pass 能力不完整 |
| 兼容性 | 2.5 / 5 | Chrome 和当前 GLB 语料有证据，跨浏览器和跨设备证据不足 |
| 工程化与发布 | 3.0 / 5 | CI、打包、消费者验证和浏览器 gate 已建立，浏览器 gate 仍是 runner 能力依赖 |

综合成熟度约为 2.8 / 5，即约 55% 至 60%。这个数字表示实现、证据、边界和生产防护的综合程度，不表示测试通过率。

## 4. 已有真实证据

### 4.1 Node 回归

当前测试结果：

```text
19 test files passed
282 tests passed
```

已覆盖的执行不变量包括：

- pipeline、geometry、bind group 变化时正确拆分批次
- instance count、draw count、batch count 统计
- group=1 dynamic offset 的非零偏移
- RingBuffer 扩容后 bind group 重建
- draw args 在 compute dispatch 前写入
- geometry、pipeline、bind group 分组
- 不同材质不会错误合并
- world-space bounds 和 model matrix offset
- 真实 GLB 到 RenderItem 的导入路径

主要测试文件：

- `test/render-chain.test.ts`
- `test/execution-grouping.test.ts`
- `test/pipeline-descriptor.test.ts`
- `test/asset-corpus.test.ts`
- `test/culling.test.ts`

### 4.2 真实资产

`test/asset-corpus.test.ts` 对 23 个 GLB 执行：

```text
parseGltf
  -> importGltfAsset
  -> sceneToRenderItems
  -> submit / submitCulled
```

并验证几何规模、bounds、group 数量、dynamic offset 和命令顺序。

当前审计结果：

```text
23/23 GLB parse
BROKEN = none
```

这证明当前语料和当前 parser/importer 组合可以进入执行链，不证明任意 glTF 都可以正确解析。

### 4.3 真实 Chrome

仓库记录的 Chrome 153、Dawn、Windows 验证覆盖：

- 23/23 模型
- validation error = 0
- texture skipped = 0
- Direct/Culled 亮度网格一致
- 多材质模型不串 group=2

相关实现：

- `benchmark/browser-material-check.mjs`
- `demo/glb-viewer.ts`

该证据属于单一浏览器、单一操作系统和单一 GPU 环境，不能替代跨设备验证。

本轮修复后的本地 Chrome 验证重新执行了完整 23 模型 gate：两种模式均有 draw call 和非空像素签名，validation error = 0，贴图跳过 = 0，Direct/Culled 亮度网格差异为 0。该 gate 现在同时拒绝空画布、无匹配资产、模式错误，并在同一提交队列的 GPU completion 后发布像素结果。

## 5. P0：生产可用前必须处理

以下条目保留为审计背景；截至本轮 hardening，5.1、5.2、5.4、5.5 已关闭，5.3 已建立显式契约边界但自定义 WGSL 的真实语义仍需 GPU 证明，当前状态汇总见第 11 节。

### 5.1 group=2 静默状态复用（已关闭）

`ExecutionBackend` 只有在 batch 有 bind group 时才调用 `setBindGroup(2, ...)`：

- `src/core/executor.ts:75`
- `src/core/renderer.ts:530`
- `src/core/renderer.ts:838`

如果同一 pipeline 先后绘制：

```text
item A：bindGroup = materialA
item B：bindGroup = undefined
```

item B 可能继续使用 materialA 的 group=2。该问题可以产生串材质画面，但不一定产生 validation error。

当前测试覆盖了有材质到无材质的拒绝路径，以及非空 bind group 的切换。

### 5.2 PipelineCache 未使用真实 GPU 对象身份（已关闭）

`PipelineCache` 的 canonical key 现在同时使用 shader 文本、layout/buffer 的真实对象身份、size、byte offset 和 byte length；cache 还隔离 GPUDevice，并保存不可变 descriptor snapshot。Renderer 的 global bind group 缓存改用 pipeline 对象身份，且拒绝跨 Renderer pipeline。

回归测试覆盖：

- 不同 GPUBuffer、相同 size
- 不同 BindGroupLayout、相同 label
- 相同对象/相同 descriptor 的缓存命中
- descriptor snapshot 与跨 device 拒绝

### 5.3 compaction shader 契约没有被 runtime 强制验证（部分关闭）

`submitCulled()` 要求使用 compaction pipeline：

- `src/core/renderer.ts:574`

但注册和提交阶段现在要求内置 compaction shader，或要求自定义 WGSL 显式声明 `compactionContract: 'hpg-compaction-v1'`；普通提交入口会拒绝 compaction pipeline。该契约能阻止误用，但仍不能替代真实 shader 编译/执行证据，自定义 WGSL 的语义验证仍是发布前责任。

### 5.4 任意 transform 下的包围球可能不保守（已关闭：culling 限定 affine）

当前 culling radius 使用矩阵列范数：

- `src/core/renderer.ts:721`
- `src/core/renderer.ts:727`

公共 `RenderItem.transforms` 仍允许一般矩阵，但 `submitCulled()` 明确限制为 affine（末行 `[0,0,0,1]`）；affine shear 由矩阵 1/无穷范数乘积覆盖，projective transform 会在提交前报错。Direct 路径不使用该 CPU 包围球契约。

### 5.5 真实 WebGPU 验证未进入自动发布门禁（已关闭：release/publish 依赖 browser gate）

当前 CI 主要验证：

- Node
- fake GPU
- TypeScript
- 资产审计
- 构建
- tarball
- 消费者安装

workflow：

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- `.github/workflows/publish-npm.yml`
- `.github/workflows/browser-gate.yml`

真实 Chrome/WebGPU 现在通过 `browser-gate.yml` 提供手动和 reusable workflow 两种入口；GitHub Release 与 npm publish job 都依赖同一 gate，hosted runner 是否提供 WebGPU 仍由环境决定。

## 6. P1：重要生产差距

### 6.1 输入契约（已大幅收紧，仍非完整反射系统）

当前已验证 transforms、instanceCount、instanceData 长度与有限性、bytesPerInstance/modelMatrixOffset 对齐、bounding/depth、vpMatrix、geometry/pipeline layout、primitive、target format、index range 和 affine culling transform。动态用户 bind group layout、任意多 slot geometry 和 WGSL 反射仍不在支持域内。

### 6.2 公共 API 宽于实际 Executor

类型允许多个 target 和 bind group layout：

- `src/types.ts:93`
- `src/types.ts:97`

实际执行主要支持单一 color attachment、group 0/1/2 和默认 triangle-list。`geometry.primitive` 虽然被保存，但没有完整驱动 primitive state：

- `src/types.ts:57`
- `src/core/executor.ts:90`

### 6.3 资源和设备生命周期没有生产证据

尚未验证：

- 长时间逐帧运行
- 多次 RingBuffer 扩容
- 多次 Arena pool 增长
- 重复 import/unload
- device limit
- device lost
- query feature 缺失
- resize 和 context 状态变化

### 6.4 culling 缺少真实部分可见矩阵

尚未系统验证：

- 0% 可见
- 10% 和 50% 可见
- frustum 边界
- 多 geometry 的 slot base
- partial visibility 下的 compaction mapping
- indirect instance count

### 6.5 compute 和 render 分成两个 submit（已关闭）

`Renderer.submitCulled()` 现在把 culling compute pass 和 render pass 录制到同一个 `GPUCommandEncoder`，只提交一次；`CullingPipeline.cull()` 直接调用仍保留独立提交以维持独立 API。

### 6.6 culling timestamp 不完整

`submitCulled()` 没有将 timestamp 传入 compute：

- `src/core/renderer.ts:779`

当前 GPU timing 主要覆盖 render pass，可能低估 culling 的 GPU 成本。

### 6.7 glTF 语义仍有未实现项

当前 parser 已对 sparse accessor、截断 buffer/view/accessor、错误索引、默认材质和缺失 NORMAL 给出显式处理；仍未实现或仅告警的项目包括 double-sided 执行语义（现在会显式 warning）、完整 PBR texture、texCoord 选择、mipmap 链和动画。

### 6.8 热路径存在大量临时分配

`submitCulled()` 和 `CullingPipeline.cull()` 会创建 Map、TypedArray 和临时对象：

- `src/core/renderer.ts:601`
- `src/core/renderer.ts:766`
- `src/core/culling.ts:255`

这与“热路径零分配”的历史表述不一致，500K 候选规模下可能产生 CPU/GC 瓶颈。

## 7. P2：工程和方法学差距

- browser parity 使用亮度 grid 和阈值，不是严格逐像素相等：`benchmark/browser-material-check.mjs`
- Phase 5 和 Phase 6 benchmark 的 transform 与空间位置口径需要重新核对
- coverage provider 和 coverage script 尚未纳入 package contract：`vitest.config.ts`
- 没有 lint gate
- 没有跨浏览器和跨 GPU CI
- consumer check 主要验证 import、代表性 runtime exports 和 bundler TypeScript：`scripts/verify-package.mjs`
- Node16/NodeNext declaration 解析仍是已知限制

## 8. UNKNOWN，不应直接转为 TODO

以下项目目前没有足够证据决定优先级：

- 长时运行和 resource pressure
- 生产规模 workload
- 目标硬件性能基线
- 真实项目接入
- 跨浏览器和跨设备行为
- 生产规模 culling 的实际收益
- skin / animation
- alphaMode=BLEND
- 其他 PBR texture
- 多 pass / render graph

这些应由真实 workload、correctness 问题或性能测量触发，而不是由功能清单驱动。

## 9. 最小下一阶段路线

### 阶段一：消除静默错误（已完成）

1. group=2 从有材质切换到无材质
2. PipelineCache 的真实 GPU 对象身份
3. compaction shader 契约验证
4. 任意 transform 下的保守包围球
5. RenderItem 输入校验
6. primitive、target、bind group 能力与 Executor 对齐

### 阶段二：真实 GPU 正确性矩阵

至少覆盖：

- 全可见
- 0% 可见
- 部分可见
- 多 geometry
- 多 material
- 有材质/无材质混合
- RingBuffer 扩容
- 非索引 indirect
- resize
- 更严格的像素或 image signature

### 阶段三：资源和设备压力

只测当前窄域 workload：

- 重复 import/unload
- 长时间逐帧 submit
- 多次 RingBuffer 扩容
- 多次 Arena 增长
- 多材质共享 image
- device limit
- dispose 前后行为
- device lost 和错误恢复

### 阶段四：平台和发布契约

明确：

- 支持的浏览器和设备
- timestamp feature fallback
- device lost/error API
- 资源 disposal policy
- package 的 Node16/NodeNext 支持策略
- npm 和 GitHub Release 的发布边界

## 10. 最终判断

当前项目可以称为：

> 真实可执行的 WebGPU Render Work -> GPU Execution 技术基线。

当前项目不能称为：

> 已经 production-ready、跨设备可靠、长期资源行为可证明的通用 WebGPU rendering backend。

距离生产可用的核心差距不是缺少 skin、动画或完整 PBR，而是：

```text
公共输入契约
  -> 静默错误是否全部消除
  -> 真实 GPU 是否覆盖边界语义
  -> 资源和设备生命周期是否可证明
  -> benchmark 是否测到真实执行链
  -> 支持矩阵和发布策略是否明确
```

## 11. 阶段一与第二轮 hardening 状态

本轮已关闭的确定性问题：

- WebGPU 投影矩阵改为 NDC 深度 `[0,1]`，near/far 映射为 `0/1`。
- 普通 `submit()` / `submitDirect()` 拒绝 compaction pipeline；自定义 compaction WGSL 必须声明 `compactionContract: 'hpg-compaction-v1'`。
- `submitCulled()` 拒绝 projective transform；affine shear 使用保守矩阵范数；culling geometry ID 长度、范围和分组顺序显式校验。
- instanceData 要求精确长度，三条写入路径都会清零额外字段；bytesPerInstance 使用 16 字节对齐。
- Geometry 拒绝多 vertex slot、instance-step、非法 stride/attribute、错误 index format、越界索引和 triangle-list 非整三角形索引；vertex/index binding 传递 byteLength。
- Renderer 拒绝 target/depth format 不匹配、stencil/color depth format、未配置或跨 device 的 context；静态 global uniform binding 校验设备对齐、范围与 UNIFORM usage。
- PipelineCache 使用设备隔离、对象身份和 descriptor snapshot；Renderer 按 pipeline 对象而非局部 numeric id 缓存和合批，并拒绝跨 Renderer pipeline。
- Renderer 拒绝未由当前 arena 创建或已销毁的 Geometry；内置实例 shader 的布局契约按规范化 WGSL 校验，不能用注释绕过。
- glTF 严格校验 GLB/container/bufferView/accessor，拒绝 sparse accessor、截断数据和非法索引；primitive 未声明 material 时使用默认材质；缺失 NORMAL 时展开 flat normals。
- 修复默认 UV 原点、共享 image 只上传一次、sRGB 选项、场景 reload 的 Geometry 回收、timestamp feature 检查和 benchmark transform 组合；reload generation 在异步文件读取前保留，避免旧请求覆盖新模型。
- culling compute 与 render 使用同一 command encoder；GPU timestamp 通过标准 `timestampWrites` pass descriptor 注入；browser gate 先做 WebGPU adapter 快速预检，再拒绝空像素/无 draw/无资产/模式错误，并在同一提交队列的 GPU completion 后发布 harness 结果。
- release 与 npm publish workflow 依赖 reusable browser gate；发布前验证并发布同一个 tarball。

当前验证：

```text
19 test files / 282 tests passed
23/23 GLB audit: parse ok, BROKEN = 0
Chrome 23/23: validation error = 0, texture skipped = 0,
Direct/Culled brightness-grid parity = 0
GPU culling matrix: 0% / 10% / 50% / 100%, mapping valid = true
```

仍不能由本轮静态/本地证据关闭的事项：

- 自定义 compaction WGSL 的真实 mapping 语义仍需 shader 编译和 GPU 执行验证；显式 contract 是信任边界，不是 WGSL 反射证明。
- 用户自定义 dynamic group layout、多 vertex slot、完整 PBR、mipmap 链、alpha BLEND、double-sided 执行语义仍未实现（double-sided 已有显式 warning）。
- `FS_DEPTH_ONLY` 仍不是无 color output 的真正 depth-only pipeline；该 API 需要单独的多 attachment 设计。
- culling timestamp 目前只覆盖 render pass，不能代表完整 compute+render GPU 成本。
- 当前已验证合成 workload 的 0%/10%/50%/100% GPU culling mapping；更复杂的多材质/大规模 culling、device lost、长时间资源压力、跨设备和跨浏览器矩阵仍需要专门 workload。
- hosted runner 是否提供 WebGPU 取决于环境；browser gate 现在会让发布 fail closed，而不是静默跳过。
