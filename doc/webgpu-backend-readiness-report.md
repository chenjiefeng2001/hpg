# WebGPU 后端成熟度与生产可用性分析报告

## 1. 评估范围

本报告评估当前 `hpg` 仓库距离一个真实可用的 WebGPU rendering backend 的差距。

本报告的代码行号以阶段一修复前的基线为主；阶段一新增代码后，部分行号会向后移动，文件和符号名称保持不变。

评估依据：

- 当前源代码和公开 API
- 19 个测试文件、237 个测试（阶段一新增 7 个回归测试）
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
| 工程化与发布 | 3.0 / 5 | CI、打包、消费者验证和发布 workflow 已建立，浏览器 gate 不在 CI |

综合成熟度约为 2.8 / 5，即约 55% 至 60%。这个数字表示实现、证据、边界和生产防护的综合程度，不表示测试通过率。

## 4. 已有真实证据

### 4.1 Node 回归

当前测试结果：

```text
19 test files passed
237 tests passed
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
- Direct/Culled 像素签名一致
- 多材质模型不串 group=2

相关实现：

- `benchmark/browser-material-check.mjs`
- `demo/glb-viewer.ts`

该证据属于单一浏览器、单一操作系统和单一 GPU 环境，不能替代跨设备验证。

## 5. P0：生产可用前必须处理

### 5.1 group=2 静默状态复用

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

现有测试覆盖了两个非空 bind group 的切换，没有覆盖有材质到无材质的转换。

### 5.2 PipelineCache 未使用真实 GPU 对象身份

`PipelineCache` 的 canonical key 主要使用：

- shader 文本
- layout label
- buffer size
- byte offset
- byte length

相关代码：

- `src/core/pipeline-cache.ts:26`
- `src/core/renderer.ts:282`

不同 GPUBuffer 如果 size 相同，或不同 BindGroupLayout 如果 label 相同，可能产生相同 key。Renderer 随后按 pipeline id 缓存 global bind group，存在错误资源复用的可能。

当前缺少以下测试：

- 不同 GPUBuffer、相同 size
- 不同 BindGroupLayout、相同 label
- 不同对象身份但相同 descriptor

### 5.3 compaction shader 契约没有被 runtime 强制验证

`submitCulled()` 要求使用 compaction pipeline：

- `src/core/renderer.ts:574`

但注册和提交阶段没有验证 vertex shader 是否真正读取 compaction mapping。

fake GPU：

- 不编译 WGSL
- 不执行 compute shader
- 不执行 atomic

因此部分 Node 测试只能证明命令录制形状，不能证明真实 compaction 语义。

真实 viewer 使用了正确 shader，但 demo 的正确使用不能替代 runtime contract。

### 5.4 任意 transform 下的包围球可能不保守

当前 culling radius 使用矩阵列范数：

- `src/core/renderer.ts:721`
- `src/core/renderer.ts:727`

公共 `RenderItem.transforms` 没有限制 shear：

- `src/types.ts:131`

对于一般带 shear 的 affine transform，列范数不一定覆盖完整变换，可能低估世界空间包围球并错误剔除对象。

需要补充：

- rotation + non-uniform scale
- shear
- 多个 parent transform 组合
- frustum plane 边界

### 5.5 真实 WebGPU 验证未进入自动发布门禁

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

真实 Chrome/WebGPU 仍是本地手工 gate。发布前仍可能只通过结构测试，而未执行真实 shader、atomic 或 indirect draw。

## 6. P1：重要生产差距

### 6.1 输入契约缺少完整运行时验证

需要验证：

- transforms 长度
- instanceCount 合法性
- instanceData 长度
- bytesPerInstance 对齐
- modelMatrixOffset 合法性
- bounding 和 depth 的有限性
- vpMatrix 长度
- geometry、primitive、target 的组合

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

### 6.5 compute 和 render 分成两个 submit

`CullingPipeline.cull()` 自己提交 compute：

- `src/core/culling.ts:308`

Renderer 再提交 render：

- `src/core/renderer.ts:797`

这会限制外部 command encoder、统一错误范围和 frame-level profiling。

### 6.6 culling timestamp 不完整

`submitCulled()` 没有将 timestamp 传入 compute：

- `src/core/renderer.ts:779`

当前 GPU timing 主要覆盖 render pass，可能低估 culling 的 GPU 成本。

### 6.7 glTF 语义仍有静默缺失

明确例子：

- sparse accessor 没有读取分支：`src/core/gltf.ts:335`
- `doubleSided`、metallic、roughness 被解析但没有完整进入执行语义
- `texCoord` 没有完整驱动 UV 选择
- 部分 parser 失败只 console 输出，没有进入 `asset.warnings`

### 6.8 热路径存在大量临时分配

`submitCulled()` 和 `CullingPipeline.cull()` 会创建 Map、TypedArray 和临时对象：

- `src/core/renderer.ts:601`
- `src/core/renderer.ts:766`
- `src/core/culling.ts:255`

这与“热路径零分配”的历史表述不一致，500K 候选规模下可能产生 CPU/GC 瓶颈。

## 7. P2：工程和方法学差距

- browser parity 使用亮度 grid 和阈值，不是严格逐像素相等：`benchmark/browser-material-check.mjs:220`
- pixel signature 为空时，gate 不一定失败：`demo/glb-viewer.ts:823`
- Phase 5 和 Phase 6 benchmark 的 transform 与空间位置口径需要重新核对
- coverage provider 和 coverage script 尚未纳入 package contract：`vitest.config.ts:8`
- 没有 lint gate
- 没有跨浏览器和跨 GPU CI
- `emptyOutDir: false` 可能保留旧的 dist 文件：`vite.config.ts:16`
- consumer check 主要验证 import、identity 和 bundler TypeScript：`scripts/verify-package.mjs:50`
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

### 阶段一：消除静默错误

优先处理：

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

## 11. 阶段一修复状态

本轮已完成以下确定性修复：

- group=2 材质 bind group 缺失时，在 submit、submitDirect、submitCulled 和 Executor 入口显式报错，避免继承上一项的材质状态。
- PipelineCache 的 bind group layout 和 global buffer key 纳入真实 GPU 对象身份，避免相同 label/size 的不同资源发生碰撞。
- submitCulled 强制要求 `compaction: true`，并将 fake GPU 测试中的 culling helper 改为使用 `VS_INSTANCED_COMPACTION`。
- culling 包围球使用矩阵 1/无穷范数乘积作为保守线性变换上界，覆盖 shear transform。
- 增加 RenderItem transforms、instanceCount、instanceData、bounding、depth、vpMatrix 和 geometry/pipeline primitive 校验。
- 管线注册阶段拒绝非法 modelMatrixOffset、bytesPerInstance、空/多 target 和未实现的额外 bind group layout。
- 新增对应回归测试，当前测试总数为 19 个文件、237 个测试。

阶段一仍未完全关闭的事项：

- 本地 Chrome/WebGPU 小规模回归已通过：3 个模型、Direct/Culled、0 validation error、0 texture skip、像素签名一致。
- 真实 Chrome/WebGPU 浏览器验证仍未进入 GitHub hosted runner 的自动发布门禁；已增加 `browser-gate.yml` 作为发布前可手动触发的独立 gate。
- 0%/部分可见 culling、device lost、资源压力和跨设备矩阵仍需要真实 workload 证据。
- compaction flag 已强制，但自定义 WGSL 是否正确读取 compaction mapping 仍必须由真实 shader 编译和浏览器验证证明。
