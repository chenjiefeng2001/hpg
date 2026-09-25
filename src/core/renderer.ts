/**
 * Renderer —— API 入口。
 *
 * 生命周期：validate → spatialKey → packKey → counting sort → collect(batch)
 * → assemble(统一实例存储) → ringBuffer → ExecutionBackend → WebGPU queue。
 *
 * 热路径零堆分配（实例存储与排序 scratch 均为复用数组）。
 */

import type { CullingDebugResult, Geometry, GlobalBinding, PipelineDesc, RenderItem, RenderStats, ResolvedPipeline } from '../types';
import { PipelineCache } from './pipeline-cache';
import { GeometryArena } from './geometry';
import { RingBuffer, RING_ALIGN } from './ringbuffer';
import { Batcher, instanceCountOf } from './batcher';
import { ExecutionBackend, assertRequiredBindGroup } from './executor';
import { instanceBufferOffset } from './commands';
import type { Batch } from './commands';
import { countingSortKeys, packKeyValue, DEFAULT_LAYER } from './keygen';
import { CullingPipeline } from './culling';
import { TimestampQuery } from './timestamp';
import { VS_INSTANCED, VS_INSTANCED_MATERIAL, VS_INSTANCED_COMPACTION, VS_INSTANCED_MATERIAL_COMPACTION } from '../shaders/instance';

/** submit() 的可选参数。 */
export interface SubmitOptions {
  /**
   * 相机世界位置。提供后，未显式指定 `item.depth` 的物体按相机距离归一化排序
   * （近→远，early-z 友好）；未提供时保持提交顺序（稳定排序）。
   */
  camera?: [number, number, number];
  /** 深度归一化区间（世界单位）。默认 [0, 1e4]。 */
  depthRange?: [number, number];
}

export interface RendererOptions {
  clearColor?: [number, number, number, number];
  depthFormat?: GPUTextureFormat;
  /** 单帧瞬态池初始大小（字节）。 */
  maxRingBytes?: number;
  label?: string;
}

export interface RendererDescriptor extends RendererOptions {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

interface CullingDebugSnapshot {
  drawArgsBuffer: GPUBuffer;
  drawArgsCount: number;
  compactedIndicesBuffer: GPUBuffer;
  compactedIndicesByteLength: number;
  compactedSlotCount: number;
  slotBases: Uint32Array;
  candidateCounts: Uint32Array;
  totalCandidateInstances: number;
}

const DEFAULT_CLEAR: [number, number, number, number] = [0.05, 0.06, 0.09, 1];

function normalizeShaderSource(code: string): string {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/.*$/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function isCompactionShader(code: string): boolean {
  const normalized = normalizeShaderSource(code);
  return normalized === normalizeShaderSource(VS_INSTANCED_COMPACTION) || normalized === normalizeShaderSource(VS_INSTANCED_MATERIAL_COMPACTION);
}

function isBuiltInInstanceShader(code: string): boolean {
  const normalized = normalizeShaderSource(code);
  return normalized === normalizeShaderSource(VS_INSTANCED) || normalized === normalizeShaderSource(VS_INSTANCED_MATERIAL) || isCompactionShader(code);
}

function assertCompactionPipeline(item: RenderItem): void {
  if (item.pipeline.desc.compaction !== true) {
    throw new Error(`Pipeline "${item.pipeline.label}" must be registered with compaction: true for submitCulled().`);
  }
}

function assertDirectPipeline(item: RenderItem): void {
  if (item.pipeline.desc.compaction === true) {
    throw new Error(`Pipeline "${item.pipeline.label}" is a compaction pipeline and can only be used with submitCulled().`);
  }
}

function assertAffineTransforms(item: RenderItem): void {
  if (!item.transforms) return;
  for (let i = 0; i < item.transforms.length; i += 16) {
    const m = item.transforms;
    if (Math.abs(m[i + 3] as number) > 1e-6 || Math.abs(m[i + 7] as number) > 1e-6 || Math.abs(m[i + 11] as number) > 1e-6 || Math.abs((m[i + 15] as number) - 1) > 1e-6) {
      throw new Error('submitCulled only supports affine transforms.');
    }
  }
}

function expectedInstanceDataFloats(item: RenderItem): number {
  const extraBytes = item.pipeline.bytesPerInstance - item.pipeline.modelMatrixOffset - 64;
  return extraBytes / 4;
}

function assertFiniteValues(values: ArrayLike<number>, name: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error(`${name}[${i}] must be finite.`);
  }
}

function validateRenderItem(item: RenderItem): void {
  if (item.transforms && item.transforms.length % 16 !== 0) {
    throw new Error('RenderItem.transforms length must be a multiple of 16.');
  }
  const count = instanceCountOf(item);
  if (!Number.isSafeInteger(count) || count <= 0) {
    throw new Error('RenderItem.instanceCount must be a positive safe integer.');
  }
  if (item.transforms) {
    if (count > item.transforms.length / 16) {
      throw new Error('RenderItem.instanceCount cannot exceed transforms.length / 16.');
    }
    assertFiniteValues(item.transforms, 'RenderItem.transforms');
  }
  if (item.instanceData) {
    assertFiniteValues(item.instanceData, 'RenderItem.instanceData');
    const expected = count * expectedInstanceDataFloats(item);
    if (item.instanceData.length !== expected) {
      throw new Error(`RenderItem.instanceData length must equal instanceCount * extra floats (${expected}).`);
    }
  }
  const topology = item.pipeline.desc.primitive?.topology ?? 'triangle-list';
  if (item.geometry.primitive !== topology) {
    throw new Error(`RenderItem geometry primitive ${item.geometry.primitive} does not match pipeline topology ${topology}.`);
  }
  const pipelineLayouts = item.pipeline.desc.vertexLayouts;
  if (pipelineLayouts.length !== item.geometry.vertexLayouts.length) {
    throw new Error('Geometry and pipeline vertex layout counts do not match.');
  }
  for (let i = 0; i < pipelineLayouts.length; i++) {
    const geometryLayout = item.geometry.vertexLayouts[i]!;
    const pipelineLayout = pipelineLayouts[i]!;
    if (geometryLayout.arrayStride !== pipelineLayout.arrayStride || geometryLayout.stepMode !== pipelineLayout.stepMode) {
      throw new Error(`Geometry and pipeline vertex layout ${i} do not match.`);
    }
    for (const pipelineAttribute of pipelineLayout.attributes) {
      const geometryAttribute = geometryLayout.attributes.find((attribute) => attribute.shaderLocation === pipelineAttribute.shaderLocation);
      if (!geometryAttribute || geometryAttribute.offset !== pipelineAttribute.offset || geometryAttribute.format !== pipelineAttribute.format) {
        throw new Error(`Geometry and pipeline vertex attribute ${i}:${pipelineAttribute.shaderLocation} do not match.`);
      }
    }
  }
  if (topology.endsWith('-strip') && item.geometry.indexSlice && item.pipeline.desc.primitive?.stripIndexFormat !== item.geometry.indexFormat) {
    throw new Error('Indexed strip geometry requires a matching pipeline stripIndexFormat.');
  }
  if (item.geometry.indexSlice && !item.geometry.indexBuffer) {
    throw new Error('Geometry indexSlice requires an indexBuffer.');
  }
  if (item.depth !== undefined && !Number.isFinite(item.depth)) {
    throw new Error('RenderItem.depth must be finite.');
  }
  if (item.bounding) {
    assertFiniteValues([item.bounding.centerX, item.bounding.centerY, item.bounding.centerZ], 'RenderItem.bounding center');
    if (!Number.isFinite(item.bounding.radius) || item.bounding.radius < 0) {
      throw new Error('RenderItem.bounding.radius must be a non-negative finite number.');
    }
  }
}

export class Renderer {
  private cache = new PipelineCache();
  private arena: GeometryArena;
  private ring: RingBuffer;
  private batcher = new Batcher();
  private executor = new ExecutionBackend();

  private instanceLayout: GPUBindGroupLayout;
  /** 复用的 group=1 实例 bind group（随 buffer 身份 / 绑定尺寸变化重建）。 */
  private instanceBG: { bg: GPUBindGroup; ring: GPUBuffer; size: number } | null = null;
  /** Culling path: 2-binding layout [instance, compactedIndices]。 */
  private compactionLayout: GPUBindGroupLayout;
  /** Culling path: 复用的 compaction bind group（随 buffer 身份 / 绑定尺寸变化重建）。 */
  private compactionBG: {
    bg: GPUBindGroup;
    ring: GPUBuffer;
    compacted: GPUBuffer;
    instanceSize: number;
    slotSize: number;
  } | null = null;
  private globalBGs = new Map<ResolvedPipeline, GPUBindGroup>();
  private ownedPipelines = new WeakSet<ResolvedPipeline>();

  private sortedItems: RenderItem[] = [];
  private orderScratch: Int32Array = new Int32Array(0);
  private indexScratch: number[] = [];
  private keysScratch: number[] = [];
  private instanceStore = new Float32Array(0);
  private _pipelineSet = new Set<number>();
  private _disposed = false;

  // GPU Culling 相关。
  private _culling: CullingPipeline | null = null;
  private _cullingSnapshot: CullingDebugSnapshot | null = null;

  // GPU 时间戳查询（按需创建）。
  private _timestamps: TimestampQuery | null = null;

  private clearColor: [number, number, number, number];
  private depthFormat: GPUTextureFormat;
  private readonly label: string;

  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly presentationFormat: GPUTextureFormat;

  /** 便捷工厂：单对象描述符创建 Renderer。 */
  static create(desc: RendererDescriptor): Renderer {
    return new Renderer(desc.device, desc.context, desc.format, desc);
  }

  /** 暴露 GeometryArena 实例（供 stats() / createGeometry / destroyGeometry）。 */
  get geometryArena(): GeometryArena {
    return this.arena;
  }

  /**
   * 当前瞬态实例缓冲（ring 的底层 GPUBuffer）。
   * 仅用于调试 / 统计 —— 容量不足时会在帧内重建，不要跨帧持有。
   */
  get instanceBuffer(): GPUBuffer {
    return this.ring.buffer;
  }

  constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    opts: RendererOptions = {},
  ) {
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;
    this.clearColor = opts.clearColor ?? DEFAULT_CLEAR;
    this.depthFormat = opts.depthFormat ?? 'depth24plus';
    if (this.depthFormat.includes('stencil')) {
      throw new Error('Renderer currently supports depth-only formats without stencil attachments.');
    }
    if (this.depthFormat !== 'depth16unorm' && this.depthFormat !== 'depth24plus' && this.depthFormat !== 'depth32float') {
      throw new Error(`Renderer depthFormat must be a depth-only format, received ${this.depthFormat}.`);
    }
    this.label = opts.label ?? 'hpg';

    this.arena = new GeometryArena(device);
    this.ring = new RingBuffer(device, opts.maxRingBytes ?? 1 << 20);

    this.instanceLayout = device.createBindGroupLayout({
      label: `${this.label}:instance-layout`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage', hasDynamicOffset: true },
        },
      ],
    });
    // Culling path: group(1) = [instanceBuffer, compactedIndicesBuffer]
    this.compactionLayout = device.createBindGroupLayout({
      label: `${this.label}:compaction-layout`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage', hasDynamicOffset: true },
        },
        {
          // compaction mapping：按 geometry 的 slot 区绑定（dynamic offset）。
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'read-only-storage', hasDynamicOffset: true },
        },
      ],
    });
  }

  /**
   * 取/建 group=1 实例 bind group。
   *
   * **绑定尺寸（size）必须显式声明，且不能是整块 ring buffer**：
   * WebGPU 要求 `bindingOffset + dynamicOffset + bindingSize ≤ bufferSize`。
   * 绑定整块 buffer 时 size = capacity，任何非零 dynamic offset 都越界 ——
   * Dawn 会报 “Dynamic Offset[0] (256) is out of bounds … Did you forget to specify
   * the binding's size?”，整个 render pass 失效 → **整帧无输出**。
   * 单批帧 offset 恒为 0 所以看不出问题，多 mesh 模型必黑。
   *
   * 尺寸取「本帧最大批的实例字节数」：既能覆盖任意批的全部实例数据，
   * 又只需调用方为 ring 预留同样大小的余量即可让所有批的 offset 都落在范围内。
   *
   * ring 扩容会新建底层 GPUBuffer（旧 bind group 会读到废弃 buffer）→ 一起重建。
   */
  private ensureInstanceBindGroup(size: number): GPUBindGroup {
    const cached = this.instanceBG;
    if (cached && cached.ring === this.ring.buffer && cached.size === size) return cached.bg;
    const bg = this.device.createBindGroup({
      label: `${this.label}:instance-bg`,
      layout: this.instanceLayout,
      entries: [{ binding: 0, resource: { buffer: this.ring.buffer, size } }],
    });
    this.instanceBG = { bg, ring: this.ring.buffer, size };
    return bg;
  }

  /**
   * 复用 compaction bind group：[实例区, compaction mapping]。
   *
   * 两个 binding 都带 dynamic offset，所以同一个 bind group 可服务所有 geometry，
   * 只有底层 buffer 身份 / 绑定尺寸变化时才重建（ring 扩容、剔除缓冲重建）。
   * 两者都必须显式声明 size（理由同 ensureInstanceBindGroup）：
   *   binding 0 → 本帧最大 geometry 组的实例字节数
   *   binding 1 → 单个 geometry 的最大 slot 区字节数（CullingPipeline 已按此预留余量）
   */
  private ensureCompactionBindGroup(
    compacted: GPUBuffer,
    instanceSize: number,
    slotSize: number,
  ): GPUBindGroup {
    const ring = this.ring.buffer;
    const cached = this.compactionBG;
    if (
      cached &&
      cached.ring === ring &&
      cached.compacted === compacted &&
      cached.instanceSize === instanceSize &&
      cached.slotSize === slotSize
    ) {
      return cached.bg;
    }
    const bg = this.device.createBindGroup({
      label: `${this.label}:compaction-bg`,
      layout: this.compactionLayout,
      entries: [
        { binding: 0, resource: { buffer: ring, size: instanceSize } },
        { binding: 1, resource: { buffer: compacted, size: slotSize } },
      ],
    });
    this.compactionBG = { bg, ring, compacted, instanceSize, slotSize };
    return bg;
  }

  /** dispose 之后调用任何提交入口都是编程错误，显式报错替代晦涩的空引用异常。 */
  private assertUsable(method: string): void {
    if (this._disposed) throw new Error(`[hpg] Renderer 已 dispose，${method}() 不可再用。`);
  }

  private assertPresentationFormat(method: string): void {
    const configuration = this.context.getConfiguration?.();
    if (!configuration) {
      throw new Error(`[hpg] ${method}() requires a configured GPUCanvasContext.`);
    }
    if (configuration.format !== this.presentationFormat) {
      throw new Error(`[hpg] ${method}() context format ${configuration.format} does not match presentation format ${this.presentationFormat}.`);
    }
    if (configuration.device && configuration.device !== this.device) {
      throw new Error(`[hpg] ${method}() context is configured for a different GPUDevice.`);
    }
    const usage = configuration.usage ?? GPUTextureUsage.RENDER_ATTACHMENT;
    if ((usage & GPUTextureUsage.RENDER_ATTACHMENT) === 0) {
      throw new Error(`[hpg] ${method}() context usage must include RENDER_ATTACHMENT.`);
    }
  }

  /**
   * 注册管线（内容哈希缓存）。返回句柄可直接在 RenderItem 中使用。
   *
   * **Bind group 槽位是固定的**（executor 按此录制，调用方不需要关心）：
   *   - group 0 = 全局（`bindGroupLayouts[0]` + `globalBindings`）
   *   - group 1 = 实例数据（内部自动插入；普通路径 1 binding，`compaction: true` 时 2 binding）
   *   - group 2+ = 调用方额外布局（如材质），对应 `bindGroupLayouts[1..]`
   *
   * 因此传入 `[globalLayout]` 得到 `[global, instance]`，传入 `[globalLayout, materialLayout]`
   * 得到 `[global, instance, material]` —— 材质恰好落在 group 2（`RenderItem.bindGroup`）。
   */
  registerPipeline(desc: PipelineDesc): ResolvedPipeline {
    this.assertUsable('registerPipeline');
    if (desc.bindGroupLayouts.length === 0) {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 至少需要一个 group 0 layout。`);
    }
    if (desc.targets.length === 0) {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 至少需要一个 color target。`);
    }
    if (desc.targets.length !== 1) {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 当前只支持一个 color target。`);
    }
    if (desc.targets[0]!.format !== this.presentationFormat) {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 的 target format ${desc.targets[0]!.format} 与 presentation format ${this.presentationFormat} 不一致。`);
    }
    if (desc.vertexLayouts.length !== 1 || desc.vertexLayouts[0]!.stepMode !== 'vertex') {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 当前只支持一个 vertex-step layout。`);
    }
    assertGlobalBindings(this.device, desc.globalBindings);
    if (desc.bindGroupLayouts.length > 2) {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 当前只支持 group 0 和一个额外 group 2。`);
    }
    // 深度格式必须与渲染通道的深度附件一致，否则 draw 阶段才会报校验错误（很难定位）。
    if (desc.depth && desc.depth.format !== this.depthFormat) {
      throw new Error(
        `[hpg] 管线 "${desc.label ?? 'unnamed'}" 的深度格式 ${desc.depth.format} 与 Renderer 的 depthFormat ${this.depthFormat} 不一致；` +
          '请在 Renderer.create({ depthFormat }) 中对齐（或修正管线的 depth.format）。',
      );
    }
    // 实例记录布局的合法性：mat4x4<f32> 在 storage buffer 里要求 16 字节对齐，
    // 且记录必须能装下「预留区 + mat4」。静默接受会让着色器读到错位的矩阵。
    const bpi = desc.bytesPerInstance ?? 80;
    const mmo = desc.modelMatrixOffset ?? 0;
    if (!Number.isSafeInteger(mmo) || mmo < 0 || mmo % 16 !== 0) {
      throw new Error(
        `[hpg] 管线 "${desc.label ?? 'unnamed'}" 的 modelMatrixOffset=${mmo} 必须是非负的 16 的倍数（mat4x4<f32> 对齐）。`,
      );
    }
    if (!Number.isSafeInteger(bpi) || bpi < 64 || bpi % 16 !== 0) {
      throw new Error(
        `[hpg] 管线 "${desc.label ?? 'unnamed'}" 的 bytesPerInstance=${bpi} 必须是至少 64 且按 16 字节对齐的安全整数。`,
      );
    }
    if (bpi < mmo + 64) {
      throw new Error(
        `[hpg] 管线 "${desc.label ?? 'unnamed'}" 的 bytesPerInstance=${bpi} 必须 ≥ modelMatrixOffset(${mmo}) + 64。`,
      );
    }
    if (isBuiltInInstanceShader(desc.vsCode) && (bpi !== 80 || mmo !== 0)) {
      throw new Error(`[hpg] 内置实例着色器要求 bytesPerInstance=80 且 modelMatrixOffset=0。`);
    }
    if (desc.compaction === true && !isCompactionShader(desc.vsCode) && desc.compactionContract !== 'hpg-compaction-v1') {
      throw new Error(`[hpg] 管线 "${desc.label ?? 'unnamed'}" 使用自定义 compaction shader 时必须声明 compactionContract: 'hpg-compaction-v1'.`);
    }
    if (desc.compaction !== true && isCompactionShader(desc.vsCode)) {
      throw new Error(`[hpg] 内置 compaction 着色器必须搭配 compaction: true。`);
    }

    const instanceGroup = desc.compaction ? this.compactionLayout : this.instanceLayout;
    // 实例布局固定插在 group 1：用户的第一个 layout 做 group 0（全局），
    // 其余 layout 顺延到 group 2+（材质等）。这样 executor 的 1=instance / 2=material 恒成立。
    const userLayouts = desc.bindGroupLayouts;
    const full = {
      ...desc,
      bindGroupLayouts: userLayouts.length > 0
        ? [userLayouts[0] as GPUBindGroupLayout, instanceGroup, ...userLayouts.slice(1)]
        : [instanceGroup],
    };
    const resolved = this.cache.getOrCreate(this.device, full);
    this.ownedPipelines.add(resolved);
    return resolved;
  }

  createGeometry(
    vertexData: Float32Array,
    vertexLayouts: PipelineDesc['vertexLayouts'],
    indexData?: Uint16Array | Uint32Array,
    indexFormat?: GPUIndexFormat,
    primitive: GPUPrimitiveTopology = 'triangle-list',
  ) {
    this.assertUsable('createGeometry');
    return this.arena.createGeometry(vertexData, vertexLayouts, indexData, indexFormat, primitive);
  }

  private assertOwnedGeometry(geometry: Geometry): void {
    if (!this.arena.ownsGeometry(geometry)) {
      throw new Error('Geometry was not created by this Renderer or has been destroyed.');
    }
  }

  private assertOwnedPipeline(item: RenderItem): void {
    if (!this.ownedPipelines.has(item.pipeline) || (item.pipeline.device && item.pipeline.device !== this.device)) {
      throw new Error(`Pipeline "${item.pipeline.label}" was not registered with this Renderer.`);
    }
  }

  private globalBindGroup(p: ResolvedPipeline): GPUBindGroup {
    const hit = this.globalBGs.get(p);
    if (hit) return hit;
    const bg = this.device.createBindGroup({
      label: `${this.label}:global:${p.label}`,
      layout: p.bindGroupLayouts[0] as GPUBindGroupLayout,
      entries: p.desc.globalBindings.map((b) => ({
        binding: b.binding,
        resource: globalBindingResource(b),
      })),
    });
    this.globalBGs.set(p, bg);
    return bg;
  }

  /**
   * 提交一批渲染意图包，立即渲染到当前帧。
   * @param opts 可选：传入相机位置后，未指定 depth 的物体按相机距离做近→远排序。
   * @returns 该帧统计
   */
  submit(items: RenderItem[], opts?: SubmitOptions): RenderStats {
    this.assertUsable('submit');
    this.assertPresentationFormat('submit');
    const n = items.length;
    const stats: RenderStats = {
      itemsSubmitted: n,
      itemsDrawn: 0,
      instances: 0,
      drawCalls: 0,
      batches: 0,
      pipelinesUsed: 0,
    };

    if (n === 0) return stats;

    for (const item of items) {
       this.assertOwnedPipeline(item);
       this.assertOwnedGeometry(item.geometry);
       validateRenderItem(item);
      assertDirectPipeline(item);
      assertRequiredBindGroup(item);
    }

    if (opts?.depthRange) {
      const [near, far] = opts.depthRange;
      if (!Number.isFinite(near) || !Number.isFinite(far) || far <= near) throw new Error('SubmitOptions.depthRange must be finite with far > near.');
    }
    if (opts?.camera && opts.camera.some((value) => !Number.isFinite(value))) {
      throw new Error('SubmitOptions.camera values must be finite.');
    }

    // 1. 排序键 + 计数排序（稳定）。
    const [near, far] = opts?.depthRange ?? [0, 1e4];
    this.ensureScratch(n);
    const keys = this.keysScratch;
    for (let i = 0; i < n; i++) {
      const item = items[i] as RenderItem;
      const depth01 = resolveDepth01(item, opts?.camera, near, far);
      keys[i] = packKeyValue(DEFAULT_LAYER, item.pipeline.id, 0, depth01);
    }
    const order = this.orderScratch;
    for (let i = 0; i < n; i++) order[i] = i;
    countingSortKeys(order as unknown as number[], keys, this.indexScratch);
    const sorted = this.sortedItems;
    for (let i = 0; i < n; i++) sorted[i] = items[order[i] as number] as RenderItem;

    // 2. 合批 + 实例重排。
    //    实例落位跨步由各批自己的管线决定（见 Batcher.collect）——
    //    帧内混用不同 bytesPerInstance 的管线时，不能用帧内最大跨步统一排布。
    const { batches, bytesToWrite } = this.batcher.collect(sorted);

    // 3. 组装统一实例存储。
    this.assemble(batches, bytesToWrite);

    // 4. 分配瞬态缓冲 + 上传（256 对齐：作为 storage dynamic offset 使用）。
    //    额外预留「本帧最大批」的字节数：实例 bind group 的绑定尺寸取该值，
    //    所有批都必须满足 dynamicOffset + 绑定尺寸 ≤ capacity（见 ensureInstanceBindGroup）。
    const maxBatchBytes = maxBatchBytesOf(batches);
    const base = this.ring.alloc(bytesToWrite + maxBatchBytes, RING_ALIGN);
    try {
      this.ring.write(base, this.instanceStore, 0, bytesToWrite);
    // alloc() 可能触发 ring 扩容（底层换新 buffer）→ 必须在录制前重建实例 bind group。
    const instanceBindGroup = this.ensureInstanceBindGroup(maxBatchBytes);

    // 6. 录制 + 提交。
    const encoder = this.device.createCommandEncoder({ label: `${this.label}:frame` });
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      label: `${this.label}:main`,
      colorAttachments: [
        {
          view,
          clearValue: { r: this.clearColor[0], g: this.clearColor[1], b: this.clearColor[2], a: this.clearColor[3] },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.ensureDepthView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    const pipelines = this._pipelineSet;
    pipelines.clear();
    for (const b of batches) pipelines.add(b.pipeline.id);

     this.executor.run(
       pass,
       {
         batches,
         items: sorted,
         instanceBindGroup,
         globalBindGroup: (p) => this.globalBindGroup(p),
         frameBase: base,
       },
       stats,
     );
     stats.batches = batches.length;
     stats.pipelinesUsed = pipelines.size;
     pass.end();

     this.device.queue.submit([encoder.finish()]);
     return stats;
   } finally {
     this.ring.endFrame();
   }
  }

  /**
   * 逐 item 直接绘制（绕过 batcher），用作 benchmark baseline。
   *
   * 每个 RenderItem 产生独立的 draw call，不做合批。
   * 仅用于性能测量，生产环境应使用 submit() 或 submitCulled()。
   *
   * @param timestamps 可选 GPU 时间戳查询；传入则在 render pass 前后各写一次。
   * @returns 该帧统计
   */
  submitDirect(items: RenderItem[], timestamps?: TimestampQuery): RenderStats {
    this.assertUsable('submitDirect');
    this.assertPresentationFormat('submitDirect');
    const n = items.length;
    const stats: RenderStats = {
      itemsSubmitted: n,
      itemsDrawn: 0,
      instances: 0,
      drawCalls: 0,
      batches: 0,
      pipelinesUsed: 0,
    };

    if (n === 0) return stats;

    for (const item of items) {
       this.assertOwnedPipeline(item);
       this.assertOwnedGeometry(item.geometry);
       validateRenderItem(item);
      assertDirectPipeline(item);
      assertRequiredBindGroup(item);
    }

    // 逐 item 组装到 instanceStore。
    // 每个 item 按**自己管线的实例跨步**落位（跨步 = 该管线 WGSL 中 InstanceData 的大小），
    // 批起始 256 对齐；帧内混用不同跨步的管线时各自独立，不会互相错位。
    //
    // 容量必须先于填充：instanceStore 初始长度为 0，向长度不足的 TypedArray 写入是
    // 静默 no-op —— 若先填充后扩容，本帧实例数据会全部变成零（矩阵退化 → 什么都画不出）。
    let required = 0;
    let maxItemBytes = 0;
    for (let i = 0; i < n; i++) {
      const it = items[i] as RenderItem;
      const stride = it.pipeline.bytesPerInstance || 80;
      const bytes = instanceCountOf(it) * stride;
      if (bytes > maxItemBytes) maxItemBytes = bytes;
      required = instanceBufferOffset(required) + bytes;
    }
    this.ensureStore(required);

    const itemOffsets: number[] = [];
    let gpuOffset = 0;

    for (let i = 0; i < n; i++) {
      const alignedOffset = instanceBufferOffset(gpuOffset);
      itemOffsets.push(alignedOffset);
      const item = items[i] as RenderItem;
      const frameStride = item.pipeline.bytesPerInstance || 80;
      // 实例记录布局：[0, mmBytes) 预留区 → mat4(64B) → 额外数据。
      const mmBytes = item.pipeline.modelMatrixOffset || 0;
      const mmFloat = mmBytes >> 2;
       const extra = frameStride - mmBytes - 64;
       const extraFloats = extra >> 2;
       const cnt = instanceCountOf(item);
       const m0 = item.transforms;
       const extraData = item.instanceData;
       const hasExtra = extraData !== undefined;
       let instanceOffset = alignedOffset;
       for (let m = 0; m < cnt; m++) {
         const base = instanceOffset / 4;
         for (let c = 0; c < mmFloat; c++) this.instanceStore[base + c] = 0;
         const f = base + mmFloat;
         if (m0) {
           const src = m * 16;
           for (let c = 0; c < 16; c++) this.instanceStore[f + c] = m0[src + c] as number;
         } else {
           for (let c = 0; c < 16; c++) this.instanceStore[f + c] = 0;
           this.instanceStore[f] = 1;
           this.instanceStore[f + 5] = 1;
           this.instanceStore[f + 10] = 1;
           this.instanceStore[f + 15] = 1;
         }
           for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = 0;
           if (hasExtra && extraData) {
             const src = m * extraFloats;
             for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = extraData[src + c] as number;
           } else if (extra >= 16) {
             this.instanceStore[f + 16] = 1;
             this.instanceStore[f + 17] = 1;
             this.instanceStore[f + 18] = 1;
             this.instanceStore[f + 19] = 1;
           }
         instanceOffset += frameStride;
       }
       gpuOffset = instanceOffset;
    }

    const bytesToWrite = gpuOffset;
    this.ensureStore(bytesToWrite); // 冗余保险：容量已在填充前确保。

    // 每个 item 一个 draw call + 一个 dynamic offset，故预留「最大 item」的字节数
    // 作为实例 bind group 的绑定尺寸与 buffer 余量。
    const base = this.ring.alloc(bytesToWrite + maxItemBytes, RING_ALIGN);
    try {
      this.ring.write(base, this.instanceStore, 0, bytesToWrite);
    const instanceBindGroup = this.ensureInstanceBindGroup(maxItemBytes);

    // 录制：逐 item 独立 draw call。
    const encoder = this.device.createCommandEncoder({ label: `${this.label}:frame-direct` });
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      label: `${this.label}:main-direct`,
      colorAttachments: [
        {
          view,
          clearValue: { r: this.clearColor[0], g: this.clearColor[1], b: this.clearColor[2], a: this.clearColor[3] },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.ensureDepthView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      ...(timestamps ? { timestampWrites: timestamps.timestampWrites(0, 1) } : {}),
    });



    const pipelines = this._pipelineSet;
    pipelines.clear();

    let lastPipeline: ResolvedPipeline | null = null;
    for (let i = 0; i < n; i++) {
      const item = items[i] as RenderItem;
      const pipeline = item.pipeline;
      const geometry = item.geometry;

      if (pipeline !== lastPipeline) {
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, this.globalBindGroup(pipeline));
        pipelines.add(pipeline.id);
        lastPipeline = pipeline;
      }

      pass.setBindGroup(1, instanceBindGroup, [
        instanceBufferOffset(base + itemOffsets[i]!),
      ]);
      // 材质 bind group（group=2），逐 item 重绑（不同 item 可能用不同材质）。
      if (item.bindGroup) {
        pass.setBindGroup(2, item.bindGroup);
      }
      for (let s = 0; s < geometry.vertexBuffers.length; s++) {
        const vb = geometry.vertexBuffers[s]!;
        pass.setVertexBuffer(s, vb.buffer, vb.byteOffset, vb.byteLength);
      }

      const cnt = instanceCountOf(item);
      if (geometry.indexSlice && geometry.indexBuffer) {
        pass.setIndexBuffer(geometry.indexBuffer, geometry.indexFormat, geometry.indexSlice.byteOffset, geometry.indexSlice.byteLength);
        pass.drawIndexed(geometry.indexCount, cnt, 0, 0, 0);
      } else {
        pass.draw(geometry.vertexCount, cnt, 0, 0);
      }

      stats.drawCalls++;
      stats.instances += cnt;
      stats.itemsDrawn++;
    }

    stats.batches = n;
    stats.pipelinesUsed = pipelines.size;

    pass.end();
     if (timestamps) timestamps.resolve(encoder);

     this.device.queue.submit([encoder.finish()]);
     return stats;
   } finally {
     this.ring.endFrame();
   }
  }

  /**
   * GPU 视锥剔除 + Indirect Draw 提交路径（Phase 6B：全 GPU-driven，零 readback）。
   *
   * 流程：
   *   1. 按 (geometry, pipeline) 分组，逐实例推导世界空间包围球 + geometry index
   *   2. 预写 draw args（indexCount 等），instanceCount 置零
   *   3. Compute Shader 视锥剔除 → 原子填充 instanceCount + compaction mapping
   *   4. drawIndexedIndirect 读取间接参数
   *
   * CPU 不读取任何 GPU 结果。
   *
   * 注意：传入的 item 必须使用 `registerPipeline({ compaction: true, vsCode: VS_INSTANCED_COMPACTION })`
   * 注册的管线 —— 顶点着色器需通过 `compactedIndices[instance_index]` 间接索引实例数据。
   * 用普通管线调用会在提交前被显式拒绝（必须注册为 compaction pipeline）。
   *
   * @param vpMatrix 列主序 4×4 VP 矩阵（用于视锥裁剪面提取）
   * @param timestamps 可选 GPU 时间戳查询；当前只覆盖 render pass，culling compute 不计入该区间。
   * @returns 该帧统计（`instances` 为候选实例数；可见数在 GPU 侧确定，不回读）
   */
  submitCulled(items: RenderItem[], vpMatrix: Float32Array, timestamps?: TimestampQuery): RenderStats {
    this.assertUsable('submitCulled');
    this.assertPresentationFormat('submitCulled');
    const n = items.length;
    const stats: RenderStats = {
      itemsSubmitted: n,
      itemsDrawn: 0,
      instances: 0,
      drawCalls: 0,
      batches: 0,
      pipelinesUsed: 0,
    };

    if (n === 0) return stats;

    if (vpMatrix.length !== 16) {
      throw new Error('vpMatrix must contain exactly 16 values.');
    }
    assertFiniteValues(vpMatrix, 'vpMatrix');

    for (const item of items) {
       this.assertOwnedPipeline(item);
       this.assertOwnedGeometry(item.geometry);
       validateRenderItem(item);
      assertRequiredBindGroup(item);
      assertCompactionPipeline(item);
      assertAffineTransforms(item);
    }

    // 1. 按 (geometry, pipeline, bindGroup) 分组。
    //    indirect draw 的 draw args 是逐 geometry 的，所以分组键必须含 geometry ——
    //    用 pipeline 分组会把共享同一管线的多个 mesh 当成一个，只画其中一个。
    //    bindGroup（材质）也进分组键：一个 geometry 组只能绑定一个 group=2，
    //    不同材质的实例必须拆成各自的 draw（否则会串用同一张贴图）。
     type GeoGroup = { geoIdx: number; item: RenderItem; count: number; itemCount: number };
    type ByBindGroup = Map<GPUBindGroup | undefined, GeoGroup>;
    const geoMap = new Map<Geometry, Map<ResolvedPipeline, ByBindGroup>>();
    const geoOrder: GeoGroup[] = [];
    let totalInstances = 0;

    for (let i = 0; i < n; i++) {
      const item = items[i] as RenderItem;
      let byPipeline = geoMap.get(item.geometry);
      if (!byPipeline) {
        byPipeline = new Map<ResolvedPipeline, ByBindGroup>();
        geoMap.set(item.geometry, byPipeline);
      }
      let byBindGroup = byPipeline.get(item.pipeline);
      if (!byBindGroup) {
        byBindGroup = new Map<GPUBindGroup | undefined, GeoGroup>();
        byPipeline.set(item.pipeline, byBindGroup);
      }
      let group = byBindGroup.get(item.bindGroup);
      const cnt = instanceCountOf(item);
      if (!group) {
         group = { geoIdx: geoOrder.length, item, count: 0, itemCount: 0 };
        byBindGroup.set(item.bindGroup, group);
        geoOrder.push(group);
      }
       group.count += cnt;
       group.itemCount++;
       totalInstances += cnt;
    }

    const geometryCount = geoOrder.length;

    // 2. 逐实例包围球（与实例数据写入顺序保持一致，在下面同一个循环里填充）。
    const spheresData = new Float32Array(totalInstances * 4);
    const geometryIds = new Uint32Array(totalInstances);

    // 3. 按 geometry 分组装写实例数据（跳过 batcher）+ 同步填充逐实例包围球。
    //    每个 geometry 组按**自己管线的实例跨步**落位（组起始 256 对齐）。
    //
    //    与 submitDirect 同理：实例存储容量必须先于填充确保，
    //    否则首帧（或容量不足的帧）会把零矩阵写进实例缓冲。
    let required = 0;
    for (const group of geoOrder) {
      required = instanceBufferOffset(required) + group.count * (group.item.pipeline.bytesPerInstance || 80);
    }
    this.ensureStore(required);

    let gpuOffset = 0;
    let sphereCursor = 0;
    const batchInfos: { offsetBytes: number; geoIdx: number }[] = [];

    for (const group of geoOrder) {
      const groupGeometry = group.item.geometry;
      const groupPipeline = group.item.pipeline;
      const groupStart = instanceBufferOffset(gpuOffset);
      let instanceOffset = groupStart;
      // 组内所有 item 共享同一管线 → 跨步/额外数据长度取该管线。
      const frameStride = groupPipeline.bytesPerInstance || 80;
      // 实例记录布局：[0, mmBytes) 预留区 → mat4(64B) → 额外数据。
      const mmBytes = groupPipeline.modelMatrixOffset || 0;
      const mmFloat = mmBytes >> 2;
      const extra = frameStride - mmBytes - 64;
      const extraFloats = extra >> 2;

      // 局部 AABB → 中心 + 半径（供无 item.bounding 时推导世界空间包围球）。
      const local = groupGeometry.bounds;
      const lcX = local ? (local.min[0] + local.max[0]) * 0.5 : 0;
      const lcY = local ? (local.min[1] + local.max[1]) * 0.5 : 0;
      const lcZ = local ? (local.min[2] + local.max[2]) * 0.5 : 0;
      const lr = local
        ? Math.hypot(local.max[0] - local.min[0], local.max[1] - local.min[1], local.max[2] - local.min[2]) * 0.5
        : 0;

      // 遍历所有 item，收集属于该 (geometry, pipeline, bindGroup) 的实例。
      const groupBindGroup = group.item.bindGroup;
      for (let i = 0; i < n; i++) {
        const item = items[i] as RenderItem;
        if (
          item.geometry !== groupGeometry ||
          item.pipeline !== groupPipeline ||
          item.bindGroup !== groupBindGroup
        ) continue;
        const cnt = instanceCountOf(item);
        const m0 = item.transforms;
         const extraData = item.instanceData;
         const hasExtra = extraData !== undefined;
         const b = item.bounding;
        for (let m = 0; m < cnt; m++) {
          const base = instanceOffset / 4;
          for (let c = 0; c < mmFloat; c++) this.instanceStore[base + c] = 0;
          const f = base + mmFloat;
          const s = sphereCursor * 4;
          if (m0) {
            const src = m * 16;
            for (let c = 0; c < 16; c++) this.instanceStore[f + c] = m0[src + c] as number;
          } else {
            for (let c = 0; c < 16; c++) this.instanceStore[f + c] = 0;
            this.instanceStore[f] = 1;
            this.instanceStore[f + 5] = 1;
            this.instanceStore[f + 10] = 1;
            this.instanceStore[f + 15] = 1;
          }
           for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = 0;
           if (hasExtra && extraData) {
             const src = m * extraFloats;
             for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = extraData[src + c] as number;
           } else if (extra >= 16) {
             this.instanceStore[f + 16] = 1;
             this.instanceStore[f + 17] = 1;
             this.instanceStore[f + 18] = 1;
             this.instanceStore[f + 19] = 1;
           }

          // 包围球：显式 bounding 优先；否则用局部 AABB × 实例矩阵（世界空间）。
          if (b) {
            spheresData[s] = b.centerX;
            spheresData[s + 1] = b.centerY;
            spheresData[s + 2] = b.centerZ;
            spheresData[s + 3] = b.radius;
          } else if (m0) {
            const src = m * 16;
            const r0 = m0[src] as number, r1 = m0[src + 1] as number, r2 = m0[src + 2] as number;
            const r4 = m0[src + 4] as number, r5 = m0[src + 5] as number, r6 = m0[src + 6] as number;
            const r8 = m0[src + 8] as number, r9 = m0[src + 9] as number, r10 = m0[src + 10] as number;
            spheresData[s] = r0 * lcX + r4 * lcY + r8 * lcZ + (m0[src + 12] as number);
            spheresData[s + 1] = r1 * lcX + r5 * lcY + r9 * lcZ + (m0[src + 13] as number);
            spheresData[s + 2] = r2 * lcX + r6 * lcY + r10 * lcZ + (m0[src + 14] as number);
            const norm1 = Math.max(
              Math.abs(r0) + Math.abs(r4) + Math.abs(r8),
              Math.abs(r1) + Math.abs(r5) + Math.abs(r9),
              Math.abs(r2) + Math.abs(r6) + Math.abs(r10),
            );
            const normInf = Math.max(
              Math.abs(r0) + Math.abs(r1) + Math.abs(r2),
              Math.abs(r4) + Math.abs(r5) + Math.abs(r6),
              Math.abs(r8) + Math.abs(r9) + Math.abs(r10),
            );
            const scale = Math.sqrt(norm1 * normInf);
            spheresData[s + 3] = lr * scale;
            if (!local) {
              // 无几何包围盒信息：保守处理（永不被剔除），保证剔除失败不会造成漏画。
              spheresData[s + 3] = 1e30;
            }
          } else if (local) {
            spheresData[s] = lcX;
            spheresData[s + 1] = lcY;
            spheresData[s + 2] = lcZ;
            spheresData[s + 3] = lr;
          } else {
            spheresData[s] = 0;
            spheresData[s + 1] = 0;
            spheresData[s + 2] = 0;
            spheresData[s + 3] = 1e30;
          }
          geometryIds[sphereCursor] = group.geoIdx;
          sphereCursor++;
          instanceOffset += frameStride;
        }
      }
      gpuOffset = instanceOffset;
      batchInfos.push({ offsetBytes: groupStart, geoIdx: group.geoIdx });
    }

    // 5. 上传实例数据到 ring buffer（256 对齐：storage dynamic offset）。
    const bytesToWrite = gpuOffset;
    this.ensureStore(bytesToWrite); // 冗余保险：容量已在填充前确保。
    const maxBatchBytes = maxGroupBytesOf(geoOrder);
    const base = this.ring.alloc(bytesToWrite + maxBatchBytes, RING_ALIGN);
    try {
      this.ring.write(base, this.instanceStore, 0, bytesToWrite);

    // 6. GPU 视锥剔除：compute shader 原子填充 instanceCount + compaction mapping。
    //
    //    draw args 的 CPU 字段（indexCount / firstIndex / baseVertex / firstInstance）
    //    必须作为**模板随 dispatch 一起写入**，不能在 dispatch 之后单独 writeBuffer：
    //    queue.writeBuffer 按入队顺序执行，后写会把 GPU 刚原子填充的 instanceCount 覆盖回 0
    //    （drawIndexedIndirect 于是画 0 个实例 → 整帧全黑，且无任何校验错误）。
    const argsU32 = new Uint32Array(geometryCount * 5);
    for (const info of batchInfos) {
      const item = geoOrder[info.geoIdx]!.item;
      const geo = item.geometry;
       const base32 = info.geoIdx * 5;
       argsU32[base32 + 0] = geo.indexSlice ? geo.indexCount : geo.vertexCount;
      // [base32 + 1] instanceCount 由 compute shader 原子填充
      argsU32[base32 + 2] = 0; // firstIndex
      argsU32[base32 + 3] = 0; // baseVertex
      argsU32[base32 + 4] = 0; // firstInstance
    }

     if (!this._culling) this._culling = new CullingPipeline(this.device);
    const encoder = this.device.createCommandEncoder({ label: `${this.label}:frame-culled` });
    const {
      drawArgsBuffer,
      compactedIndicesBuffer,
      drawArgsCount,
      slotBases,
      candidateCounts,
      compactedSlotCount,
      maxSlotBytes,
    } = this._culling.cull(
      vpMatrix,
      spheresData,
      geometryIds,
      geometryCount,
       argsU32,
       // 不传 timestamps：compute pass 与 render pass 共用同一个 query set，
       // 两边都写会互相覆盖（详见 TimestampQuery 的槽位约定）。
       undefined,
       encoder,
     );

    // 8. compaction bind group：[instanceBuffer + compactedIndicesBuffer]（按 buffer 身份与尺寸缓存复用）。
    const compactionBindGroup = this.ensureCompactionBindGroup(
      compactedIndicesBuffer,
      maxBatchBytes,
      maxSlotBytes,
    );

    // 9. 录制 render pass + indirect draw。
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      label: `${this.label}:main-culled`,
      colorAttachments: [
        {
          view,
          clearValue: { r: this.clearColor[0], g: this.clearColor[1], b: this.clearColor[2], a: this.clearColor[3] },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.ensureDepthView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      ...(timestamps ? { timestampWrites: timestamps.timestampWrites(0, 1) } : {}),
    });



    const pipelines = this._pipelineSet;
    pipelines.clear();
    for (const group of geoOrder) pipelines.add(group.item.pipeline.id);

    // 逐 geometry 设置管线 + 绑定 + indirect draw。
    let lastPipeline: ResolvedPipeline | null = null;
    let lastMaterial: GPUBindGroup | undefined;
    for (const info of batchInfos) {
      const item = geoOrder[info.geoIdx]!.item;
      const pipeline = item.pipeline;
      const geometry = item.geometry;

      if (pipeline !== lastPipeline) {
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, this.globalBindGroup(pipeline));
        lastPipeline = pipeline;
        lastMaterial = undefined;
      }

      // 材质 bind group（group=2）—— 每个 geometry 组一个（已在分组键里）。
      if (item.bindGroup && item.bindGroup !== lastMaterial) {
        pass.setBindGroup(2, item.bindGroup);
        lastMaterial = item.bindGroup;
      }

      // Culling path: group(1) = [instanceBuffer, compactedIndicesBuffer]，
      // 两个 binding 都带 dynamic offset（均需 256 对齐）：
      //   binding 0 → 该 geometry 的实例区
      //   binding 1 → 该 geometry 的 compaction slot 区
      // 两者必须逐 geometry 重绑（同一个 bind group + 不同 offset）。
      pass.setBindGroup(1, compactionBindGroup, [
        instanceBufferOffset(base + info.offsetBytes),
        (slotBases[info.geoIdx] as number) * 4,
      ]);

      for (let s = 0; s < geometry.vertexBuffers.length; s++) {
        const vb = geometry.vertexBuffers[s]!;
        pass.setVertexBuffer(s, vb.buffer, vb.byteOffset, vb.byteLength);
      }
      if (geometry.indexSlice && geometry.indexBuffer) {
        pass.setIndexBuffer(geometry.indexBuffer, geometry.indexFormat, geometry.indexSlice.byteOffset, geometry.indexSlice.byteLength);
        pass.drawIndexedIndirect(drawArgsBuffer, info.geoIdx * 20);
      } else {
        pass.drawIndirect(drawArgsBuffer, info.geoIdx * 20);
      }

       stats.drawCalls++;
       stats.itemsDrawn += (geoOrder[info.geoIdx] as GeoGroup).itemCount;
    }

    stats.instances = totalInstances;
    stats.batches = geometryCount;
    stats.pipelinesUsed = pipelines.size;

    pass.end();
    if (timestamps) timestamps.resolve(encoder);

      this.device.queue.submit([encoder.finish()]);
    const snapshot = this._cullingSnapshot;
    if (snapshot) {
      snapshot.drawArgsBuffer = drawArgsBuffer;
      snapshot.drawArgsCount = drawArgsCount;
      snapshot.compactedIndicesBuffer = compactedIndicesBuffer;
      snapshot.compactedIndicesByteLength = compactedIndicesBuffer.size;
      snapshot.compactedSlotCount = compactedSlotCount;
      snapshot.slotBases = slotBases;
      snapshot.candidateCounts = candidateCounts;
      snapshot.totalCandidateInstances = totalInstances;
    } else {
      this._cullingSnapshot = {
        drawArgsBuffer,
        drawArgsCount,
        compactedIndicesBuffer,
        compactedIndicesByteLength: compactedIndicesBuffer.size,
        compactedSlotCount,
        slotBases,
        candidateCounts,
        totalCandidateInstances: totalInstances,
      };
    }
      return stats;
    } finally {
       this.ring.endFrame();
     }
   }

  async readCullingDebug(device: GPUDevice): Promise<CullingDebugResult> {
    this.assertUsable('readCullingDebug');
    if (!device || device !== this.device) {
      throw new Error('[hpg] readCullingDebug() requires the device used to create this Renderer.');
    }
    const snapshot = this._cullingSnapshot;
    if (!snapshot) {
      throw new Error('[hpg] readCullingDebug() requires a prior successful submitCulled() call.');
    }

    const drawArgsBuffer = snapshot.drawArgsBuffer;
    const drawArgsCount = snapshot.drawArgsCount;
    const compactedIndicesBuffer = snapshot.compactedIndicesBuffer;
    const compactedByteLength = snapshot.compactedIndicesByteLength;
    const compactedSlotCount = snapshot.compactedSlotCount;
    const slotBases = snapshot.slotBases;
    const candidateCounts = snapshot.candidateCounts;
    const totalCandidateInstances = snapshot.totalCandidateInstances;
    const drawArgsByteLength = drawArgsCount * 20;
    const totalByteLength = drawArgsByteLength + compactedByteLength;
    let stagingBuffer: GPUBuffer | null = null;
    let mapped = false;
    try {
      stagingBuffer = device.createBuffer({
        label: `${this.label}:culling-debug-staging`,
        size: Math.max(totalByteLength, 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder({ label: `${this.label}:culling-debug-copy` });
      if (drawArgsByteLength > 0) {
        encoder.copyBufferToBuffer(drawArgsBuffer, 0, stagingBuffer, 0, drawArgsByteLength);
      }
      if (compactedByteLength > 0) {
        encoder.copyBufferToBuffer(
          compactedIndicesBuffer,
          0,
          stagingBuffer,
          drawArgsByteLength,
          compactedByteLength,
        );
      }
      device.queue.submit([encoder.finish()]);
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;
      const mappedRange = stagingBuffer.getMappedRange();
      const drawArgs = new Uint32Array(mappedRange.slice(0, drawArgsByteLength));
      const compactedIndices = new Uint32Array(
        mappedRange.slice(drawArgsByteLength, drawArgsByteLength + compactedByteLength),
      );
      return analyzeCullingDebug(
        drawArgs,
        compactedIndices,
        drawArgsCount,
        slotBases,
        candidateCounts,
        compactedSlotCount,
        totalCandidateInstances,
      );
    } finally {
      if (stagingBuffer) {
        if (mapped) {
          try {
            stagingBuffer.unmap();
          } catch {}
        }
        try {
          stagingBuffer.destroy();
        } catch {}
      }
    }
  }

  private assemble(batches: Batch[], totalBytes: number): void {
    this.ensureStore(totalBytes);

    let write = 0; // 字节游标（= batch.offsetBytes + 偏移）
    for (const b of batches) {
      // 批内所有 item 共享同一管线 → 跨步与额外数据长度均取该管线。
      const frameStride = b.pipeline.bytesPerInstance || 80;
      // 实例记录布局：[0, mmBytes) 预留区 → mat4(64B) → 额外数据（实例颜色等）。
      const mmBytes = b.pipeline.modelMatrixOffset || 0;
      const mmFloat = mmBytes >> 2;
      const extra = frameStride - mmBytes - 64;
      const extraFloats = extra >> 2;
      write = b.offsetBytes;
      for (let k = b.itemStart; k < b.itemEnd; k++) {
        const item = this.sortedItems[k] as RenderItem;
        const cnt = instanceCountOf(item);
        const m0 = item.transforms;
         const extraData = item.instanceData;
         const hasExtra = extraData !== undefined;
         for (let m = 0; m < cnt; m++) {
          const base = write / 4;
          // 预留区显式清零：同一条 ring 区间会跨帧复用，不清会残留上一帧数据。
          for (let c = 0; c < mmFloat; c++) this.instanceStore[base + c] = 0;
          const f = base + mmFloat;
          if (m0) {
            const src = m * 16;
            for (let c = 0; c < 16; c++) this.instanceStore[f + c] = m0[src + c] as number;
          } else {
            // 单位矩阵。
            for (let c = 0; c < 16; c++) this.instanceStore[f + c] = 0;
            this.instanceStore[f] = 1;
            this.instanceStore[f + 5] = 1;
            this.instanceStore[f + 10] = 1;
            this.instanceStore[f + 15] = 1;
          }
           for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = 0;
           if (hasExtra && extraData) {
             const src = m * extraFloats;
             for (let c = 0; c < extraFloats; c++) this.instanceStore[f + 16 + c] = extraData[src + c] as number;
           } else if (extra >= 16) {
             this.instanceStore[f + 16] = 1;
             this.instanceStore[f + 17] = 1;
             this.instanceStore[f + 18] = 1;
             this.instanceStore[f + 19] = 1;
           }
          write += frameStride;
        }
      }
    }
  }

  private ensureScratch(n: number): void {
    this.sortedItems.length = n;
    if (this.orderScratch.length >= n) return;
    this.orderScratch = new Int32Array(n);
    this.keysScratch = new Array(n);
    this.indexScratch = new Array(n);
  }

  private ensureStore(bytes: number): void {
    const floats = Math.ceil(bytes / 4);
    if (this.instanceStore.length >= floats) return;
    this.instanceStore = new Float32Array(Math.max(floats, 1 << 14));
  }

  private depthTexture: GPUTexture | null = null;
  private depthWidth = 0;
  private depthHeight = 0;

  private ensureDepthView(): GPUTextureView {
    const w = this.context.canvas.width;
    const h = this.context.canvas.height;
    if (this.depthTexture && this.depthWidth === w && this.depthHeight === h) {
      return this.depthTexture.createView();
    }
    if (this.depthTexture) {
      this.depthTexture.destroy();
    }
    this.depthTexture = this.device.createTexture({
      label: `${this.label}:depth`,
      size: { width: w, height: h, depthOrArrayLayers: 1 },
      format: this.depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthWidth = w;
    this.depthHeight = h;
    return this.depthTexture.createView();
  }

  /** 销毁 Renderer 持有的所有 GPU 资源。调用后不应再使用该 Renderer。 */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    if (this.depthTexture) {
      this.depthTexture.destroy();
      this.depthTexture = null;
    }
    this._culling?.dispose();
    this._cullingSnapshot = null;
    this.arena.dispose();
    this.ring.dispose();
     this.compactionBG = null;
     this.instanceBG = null;
     this.globalBGs.clear();
     this.cache.clear();
   }
}

function analyzeCullingDebug(
  drawArgs: Uint32Array,
  compactedIndices: Uint32Array,
  drawArgsCount: number,
  slotBases: Uint32Array,
  candidateCounts: Uint32Array,
  compactedSlotCount: number,
  totalCandidateInstances: number,
): CullingDebugResult {
  const visibleInstances = new Uint32Array(drawArgsCount);
  const mappedInstances = new Uint32Array(drawArgsCount);
  const outOfRangeIndices: number[] = [];
  const duplicateIndices: number[] = [];
  let mappingValid = drawArgs.length === drawArgsCount * 5;
  let candidateTotal = 0;

  for (let g = 0; g < drawArgsCount; g++) {
    const slotBase = slotBases[g];
    const nextSlotBase = g + 1 < drawArgsCount ? slotBases[g + 1] : compactedSlotCount;
    const candidateCount = candidateCounts[g] ?? 0;
    const instanceCount = drawArgs[g * 5 + 1] ?? 0;
    candidateTotal += candidateCount;
    visibleInstances[g] = instanceCount;

    if (
      slotBase === undefined ||
      nextSlotBase === undefined ||
      slotBase > nextSlotBase ||
      nextSlotBase > compactedSlotCount ||
      compactedSlotCount > compactedIndices.length
    ) {
      mappingValid = false;
      continue;
    }

    const availableSlots = nextSlotBase - slotBase;
    if (instanceCount > availableSlots || instanceCount > candidateCount) mappingValid = false;
    const seen = new Set<number>();
    const scanCount = Math.min(instanceCount, availableSlots);
    for (let slot = 0; slot < scanCount; slot++) {
      const index = compactedIndices[slotBase + slot] as number;
      if (index >= candidateCount) {
        outOfRangeIndices.push(index);
        continue;
      }
      if (seen.has(index)) {
        duplicateIndices.push(index);
      } else {
        seen.add(index);
      }
      mappedInstances[g]++;
    }
    if (mappedInstances[g] !== instanceCount) mappingValid = false;
  }

  if (candidateTotal !== totalCandidateInstances) mappingValid = false;
  if (outOfRangeIndices.length > 0 || duplicateIndices.length > 0) mappingValid = false;
  return {
    drawArgs,
    compactedIndices,
    visibleInstances,
    mappedInstances,
    outOfRangeIndices: Uint32Array.from(outOfRangeIndices),
    duplicateIndices: Uint32Array.from(duplicateIndices),
    mappingValid,
  };
}

/** 夹到 [0, 1]。 */
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** `array<InstanceData>`（mat4 + vec4）的 minBindingSize；绑定尺寸不得低于该值。 */
const MIN_INSTANCE_BINDING_SIZE = 80;

/**
 * 实例 bind group 的绑定尺寸 = 本帧最大批的实例字节数。
 *
 * 该值同时也是调用方必须为 ring 预留的余量（见 ensureInstanceBindGroup）。
 */
function maxBatchBytesOf(batches: Batch[]): number {
  let max = 0;
  for (const b of batches) {
    const bytes = b.instanceCount * (b.pipeline.bytesPerInstance || MIN_INSTANCE_BINDING_SIZE);
    if (bytes > max) max = bytes;
  }
  return Math.max(max, MIN_INSTANCE_BINDING_SIZE);
}

/** submitCulled：本帧最大 geometry 组的实例字节数（= compaction bind group 的实例绑定尺寸）。 */
function maxGroupBytesOf(groups: { item: RenderItem; count: number }[]): number {
  let max = 0;
  for (const g of groups) {
    const bytes = g.count * (g.item.pipeline.bytesPerInstance || MIN_INSTANCE_BINDING_SIZE);
    if (bytes > max) max = bytes;
  }
  return Math.max(max, MIN_INSTANCE_BINDING_SIZE);
}

/**
 * 归一化排序深度（0 = 最近）。
 *
 * 优先使用 `item.depth`；否则仅在提供相机位置时按包围球中心到相机的距离归一化；
 * 否则返回 0（计数排序是稳定的，等价于保持提交顺序）。
 *
 * 注意：`geometry.bounds` 是**局部空间**的，必须用实例矩阵变换到世界空间后再量距离；
 * 多实例 item 取离相机最近的那个实例作为排序基准（与 early-z 的目标一致）。
 */
function resolveDepth01(
  item: RenderItem,
  camera: [number, number, number] | undefined,
  near: number,
  far: number,
): number {
  if (item.depth !== undefined) return clamp01(item.depth);
  if (!camera) return 0;

  const b = item.bounding;
  let cx: number, cy: number, cz: number;
  if (b) {
    cx = b.centerX;
    cy = b.centerY;
    cz = b.centerZ;
  } else {
    const bounds = item.geometry.bounds;
    if (!bounds) return 0;
    cx = (bounds.min[0] + bounds.max[0]) * 0.5;
    cy = (bounds.min[1] + bounds.max[1]) * 0.5;
    cz = (bounds.min[2] + bounds.max[2]) * 0.5;
  }

  const t = item.transforms;
  let d: number;
  if (t && t.length >= 16) {
    // 局部中心 × 实例矩阵 → 世界空间（列主序：m[col*4 + row]）。
    const camX = camera[0], camY = camera[1], camZ = camera[2];
    const count = t.length >> 4;
    d = Infinity;
    for (let i = 0; i < count; i++) {
      const o = i * 16;
      const wx = (t[o] as number) * cx + (t[o + 4] as number) * cy + (t[o + 8] as number) * cz + (t[o + 12] as number);
      const wy = (t[o + 1] as number) * cx + (t[o + 5] as number) * cy + (t[o + 9] as number) * cz + (t[o + 13] as number);
      const wz = (t[o + 2] as number) * cx + (t[o + 6] as number) * cy + (t[o + 10] as number) * cz + (t[o + 14] as number);
      const di = Math.hypot(wx - camX, wy - camY, wz - camZ);
      if (di < d) d = di;
    }
    if (!Number.isFinite(d)) d = 0;
  } else {
    d = Math.hypot(cx - camera[0], cy - camera[1], cz - camera[2]);
  }

  const span = far - near;
  if (!(span > 0)) return 0;
  return clamp01((d - near) / span);
}

/** 便捷：全局 uniform buffer 布局帮助函数。 */
export function uniformBindGroupLayout(
  device: GPUDevice,
  entries: { binding: number; visibility: number }[],
): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: entries.map((e) => ({
      binding: e.binding,
      visibility: e.visibility,
      buffer: { type: 'uniform' },
    })),
  });
}

/** 便捷：把 global bindings 与对应 layout 封装。 */
export function createGlobalBindGroup(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  bindings: GlobalBinding[],
): GPUBindGroup {
  assertGlobalBindings(device, bindings);
  return device.createBindGroup({
    layout,
    entries: bindings.map((b) => ({ binding: b.binding, resource: globalBindingResource(b) })),
  });
}

/**
 * `GlobalBinding` → `GPUBufferBinding`。
 *
 * `byteOffset` / `byteLength` 必须真正生效：只绑整块 buffer 会让「同一 buffer 内的多个
 * uniform 切片」（如 uniform 数组）静默读到错误数据。
 */
function assertGlobalBindings(device: GPUDevice, bindings: GlobalBinding[]): void {
  const seen = new Set<number>();
  for (const binding of bindings) {
    if (!Number.isSafeInteger(binding.binding) || binding.binding < 0 || seen.has(binding.binding)) {
      throw new Error('GlobalBinding.binding must be a unique non-negative integer.');
    }
    seen.add(binding.binding);
    const offset = binding.byteOffset ?? 0;
    const size = binding.byteLength ?? binding.buffer.size - offset;
    const alignment = device.limits?.minUniformBufferOffsetAlignment ?? 256;
    if (!Number.isSafeInteger(offset) || offset < 0 || offset % alignment !== 0) {
      throw new Error(`GlobalBinding byteOffset ${offset} must be aligned to ${alignment}.`);
    }
    if (!Number.isSafeInteger(size) || size <= 0 || size % 4 !== 0 || offset + size > binding.buffer.size) {
      throw new Error('GlobalBinding byteOffset and byteLength must be 4-byte aligned and within buffer bounds.');
    }
    const usage = binding.buffer.usage;
    if ((usage & GPUBufferUsage.UNIFORM) === 0) {
      throw new Error('GlobalBinding requires a buffer with UNIFORM usage.');
    }
  }
}

function globalBindingResource(b: GlobalBinding): GPUBufferBinding | GPUBuffer {
  if (b.byteOffset === undefined && b.byteLength === undefined) return b.buffer;
  const resource: GPUBufferBinding = { buffer: b.buffer };
  if (b.byteOffset !== undefined) resource.offset = b.byteOffset;
  if (b.byteLength !== undefined) resource.size = b.byteLength;
  return resource;
}