/**
 * 记录型 GPU 替身 —— 用于无需真实 GPU 的集成测试。
 * 记录 createBuffer / createCommandEncoder / draw 调用，供断言。
 */

export interface FakeDrawCall {
  indexCount: number;
  instanceCount: number;
  vertexOffset: number;
  pipeline: unknown;
}

export interface FakeBindGroupRecord {
  label: string;
  bindings: {
    binding: number;
    buffer: unknown;
    /** 静态绑定偏移（GPUBufferBinding.offset）。 */
    offset: number;
    /** 声明的绑定尺寸（未声明时 = bufferSize - offset）。 */
    size: number | null;
    /** 该 binding 是否要求 dynamic offset（取自 bind group layout）。 */
    dynamic: boolean;
  }[];
}

export interface FakeIndirectDraw {
  buffer: unknown;
  offset: number;
  indexed: boolean;
}

export interface FakeRecorded {
  buffers: { size: number }[];
  drawCalls: FakeDrawCall[];
  writes: { offset: number; bytes: Uint8Array; buffer?: unknown }[];
  pipelinesCreated: number;
  passCount: number;
  /** 每次 createBindGroup 的 buffer 绑定（用于验证 ring 扩容后是否重建）。 */
  bindGroups: FakeBindGroupRecord[];
  /** drawIndexedIndirect / drawIndirect 调用。 */
  indirectDraws: FakeIndirectDraw[];
  /** setIndexBuffer 调用次数。 */
  indexBufferBinds: number;
  /** render pass 内的 setBindGroup(group, bindGroup, dynamicOffsets)。 */
  renderBinds: { group: number; offsets: number[]; bindGroup?: unknown }[];
  /** compute dispatch 次数。 */
  dispatches: number;
  /**
   * 模拟 Dawn 的 dynamic offset 越界校验：
   * `bindingOffset + dynamicOffset + bindingSize <= bufferSize`。
   * 违反时报错字符串入此数组 —— 真实浏览器里这会直接让整个 render pass 失效（整帧无输出），
   * 所以测试应断言本数组为空。
   */
  gpuErrors: string[];
  /**
   * 按入队顺序记录的 GPU 操作（writeBuffer / compute dispatch）。
   * 用于断言顺序敏感的不变量，例如 draw args 的 CPU 预写必须早于 dispatch。
   */
  ops: { kind: 'write' | 'dispatch'; buffer?: unknown; bytes?: number }[];
  /** createTexture 调用（材质路径：格式 / 尺寸 / usage）。 */
  textures: { label: string; format: string; width: number; height: number; usage: number }[];
  /** queue.writeTexture 调用（贴图上传）。 */
  textureWrites: { width: number; height: number; byteLength: number }[];
  /** createSampler 调用（材质路径：过滤 / 寻址模式）。 */
  samplers: { magFilter?: string; minFilter?: string; addressModeU?: string; addressModeV?: string }[];
}

export function createFakeGPU(): { device: GPUDevice; context: GPUCanvasContext; recorded: FakeRecorded } {
  const recorded: FakeRecorded = {
    buffers: [],
    drawCalls: [],
    writes: [],
    pipelinesCreated: 0,
    passCount: 0,
    bindGroups: [],
    indirectDraws: [],
    indexBufferBinds: 0,
    dispatches: 0,
    renderBinds: [],
    gpuErrors: [],
    ops: [],
    textures: [],
    textureWrites: [],
    samplers: [],
  };

  // bind group → 记录，用于在 setBindGroup 时校验 dynamic offset 范围。
  const bindGroupRecords = new WeakMap<object, FakeBindGroupRecord>();
  // bind group layout 对象 → 各 binding 是否带 dynamic offset。
  const layoutDynamic = new WeakMap<object, Map<number, boolean>>();

  const fakePass = {
    setPipeline: () => undefined,
    setBindGroup: (group: number, bg: GPUBindGroup, offsets?: number[]) => {
      recorded.renderBinds.push({ group, offsets: offsets ? [...offsets] : [], bindGroup: bg });
      validateDynamicOffsets(bg, offsets);
    },
    setVertexBuffer: () => undefined,
    setIndexBuffer: () => { recorded.indexBufferBinds++; },
    draw: () => undefined,
    drawIndexed: (indexCount: number, instanceCount: number, _v0: number, _v1: number) => {
      recorded.drawCalls.push({ indexCount, instanceCount, vertexOffset: _v0, pipeline: {} });
    },
    drawIndirect: (buffer: GPUBuffer, offset: number) => {
      recorded.indirectDraws.push({ buffer, offset, indexed: false });
    },
    drawIndexedIndirect: (buffer: GPUBuffer, offset: number) => {
      recorded.indirectDraws.push({ buffer, offset, indexed: true });
    },
    writeTimestamp: () => undefined,
    end: () => undefined,
  } as unknown as GPURenderPassEncoder;

  const fakeComputePass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: () => {
      recorded.dispatches++;
      recorded.ops.push({ kind: 'dispatch' });
    },
    writeTimestamp: () => undefined,
    end: () => undefined,
  } as unknown as GPUComputePassEncoder;

  const fakeEncoder = {
    beginRenderPass: () => { recorded.passCount++; return fakePass; },
    beginComputePass: () => fakeComputePass,
    finish: () => ({}) as unknown as GPUCommandBuffer,
    resolveQuerySet: () => undefined,
    copyBufferToBuffer: () => undefined,
  } as unknown as GPUCommandEncoder;

  /**
   * 模拟 Dawn 的 dynamic offset 越界校验。
   *
   * 参考错误（真实 Chrome）：
   *   Dynamic Offset[0] (256) is out of bounds of [Buffer "hpg:ring"] with a size of 1048576
   *   and a bound range of (offset: 0, size: 1048576). Did you forget to specify the binding's size?
   */
  const validateDynamicOffsets = (bg: GPUBindGroup, offsets?: number[]): void => {
    if (!offsets || offsets.length === 0 || !bg) return;
    const rec = bindGroupRecords.get(bg as unknown as object);
    if (!rec) return;
    let dynIndex = 0;
    for (const e of rec.bindings) {
      if (!e.dynamic) continue;
      const dyn = offsets[dynIndex] ?? 0;
      dynIndex++;
      const bufferSize = (e.buffer as { size?: number } | undefined)?.size ?? 0;
      const bindingSize = e.size ?? bufferSize - e.offset;
      if (e.offset + dyn + bindingSize > bufferSize) {
        recorded.gpuErrors.push(
          `Dynamic Offset[${dynIndex - 1}] (${dyn}) is out of bounds of buffer ` +
            `(size: ${bufferSize}, bound range: offset ${e.offset}, size ${bindingSize}).`,
        );
      }
    }
  };

  const device = {
    createBuffer(desc: GPUBufferDescriptor) {
      recorded.buffers.push({ size: desc.size });
      return { size: desc.size, usage: desc.usage, destroy() {} } as unknown as GPUBuffer;
    },
    createBindGroupLayout: (desc: GPUBindGroupLayoutDescriptor) => {
      const layout = {} as unknown as GPUBindGroupLayout;
      const dyn = new Map<number, boolean>();
      for (const e of desc.entries ?? []) {
        dyn.set(e.binding, e.buffer?.hasDynamicOffset ?? false);
      }
      layoutDynamic.set(layout as unknown as object, dyn);
      return layout;
    },
    createPipelineLayout: () => ({}) as unknown as GPUPipelineLayout,
    createShaderModule: () => ({}) as unknown as GPUShaderModule,
    createRenderPipeline: () => {
      recorded.pipelinesCreated++;
      return { getBindGroupLayout: () => ({}) } as unknown as GPURenderPipeline;
    },
    createComputePipeline: () => ({}) as unknown as GPUComputePipeline,
    createBindGroup: (desc: GPUBindGroupDescriptor) => {
      const dyn = layoutDynamic.get(desc.layout as unknown as object);
      const record: FakeBindGroupRecord = {
        label: desc.label ?? '',
        bindings: Array.from(desc.entries ?? []).map((e) => {
          const res = e.resource as GPUBufferBinding | GPUBuffer | undefined;
          const isBinding = !!res && typeof res === 'object' && 'buffer' in (res as GPUBufferBinding);
          return {
            binding: e.binding,
            buffer: isBinding ? (res as GPUBufferBinding).buffer : (res as GPUBuffer | undefined),
            offset: isBinding ? ((res as GPUBufferBinding).offset ?? 0) : 0,
            size: isBinding ? ((res as GPUBufferBinding).size ?? null) : null,
            dynamic: dyn?.get(e.binding) ?? false,
          };
        }),
      };
      recorded.bindGroups.push(record);
      const bg = {} as unknown as GPUBindGroup;
      bindGroupRecords.set(bg as unknown as object, record);
      return bg;
    },
    createCommandEncoder: () => fakeEncoder,
    createTexture: (desc: GPUTextureDescriptor) => {
      const size = desc.size as { width: number; height: number };
      recorded.textures.push({
        label: desc.label ?? '',
        format: desc.format,
        width: size?.width ?? 1,
        height: size?.height ?? 1,
        usage: desc.usage,
      });
      return { createView: () => ({}), destroy() {} } as unknown as GPUTexture;
    },
    createSampler: (desc: GPUSamplerDescriptor) => {
      recorded.samplers.push({
        magFilter: desc.magFilter,
        minFilter: desc.minFilter,
        addressModeU: desc.addressModeU,
        addressModeV: desc.addressModeV,
      });
      return { ...desc } as unknown as GPUSampler;
    },
    createQuerySet: () => ({
      destroy() {},
    }) as unknown as GPUQuerySet,
    queue: {
      submit: () => undefined,
      writeTexture: (
        destination: { texture: GPUTexture },
        data: ArrayBuffer | ArrayBufferView,
        dataLayout: { bytesPerRow?: number; rowsPerImage?: number },
        size: { width: number; height: number },
      ) => {
        void destination;
        void dataLayout;
        recorded.textureWrites.push({
          width: size.width,
          height: size.height,
          byteLength: ArrayBuffer.isView(data) ? data.byteLength : (data as ArrayBuffer).byteLength,
        });
      },
      writeBuffer: (
        buffer: GPUBuffer,
        offset: number,
        data: ArrayBuffer | ArrayBufferView,
        dataOffset?: number,
        len?: number,
      ) => {
        let ab: ArrayBuffer;
        let byteOffset: number;
        let byteLength: number;
        if (ArrayBuffer.isView(data)) {
          ab = data.buffer;
          byteOffset = data.byteOffset;
          byteLength = data.byteLength;
        } else {
          ab = data as ArrayBuffer;
          byteOffset = 0;
          byteLength = ab.byteLength;
        }
        // 兼容 3 参数形式 writeBuffer(buffer, offset, data)（dataOffset/len 缺省）。
        const from = typeof dataOffset === 'number' ? dataOffset : 0;
        const size = typeof len === 'number' ? len : byteLength - from;
        const bytes = new Uint8Array(ab.slice(byteOffset + from, byteOffset + from + size));
        recorded.writes.push({ offset, bytes, buffer });
        recorded.ops.push({ kind: 'write', buffer, bytes: size });
      },
    },
  } as unknown as GPUDevice;

  const context = {
    canvas: { width: 640, height: 480 },
    getCurrentTexture: () => ({ createView: () => ({}) }) as unknown as GPUTexture,
    configure: () => undefined,
    unconfigure: () => undefined,
  } as unknown as GPUCanvasContext;

  return { device, context, recorded };
}