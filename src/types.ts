/// <reference types="@webgpu/types" />

/**
 * 数据模型 —— 渲染意图包（Draw Packet）。
 * 上层框架（ECS / SceneGraph / 响应式状态）只需每帧摊平 `RenderItem[]` 交给渲染器。
 */

/** 包装一个显存切片（buffer + byte 范围）。 */
export interface BufferSlice {
  buffer: GPUBuffer;
  byteOffset: number;
  byteLength: number;
}

/** 顶点布局：WebGPU vertex buffer layout 的两段式描述。 */
export interface VertexAttributeDesc {
  shaderLocation: number;
  offset: number;
  format: GPUVertexFormat;
}

export interface VertexLayoutDesc {
  arrayStride: number;
  stepMode: 'vertex' | 'instance';
  attributes: VertexAttributeDesc[];
}

/** 轴对齐包围盒（AABB）。 */
export interface Aabb {
  min: [number, number, number];
  max: [number, number, number];
}

/**
 * 静态几何体：持有已上传的顶点/索引 Buffer 与绘制描述。
 * 通过 GeometryArena 注册，得到的实例可作为资源句柄。
 *
 * 几何描述保留 vertexBuffers[] slot 形状；当前 GeometryArena 上传 API 只接受一个 vertex-step layout。
 * vertexBuffer / vertexSlice 始终指向 vertexBuffers[0]（向后兼容）。
 */
export interface Geometry {
  vertexBuffer: GPUBuffer;
  vertexSlice: BufferSlice;
  /** 多顶点 buffer slots（slot 0 = vertexBuffer）。单 buffer 几何体长度为 1。 */
  vertexBuffers: BufferSlice[];
  vertexLayouts: VertexLayoutDesc[];
  indexBuffer?: GPUBuffer;
  indexSlice?: BufferSlice;
  indexFormat: GPUIndexFormat;
  indexCount: number;
  vertexCount: number;
  primitive: GPUPrimitiveTopology;
  /**
   * 局部空间 AABB（由 GeometryArena 从 position 属性推导）。
   * submitCulled() 用它推导逐实例包围球；缺失时该几何体保守地不做剔除。
   */
  bounds?: Aabb;
}

/** 绑定组常量绑定（如全局 uniform），render 时固定注入 group=0。 */
export interface GlobalBinding {
  binding: number;
  buffer: GPUBuffer;
  /** uniform buffer 的字节偏移。 */
  byteOffset?: number;
  byteLength?: number;
}

/** 管线：以内容（着色器 + 布局）为哈希键的不可变描述。 */
export interface PipelineDesc {
  label?: string;
  vsCode: string;
  fsCode: string;
  /**
   * 每实例数据字节跨步，默认 80。
   *
   * **必须等于该管线 WGSL 中 `InstanceData` 的 sizeof** —— 着色器总是按自己的
   * 结构体跨步索引实例（`instances[instance_index]`），跨步不一致会读到错位数据。
   * 库内置的 VS_INSTANCED / VS_INSTANCED_COMPACTION 固定为
   * `mat4x4<f32> + vec4<f32>` = 80 字节；内置 shader 只能使用 80/0 布局，
   * 自定义结构体必须同步声明此值。
   */
  bytesPerInstance?: number;
  /** 实例数据中 modelMatrix 的字节偏移（列主序）。默认 0。 */
  modelMatrixOffset?: number;
  /**
   * 标记为 GPU 剔除（compaction）管线：group=1 使用
   * [instanceBuffer(dynamic offset), compactedIndices] 双绑定布局。
   *
   * 顶点着色器必须通过 `compactedIndices[instance_index]` 间接索引实例数据
   * （例如 src/shaders/instance.ts 的 VS_INSTANCED_COMPACTION），只能配合
   * Renderer.submitCulled() 使用。默认 false（普通实例绑定）。
   */
  compaction?: boolean;
  /**
   * 自定义 compaction 顶点着色器的受信契约标记。内置两个 compaction shader
   * 不需要此字段；自定义 WGSL 必须显式声明 runtime 已验证其消费 mapping。
   */
  compactionContract?: 'hpg-compaction-v1';
  vertexLayouts: VertexLayoutDesc[];
  bindGroupLayouts: GPUBindGroupLayout[];
  globalBindings: GlobalBinding[];
  depth?: Partial<GPUDepthStencilState>;
  targets: GPUColorTargetState[];
  primitive?: Partial<GPUPrimitiveState>;
}

/** 已解析（可被缓存池命中）的管线。 */
export interface ResolvedPipeline {
  id: number;
  device?: GPUDevice;
  desc: PipelineDesc;
  pipeline: GPURenderPipeline;
  layout: GPUBindGroupLayout;
  bindGroupLayouts: GPUBindGroupLayout[];
  bytesPerInstance: number;
  modelMatrixOffset: number;
  /** 统计用友好名。 */
  label: string;
}

/** 空间包围体：声明后才有资格进入 GPU 剔除 / 深度排序。 */
export interface BoundingSphere {
  centerX: number;
  centerY: number;
  centerZ: number;
  radius: number;
}

/** 渲染意图包 —— 上层唯一需要理解的最小工作单元。 */
export interface RenderItem {
  /** 静态几何（GeometryArena 注册的引用）。 */
  geometry: Geometry;
  /** 管线句柄（内容哈希缓存命中）。 */
  pipeline: ResolvedPipeline;
  /** 管线声明之外的额外绑定组；管线声明 group 2 时必须提供。 */
  bindGroup?: GPUBindGroup;
  /**
   * 一个或多个列主序 4x4 变换矩阵。
   * 缺省：单位矩阵 ×1。长度必须为 16 的倍数。
   */
  transforms?: Float32Array;
  /** 实例个数 = transforms.length / 16，缺省自动推导。 */
  instanceCount?: number;
  /**
   * 每实例 modelMatrix 之后的扁平数据。提供时长度必须精确等于
   * instanceCount × (bytesPerInstance - modelMatrixOffset - 64) / 4。
   */
  instanceData?: Float32Array;
  /** 声明后才有资格进入后续 GPU 剔除。 */
  bounding?: BoundingSphere;
  /** 覆盖默认排序深度。 */
  depth?: number;
}

/** 逐帧统计（零分配累积）。 */
export interface RenderStats {
  itemsSubmitted: number;
  itemsDrawn: number;
  instances: number;
  drawCalls: number;
  batches: number;
  pipelinesUsed: number;
}