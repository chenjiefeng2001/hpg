/**
 * Minimal glTF 2.0 / GLB loader — 只解析 hpg 所需的几何与材质数据。
 *
 * 不依赖外部 glTF 库，不支持动画 / skin / morph / extras。
 * 支持: POSITION, NORMAL, TEXCOORD_0, TANGENT, indices, PBR metallic roughness。
 *
 * 用法:
 *   const asset = await loadGlb(arrayBuffer);
 *   const items = convertToRenderItems(asset, renderer);
 */

/// <reference types="@webgpu/types" />

// ─── glTF JSON Types ────────────────────────────────────────

interface GltfAccessor {
  bufferView: number;
  byteOffset?: number;
  componentType: number; // 5120=BYTE, 5121=UBYTE, 5122=SHORT, 5123=USHORT, 5125=UINT, 5126=FLOAT
  count: number;
  type: string; // "SCALAR", "VEC2", "VEC3", "VEC4", "MAT4"
  normalized?: boolean;
  min?: number[];
  max?: number[];
}

interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
  target?: number; // 34962=ARRAY_BUFFER, 34963=ELEMENT_ARRAY_BUFFER
}

interface GltfBuffer {
  byteLength: number;
  uri?: string;
}

interface GltfPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
  mode?: number; // 4=TRIANGLES (default)
  targets?: unknown[]; // morph targets
}

interface GltfMesh {
  name?: string;
  primitives: GltfPrimitive[];
}

interface GltfMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: [number, number, number, number];
    metallicFactor?: number;
    roughnessFactor?: number;
    baseColorTexture?: { index: number; texCoord?: number };
    metallicRoughnessTexture?: { index: number; texCoord?: number };
  };
  normalTexture?: { index: number; texCoord?: number; scale?: number };
  occlusionTexture?: { index: number; texCoord?: number; strength?: number };
  emissiveTexture?: { index: number; texCoord?: number };
  emissiveFactor?: [number, number, number];
  alphaMode?: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: Record<string, unknown>;
}

interface GltfTexture {
  source: number;
  sampler?: number;
}

interface GltfImage {
  uri?: string;
  mimeType?: string;
  bufferView?: number;
}

interface GltfSampler {
  magFilter?: number; // 9728=NEAREST, 9729=LINEAR
  minFilter?: number; // 9728/9729/9984..9987
  wrapS?: number; // 33071=CLAMP, 33648=MIRROR, 10497=REPEAT
  wrapT?: number;
}

interface GltfNode {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number];
}

interface GltfScene {
  name?: string;
  nodes: number[];
}

interface GltfRoot {
  asset: { version: string; generator?: string };
  scene?: number;
  scenes?: GltfScene[];
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  materials?: GltfMaterial[];
  textures?: GltfTexture[];
  images?: GltfImage[];
  samplers?: GltfSampler[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
  animations?: unknown[];
  skins?: unknown[];
  extensionsRequired?: string[];
  extensionsUsed?: string[];
}

/** 需要解码/解压才能读几何的扩展 —— 无法忽略，直接给出可读错误。 */
const UNSUPPORTED_EXTENSIONS = [
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
] as const;

/**
 * 可忽略（但会让输出与作者意图不一致）的扩展 → 提示文案。
 *
 * 这些扩展不携带必需数据，忽略后依然能渲染；但结果是「看起来对、其实不对」，
 * 必须显式告知调用方，而不是静默丢弃。
 */
const IGNORED_EXTENSIONS: Record<string, string> = {
  KHR_texture_transform: 'UV 变换（offset/rotation/scale）被忽略，贴图坐标会错位',
  KHR_texture_basisu: 'Basis Universal 压缩贴图未解码',
  EXT_texture_webp: 'WebP 贴图未解码',
  KHR_materials_pbrSpecularGlossiness: 'specular-glossiness 材质退化为标准 metallic-roughness',
  KHR_materials_unlit: 'unlit（无光照）材质按标准 PBR 渲染',
  KHR_materials_emissive_strength: '自发光强度倍率被忽略',
  KHR_materials_variants: '材质变体（variants）未实现',
};

/** `KHR_materials_*` 的可读提示：一律退化为标准 PBR。 */
function extensionHint(ext: string): string | null {
  if (ext === 'KHR_mesh_quantization') return null; // 整数归一化属性已支持
  const known = IGNORED_EXTENSIONS[ext];
  if (known) return known;
  if (ext.startsWith('KHR_materials_')) return '该材质扩展未实现，将退化为标准 metallic-roughness';
  if (ext.startsWith('KHR_texture_') || ext.startsWith('EXT_texture_')) return '该贴图扩展未实现，贴图不会生效';
  if (ext.startsWith('KHR_animation_') || ext.startsWith('KHR_interactivity')) return '运行时特性未实现';
  return null;
}

/** 已实现的扩展（其余会进入 warnings）。 */
const IMPLEMENTED_EXTENSIONS = new Set(['KHR_mesh_quantization']);

// ─── Parsed Asset ───────────────────────────────────────────

export interface AssetMesh {
  name: string;
  primitives: AssetPrimitive[];
}

export interface AssetPrimitive {
  /** 交错顶点数据 (POSITION + NORMAL + TEXCOORD_0 + TANGENT)。 */
  vertices: Float32Array;
  /** 顶点布局描述。 */
  vertexLayout: {
    arrayStride: number;
    attributes: { shaderLocation: number; offset: number; format: GPUVertexFormat }[];
  };
  /** 索引数据 (uint16 或 uint32)。 */
  indices: Uint16Array | Uint32Array;
  indexFormat: GPUIndexFormat;
  materialIndex: number;
}

/**
 * 采样器状态（已折算成 WebGPU 描述）。
 * 未在 glTF 中声明时使用规范默认值：线性过滤 + repeat 寻址。
 */
export interface AssetSampler {
  magFilter: GPUFilterMode;
  minFilter: GPUFilterMode;
  mipmapFilter: GPUMipmapFilterMode;
  addressModeU: GPUAddressMode;
  addressModeV: GPUAddressMode;
}

/**
 * GLB 内嵌图片的**编码字节**（PNG / JPEG …）。
 *
 * 解析器刻意不解码：`createImageBitmap` / canvas 只存在于浏览器，
 * 解码必须由调用方注入的 ImageDecoder 完成（Node 测试 / 资源管线可替换）。
 */
export interface AssetImage {
  name: string;
  mimeType: string;
  data: Uint8Array;
}

/** 材质对某张贴图的引用（image + sampler + uv 通道）。 */
export interface AssetTextureRef {
  imageIndex: number;
  sampler: AssetSampler;
  /** 使用的 UV 通道（0 = TEXCOORD_0）。 */
  texCoord: number;
}

export interface AssetMaterial {
  name: string;
  baseColorFactor: [number, number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  doubleSided: boolean;
  /** base color 贴图（需外部 ImageDecoder 解码后上传；缺失则只用 baseColorFactor）。 */
  baseColorTexture?: AssetTextureRef;
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff: number;
}

export interface AssetNode {
  name: string;
  meshIndex?: number;
  localMatrix: Float32Array;
  children: number[];
}

export interface GltfAsset {
  meshes: AssetMesh[];
  materials: AssetMaterial[];
  /**
   * 内嵌图片（按 glTF image index 存放；无法加载的项在数组中留空）。
   * 只保留编码字节，解码交给注入的 ImageDecoder —— 解析器保持环境无关。
   */
  images: AssetImage[];
  nodes: AssetNode[];
  rootNodes: number[];
  /**
   * 解析期发现的「hpg 未实现」feature（按 feature 去重，经 @see noteFeature 收集）。
   *
   * 真实资产（Blender / glTF-Transform / COLLADA2GLTF 导出）常见 JOINTS_0、
   * baseColorTexture、alphaMode=BLEND 等 —— 这些不会让加载失败，但会让渲染结果
   * 与作者意图不符。调用方应展示此列表（demo 的警告面板即如此），而不是只看控制台日志。
   */
  warnings: string[];
}

// ─── GLB Header Parsing ─────────────────────────────────────

function parseGlb(buffer: ArrayBuffer): { json: GltfRoot; bin: ArrayBuffer } {
  if (buffer.byteLength < 12) {
    throw new Error(`Invalid GLB magic: buffer too small (${buffer.byteLength} bytes). Expected a binary .glb container.`);
  }
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x46546C67) {
    throw new Error(
      `Invalid GLB magic: 0x${magic.toString(16)}. ` +
        `Expected a binary .glb container; text .gltf (with external .bin / .jpg) is not supported.`,
    );
  }
  const version = view.getUint32(4, true);
  if (version !== 2) throw new Error(`Unsupported glTF version: ${version}`);
  const totalLength = view.getUint32(8, true);

  let jsonChunk: GltfRoot | null = null;
  let binChunk: ArrayBuffer | null = null;
  let offset = 12;

  while (offset < totalLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const chunkData = buffer.slice(offset + 8, offset + 8 + chunkLength);

    if (chunkType === 0x4E4F534A) {
      // JSON chunk
      const text = new TextDecoder().decode(chunkData);
      jsonChunk = JSON.parse(text) as GltfRoot;
    } else if (chunkType === 0x004E4942) {
      // BIN chunk
      binChunk = chunkData;
    }

    offset += 8 + chunkLength;
  }

  if (!jsonChunk) throw new Error('No JSON chunk found in GLB');
  return { json: jsonChunk, bin: binChunk! };
}

// ─── Buffer Resolution ──────────────────────────────────────

function accessorComponentSize(componentType: number): number {
  switch (componentType) {
    case 5120: return 1; // BYTE
    case 5121: return 1; // UNSIGNED_BYTE
    case 5122: return 2; // SHORT
    case 5123: return 2; // UNSIGNED_SHORT
    case 5125: return 4; // UNSIGNED_INT
    case 5126: return 4; // FLOAT
    default: throw new Error(`Unknown componentType: ${componentType}`);
  }
}

function accessorTypeCount(type: string): number {
  switch (type) {
    case 'SCALAR': return 1;
    case 'VEC2': return 2;
    case 'VEC3': return 3;
    case 'VEC4': return 4;
    case 'MAT4': return 16;
    default: throw new Error(`Unknown accessor type: ${type}`);
  }
}

// ─── Accessor Reading ───────────────────────────────────────

/** 读取单个分量并做归一化（glTF normalized 语义）。 */
function readComponent(dv: DataView, offset: number, componentType: number, normalized: boolean): number {
  switch (componentType) {
    case 5120: { const v = dv.getInt8(offset); return normalized ? Math.max(v / 127, -1) : v; }
    case 5121: { const v = dv.getUint8(offset); return normalized ? v / 255 : v; }
    case 5122: { const v = dv.getInt16(offset, true); return normalized ? Math.max(v / 32767, -1) : v; }
    case 5123: { const v = dv.getUint16(offset, true); return normalized ? v / 65535 : v; }
    case 5125: return dv.getUint32(offset, true);
    case 5126: return dv.getFloat32(offset, true);
    default: throw new Error(`Unsupported componentType: ${componentType}`);
  }
}

/** accessor → { buffer, baseOffset, stride, count } 公共解析。 */
function accessorLocation(
  accessor: GltfAccessor,
  bufferViews: GltfBufferView[],
  buffers: ArrayBuffer[],
): { buffer: ArrayBuffer; baseOffset: number; stride: number; elementSize: number; typeCount: number } {
  const bv = bufferViews[accessor.bufferView]!;
  const buffer = buffers[bv.buffer]!;
  const compSize = accessorComponentSize(accessor.componentType);
  const typeCount = accessorTypeCount(accessor.type);
  const elementSize = compSize * typeCount;
  return {
    buffer,
    baseOffset: (bv.byteOffset ?? 0) + (accessor.byteOffset ?? 0),
    stride: bv.byteStride && bv.byteStride > 0 ? bv.byteStride : elementSize,
    elementSize,
    typeCount,
  };
}

/**
 * 读取顶点属性为 Float32Array。
 *
 * 支持：交错（byteStride）/ 非对齐 accessor / 整数归一化（KHR_mesh_quantization）。
 * 只有当数据连续、类型为 FLOAT、4 字节对齐且不越界时才走零拷贝视图，
 * 其余情况统一用 DataView 逐分量读取 —— 避免 TypedArray 构造函数的对齐/越界异常。
 */
function readAccessorFloats(
  accessor: GltfAccessor,
  bufferViews: GltfBufferView[],
  buffers: ArrayBuffer[],
): Float32Array {
  const { buffer, baseOffset, stride, elementSize, typeCount } = accessorLocation(accessor, bufferViews, buffers);
  const count = accessor.count;

  const contiguousFloat =
    stride === elementSize &&
    accessor.componentType === 5126 &&
    baseOffset % 4 === 0 &&
    baseOffset + count * elementSize <= buffer.byteLength;
  if (contiguousFloat) {
    return new Float32Array(buffer, baseOffset, count * typeCount);
  }

  const out = new Float32Array(count * typeCount);
  const dv = new DataView(buffer);
  const normalized = accessor.normalized === true;
  for (let i = 0; i < count; i++) {
    const rowBase = baseOffset + i * stride;
    for (let c = 0; c < typeCount; c++) {
      const off = rowBase + c * (elementSize / typeCount);
      if (off + elementSize / typeCount > buffer.byteLength) return out; // 截断保护
      out[i * typeCount + c] = readComponent(dv, off, accessor.componentType, normalized);
    }
  }
  return out;
}

/**
 * 读取索引 accessor。WebGPU 只支持 uint16 / uint32 索引，UBYTE 会被提升为 uint16。
 * 非对齐 / 越界的源数据会先复制到对齐的新数组（glTF 允许，但 TypedArray 构造不允许）。
 */
function readIndices(
  accessor: GltfAccessor,
  bufferViews: GltfBufferView[],
  buffers: ArrayBuffer[],
): { indices: Uint16Array | Uint32Array; format: GPUIndexFormat } {
  const componentType = accessor.componentType;
  if (componentType !== 5121 && componentType !== 5123 && componentType !== 5125) {
    throw new Error(`Unsupported index componentType: ${componentType} (expected 5121/5123/5125)`);
  }
  const { buffer, baseOffset, stride, elementSize } = accessorLocation(accessor, bufferViews, buffers);
  const count = accessor.count;
  const bytes = count * elementSize;
  const aligned = baseOffset % 4 === 0 && baseOffset + bytes <= buffer.byteLength;

  if (aligned) {
    if (componentType === 5125) {
      return { indices: new Uint32Array(buffer, baseOffset, count), format: 'uint32' };
    }
    if (componentType === 5123 && baseOffset % 2 === 0) {
      return { indices: new Uint16Array(buffer, baseOffset, count), format: 'uint16' };
    }
    if (componentType === 5121) {
      const out = new Uint16Array(count);
      out.set(new Uint8Array(buffer, baseOffset, count));
      return { indices: out, format: 'uint16' };
    }
  }

  // 通用（复制）：处理非对齐 accessor。
  const dv = new DataView(buffer);
  if (componentType === 5125) {
    const out = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      const off = baseOffset + i * stride;
      if (off + 4 > buffer.byteLength) break;
      out[i] = dv.getUint32(off, true);
    }
    return { indices: out, format: 'uint32' };
  }
  const out = new Uint16Array(count);
  for (let i = 0; i < count; i++) {
    const off = baseOffset + i * stride;
    if (off + elementSize > buffer.byteLength) break;
    out[i] = componentType === 5121 ? dv.getUint8(off) : dv.getUint16(off, true);
  }
  return { indices: out, format: 'uint16' };
}

// ─── Matrix Helpers ─────────────────────────────────────────

function mat4FromTRS(
  translation?: [number, number, number],
  rotation?: [number, number, number, number],
  scale?: [number, number, number],
): Float32Array {
  const m = new Float32Array(16);
  // Identity
  m[0] = 1; m[5] = 1; m[10] = 1; m[15] = 1;

  const tx = translation?.[0] ?? 0;
  const ty = translation?.[1] ?? 0;
  const tz = translation?.[2] ?? 0;

  const sx = scale?.[0] ?? 1;
  const sy = scale?.[1] ?? 1;
  const sz = scale?.[2] ?? 1;

  if (rotation) {
    const [qx, qy, qz, qw] = rotation;
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;

    m[0] = (1 - 2 * (yy + zz)) * sx;
    m[1] = 2 * (xy + wz) * sx;
    m[2] = 2 * (xz - wy) * sx;
    m[4] = 2 * (xy - wz) * sy;
    m[5] = (1 - 2 * (xx + zz)) * sy;
    m[6] = 2 * (yz + wx) * sy;
    m[8] = 2 * (xz + wy) * sz;
    m[9] = 2 * (yz - wx) * sz;
    m[10] = (1 - 2 * (xx + yy)) * sz;
  } else {
    m[0] = sx;
    m[5] = sy;
    m[10] = sz;
  }

  m[12] = tx;
  m[13] = ty;
  m[14] = tz;

  return m;
}

function mat4FromMatrix(arr: number[]): Float32Array {
  const m = new Float32Array(16);
  for (let i = 0; i < 16; i++) m[i] = arr[i];
  return m;
}

function multiplyMat4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      out[j * 4 + i] =
        a[i] * b[j * 4] +
        a[4 + i] * b[j * 4 + 1] +
        a[8 + i] * b[j * 4 + 2] +
        a[12 + i] * b[j * 4 + 3];
    }
  }
  return out;
}

// ─── Helpers ────────────────────────────────────────────────

const _warned = new Set<string>();

/** 同类问题只提示一次，避免逐 mesh 刷屏。 */
function warnOnce(key: string, message: string): void {
  if (_warned.has(key)) return;
  _warned.add(key);
  console.warn(`[hpg:gltf] ${message}`);
}

/** 蒙皮相关的提示文案（属性扫描与 skin 扫描共用，用于去重判定）。 */
const SKINNING_WARNING =
  'JOINTS_0/WEIGHTS_0 未实现：蒙皮网格按绑定姿势（A/T-pose）渲染，动画姿态不会生效。';

/**
 * 记录一条「模型用到但 hpg 未实现」的 feature。
 *
 * 同时进两处：
 *   - `asset.warnings`（结构化，每个 asset 完整 —— 供应用展示）
 *   - console（按 feature 全局去重一次 —— 避免批量加载时刷屏）
 */
function noteFeature(warnings: Set<string>, key: string, message: string): void {
  warnings.add(message);
  warnOnce(key, message);
}

/** 非索引 primitive 的 index 序列（0,1,2,...）。 */
function generateSequentialIndices(vertexCount: number): Uint16Array | Uint32Array {
  if (vertexCount > 65535) {
    const out = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) out[i] = i;
    return out;
  }
  const out = new Uint16Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) out[i] = i;
  return out;
}

// ─── Texture / Image Parsing ────────────────────────────────

/** glTF sampler 未声明时的默认值（规范：线性过滤 + repeat 寻址）。 */
const DEFAULT_SAMPLER: AssetSampler = {
  magFilter: 'linear',
  minFilter: 'linear',
  mipmapFilter: 'linear',
  addressModeU: 'repeat',
  addressModeV: 'repeat',
};

/** glTF wrap 枚举 → WebGPU address mode。 */
function mapAddressMode(v: number | undefined): GPUAddressMode {
  switch (v) {
    case 33071: return 'clamp-to-edge';
    case 33648: return 'mirror-repeat';
    case 10497:
    default: return 'repeat';
  }
}

/** glTF sampler → AssetSampler（min/mip 由 minFilter 枚举一并推导）。 */
function mapSampler(s: GltfSampler | undefined): AssetSampler {
  if (!s) return DEFAULT_SAMPLER;
  let min: GPUFilterMode = 'linear';
  let mip: GPUMipmapFilterMode = 'linear';
  switch (s.minFilter) {
    case 9728: min = 'nearest'; mip = 'nearest'; break;
    case 9729: min = 'linear'; mip = 'nearest'; break;
    case 9984: min = 'nearest'; mip = 'nearest'; break;
    case 9985: min = 'linear'; mip = 'nearest'; break;
    case 9986: min = 'nearest'; mip = 'linear'; break;
    default: min = 'linear'; mip = 'linear'; break; // 9987 / undefined
  }
  return {
    magFilter: s.magFilter === 9728 ? 'nearest' : 'linear',
    minFilter: min,
    mipmapFilter: mip,
    addressModeU: mapAddressMode(s.wrapS),
    addressModeV: mapAddressMode(s.wrapT),
  };
}

/** 由魔数推断图片 mime（glTF 允许省略 mimeType）。 */
function sniffImageMime(data: Uint8Array): string {
  if (data.length >= 8 && data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47) return 'image/png';
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) return 'image/jpeg';
  if (data.length >= 12 && data[8] === 0x57 && data[9] === 0x45 && data[10] === 0x42 && data[11] === 0x50) return 'image/webp';
  return 'application/octet-stream';
}

/** 解析 data: URI（glTF 允许把图片内联在 JSON 里）。 */
function decodeDataUri(uri: string): { mimeType: string; data: Uint8Array } | null {
  const m = /^data:([^;,]*)?(;base64)?,([\s\S]*)$/.exec(uri);
  if (!m) return null;
  const mimeType = m[1] || 'application/octet-stream';
  const payload = m[3] ?? '';
  if (m[2]) {
    const binary = atob(payload);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return { mimeType, data: out };
  }
  return { mimeType, data: new TextEncoder().encode(decodeURIComponent(payload)) };
}

/** 取出 GLB 内嵌图片的编码字节（bufferView 优先，其次 data: URI）。 */
function parseImages(
  json: GltfRoot,
  bufferViews: GltfBufferView[],
  buffers: ArrayBuffer[],
  warnings: Set<string>,
): AssetImage[] {
  const images: AssetImage[] = [];
  const declared = json.images ?? [];
  for (let i = 0; i < declared.length; i++) {
    const img = declared[i] as GltfImage;
    if (img.bufferView != null) {
      const bv = bufferViews[img.bufferView];
      const buffer = bv ? buffers[bv.buffer] : undefined;
      if (!bv || !buffer) continue;
      const start = bv.byteOffset ?? 0;
      const end = Math.min(start + bv.byteLength, buffer.byteLength);
      if (end <= start) continue;
      const data = new Uint8Array(buffer.slice(start, end));
      images[i] = { name: `image-${i}`, mimeType: img.mimeType ?? sniffImageMime(data), data };
      continue;
    }
    if (img.uri?.startsWith('data:')) {
      const decoded = decodeDataUri(img.uri);
      if (!decoded) continue;
      images[i] = { name: `image-${i}`, mimeType: img.mimeType ?? decoded.mimeType, data: decoded.data };
      continue;
    }
    // 指向 GLB 之外的外部图片文件：自包含容器无法解析。
    noteFeature(warnings, 'external-image', '外部图片文件（uri 指向 GLB 之外）无法加载：对应贴图不会生效。');
  }
  return images;
}

/** glTF texture 引用 → AssetTextureRef（含 sampler 状态）。 */
function resolveTextureRef(
  ref: { index: number; texCoord?: number } | undefined,
  json: GltfRoot,
): AssetTextureRef | undefined {
  if (!ref) return undefined;
  const tex = json.textures?.[ref.index];
  if (!tex || tex.source == null) return undefined;
  const sampler = tex.sampler != null ? json.samplers?.[tex.sampler] : undefined;
  return { imageIndex: tex.source, sampler: mapSampler(sampler), texCoord: ref.texCoord ?? 0 };
}

// ─── Main Loader ────────────────────────────────────────────

/**
 * 解析 glTF/GLB ArrayBuffer，返回 hpg 可用的 GltfAsset。
 */
export function parseGltf(buffer: ArrayBuffer): GltfAsset {
  const { json, bin } = parseGlb(buffer);

  // 校验必需扩展：Draco / meshopt 等压缩网格无法按原始 bufferView 解析。
  const unsupported = (json.extensionsRequired ?? []).filter((e) =>
    (UNSUPPORTED_EXTENSIONS as readonly string[]).includes(e),
  );
  if (unsupported.length > 0) {
    throw new Error(
      `Unsupported glTF extension(s) required: ${unsupported.join(', ')}. ` +
        `Re-export the model without Draco / meshopt compression.`,
    );
  }

  // ── 未实现 feature 审计 ──
  // 真实资产里大量 feature 是「忽略后照样能画出东西、但结果是错的」。
  // 这类问题必须显式告知，不能静默丢弃（否则用户只能猜为什么模型不像样）。
  const warnings = new Set<string>();

  for (const ext of json.extensionsUsed ?? []) {
    if (IMPLEMENTED_EXTENSIONS.has(ext)) continue;
    const hint = extensionHint(ext);
    if (hint) noteFeature(warnings, `ext:${ext}`, `glTF 扩展 ${ext} 未实现：${hint}。`);
  }

  const requiredUnimplemented = (json.extensionsRequired ?? []).filter(
    (e) => !IMPLEMENTED_EXTENSIONS.has(e),
  );
  if (requiredUnimplemented.length > 0) {
    noteFeature(
      warnings,
      'required-ext',
      `模型声明了必需扩展（${requiredUnimplemented.join(', ')}），hpg 未完整实现 —— 显示结果可能与作者意图不符。`,
    );
  }

  if ((json.animations?.length ?? 0) > 0) {
    noteFeature(warnings, 'animation', `glTF 动画未播放（${json.animations!.length} 条）：只渲染静态场景。`);
  }

  for (const mat of json.materials ?? []) {
    const pbr = mat.pbrMetallicRoughness ?? {};
    // MASK 已由材质管线实现（fragment discard）；BLEND 仍需要透明排序 / 混合策略，暂不支持。
    if (mat.alphaMode === 'BLEND') {
      noteFeature(
        warnings,
        'alpha:BLEND',
        'alphaMode=BLEND 未实现：被当作不透明渲染，会露出被遮挡的面。',
      );
    }
    if (pbr.metallicRoughnessTexture || mat.normalTexture || mat.occlusionTexture || mat.emissiveTexture) {
      noteFeature(warnings, 'pbr-textures', 'metallic-roughness / normal / occlusion / emissive 贴图未采样：光照细节丢失。');
    }
    for (const ext of Object.keys(mat.extensions ?? {})) {
      // 与 asset 级扫描共用同一个 key / 文案：同一扩展只报一次。
      const hint = extensionHint(ext);
      if (hint) noteFeature(warnings, `ext:${ext}`, `glTF 扩展 ${ext} 未实现：${hint}。`);
    }
  }

  const warnsSkinning = (json.skins?.length ?? 0) > 0;

  // Resolve buffers —— 当前只支持 GLB 内嵌 BIN chunk。
  const buffers: ArrayBuffer[] = [];
  const declaredBuffers = json.buffers ?? [];
  if (declaredBuffers.length === 0) throw new Error('glTF asset declares no buffers.');
  for (let i = 0; i < declaredBuffers.length; i++) {
    const buf = declaredBuffers[i] as GltfBuffer;
    if (i === 0 && bin) {
      buffers.push(bin);
    } else if (buf.uri) {
      throw new Error(
        `External buffer URI not supported: ${buf.uri}. Only self-contained .glb containers are supported.`,
      );
    } else {
      throw new Error(`GLB has no BIN chunk for buffer ${i}.`);
    }
  }

  const bufferViews = json.bufferViews ?? [];
  const accessors = json.accessors ?? [];

  // Parse meshes
  const meshes: AssetMesh[] = [];
  for (const mesh of json.meshes ?? []) {
    const primitives: AssetPrimitive[] = [];
    for (const prim of mesh.primitives) {
      // 只支持 TRIANGLES（mode 4）；其余拓扑类型直接跳过并提示。
      if (prim.mode !== undefined && prim.mode !== 4) {
        warnOnce(
          `mode-${prim.mode}`,
          `Unsupported primitive mode ${prim.mode} (only 4 = TRIANGLES is supported); primitive skipped.`,
        );
        continue;
      }

      // Interleave attributes: POSITION + NORMAL + TEXCOORD_0 + TANGENT
      const posIndex = prim.attributes.POSITION;
      const posAcc = posIndex != null ? accessors[posIndex] : undefined;
      if (!posAcc) {
        warnOnce('no-position', 'Primitive without POSITION attribute skipped.');
        continue;
      }
      const normAcc = prim.attributes.NORMAL != null ? accessors[prim.attributes.NORMAL] : null;
      const uvAcc = prim.attributes.TEXCOORD_0 != null ? accessors[prim.attributes.TEXCOORD_0] : null;
      const tanAcc = prim.attributes.TANGENT != null ? accessors[prim.attributes.TANGENT] : null;

      // 顶点属性 / morph：逐 primitive 扫描一次（按 feature 去重，不会刷屏）。
      for (const name of Object.keys(prim.attributes)) {
        if (name === 'POSITION' || name === 'NORMAL' || name === 'TEXCOORD_0' || name === 'TANGENT') continue;
        if (name === 'JOINTS_0' || name === 'WEIGHTS_0') {
          noteFeature(warnings, 'skinning-attrs', SKINNING_WARNING);
        } else if (name === 'COLOR_0') {
          noteFeature(warnings, 'color0', 'COLOR_0（顶点色）被忽略。');
        } else if (name.startsWith('TEXCOORD_')) {
          noteFeature(warnings, 'uv1+', `${name} 被忽略：只使用 TEXCOORD_0。`);
        } else {
          noteFeature(warnings, `attr:${name}`, `顶点属性 ${name} 未实现，已忽略。`);
        }
      }
      if ((prim.targets?.length ?? 0) > 0) {
        noteFeature(warnings, 'morph-targets', 'morph target（变形目标）未实现：只渲染基础形状。');
      }

      const posData = readAccessorFloats(posAcc, bufferViews, buffers);
      const normData = normAcc ? readAccessorFloats(normAcc, bufferViews, buffers) : null;
      const uvData = uvAcc ? readAccessorFloats(uvAcc, bufferViews, buffers) : null;
      const tanData = tanAcc ? readAccessorFloats(tanAcc, bufferViews, buffers) : null;

      const vertexCount = Math.min(posAcc.count, Math.floor(posData.length / 3));

      // Build layout
      const attributes: { shaderLocation: number; offset: number; format: GPUVertexFormat }[] = [];
      let offset = 0;
      const FLOAT32_BYTES = 4;

      // Location 0: POSITION (vec3)
      attributes.push({ shaderLocation: 0, offset, format: 'float32x3' });
      offset += 3 * FLOAT32_BYTES;

      // Location 1: NORMAL (vec3) — optional
      if (normData) {
        attributes.push({ shaderLocation: 1, offset, format: 'float32x3' });
        offset += 3 * FLOAT32_BYTES;
      }

      // Location 2: TEXCOORD_0 (vec2) — optional
      if (uvData) {
        attributes.push({ shaderLocation: 2, offset, format: 'float32x2' });
        offset += 2 * FLOAT32_BYTES;
      }

      // Location 3: TANGENT (vec4) — optional
      if (tanData) {
        attributes.push({ shaderLocation: 3, offset, format: 'float32x4' });
        offset += 4 * FLOAT32_BYTES;
      }

      const arrayStride = offset;

      // Interleave vertex data
      const vertices = new Float32Array(vertexCount * (arrayStride / FLOAT32_BYTES));
      for (let v = 0; v < vertexCount; v++) {
        let dst = v * (arrayStride / FLOAT32_BYTES);

        // POSITION
        vertices[dst++] = posData[v * 3 + 0];
        vertices[dst++] = posData[v * 3 + 1];
        vertices[dst++] = posData[v * 3 + 2];

        // NORMAL
        if (normData) {
          vertices[dst++] = normData[v * 3 + 0];
          vertices[dst++] = normData[v * 3 + 1];
          vertices[dst++] = normData[v * 3 + 2];
        }

        // TEXCOORD_0
        if (uvData) {
          vertices[dst++] = uvData[v * 2 + 0];
          vertices[dst++] = uvData[v * 2 + 1];
        }

        // TANGENT
        if (tanData) {
          vertices[dst++] = tanData[v * 4 + 0];
          vertices[dst++] = tanData[v * 4 + 1];
          vertices[dst++] = tanData[v * 4 + 2];
          vertices[dst++] = tanData[v * 4 + 3];
        }
      }

      // Indices
      let indices: Uint16Array | Uint32Array;
      let indexFormat: GPUIndexFormat;
      if (prim.indices != null) {
        const idxAcc = accessors[prim.indices];
        if (idxAcc) {
          const read = readIndices(idxAcc, bufferViews, buffers);
          indices = read.indices;
          indexFormat = read.format;
        } else {
          warnOnce('bad-index-accessor', 'Primitive references a missing index accessor; indices generated.');
          indices = generateSequentialIndices(vertexCount);
          indexFormat = vertexCount > 65535 ? 'uint32' : 'uint16';
        }
      } else {
        // Non-indexed: generate 0,1,2,...（顶点数超过 uint16 范围时用 uint32）
        indices = generateSequentialIndices(vertexCount);
        indexFormat = vertexCount > 65535 ? 'uint32' : 'uint16';
      }

      if (indices.length === 0) {
        warnOnce('empty-primitive', 'Primitive produced 0 indices; skipped.');
        continue;
      }

      primitives.push({
        vertices,
        vertexLayout: { arrayStride, attributes },
        indices,
        indexFormat,
        materialIndex: prim.material ?? 0,
      });
    }
    meshes.push({ name: mesh.name ?? 'unnamed', primitives });
  }

  // Parse materials
  const materials: AssetMaterial[] = (json.materials ?? []).map((mat) => {
    const pbr = mat.pbrMetallicRoughness ?? {};
    return {
      name: mat.name ?? 'unnamed',
      baseColorFactor: pbr.baseColorFactor ?? [1, 1, 1, 1],
      metallicFactor: pbr.metallicFactor ?? 1.0,
      roughnessFactor: pbr.roughnessFactor ?? 1.0,
      doubleSided: mat.doubleSided ?? false,
      baseColorTexture: resolveTextureRef(pbr.baseColorTexture, json),
      alphaMode: mat.alphaMode ?? 'OPAQUE',
      alphaCutoff: mat.alphaCutoff ?? 0.5,
    };
  });

  // Parse embedded images（只保留编码字节，解码由注入的 ImageDecoder 完成）。
  const images = parseImages(json, bufferViews, buffers, warnings);

  // Parse nodes
  const nodes: AssetNode[] = (json.nodes ?? []).map((node) => {
    let localMatrix: Float32Array;
    if (node.matrix) {
      localMatrix = mat4FromMatrix(node.matrix);
    } else {
      localMatrix = mat4FromTRS(node.translation, node.rotation, node.scale);
    }
    return {
      name: node.name ?? 'unnamed',
      meshIndex: node.mesh,
      localMatrix,
      children: node.children ?? [],
    };
  });

  // Root nodes
  const scene = json.scenes?.[json.scene ?? 0];
  const rootNodes = scene?.nodes ?? [];

  // 声明了 skin 但 primitive 上没有 JOINTS_0（例如 skin 挂在未使用的 mesh 上）时也要提示。
  if (warnsSkinning && !warnings.has(SKINNING_WARNING)) {
    noteFeature(warnings, 'skinning-attrs', SKINNING_WARNING);
  }

  return { meshes, materials, images, nodes, rootNodes, warnings: [...warnings] };
}

// ─── Scene Flattening ───────────────────────────────────────

export interface FlattenedNode {
  name: string;
  meshIndex?: number;
  worldMatrix: Float32Array;
}

/**
 * 展平场景树，计算每个节点的世界矩阵。
 */
export function flattenScene(asset: GltfAsset): FlattenedNode[] {
  const result: FlattenedNode[] = [];

  function walk(nodeIdx: number, parentMatrix: Float32Array) {
    const node = asset.nodes[nodeIdx];
    if (!node) return;
    const worldMatrix = multiplyMat4(parentMatrix, node.localMatrix);

    if (node.meshIndex != null) {
      result.push({
        name: node.name,
        meshIndex: node.meshIndex,
        worldMatrix,
      });
    }

    for (const childIdx of node.children) {
      walk(childIdx, worldMatrix);
    }
  }

  const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  for (const rootIdx of asset.rootNodes) {
    walk(rootIdx, identity);
  }

  return result;
}
