/**
 * CullingPipeline —— GPU 视锥剔除的 Compute Pipeline 封装（Phase 6B+：含 compaction）。
 *
 * 架构：
 *   CPU 上传 → bounding spheres + geometry indices + VP matrix
 *   Compute Shader → visibility 测试 + compaction mapping + 原子填充 indirect args
 *   Render Pass → vertex shader 通过 compactedIndices 间接访问 instance buffer
 *
 * CPU 不读取任何 GPU 结果（零 readback）。
 */

import { CS_FRUSTUM_CULL } from '../shaders/culling';
import type { TimestampQuery } from './timestamp';

/**
 * 从列主序 VP 矩阵提取 6 个归一化视锥平面（CPU 参考实现）。
 *
 * **必须与 src/shaders/culling.ts 的 extractPlane 保持数学一致**。
 * 返回 Float32Array(24)：[left, right, bottom, top, near, far] × (nx, ny, nz, d)，
 * 其中内侧判定为 `dot(n, c) + d >= 0`（球体再加 radius）。
 *
 * 注意：WGSL 中 `vp[i]` 是列，所以数学第 r 行 = (vp[0][r], vp[1][r], vp[2][r], vp[3][r])。
 */
export function extractFrustumPlanes(vpColumnMajor: Float32Array): Float32Array {
  const col = (c: number, r: number) => vpColumnMajor[c * 4 + r] as number;
  const rowOf = (r: number): [number, number, number, number] => [col(0, r), col(1, r), col(2, r), col(3, r)];

  const rx = rowOf(0);
  const ry = rowOf(1);
  const rz = rowOf(2);
  const rw = rowOf(3);

  const out = new Float32Array(24);
  const write = (i: number, row: readonly number[], sign: number) => {
    let nx = sign * row[0]! + rw[0]!;
    let ny = sign * row[1]! + rw[1]!;
    let nz = sign * row[2]! + rw[2]!;
    const d = sign * row[3]! + rw[3]!;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      nx /= len;
      ny /= len;
      nz /= len;
    }
    out[i * 4] = nx;
    out[i * 4 + 1] = ny;
    out[i * 4 + 2] = nz;
    out[i * 4 + 3] = d / (len > 0 ? len : 1);
  };

  write(0, rx, 1);
  write(1, rx, -1);
  write(2, ry, 1);
  write(3, ry, -1);
  write(4, rz, 1);
  write(5, rz, -1);
  return out;
}

/**
 * 判断球体是否与 6 个平面相交（内侧）。仅用于测试 / CPU 侧可见性校验。
 */
export function sphereInFrustum(planes: Float32Array, cx: number, cy: number, cz: number, radius: number): boolean {
  for (let i = 0; i < 6; i++) {
    const nx = planes[i * 4] as number;
    const ny = planes[i * 4 + 1] as number;
    const nz = planes[i * 4 + 2] as number;
    const d = planes[i * 4 + 3] as number;
    if (nx * cx + ny * cy + nz * cz + d + radius < 0) return false;
  }
  return true;
}

const WORKGROUP_SIZE = 64;
const FLOATS_PER_SPHERE = 4;
const BYTES_PER_SPHERE = FLOATS_PER_SPHERE * 4;
// Uniform: vp(64) + sphereCount(4) + geometryCount(4) + pad(8) = 80
const UNIFORM_SIZE = 80;

export interface CullingBuffers {
  /** 每实例 bounding sphere: N × vec4(cx, cy, cz, radius)。 */
  spheres: GPUBuffer;
  /** 每实例所属 geometry index: N × u32。 */
  geometryIndices: GPUBuffer;
  /**
   * Compaction mapping: slotCount × u32，
   * `compactedIndices[slotBase + slot] = 组内原始实例索引`。
   * 每个 geometry 的 slotBase 保证 256 字节对齐（供 dynamic offset 使用）。
   */
  compactedIndices: GPUBuffer;
  /** 每个 geometry 的 [candidateBase, slotBase]（u32 × 2）。 */
  geometryBases: GPUBuffer;
  /** Indirect draw args: G × DrawIndexedIndirectArgs(20B)。 */
  drawArgs: GPUBuffer;
  /** Compaction counters: G × u32，每 geometry 的 compaction slot 计数器。 */
  compactionCounters: GPUBuffer;
  /** Uniform: VP matrix + counts。 */
  uniforms: GPUBuffer;
}

/** compactedIndices 中每个 geometry 的 slot 区步长（实例数），256 字节对齐。 */
const SLOT_ALIGN = 64;

/** 向上对齐到 SLOT_ALIGN。 */
function alignSlots(n: number): number {
  return Math.ceil(n / SLOT_ALIGN) * SLOT_ALIGN;
}

/**
 * compactedIndices 需要为「统一绑定尺寸」预留的额外 slot 数。
 *
 * 渲染侧用一个 bind group + 逐 geometry dynamic offset 绑定 slot 区，
 * 绑定尺寸取「单个 geometry 的最大 slot 区」（覆盖任意组的可见实例数）。
 * WebGPU 要求 `slotBase + 绑定尺寸 ≤ bufferSize`，而各组的 slot 区长度不等，
 * 因此尾部必须按最大 slot 区预留余量，否则最后一个（或任意一个）组的偏移会越界。
 */
function slotHeadroom(counts: Uint32Array): number {
  let max = 0;
  for (let g = 0; g < counts.length; g++) {
    const aligned = alignSlots(counts[g] as number);
    if (aligned > max) max = aligned;
  }
  return max;
}

export class CullingPipeline {
  private _pipeline: GPUComputePipeline;
  private _bindGroupLayout: GPUBindGroupLayout;
  private _maxInstances = 0;
  private _maxGeometries = 0;
  private _maxSlots = 0;
  private _buffers: CullingBuffers | null = null;
  private _bindGroup: GPUBindGroup | null = null;

  constructor(private device: GPUDevice) {
    const module = device.createShaderModule({
      label: 'hpg:cs-frustum-cull',
      code: CS_FRUSTUM_CULL,
    });

    // 7 bindings: uniforms, spheres, geometryIndices, compactedIndices, drawArgs,
    //             compactionCounters, geometryBases
    this._bindGroupLayout = device.createBindGroupLayout({
      label: 'hpg:cull-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });

    const layout = device.createPipelineLayout({
      label: 'hpg:cull-pl',
      bindGroupLayouts: [this._bindGroupLayout],
    });

    this._pipeline = device.createComputePipeline({
      label: 'hpg:cull-pipeline',
      layout,
      compute: { module, entryPoint: 'cs_main' },
    });
  }

  /** 确保所有 buffer 容量足够。不足时重新创建。 */
  private ensureBuffers(instanceCount: number, geometryCount: number, slotCount: number): void {
    if (
      this._buffers &&
      this._maxInstances >= instanceCount &&
      this._maxGeometries >= geometryCount &&
      this._maxSlots >= slotCount
    ) {
      return;
    }

    this._destroyBuffers();

    const createBuf = (size: number, usage: GPUBufferUsageFlags, label: string) =>
      this.device.createBuffer({
        label: `hpg:cull-${label}`,
        size: Math.max(size, 256),
        usage,
      });

    this._buffers = {
      spheres: createBuf(instanceCount * BYTES_PER_SPHERE, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'spheres'),
      geometryIndices: createBuf(instanceCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'geo-indices'),
      compactedIndices: createBuf(slotCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'compacted'),
      geometryBases: createBuf(geometryCount * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'geo-bases'),
      drawArgs: createBuf(geometryCount * 20, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'draw-args'),
      compactionCounters: createBuf(geometryCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'compaction-counters'),
      uniforms: createBuf(UNIFORM_SIZE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'uniforms'),
    };

    this._maxInstances = instanceCount;
    this._maxGeometries = geometryCount;
    this._maxSlots = slotCount;
    this._bindGroup = null; // buffer 已重建 → 缓存的 bind group 失效
  }

  /** 复用的 compute bind group（buffer 未重建时跨帧复用，避免逐帧 createBindGroup）。 */
  private ensureBindGroup(): GPUBindGroup {
    if (this._bindGroup) return this._bindGroup;
    const bufs = this._buffers!;
    this._bindGroup = this.device.createBindGroup({
      label: 'hpg:cull-bg',
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: bufs.uniforms } },
        { binding: 1, resource: { buffer: bufs.spheres } },
        { binding: 2, resource: { buffer: bufs.geometryIndices } },
        { binding: 3, resource: { buffer: bufs.compactedIndices } },
        { binding: 4, resource: { buffer: bufs.drawArgs } },
        { binding: 5, resource: { buffer: bufs.compactionCounters } },
        { binding: 6, resource: { buffer: bufs.geometryBases } },
      ],
    });
    return this._bindGroup;
  }

  /**
   * 执行 GPU 视锥剔除 + compaction。
   *
   * @param vpMatrix    列主序 4×4 VP 矩阵
   * @param spheresData N × vec4(cx, cy, cz, radius)
   * @param geometryIds N × u32，每个 instance 所属 geometry 的索引
   * @param geometryCount 总 geometry 数量
    * @param drawArgsTemplate G × 5 的 u32 模板（indexCount / instanceCount / firstIndex /
    *        baseVertex / firstInstance），geometryCount > 0 时必须提供且长度精确匹配。instanceCount 位忽略（强制置 0，由 compute 原子填充）。
   *
   *        **必须在 dispatch 之前写入**：`queue.writeBuffer` 按入队顺序执行，
   *        dispatch 之后再写会把 GPU 刚原子填充的 instanceCount 覆盖回 0 →
   *        drawIndexedIndirect 画 0 个实例 → 整帧全黑（且无任何校验错误）。
   * @returns drawArgsBuffer / compactedIndicesBuffer / slotBases（元素下标，用于 dynamic offset）
   */
  cull(
    vpMatrix: Float32Array,
    spheresData: Float32Array,
    geometryIds: Uint32Array,
    geometryCount: number,
    drawArgsTemplate?: Uint32Array,
    timestamps?: TimestampQuery,
    commandEncoder?: GPUCommandEncoder,
  ): {
    drawArgsBuffer: GPUBuffer;
    compactedIndicesBuffer: GPUBuffer;
    drawArgsCount: number;
    /** 每个 geometry 在 compactedIndices 中的起始元素下标（256 字节对齐）。 */
    slotBases: Uint32Array;
    candidateCounts: Uint32Array;
    compactedSlotCount: number;
    /** compactedIndices 的绑定尺寸（字节）= 单个 geometry 的最大 slot 区。 */
    maxSlotBytes: number;
  } {
    if (!Number.isSafeInteger(geometryCount) || geometryCount < 0) {
      throw new Error('geometryCount must be a non-negative safe integer.');
    }
    if (geometryCount > 0 && !drawArgsTemplate) {
      throw new Error('drawArgsTemplate is required when geometryCount is greater than zero.');
    }
    if (drawArgsTemplate && drawArgsTemplate.length !== geometryCount * 5) {
      throw new Error(`drawArgsTemplate length must equal geometryCount * 5 (${geometryCount * 5}).`);
    }
    if (spheresData.length % FLOATS_PER_SPHERE !== 0) {
      throw new Error('spheresData length must be a multiple of 4.');
    }
    if (vpMatrix.length !== 16) throw new Error('vpMatrix must contain exactly 16 values.');
    for (let i = 0; i < vpMatrix.length; i++) {
      if (!Number.isFinite(vpMatrix[i])) throw new Error(`vpMatrix[${i}] must be finite.`);
    }
    const n = spheresData.length / FLOATS_PER_SPHERE;
    if (geometryIds.length !== n) {
      throw new Error(`geometryIds length (${geometryIds.length}) must equal sphere count (${n}).`);
    }
    let previousGeometry = 0;
    for (let i = 0; i < n; i++) {
      const g = geometryIds[i] as number;
      if (g >= geometryCount) throw new Error(`geometryIds[${i}] exceeds geometryCount.`);
      if (i > 0 && g < previousGeometry) {
        throw new Error('geometryIds must be grouped in non-decreasing order.');
      }
      previousGeometry = g;
      for (let c = 0; c < FLOATS_PER_SPHERE; c++) {
        const value = spheresData[i * FLOATS_PER_SPHERE + c] as number;
        if (!Number.isFinite(value)) throw new Error(`spheresData[${i * FLOATS_PER_SPHERE + c}] must be finite.`);
      }
      if (spheresData[i * FLOATS_PER_SPHERE + 3]! < 0) {
        throw new Error(`spheresData[${i * FLOATS_PER_SPHERE + 3}] must be non-negative.`);
      }
    }

    // 逐个 geometry 统计候选数，并计算 [candidateBase, slotBase]。
    // candidateBase 用于把全局索引换算成组内索引；slotBase 给 VS 的 dynamic offset 用。
    const counts = new Uint32Array(geometryCount);
    for (let i = 0; i < n; i++) {
      const g = geometryIds[i] as number;
      if (g < geometryCount) counts[g] = (counts[g] as number) + 1;
    }
    const bases = new Uint32Array(geometryCount * 2);
    const slotBases = new Uint32Array(geometryCount);
    let candidateBase = 0;
    let slotBase = 0;
    for (let g = 0; g < geometryCount; g++) {
      bases[g * 2] = candidateBase;
      bases[g * 2 + 1] = slotBase;
      slotBases[g] = slotBase;
      candidateBase += counts[g] as number;
      slotBase += alignSlots(counts[g] as number);
    }
    const slotCount = slotBase;
    const headroom = slotHeadroom(counts);

    // 尾部预留 headroom 个 slot：统一绑定尺寸（见 slotHeadroom 注释）。
    this.ensureBuffers(n, geometryCount, slotCount + headroom);
    const bufs = this._buffers!;

    // 上传数据（分块写入保护）。
    this.writeChunked(bufs.spheres, 0, spheresData);
    this.writeChunked(bufs.geometryIndices, 0, geometryIds);
    if (geometryCount > 0) this.writeChunked(bufs.geometryBases, 0, bases);

    // 预写 draw args（instanceCount 强制为 0，其余字段取 CPU 模板）+ 重置 compaction counters。
    // 顺序关键：必须在下面的 compute dispatch **之前**写完。
    const argsData = new Uint32Array(geometryCount * 5);
    if (drawArgsTemplate) {
      argsData.set(drawArgsTemplate.subarray(0, Math.min(drawArgsTemplate.length, geometryCount * 5)));
    }
    for (let g = 0; g < geometryCount; g++) argsData[g * 5 + 1] = 0; // instanceCount
    this.writeChunked(bufs.drawArgs, 0, argsData);
    const zeroCounters = new Uint32Array(geometryCount);
    this.writeChunked(bufs.compactionCounters, 0, zeroCounters);

    // 上传 uniforms: VP(64B) + sphereCount(4B) + geometryCount(4B) + pad(8B) = 80B
    const uniformData = new Uint8Array(UNIFORM_SIZE);
    new Float32Array(uniformData.buffer).set(vpMatrix.subarray(0, 16), 0);
    new Uint32Array(uniformData.buffer)[16] = n;
    new Uint32Array(uniformData.buffer)[17] = geometryCount;
    this.writeChunked(bufs.uniforms, 0, uniformData);

    // bind group（7 bindings，跨帧复用）。
    const bindGroup = this.ensureBindGroup();

    // Dispatch compute。
    const ownsEncoder = commandEncoder === undefined;
    const encoder = commandEncoder ?? this.device.createCommandEncoder({ label: 'hpg:cull-encoder' });
     const pass = encoder.beginComputePass({
       label: 'hpg:cull-pass',
       ...(timestamps ? { timestampWrites: timestamps.timestampWrites(0, 1) } : {}),
     });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));

    pass.end();
    if (timestamps) timestamps.resolve(encoder);
    if (ownsEncoder) this.device.queue.submit([encoder.finish()]);

    return {
      drawArgsBuffer: bufs.drawArgs,
      compactedIndicesBuffer: bufs.compactedIndices,
      drawArgsCount: geometryCount,
      slotBases,
      candidateCounts: counts,
      compactedSlotCount: slotCount,
      maxSlotBytes: Math.max(headroom * 4, SLOT_ALIGN * 4),
    };
  }

  get compactedIndicesBuffer(): GPUBuffer | null {
    return this._buffers?.compactedIndices ?? null;
  }

  /** 分块写入保护：使用底层 ArrayBuffer 避免浏览器 TypedArray bug。 */
  private writeChunked(buffer: GPUBuffer, offset: number, data: BufferSource): void {
    let ab: ArrayBuffer;
    let abOffset: number;
    if (ArrayBuffer.isView(data)) {
      ab = data.buffer;
      abOffset = data.byteOffset;
    } else {
      ab = data as ArrayBuffer;
      abOffset = 0;
    }
    const bytes = (data as any).byteLength as number;
    const CHUNK = 4 * 1024 * 1024;
    let written = 0;
    while (written < bytes) {
      const chunk = Math.min(CHUNK, bytes - written);
      this.device.queue.writeBuffer(buffer, offset + written, ab, abOffset + written, chunk);
      written += chunk;
    }
  }

  private _destroyBuffers(): void {
    if (!this._buffers) return;
    for (const b of Object.values(this._buffers)) {
      if (b) try { b.destroy(); } catch { /* ignore */ }
    }
    this._buffers = null;
    this._bindGroup = null;
  }

  dispose(): void {
    this._destroyBuffers();
    this._maxInstances = 0;
    this._maxGeometries = 0;
    this._maxSlots = 0;
  }
}
