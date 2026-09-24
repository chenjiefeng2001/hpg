/**
 * Geometry Arena —— 静态几何显存池。
 *
 * 在共享 vertex/index 大 Buffer 上做切分式分配（Slice），并为已释放的切片维护
 * free-list 以便复用；多数场景零额外分配。用户拿到的 `Geometry` 引用可作为句柄
 * 反复提交，显存仅注册一次。
 *
 * 支持 destroyGeometry() 回收显存供后续复用。
 * 池链设计：旧池保留至所有引用其 Geometry 的对象释放。
 */

import type { Aabb, Geometry, VertexLayoutDesc, BufferSlice } from '../types';

/** Geometry Arena 池化显存使用统计。 */
export interface GeometryArenaStats {
  vertexPools: number;
  indexPools: number;
  /** vertex 池总容量（字节）。 */
  vertexPoolBytes: number;
  /** index 池总容量（字节）。 */
  indexPoolBytes: number;
  /** vertex 已分配字节（含碎片）。 */
  vertexUsedBytes: number;
  /** index 已分配字节（含碎片）。 */
  indexUsedBytes: number;
  /** vertex free-list 碎片总字节。 */
  vertexFreeBytes: number;
  /** index free-list 碎片总字节。 */
  indexFreeBytes: number;
  /** 活跃 Geometry 数量。 */
  geometries: number;
}

/** vertex + index 两块池各自的增长步长（字节）。 */
const VERTEX_CHUNK = 1 << 22; // 4 MiB
const INDEX_CHUNK = 1 << 20; // 1 MiB

interface FreeBlock {
  /** 该空闲区间所属的池 —— 偏移只在所属池内有效，必须带上池身份才能安全复用。 */
  pool: ArenaBuffers;
  offset: number;
  length: number;
}

interface ArenaBuffers {
  /** 池序号（用于区分不同池的偏移空间，以及 free-list 的分池合并）。 */
  id: number;
  buffer: GPUBuffer;
  slice: BufferSlice;
}

/** 一次分配的结果：目标池 + 该池内的字节偏移。 */
interface ArenaAlloc {
  pool: ArenaBuffers;
  offset: number;
}

/** Geometry 元数据：记录切片归属（池 + 池内偏移），供 destroyGeometry() 回收。 */
interface GeometryMeta {
  vertex: ArenaAlloc;
  vertexByteLength: number;
  index?: ArenaAlloc;
  indexByteLength?: number;
}

export class GeometryArena {
  private _vertexPools: ArenaBuffers[] = [];
  private _indexPools: ArenaBuffers[] = [];
  private _vertexCursor = 0;
  private _indexCursor = 0;
  private _vertexFrees: FreeBlock[] = [];
  private _indexFrees: FreeBlock[] = [];
  private _geoMeta = new WeakMap<Geometry, GeometryMeta>();
  private _geoCount = 0;
  private _poolSeq = 0;

  constructor(private device: GPUDevice) {}

  /** 当前活跃的 vertex 池（最后一个）。 */
  private get _vertex(): ArenaBuffers | null {
    return this._vertexPools.length > 0 ? this._vertexPools[this._vertexPools.length - 1]! : null;
  }

  /** 当前活跃的 index 池（最后一个）。 */
  private get _index(): ArenaBuffers | null {
    return this._indexPools.length > 0 ? this._indexPools[this._indexPools.length - 1]! : null;
  }

  get vertexBuffer(): GPUBuffer {
    return (this._vertex ?? this.createVertexPool()).buffer;
  }

  private ensureVertexPool(): ArenaBuffers {
    return this._vertex ?? this.createVertexPool();
  }

  private ensureIndexPool(): ArenaBuffers {
    return this._index ?? this.createIndexPool();
  }

  /**
   * 上传顶点 + 可选索引并返回 Geometry。
   * @param indexData 可空；提供则走索引绘制，否则走顶点绘制。
   */
  createGeometry(
    vertexData: Float32Array,
    vertexLayouts: VertexLayoutDesc[],
    indexData?: Uint16Array | Uint32Array,
    indexFormat: GPUIndexFormat = 'uint16',
    primitive: GPUPrimitiveTopology = 'triangle-list',
  ): Geometry {
    const vBytes = vertexData.byteLength;
    // 复用 free-list 时目标池是**该空闲块所在的池**（可能是旧池）；
    // 只有全新分配才落在最新池。写回错误的池会覆盖该池中存活几何体的数据。
    const vAlloc = this.allocVertex(vBytes);
    const vPool = vAlloc.pool;
    const vOffset = vAlloc.offset;
    this.device.queue.writeBuffer(vPool.buffer, vOffset, vertexData);

    const stride = totalStride(vertexLayouts);
    const vertexCount = stride > 0 ? vBytes / stride : 0;

    const vSlice: BufferSlice = { buffer: vPool.buffer, byteOffset: vOffset, byteLength: vBytes };

    const geo: Geometry = {
      vertexBuffer: vPool.buffer,
      vertexSlice: vSlice,
      vertexBuffers: [vSlice],
      vertexLayouts,
      indexFormat,
      indexCount: vertexCount,
      vertexCount,
      primitive,
      bounds: computePositionBounds(vertexData, vertexLayouts, vertexCount),
    };

    const meta: GeometryMeta = {
      vertex: vAlloc,
      vertexByteLength: vBytes,
    };

    if (indexData) {
      let iBytes = indexData.byteLength;
      let writeData: Uint16Array | Uint32Array = indexData;
      // writeBuffer requires byte length to be a multiple of 4
      if (iBytes % 4 !== 0) {
        const padded = new Uint16Array(indexData.length + 1);
        padded.set(indexData);
        writeData = padded;
        iBytes = padded.byteLength;
      }
      const iAlloc = this.allocIndex(iBytes);
      const iPool = iAlloc.pool;
      const iOffset = iAlloc.offset;
      this.device.queue.writeBuffer(iPool.buffer, iOffset, writeData);
      const indexCount = indexData.length;

      geo.indexBuffer = iPool.buffer;
      geo.indexSlice = { buffer: iPool.buffer, byteOffset: iOffset, byteLength: iBytes };
      geo.indexCount = indexCount;

      meta.index = iAlloc;
      meta.indexByteLength = iBytes;
    }

    this._geoMeta.set(geo, meta);
    this._geoCount++;
    return geo;
  }

  /**
   * 回收 Geometry 的显存切片，供后续 allocVertex / allocIndex 复用。
   * 调用后不应再使用该 Geometry 进行渲染。
   */
  destroyGeometry(geo: Geometry): void {
    const meta = this._geoMeta.get(geo);
    if (!meta) return;
    this._geoMeta.delete(geo);
    this._geoCount--;

    // 归还 vertex 切片到 free-list。
    this._vertexFrees.push({
      pool: meta.vertex.pool,
      offset: meta.vertex.offset,
      length: meta.vertexByteLength,
    });
    this.mergeFreeList(this._vertexFrees);

    // 归还 index 切片到 free-list（如有）。
    if (meta.index && meta.indexByteLength !== undefined) {
      this._indexFrees.push({
        pool: meta.index.pool,
        offset: meta.index.offset,
        length: meta.indexByteLength,
      });
      this.mergeFreeList(this._indexFrees);
    }
  }

  /** 合并相邻 free block，减少碎片。只在**同一个池内**合并 —— 不同池的偏移空间互不相干。 */
  private mergeFreeList(frees: FreeBlock[]): void {
    frees.sort((a, b) => (a.pool.id - b.pool.id) || (a.offset - b.offset));
    let i = 0;
    while (i < frees.length - 1) {
      const curr = frees[i]!;
      const next = frees[i + 1]!;
      if (curr.pool === next.pool && curr.offset + curr.length === next.offset) {
        // 相邻且同池：合并。
        curr.length += next.length;
        frees.splice(i + 1, 1);
      } else {
        i++;
      }
    }
  }

  /**
   * 分配 vertex 空间：优先复用 free-list 中的块（返回该块所属的池），
   * 否则在最新池的游标处新分配。
   */
  private allocVertex(bytes: number): ArenaAlloc {
    const aligned = align16(bytes);
    const reused = allocFrom(this._vertexFrees, aligned);
    if (reused) return reused;
    const offset = this._vertexCursor;
    this._vertexCursor += aligned;
    // 新块落在最新池。池链不变式：最新池的容量 ≥ 全局游标，因此新偏移在该池内有效。
    this.ensureVertexCapacity(offset + aligned);
    return { pool: this.ensureVertexPool(), offset };
  }

  /** 分配 index 空间：语义同 allocVertex。 */
  private allocIndex(bytes: number): ArenaAlloc {
    const aligned = align4(bytes);
    const reused = allocFrom(this._indexFrees, aligned);
    if (reused) return reused;
    const offset = this._indexCursor;
    this._indexCursor += aligned;
    this.ensureIndexCapacity(offset + aligned);
    return { pool: this.ensureIndexPool(), offset };
  }

  private ensureVertexCapacity(needed: number): void {
    const pool = this._vertex;
    if (pool && pool.slice.byteLength >= needed) return;
    const newSize = align(nextPow2(needed), VERTEX_CHUNK);
    this.createVertexPool(newSize);
  }

  private ensureIndexCapacity(needed: number): void {
    const pool = this._index;
    if (pool && pool.slice.byteLength >= needed) return;
    const newSize = align(nextPow2(needed), INDEX_CHUNK);
    this.createIndexPool(newSize);
  }

  private createVertexPool(size = VERTEX_CHUNK): ArenaBuffers {
    const buffer = this.device.createBuffer({
      label: 'hpg:vertex-pool',
      size,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const pool: ArenaBuffers = { id: this._poolSeq++, buffer, slice: { buffer, byteOffset: 0, byteLength: size } };
    this._vertexPools.push(pool);
    return pool;
  }

  private createIndexPool(size = INDEX_CHUNK): ArenaBuffers {
    const buffer = this.device.createBuffer({
      label: 'hpg:index-pool',
      size,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    const pool: ArenaBuffers = { id: this._poolSeq++, buffer, slice: { buffer, byteOffset: 0, byteLength: size } };
    this._indexPools.push(pool);
    return pool;
  }

  /** 池化显存使用统计。 */
  stats(): GeometryArenaStats {
    let vertexPoolBytes = 0;
    for (const p of this._vertexPools) vertexPoolBytes += p.slice.byteLength;
    let indexPoolBytes = 0;
    for (const p of this._indexPools) indexPoolBytes += p.slice.byteLength;
    let vertexFreeBytes = 0;
    for (const f of this._vertexFrees) vertexFreeBytes += f.length;
    let indexFreeBytes = 0;
    for (const f of this._indexFrees) indexFreeBytes += f.length;
    return {
      vertexPools: this._vertexPools.length,
      indexPools: this._indexPools.length,
      vertexPoolBytes,
      indexPoolBytes,
      vertexUsedBytes: this._vertexCursor,
      indexUsedBytes: this._indexCursor,
      vertexFreeBytes,
      indexFreeBytes,
      geometries: this._geoCount,
    };
  }

  /** 销毁所有池，释放 GPU 内存。调用后不应再使用任何该 Arena 创建的 Geometry。 */
  dispose(): void {
    for (const p of this._vertexPools) p.buffer.destroy();
    for (const p of this._indexPools) p.buffer.destroy();
    this._vertexPools.length = 0;
    this._indexPools.length = 0;
    this._vertexCursor = 0;
    this._indexCursor = 0;
    this._vertexFrees.length = 0;
    this._indexFrees.length = 0;
    this._geoCount = 0;
  }
}

/**
 * 从顶点流中的 position 属性（shaderLocation = 0）推导局部空间 AABB。
 * 找不到 position 或数据不足时返回 undefined。
 */
function computePositionBounds(
  vertexData: Float32Array,
  layouts: VertexLayoutDesc[],
  vertexCount: number,
): Aabb | undefined {
  if (vertexCount <= 0) return undefined;

  let prefixBytes = 0;
  for (const layout of layouts) {
    const posAttr = layout.attributes.find((a) => a.shaderLocation === 0);
    if (posAttr) {
      const strideFloats = layout.arrayStride / 4;
      if (strideFloats <= 0) return undefined;
      const base = prefixBytes / 4 + posAttr.offset / 4;

      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (let v = 0; v < vertexCount; v++) {
        const o = base + v * strideFloats;
        const x = vertexData[o] as number;
        const y = vertexData[o + 1] as number;
        const z = vertexData[o + 2] as number;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }
      if (!isFinite(minX)) return undefined;
      return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
    }
    if (layout.stepMode === 'vertex') prefixBytes += layout.arrayStride;
  }
  return undefined;
}

function totalStride(layouts: VertexLayoutDesc[]): number {
  let stride = 0;
  for (const l of layouts) {
    stride += l.stepMode === 'vertex' ? l.arrayStride : 0;
  }
  return stride;
}

/**
 * 从 free-list 切出 `aligned` 字节，返回**该块所属的池 + 池内起始 offset**；无可用块时返回 null。
 *
 * 返回池身份是必须的：不同池的偏移空间互相独立，若只返回 offset 而把切片绑到最新池，
 * 复用旧池的空闲块就会写坏最新池中存活几何体的数据。
 *
 * 注意：必须先取出 offset 再调整块首（精确匹配时整块弹出）。
 */
function allocFrom(frees: FreeBlock[], aligned: number): ArenaAlloc | null {
  for (let i = 0; i < frees.length; i++) {
    const b = frees[i];
    if (!b || b.length < aligned) continue;
    const offset = b.offset;
    const pool = b.pool;
    if (b.length === aligned) frees.splice(i, 1);
    else {
      b.offset += aligned;
      b.length -= aligned;
    }
    return { pool, offset };
  }
  return null;
}

function align4(x: number): number {
  return (x + 3) & ~3;
}

function align16(x: number): number {
  return (x + 15) & ~15;
}

function align(x: number, a: number): number {
  return (x + a - 1) & ~(a - 1);
}

function nextPow2(x: number): number {
  let p = 1;
  while (p < x) {
    p *= 2;
    if (p > 0x40000000) return x; // 防溢出
  }
  return p;
}
