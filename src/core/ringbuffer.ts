/**
 * 单帧瞬态显存池（Ring Buffer）。
 *
 * 目标：在一个大 Buffer 上顺序分配、单次上传，帧尾把 Offset 归位 ——
 * 避免每帧为矩阵 / uniform 反复创建 Buffer 与触发 GC。
 *
 * 增长策略：分配不足时创建 2 倍大的新 Buffer（旧 Buffer 解除映射后交给 GPU
 * 异步释放，等 GPU 读完后被驱动回收）。保证热路径零堆分配。
 */

/** 存储类绑定的强制对齐（minStorageBufferOffsetAlignment 的最小公倍数）。 */
export const RING_ALIGN = 256;
/** 超过该容量后触发警告，暗示应迁移到 static/dynamic 资源。 */
export const RING_WARN_BYTES = 4 * 1024 * 1024;

export interface RingStats {
  bytesUsed: number;
  bytesCapacity: number;
  allocations: number;
}

export class RingBuffer {
  readonly device: GPUDevice;
  private _buffer: GPUBuffer;
  private _size: number;
  private _head = 0;
  private _frameBytes = 0;
  private _allocations = 0;

  constructor(device: GPUDevice, initialBytes = 1 << 20) {
    this.device = device;
    this._size = initialBytes;
    this._buffer = createDynamicBuffer(device, initialBytes);
  }

  get buffer(): GPUBuffer {
    return this._buffer;
  }

  get bytesUsed(): number {
    return this._head;
  }

  get bytesCapacity(): number {
    return this._size;
  }

  get allocations(): number {
    return this._allocations;
  }

  /**
   * 分配并返回对齐后的偏移。
   *
   * 默认 16 字节对齐；作为 storage buffer 动态绑定（dynamic offset）使用时
   * 必须传 `RING_ALIGN`（256），因为 minStorageBufferOffsetAlignment 要求
   * 动态偏移满足该对齐，否则 WebGPU 校验失败 / 读到错位数据。
   */
  alloc(bytes: number, alignment = 16): number {
    const offset = alignTo(this._head, alignment);
    const next = offset + bytes;
    if (next > this._size) {
      // 扩容后重试必须**沿用同一个 alignment**：丢掉它会让实例区返回 16 对齐
      // （而非 256）偏移，作为 storage dynamic offset 绑定时校验失败/读到错位数据。
      this.grow();
      return this.alloc(bytes, alignment);
    }
    this._head = next;
    this._frameBytes = this._frameBytes > next ? this._frameBytes : next;
    return offset;
  }

  /** 在给定偏移写入 4 字节对齐数据。返回写入的字节数。 */
  write(offset: number, view: Float32Array | Uint8Array, srcOffset = 0, srcLength = view.byteLength): number {
    const viewBytes = view.byteLength - srcOffset;
    const total = Math.min(srcLength, viewBytes);
    if (total <= 0) return 0;

    // 使用底层 ArrayBuffer + byteOffset 传入 writeBuffer，
    // 避免部分浏览器对 TypedArray + dataOffset/size 参数的 bug。
    const ab = view.buffer;
    const abByteOffset = view.byteOffset + srcOffset;

    const CHUNK = 4 * 1024 * 1024;
    let written = 0;
    while (written < total) {
      const chunk = Math.min(CHUNK, total - written);
      this.device.queue.writeBuffer(this._buffer, offset + written, ab, abByteOffset + written, chunk);
      written += chunk;
    }
    return total;
  }

  /** 直接写入（自动分配）。 */
  push(view: Float32Array | Uint8Array): number {
    const offset = this.alloc(view.byteLength);
    this.write(offset, view);
    return offset;
  }

  /** 当前底层 GPUBuffer。扩容后会变化，持有者需重新创建 bind group。 */
  get bufferIdentity(): GPUBuffer {
    return this._buffer;
  }

  /** 帧尾：重置游标；若容量长期不足则提示。 */
  endFrame(): RingStats {
    if (this._size > RING_WARN_BYTES && this._frameBytes > RING_WARN_BYTES) {
      console.warn(`[hpg] 单帧瞬态内存 ${this._frameBytes >> 20} MiB 超阈值 ${RING_WARN_BYTES >> 20} MiB，请将大容量数据迁至 static 资源。`);
    }
    const stats: RingStats = {
      bytesUsed: this._frameBytes,
      bytesCapacity: this._size,
      allocations: this._allocations,
    };
    this._head = 0;
    this._frameBytes = 0;
    return stats;
  }

  private grow(): void {
    this._size *= 2;
    // 不 destroy 旧 Buffer：它可能仍被已提交的命令缓冲引用；
    // 本对象释放引用后，实现/GC 会在 GPU 读完后回收。
    // 注意：调用方的 bind group 仍指向旧 Buffer，必须重建（见 Renderer.ensureInstanceBindGroup）。
    this._buffer = createDynamicBuffer(this.device, this._size);
    this._allocations++;
  }

  /** 销毁底层 GPUBuffer，释放瞬态显存。 */
  dispose(): void {
    try { this._buffer.destroy(); } catch { /* ignore */ }
  }
}

export function align16(x: number): number {
  return (x + 15) & ~15;
}

/** 向上对齐到 2 的幂 `alignment`（默认 16）。 */
export function alignTo(x: number, alignment = 16): number {
  return (x + alignment - 1) & ~(alignment - 1);
}

export function align256(x: number): number {
  return (x + 255) & ~255;
}

function createDynamicBuffer(device: GPUDevice, size: number): GPUBuffer {
  return device.createBuffer({
    label: 'hpg:ring',
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    mappedAtCreation: false,
  });
}