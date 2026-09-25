/**
 * GPU 时间戳查询工具。
 *
 * 用法：
 *   const tq = new TimestampQuery(device, 2);
 *   const pass = encoder.beginRenderPass({
 *     colorAttachments,
 *     timestampWrites: tq.timestampWrites(0, 1),
 *   });
 *   tq.resolve(encoder);
 *   device.queue.submit([encoder.finish()]);
 *   const ns = await tq.readback(device);  // 异步回读 GPU 耗时（纳秒）
 */

export class TimestampQuery {
  readonly querySet: GPUQuerySet;
  private _resolveBuffer: GPUBuffer;
  private _stagingBuffer: GPUBuffer | null = null;
  private _pending = false;
  private _disposed = false;
  private _count: number;

  constructor(device: GPUDevice, count = 2, label = 'hpg:timestamp') {
    if (device.features && !device.features.has('timestamp-query')) {
      throw new Error('TimestampQuery requires the timestamp-query device feature.');
    }
    this._count = count;
    const querySet = device.createQuerySet({
      type: 'timestamp',
      count,
      label,
    });
    let resolveBuffer: GPUBuffer;
    try {
      // 8 bytes per query (u64 timestamp).
      resolveBuffer = device.createBuffer({
        size: count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        label: `${label}:resolve`,
      });
    } catch (error) {
      try {
        querySet.destroy();
      } catch {}
      throw error;
    }
    this.querySet = querySet;
    this._resolveBuffer = resolveBuffer;
  }

  timestampWrites(beginningOfPassWriteIndex = 0, endOfPassWriteIndex = this._count - 1): {
    querySet: GPUQuerySet;
    beginningOfPassWriteIndex: number;
    endOfPassWriteIndex: number;
  } {
    if (this._disposed) throw new Error('TimestampQuery has been destroyed.');
    if (
      !Number.isSafeInteger(beginningOfPassWriteIndex) ||
      !Number.isSafeInteger(endOfPassWriteIndex) ||
      beginningOfPassWriteIndex < 0 ||
      endOfPassWriteIndex <= beginningOfPassWriteIndex ||
      endOfPassWriteIndex >= this._count
    ) {
      throw new Error('TimestampQuery timestamp write indices are out of range.');
    }
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex,
      endOfPassWriteIndex,
    };
  }

  /** 将 QuerySet 解析到 resolve buffer。在 encoder.finish() 前调用。 */
  resolve(encoder: GPUCommandEncoder): void {
    if (this._disposed) throw new Error('TimestampQuery has been destroyed.');
    encoder.resolveQuerySet(this.querySet, 0, this._count, this._resolveBuffer, 0);
  }

  /**
   * 异步回读 GPU 时间戳结果（纳秒）。
   * 必须在 resolve + submit 后调用；内部创建 staging buffer 做 COPY_BUFFER_TO_MAP。
   * 返回一个 Promise<number[]> —— 每个 slot 的 GPU 时间戳（ns）。
   */
  async readback(device: GPUDevice): Promise<number[]> {
    if (this._disposed) throw new Error('TimestampQuery has been destroyed.');
    if (this._pending) throw new Error('TimestampQuery: previous readback still pending');

    this._pending = true;
    let stagingBuffer: GPUBuffer | null = null;
    let mapped = false;
    try {
      // 创建 staging buffer（COPY_DST + MAP_READ）。
      const size = this._count * 8;
      stagingBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'hpg:timestamp:staging',
      });
      this._stagingBuffer = stagingBuffer;

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(this._resolveBuffer, 0, stagingBuffer, 0, size);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;
      const data = stagingBuffer.getMappedRange();
      const timestamps: number[] = [];
      const view = new DataView(data);
      for (let i = 0; i < this._count; i++) {
        // WebGPU timestamps 是 BigUint64，JavaScript 安全整数范围足够（<53 bits）。
        timestamps.push(Number(view.getBigUint64(i * 8, false)));
      }
      return timestamps;
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
      this._stagingBuffer = null;
      this._pending = false;
    }
  }

  /** 释放 GPU 资源。 */
  destroy(): void {
    if (this._disposed) return;
    if (this._pending) throw new Error('TimestampQuery cannot be destroyed while readback is pending.');
    this._disposed = true;
    const stagingBuffer = this._stagingBuffer;
    this._stagingBuffer = null;
    try {
      this.querySet.destroy();
    } finally {
      try {
        this._resolveBuffer.destroy();
      } finally {
        stagingBuffer?.destroy();
      }
    }
  }
}
