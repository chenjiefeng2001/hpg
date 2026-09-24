/**
 * GPU 时间戳查询工具。
 *
 * 用法：
 *   const tq = new TimestampQuery(device, 2);
 *   tq.writeTimestamp(pass, 0);   // pass 开始前
 *   // ... draw calls ...
 *   tq.writeTimestamp(pass, 1);   // pass 结束前
 *   tq.resolve(encoder);
 *   device.queue.submit([encoder.finish()]);
 *   const ns = await tq.readback();  // 异步回读 GPU 耗时（纳秒）
 */

export class TimestampQuery {
  readonly querySet: GPUQuerySet;
  private _resolveBuffer: GPUBuffer;
  private _stagingBuffer: GPUBuffer | null = null;
  private _pending = false;
  private _count: number;

  constructor(device: GPUDevice, count = 2, label = 'hpg:timestamp') {
    this._count = count;
    this.querySet = device.createQuerySet({
      type: 'timestamp',
      count,
      label,
    });
    // 8 bytes per query (u64 timestamp).
    this._resolveBuffer = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      label: `${label}:resolve`,
    });
  }

  /** 在 render/compute pass 中写入时间戳到指定 slot。 */
  writeTimestamp(pass: GPURenderPassEncoder | GPUComputePassEncoder, index: number): void {
    // WebGPU spec: writeTimestamp is on GPUBindingCommandsMixin (both render & compute passes).
    // TypeScript types may not include it yet.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (pass as any).writeTimestamp(this.querySet, index);
  }

  /** 将 QuerySet 解析到 resolve buffer。在 encoder.finish() 前调用。 */
  resolve(encoder: GPUCommandEncoder): void {
    encoder.resolveQuerySet(this.querySet, 0, this._count, this._resolveBuffer, 0);
  }

  /**
   * 异步回读 GPU 时间戳结果（纳秒）。
   * 必须在 resolve + submit 后调用；内部创建 staging buffer 做 COPY_BUFFER_TO_MAP。
   * 返回一个 Promise<number[]> —— 每个 slot 的 GPU 时间戳（ns）。
   */
  async readback(device: GPUDevice): Promise<number[]> {
    if (this._pending) throw new Error('TimestampQuery: previous readback still pending');

    // 创建 staging buffer（COPY_DST + MAP_READ）。
    const size = this._count * 8;
    this._stagingBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      label: 'hpg:timestamp:staging',
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this._resolveBuffer, 0, this._stagingBuffer, 0, size);
    device.queue.submit([encoder.finish()]);

    this._pending = true;
    await this._stagingBuffer.mapAsync(GPUMapMode.READ);
    const data = this._stagingBuffer.getMappedRange();
    const timestamps: number[] = [];
    const view = new DataView(data);
    for (let i = 0; i < this._count; i++) {
      // WebGPU timestamps 是 BigUint64，JavaScript 安全整数范围足够（<53 bits）。
      timestamps.push(Number(view.getBigUint64(i * 8, false)));
    }
    this._stagingBuffer.unmap();
    this._stagingBuffer.destroy();
    this._stagingBuffer = null;
    this._pending = false;
    return timestamps;
  }

  /** 释放 GPU 资源。 */
  destroy(): void {
    this.querySet.destroy();
    this._resolveBuffer.destroy();
    if (this._stagingBuffer) {
      this._stagingBuffer.destroy();
      this._stagingBuffer = null;
    }
  }
}
