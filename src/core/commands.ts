/**
 * 录制的渲染命令 IR —— 由 Batcher 生成、Executor 消费。
 * 相比直接驱动 GPU 调用，IR 是自描述的（可被测试、可被未来 Indirect 后端重放）。
 */

import type { ResolvedPipeline } from '../types';

/** 对齐到 256 字节的实例起始偏移（满足 minStorageBufferOffsetAlignment）。 */
export function instanceBufferOffset(rawOffset: number): number {
  return Math.ceil(rawOffset / 256) * 256;
}

export interface Batch {
  /** 该批首个实例在实例缓冲中的 256 对齐字节偏移。 */
  offsetBytes: number;
  /** 该批实例个数。 */
  instanceCount: number;
  /** 首项（前向）指针 —— 用于访问 geometry/pipeline/bindGroup。 */
  geometryIndex: number;
  /** 该批覆盖的（已排序）items 区间 [itemStart, itemEnd)。 */
  itemStart: number;
  itemEnd: number;
  pipeline: ResolvedPipeline;
  bindGroup: GPUBindGroup | undefined;
}