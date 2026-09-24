/**
 * Batcher —— 状态排序 + 自动合批 + 实例重排（Remap）。
 *
 * 输入：提交顺序的 RenderItem[] 及其排序键。
 * 处理：按 pipeline→geometry 相邻合并为 Batch；同时把"实例如何落到瞬态缓冲"
 *       的顺序记录下来，供 Executor 顺序写入矩阵/实例数据。
 *
 * 目标：让最终发给 WebGPU 的 CommandEncoder 严格按
 *       Pipeline → BindGroup → Mesh 递增排序，把状态切换成本压到物理极限。
 */

import type { ResolvedPipeline, RenderItem } from '../types';
import type { Batch } from './commands';
import { instanceBufferOffset } from './commands';

export interface BatchedResult {
  batches: Batch[];
  /** 重排后的实例顺序：每项 { sourceItem, sourceInstance }（扁平）。 */
  order: { sourceItem: number; sourceInstance: number }[];
  /** 写入瞬态缓冲的字节数（尚未按 256 对齐）。 */
  bytesToWrite: number;
}

export class Batcher {
  private _order: { sourceItem: number; sourceInstance: number }[] = [];
  private _batches: Batch[] = [];

  get batches(): Batch[] {
    return this._batches;
  }

  get order(): { sourceItem: number; sourceInstance: number }[] {
    return this._order;
  }

  /**
   * 重排 items（已按排序键升序），合并相邻同 geometry+pipeline 的实例。
   *
   * 实例落位跨步**以批所属管线的 `bytesPerInstance` 为准** —— 它等于该管线 WGSL 中
   * `InstanceData` 的结构体大小。着色器永远按自己的结构体跨步索引实例，所以帧内混用
   * 不同跨步的管线时不能用「帧内最大跨步」统一排布（会让跨步较小的管线读到错位数据）。
   * 批起始仍按 256 字节对齐，以满足 storage dynamic offset 的要求。
   *
   * @param items 已排序的提交项
   * @param fallbackBytesPerInstance 管线跨步缺失时的兜底值（默认 80）
   */
  collect(items: RenderItem[], fallbackBytesPerInstance = 80): BatchedResult {
    this._order.length = 0;
    this._batches.length = 0;

    // Track actual GPU byte position (256-aligned per batch).
    let gpuOffset = 0;

    const n = items.length;
    for (let i = 0; i < n; ) {
      const item = items[i] as RenderItem;
      const instanceCount = instanceCountOf(item);

      let j = i + 1;
      // 相邻同 pipeline + geometry 合并为一批。
      while (
        j < n &&
        items[j]!.pipeline.id === item.pipeline.id &&
        items[j]!.geometry === item.geometry &&
        items[j]!.bindGroup === item.bindGroup
      ) {
        j++;
      }
      // 该批覆盖 items[i .. j-1]。
      let batchInstances = 0;
      for (let k = i; k < j; k++) {
        batchInstances += instanceCountOf(items[k] as RenderItem);
      }

      const offset = instanceBufferOffset(gpuOffset);
      // 批内所有 item 共享同一管线 → 跨步取该管线的实例结构体大小。
      const stride = item.pipeline.bytesPerInstance || fallbackBytesPerInstance;
      gpuOffset = offset + batchInstances * stride;

      this._batches.push({
        offsetBytes: offset,
        instanceCount: batchInstances,
        geometryIndex: i,
        itemStart: i,
        itemEnd: j,
        pipeline: item.pipeline,
        bindGroup: item.bindGroup,
      });

      // 记录实例顺序（用于写入瞬态缓冲）。
      for (let k = i; k < j; k++) {
        const nk = instanceCountOf(items[k] as RenderItem);
        for (let m = 0; m < nk; m++) {
          this._order.push({ sourceItem: k, sourceInstance: m });
        }
      }
      i = j;
    }

    return {
      batches: this._batches,
      order: this._order,
      bytesToWrite: gpuOffset,
    };
  }
}

export function instanceCountOf(item: RenderItem): number {
  if (item.instanceCount !== undefined) return item.instanceCount;
  const t = item.transforms;
  return t ? t.length / 16 : 1;
}