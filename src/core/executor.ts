/**
 * ExecutionBackend —— 命令录制后端。
 *
 * 统一 Direct / Instanced / Indirect：
 *   - `run()`: 传统 drawIndexed 路径（CPU 驱动）
 *   - `runIndirect()`: drawIndexedIndirect 路径（GPU 驱动）
 *
 * 实例存储使用动态偏移（dynamicOffset）绑定：帧内所有批共享同一个
 * ring buffer + 同一个 group1 实例 bind group，仅通过 256 对齐的
 * dynamicOffset 切换各批的起始位置 —— 避免逐批创建 bind group。
 */

import type { ResolvedPipeline, RenderItem } from '../types';
import type { Batch } from './commands';
import { instanceBufferOffset } from './commands';

export interface ExecutorInput {
  batches: Batch[];
  items: RenderItem[];
  /** 实例 bind group（group=1），动态偏移指向各批起始。 */
  instanceBindGroup: GPUBindGroup;
  /** 各管线 group=0 全局 bind group。 */
  globalBindGroup: (p: ResolvedPipeline) => GPUBindGroup;
  /** 本帧实例区起始（256 对齐）字节偏移。 */
  frameBase: number;
}

/** drawIndexedIndirect 参数布局（20 字节，WebGPU 规范）。 */
export interface DrawIndexedIndirectArgs {
  indexCount: number;
  instanceCount: number;
  firstIndex: number;
  baseVertex: number;
  firstInstance: number;
}

export interface ExecutorStats {
  drawCalls: number;
  instances: number;
  itemsDrawn: number;
}

export function assertRequiredBindGroup(item: RenderItem): void {
  if (item.pipeline.bindGroupLayouts.length > 2 && !item.bindGroup) {
    throw new Error(`Pipeline "${item.pipeline.label}" requires a bindGroup for group 2.`);
  }
}

export class ExecutionBackend {
  /**
   * 传统 drawIndexed 路径（CPU 驱动）。
   */
  run(
    pass: GPURenderPassEncoder,
    { batches, items, instanceBindGroup, globalBindGroup, frameBase }: ExecutorInput,
    stats: ExecutorStats,
  ): void {
    let lastPipeline: ResolvedPipeline | null = null;
    let lastGeometry: RenderItem['geometry'] | null = null;
    let lastBindGroup: GPUBindGroup | undefined;

    for (const batch of batches) {
      const item = items[batch.geometryIndex] as RenderItem;
      assertRequiredBindGroup(item);
      const geometry = item.geometry;
      const pipeline = batch.pipeline;

      if (pipeline !== lastPipeline) {
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, globalBindGroup(pipeline));
        lastPipeline = pipeline;
        lastGeometry = null;
        lastBindGroup = undefined;
      }

      // 每批的实例区起始位置都不同（dynamic offset），必须逐批重绑 ——
      // 只在切换 pipeline/geometry 时重绑会让相邻同 geometry 的两个批读到错位的实例数据。
      pass.setBindGroup(1, instanceBindGroup, [
        instanceBufferOffset(frameBase + batch.offsetBytes),
      ]);

      // 额外 bind group（材质等）固定占用 group=2。
      if (batch.bindGroup && batch.bindGroup !== lastBindGroup) {
        pass.setBindGroup(2, batch.bindGroup);
        lastBindGroup = batch.bindGroup;
      }

      // 顶点绑定（多 slot）——仅 geometry 变化时重绑。
      if (geometry !== lastGeometry) {
        const vbs = geometry.vertexBuffers;
        for (let s = 0; s < vbs.length; s++) {
         const vb = vbs[s]!;
         pass.setVertexBuffer(s, vb.buffer, vb.byteOffset, vb.byteLength);
        }
      }

      if (geometry.indexSlice && geometry.indexBuffer) {
        pass.setIndexBuffer(geometry.indexBuffer, geometry.indexFormat, geometry.indexSlice.byteOffset, geometry.indexSlice.byteLength);
        pass.drawIndexed(geometry.indexCount, batch.instanceCount, 0, 0, 0);
      } else {
        pass.draw(geometry.vertexCount, batch.instanceCount, 0, 0);
      }

      stats.drawCalls++;
      stats.instances += batch.instanceCount;
      stats.itemsDrawn += batch.itemEnd - batch.itemStart;
      lastGeometry = geometry;
    }
  }

  /**
   * drawIndexedIndirect 路径（GPU 驱动）。
   * 从 indirectBuffer 中读取每个 batch 的绘制参数，实现 GPU 驱动剔除后的间接绘制。
   *
   * @param indirectBuffer 包含 N 个 DrawIndexedIndirectArgs 的 GPUBuffer
   * @param indirectOffsetBytes indirectBuffer 的起始偏移（字节）
   */
  runIndirect(
    pass: GPURenderPassEncoder,
    { batches, items, instanceBindGroup, globalBindGroup, frameBase }: ExecutorInput,
    stats: ExecutorStats,
    indirectBuffer: GPUBuffer,
    indirectOffsetBytes = 0,
  ): void {
    let lastPipeline: ResolvedPipeline | null = null;
    let lastGeometry: RenderItem['geometry'] | null = null;
    let lastBindGroup: GPUBindGroup | undefined;

    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i] as Batch;
      const item = items[batch.geometryIndex] as RenderItem;
      assertRequiredBindGroup(item);
      const geometry = item.geometry;
      const pipeline = batch.pipeline;

      if (pipeline !== lastPipeline) {
        pass.setPipeline(pipeline.pipeline);
        pass.setBindGroup(0, globalBindGroup(pipeline));
        lastPipeline = pipeline;
        lastGeometry = null;
        lastBindGroup = undefined;
      }

      pass.setBindGroup(1, instanceBindGroup, [
        instanceBufferOffset(frameBase + batch.offsetBytes),
      ]);
      if (batch.bindGroup && batch.bindGroup !== lastBindGroup) {
        pass.setBindGroup(2, batch.bindGroup);
        lastBindGroup = batch.bindGroup;
      }

      // 顶点绑定（多 slot）。
      if (geometry !== lastGeometry) {
        const vbs = geometry.vertexBuffers;
        for (let s = 0; s < vbs.length; s++) {
         const vb = vbs[s]!;
         pass.setVertexBuffer(s, vb.buffer, vb.byteOffset, vb.byteLength);
        }
      }

      if (geometry.indexSlice && geometry.indexBuffer) {
        pass.setIndexBuffer(geometry.indexBuffer, geometry.indexFormat, geometry.indexSlice.byteOffset, geometry.indexSlice.byteLength);
        const argsOffset = indirectOffsetBytes + i * 20; // 20 bytes per DrawIndexedIndirectArgs
        pass.drawIndexedIndirect(indirectBuffer, argsOffset);
      } else {
        pass.drawIndirect(indirectBuffer, indirectOffsetBytes + i * 20);
      }

      stats.drawCalls++;
      stats.instances += batch.instanceCount;
      stats.itemsDrawn += batch.itemEnd - batch.itemStart;
      lastGeometry = geometry;
    }
  }
}