import { describe, it, expect } from 'vitest';
import { ExecutionBackend } from '../src/core/executor';
import type { Batch } from '../src/core/commands';
import type { ResolvedPipeline, RenderItem, VertexLayoutDesc } from '../src/types';

const LAYOUT: VertexLayoutDesc[] = [
  {
    stepMode: 'vertex',
    arrayStride: 32,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
  },
];

function fakePass() {
  const calls: string[] = [];
  const binds: { group: number; offsets?: number[] }[] = [];
  return {
    calls,
    binds,
    setPipeline: () => { calls.push('setPipeline'); },
    setBindGroup: (g: number, _bg: unknown, offsets?: number[]) => {
      calls.push('setBindGroup');
      binds.push({ group: g, offsets });
    },
    setVertexBuffer: () => { calls.push('setVertexBuffer'); },
    setIndexBuffer: () => { calls.push('setIndexBuffer'); },
    draw: (vc: number, ic: number) => { calls.push(`draw:${vc}:${ic}`); },
    drawIndexed: (ic: number, inst: number) => { calls.push(`drawIndexed:${ic}:${inst}`); },
    end: () => undefined,
  } as unknown as GPURenderPassEncoder & {
    calls: string[];
    binds: { group: number; offsets?: number[] }[];
  };
}

function makePipeline(id: number): ResolvedPipeline {
  return {
    id,
    desc: {} as any,
    pipeline: {} as GPURenderPipeline,
    layout: {} as GPUBindGroupLayout,
    bindGroupLayouts: [] as GPUBindGroupLayout[],
    bytesPerInstance: 80,
    modelMatrixOffset: 0,
    label: `pipeline-${id}`,
  };
}

function makeItem(pipeline: ResolvedPipeline, vertexCount = 3): RenderItem {
  const vSlice = { buffer: {} as GPUBuffer, byteOffset: 0, byteLength: 128 };
  return {
    pipeline,
    geometry: {
      vertexBuffer: {} as GPUBuffer,
      vertexSlice: vSlice,
      vertexBuffers: [vSlice],
      vertexLayouts: LAYOUT,
      vertexCount,
      indexBuffer: {} as GPUBuffer,
      indexSlice: { buffer: {} as GPUBuffer, byteOffset: 0, byteLength: 6 },
      indexFormat: 'uint16' as GPUIndexFormat,
      indexCount: vertexCount,
      primitive: 'triangle-list',
    },
    transforms: new Float32Array(16),
    instanceCount: 1,
  };
}

describe('ExecutionBackend', () => {
  it('run with single batch calls setPipeline + drawIndexed', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item = makeItem(pipeline);
    const batch: Batch = {
      offsetBytes: 0,
      instanceCount: 1,
      geometryIndex: 0,
      itemStart: 0,
      itemEnd: 1,
      pipeline,
      bindGroup: undefined,
    };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [batch], items: [item], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    expect(p.calls).toContain('setPipeline');
    expect(p.calls).toContain('drawIndexed:3:1');
    expect(stats.drawCalls).toBe(1);
    expect(stats.itemsDrawn).toBe(1);
  });

  it('run skips setPipeline when pipeline unchanged', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item1 = makeItem(pipeline);
    const item2 = makeItem(pipeline);
    const b1: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const b2: Batch = { offsetBytes: 256, instanceCount: 1, geometryIndex: 1, itemStart: 1, itemEnd: 2, pipeline, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [b1, b2], items: [item1, item2], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    const setPipelineCount = p.calls.filter(c => c === 'setPipeline').length;
    expect(setPipelineCount).toBe(1);
    expect(stats.drawCalls).toBe(2);
  });

  it('run switches pipeline when batch pipeline differs', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const p1 = makePipeline(1);
    const p2 = makePipeline(2);
    const item1 = makeItem(p1);
    const item2 = makeItem(p2);
    const b1: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline: p1, bindGroup: undefined };
    const b2: Batch = { offsetBytes: 256, instanceCount: 1, geometryIndex: 1, itemStart: 1, itemEnd: 2, pipeline: p2, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [b1, b2], items: [item1, item2], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    const setPipelineCount = p.calls.filter(c => c === 'setPipeline').length;
    expect(setPipelineCount).toBe(2);
  });

  it('run with non-indexed geometry calls draw (not drawIndexed)', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const vSlice2 = { buffer: {} as GPUBuffer, byteOffset: 0, byteLength: 128 };
    const item: RenderItem = {
      pipeline,
      geometry: {
        vertexBuffer: {} as GPUBuffer,
        vertexSlice: vSlice2,
        vertexBuffers: [vSlice2],
        vertexLayouts: LAYOUT,
        vertexCount: 6,
        indexFormat: 'uint16' as GPUIndexFormat,
        indexCount: 6,
        primitive: 'triangle-list',
      },
      transforms: new Float32Array(16),
      instanceCount: 1,
    };
    const batch: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [batch], items: [item], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    expect(p.calls).toContain('draw:6:1');
  });

  it('run with empty batches does nothing', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [], items: [], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    expect(stats.drawCalls).toBe(0);
    expect(p.calls).toHaveLength(0);
  });

  it('run rebinds group=1 dynamic offset for every batch (even same geometry)', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item = makeItem(pipeline);
    // 同一 geometry + 同一 pipeline 的两个批（例如材质 bind group 不同导致拆批），
    // 实例区起始位置不同 —— 必须在每批前重绑，否则第二批会读到第一批的实例数据。
    const b1: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const b2: Batch = { offsetBytes: 256, instanceCount: 1, geometryIndex: 0, itemStart: 1, itemEnd: 2, pipeline, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [b1, b2], items: [item], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    const group1 = p.binds.filter(b => b.group === 1);
    expect(group1.length).toBe(2);
    // 两个批的 dynamic offset 必须不同（instanceBufferOffset(frameBase + offsetBytes)）
    expect(group1[0]!.offsets).toEqual([0]);
    expect(group1[1]!.offsets).toEqual([256]);
  });

  it('run binds item.bindGroup at group=2', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item = makeItem(pipeline);
    const material = {} as GPUBindGroup;
    const batch: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: material };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [batch], items: [item], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    const group2 = p.binds.filter(b => b.group === 2);
    expect(group2.length).toBe(1);
  });

  it('run with non-indexed geometry does not bind index buffer', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item = makeItem(pipeline);
    const b1: Batch = { offsetBytes: 0, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const b2: Batch = { offsetBytes: 256, instanceCount: 1, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [b1, b2], items: [item], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    // 顶点绑定按 geometry 去重（同一 geometry 只绑一次）
    expect(p.calls.filter(c => c === 'setVertexBuffer').length).toBe(1);
  });

  it('run accumulates stats across multiple batches', () => {
    const backend = new ExecutionBackend();
    const p = fakePass();
    const pipeline = makePipeline(1);
    const item1 = makeItem(pipeline);
    const item2 = makeItem(pipeline);
    const b1: Batch = { offsetBytes: 0, instanceCount: 2, geometryIndex: 0, itemStart: 0, itemEnd: 1, pipeline, bindGroup: undefined };
    const b2: Batch = { offsetBytes: 256, instanceCount: 3, geometryIndex: 1, itemStart: 1, itemEnd: 2, pipeline, bindGroup: undefined };
    const stats = { drawCalls: 0, instances: 0, itemsDrawn: 0 };
    const bg = () => ({}) as GPUBindGroup;

    backend.run(p, { batches: [b1, b2], items: [item1, item2], instanceBindGroup: {} as GPUBindGroup, globalBindGroup: bg, frameBase: 0 }, stats);

    expect(stats.drawCalls).toBe(2);
    expect(stats.instances).toBe(5);
    expect(stats.itemsDrawn).toBe(2);
  });
});
