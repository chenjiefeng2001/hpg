import { describe, it, expect } from 'vitest';
import { Batcher, instanceCountOf } from '../src/core/batcher';
import { instanceBufferOffset } from '../src/core/commands';
import type { Geometry, RenderItem, ResolvedPipeline } from '../src/types';

function fakePipeline(id: number, bytesPerInstance = 80): ResolvedPipeline {
  return {
    id,
    desc: { vsCode: '', fsCode: '', vertexLayouts: [], bindGroupLayouts: [], targets: [], globalBindings: [], bytesPerInstance },
    pipeline: {} as GPURenderPipeline,
    layout: {} as GPUBindGroupLayout,
    bindGroupLayouts: [],
    bytesPerInstance,
    modelMatrixOffset: 0,
    label: `p${id}`,
  };
}

function fakeGeometry(name: string): Geometry {
  const slice = { buffer: {} as GPUBuffer, byteOffset: 0, byteLength: 1 };
  return {
    vertexBuffer: {} as GPUBuffer,
    vertexSlice: slice,
    vertexBuffers: [slice],
    vertexLayouts: [],
    indexFormat: 'uint16',
    indexCount: 0,
    vertexCount: 0,
    primitive: 'triangle-list',
    ...(name === 'A' ? {} : {}),
  };
}

function item(pipeline: ResolvedPipeline, geometry: Geometry, transforms?: Float32Array): RenderItem {
  return { geometry, pipeline, transforms };
}

describe('batcher', () => {
  it('instanceBufferOffset aligns to 256', () => {
    expect(instanceBufferOffset(0)).toBe(0);
    expect(instanceBufferOffset(1)).toBe(256);
    expect(instanceBufferOffset(255)).toBe(256);
    expect(instanceBufferOffset(256)).toBe(256);
  });

  it('instanceCountOf defaults to 1 and derives from transforms', () => {
    const p = fakePipeline(0);
    const g = fakeGeometry('A');
    expect(instanceCountOf(item(p, g))).toBe(1);
    expect(instanceCountOf(item(p, g, new Float32Array(32)))).toBe(2);
  });

  it('groups adjacent same pipeline+geometry into a single batch', () => {
    const p = fakePipeline(1);
    const g = fakeGeometry('A');
    const b = new Batcher();
    const items = [
      item(p, g, new Float32Array(16)),
      item(p, g, new Float32Array(16)),
      item(p, g, new Float32Array(16)),
    ];
    const res = b.collect(items, 80);
    expect(res.batches.length).toBe(1);
    expect(res.batches[0]!.instanceCount).toBe(3);
    expect(res.batches[0]!.offsetBytes).toBe(0);
  });

  it('splits batches when pipeline or geometry differs', () => {
    const p1 = fakePipeline(1);
    const p2 = fakePipeline(2);
    const gA = fakeGeometry('A');
    const gB = fakeGeometry('B');
    const b = new Batcher();
    const items = [item(p1, gA), item(p2, gA), item(p1, gB), item(p1, gA)];
    const res = b.collect(items, 80);
    expect(res.batches.length).toBe(4);
    expect(res.batches.map((x) => x.instanceCount)).toEqual([1, 1, 1, 1]);
  });

  it('computes cumulative 256-aligned offsets for multi-instance batches', () => {
    const p = fakePipeline(1);
    const g = fakeGeometry('A');
    const b = new Batcher();
    const items = [
      item(p, g, new Float32Array(16)),
      item(p, g, new Float32Array(32)),
    ];
    const res = b.collect(items, 80);
    // 批1：2 实例（1+... wait item0 has 1, item1 has 2 → 批内合并同 pipeline+geometry → 3 实例。
    expect(res.batches.length).toBe(1);
    expect(res.batches[0]!.instanceCount).toBe(3);
    expect(res.batches[0]!.offsetBytes).toBe(0);
    expect(res.bytesToWrite).toBe(3 * 80);
  });

  it('records per-instance order for remap', () => {
    const p = fakePipeline(1);
    const g = fakeGeometry('A');
    const b = new Batcher();
    const items = [
      item(p, g, new Float32Array(32)), // 2 instances
      item(p, g, new Float32Array(16)), // 1 instance
    ];
    const res = b.collect(items, 80);
    expect(res.order).toEqual([
      { sourceItem: 0, sourceInstance: 0 },
      { sourceItem: 0, sourceInstance: 1 },
      { sourceItem: 1, sourceInstance: 0 },
    ]);
    expect(res.batches[0]!.itemStart).toBe(0);
    expect(res.batches[0]!.itemEnd).toBe(2);
  });

  it('does not collide offsets when multiple batches need 256-alignment', () => {
    const p = fakePipeline(1);
    const gA = fakeGeometry('A');
    const gB = fakeGeometry('B');
    const gC = fakeGeometry('C');
    const b = new Batcher();
    // 3 items with different geometries → 3 batches
    const items = [item(p, gA), item(p, gB), item(p, gC)];
    const res = b.collect(items, 80);
    expect(res.batches.length).toBe(3);
    // Each batch must have a unique 256-aligned offset
    expect(res.batches[0]!.offsetBytes).toBe(0);
    expect(res.batches[1]!.offsetBytes).toBe(256);
    expect(res.batches[2]!.offsetBytes).toBe(512);
    // bytesToWrite must cover all data including alignment gaps
    expect(res.bytesToWrite).toBe(592);
  });

  it('keeps pipeline-adjacent batches even after manual order', () => {
    const pA = fakePipeline(1);
    const pB = fakePipeline(2);
    const g = fakeGeometry('A');
    const b = new Batcher();
    // 模拟已排序：pA×2 → pB×1 → pA×1（相邻 pA 可合并）。
    const items = [item(pA, g), item(pA, g), item(pB, g), item(pA, g)];
    const res = b.collect(items, 80);
    expect(res.batches.length).toBe(3);
    expect(res.batches[0]!.instanceCount).toBe(2);
  });
});