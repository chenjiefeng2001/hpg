import { describe, it, expect } from 'vitest';
import { PipelineCache, canonicalize, fnv1a64 } from '../src/core/pipeline-cache';
import type { PipelineDesc } from '../src/types';

const MOCK_BGL = { label: 'mock-bgl' } as unknown as GPUBindGroupLayout;

function fakeDevice(): GPUDevice {
  return {
    createBuffer: () => ({}) as GPUBuffer,
    createBindGroupLayout: () => MOCK_BGL,
    createPipelineLayout: () => ({}) as GPUPipelineLayout,
    createShaderModule: () => ({}) as GPUShaderModule,
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }) as unknown as GPURenderPipeline,
    createBindGroup: () => ({}) as GPUBindGroup,
    queue: { submit: () => undefined, writeBuffer: () => undefined },
  } as unknown as GPUDevice;
}

function makeDesc(vs = 'vs1', fs = 'fs1'): PipelineDesc {
  return {
    vsCode: vs,
    fsCode: fs,
    vertexLayouts: [{
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
    }],
    bindGroupLayouts: [MOCK_BGL],
    globalBindings: [],
    targets: [{ format: 'bgra8unorm' as GPUColorTargetState['format'] }],
  };
}

describe('PipelineCache', () => {
  it('returns same pipeline for identical desc (deduplication)', () => {
    const cache = new PipelineCache();
    const device = fakeDevice();
    const desc = makeDesc();
    const p1 = cache.getOrCreate(device, desc);
    const p2 = cache.getOrCreate(device, desc);
    expect(p1).toBe(p2);
    expect(cache.size).toBe(1);
  });

  it('returns different pipelines for different desc', () => {
    const cache = new PipelineCache();
    const device = fakeDevice();
    const p1 = cache.getOrCreate(device, makeDesc('vs1', 'fs1'));
    const p2 = cache.getOrCreate(device, makeDesc('vs2', 'fs2'));
    expect(p1).not.toBe(p2);
    expect(cache.size).toBe(2);
  });

  it('onStats reports created=true for miss and created=false for hit', () => {
    const cache = new PipelineCache();
    const device = fakeDevice();
    const desc = makeDesc();
    const stats: boolean[] = [];
    cache.getOrCreate(device, desc, (created) => stats.push(created));
    cache.getOrCreate(device, desc, (created) => stats.push(created));
    expect(stats).toEqual([true, false]);
  });
});

describe('canonicalize', () => {
  it('deterministic: same desc → same string', () => {
    const d = makeDesc();
    expect(canonicalize(d)).toBe(canonicalize(d));
  });

  it('different desc → different string', () => {
    const d1 = makeDesc('vs1', 'fs1');
    const d2 = makeDesc('vs2', 'fs2');
    expect(canonicalize(d1)).not.toBe(canonicalize(d2));
  });
});

describe('fnv1a64', () => {
  it('produces consistent hashes', () => {
    const h1 = fnv1a64('hello');
    const h2 = fnv1a64('hello');
    expect(h1).toBe(h2);
  });

  it('different inputs produce different hashes', () => {
    expect(fnv1a64('hello')).not.toBe(fnv1a64('world'));
  });

  it('returns bigint', () => {
    expect(typeof fnv1a64('test')).toBe('bigint');
  });
});
