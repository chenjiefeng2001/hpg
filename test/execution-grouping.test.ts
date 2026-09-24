/**
 * 执行分组正确性 —— 材质 bind group 不得被错误复用。
 *
 * 这条不变量和之前的 dynamic-offset 越界、draw args 预写顺序属于同一级别：
 * **GPU 资源状态被错误复用**。分组键若为了「优化」而只用 (geometry, pipeline)，
 * 共享同一 geometry 但材质不同的实例就会共用第一个材质的 group=2 → 串贴图，
 * 而且不会有任何 validation error。
 *
 * 因此这里把不变量固化成测试：
 *
 *   same geometry + same pipeline + different bindGroup
 *        ↓
 *   必须产生独立的执行组（各自独立 draw / indirect args / dynamic offset）
 *
 * 覆盖 Direct（submit）与 GPU Culled（submitCulled），以及多 geometry × 多 material。
 */
import { describe, it, expect } from 'vitest';

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { MaterialStore } from '../src/core/texture';
import {
  VS_INSTANCED_MATERIAL,
  VS_INSTANCED_MATERIAL_COMPACTION,
  FS_MATERIAL,
} from '../src/shaders/instance';
import { createFakeGPU } from './fake-gpu';
import type { GltfAsset } from '../src/core/gltf';
import type { Geometry, RenderItem, ResolvedPipeline } from '../src/types';

const FORMAT: GPUTextureFormat = 'bgra8unorm';

const LAYOUT = [
  {
    arrayStride: 48,
    stepMode: 'vertex' as const,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
      { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },
      { shaderLocation: 2, offset: 24, format: 'float32x2' as GPUVertexFormat },
      { shaderLocation: 3, offset: 32, format: 'float32x4' as GPUVertexFormat },
    ],
  },
];

/** 单位 VP（剔除结果无关紧要，只验证执行组形态）。 */
const IDENTITY_VP = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

/** 两个材质的资产（无贴图 → 只有 material uniform，足够区分 bind group）。 */
function twoMaterialAsset(): GltfAsset {
  const material = (name: string, baseColorFactor: [number, number, number, number]) => ({
    name,
    baseColorFactor,
    metallicFactor: 0,
    roughnessFactor: 1,
    doubleSided: false,
    alphaMode: 'OPAQUE' as const,
    alphaCutoff: 0.5,
  });
  return {
    meshes: [],
    materials: [material('A', [1, 0, 0, 1]), material('B', [0, 1, 0, 1])],
    images: [],
    nodes: [],
    rootNodes: [],
    warnings: [],
  };
}

function makeRenderer() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = Renderer.create({ device, context, format: FORMAT });
  return { renderer, recorded };
}

function registerPipeline(
  renderer: Renderer,
  materialLayout: GPUBindGroupLayout,
  compaction: boolean,
): ResolvedPipeline {
  const globalLayout = uniformBindGroupLayout(renderer.device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniform = renderer.device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return renderer.registerPipeline({
    label: compaction ? 'grouping-culled' : 'grouping-direct',
    vsCode: compaction ? VS_INSTANCED_MATERIAL_COMPACTION : VS_INSTANCED_MATERIAL,
    fsCode: FS_MATERIAL,
    compaction,
    vertexLayouts: LAYOUT,
    bindGroupLayouts: [globalLayout, materialLayout],
    globalBindings: [{ binding: 0, buffer: uniform }],
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: FORMAT }],
  });
}

function makeGeometry(renderer: Renderer): Geometry {
  return renderer.createGeometry(
    new Float32Array(3 * 12),
    LAYOUT,
    new Uint16Array([0, 1, 2]),
  );
}

function item(
  geometry: Geometry,
  pipeline: ResolvedPipeline,
  bindGroup: GPUBindGroup | undefined,
  x: number,
): RenderItem {
  return {
    geometry,
    pipeline,
    bindGroup,
    transforms: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]),
  };
}

async function setup() {
  const { renderer, recorded } = makeRenderer();
  const store = await MaterialStore.create(renderer.device, twoMaterialAsset(), {
    decode: async () => ({ width: 1, height: 1, data: new Uint8Array(4) }),
  }, { label: 'grouping' });
  const bgA = store.bindGroupFor(0)!;
  const bgB = store.bindGroupFor(1)!;
  const direct = registerPipeline(renderer, store.layout, false);
  const culled = registerPipeline(renderer, store.layout, true);
  const geoA = makeGeometry(renderer);
  const geoB = makeGeometry(renderer);
  return { renderer, recorded, store, bgA, bgB, direct, culled, geoA, geoB };
}

function materialBinds(recorded: ReturnType<typeof createFakeGPU>['recorded']): unknown[] {
  return recorded.renderBinds.filter((b) => b.group === 2).map((b) => b.bindGroup);
}

describe('执行分组 — Direct（submit）', () => {
  it('同 geometry + 同 pipeline + 不同 bindGroup ⇒ 独立执行组（不合并、不串材质）', async () => {
    const { renderer, recorded, store, bgA, bgB, direct, geoA } = await setup();

    const stats = renderer.submit([item(geoA, direct, bgA, 0), item(geoA, direct, bgB, 1)]);

    expect(stats.batches).toBe(2);
    expect(stats.drawCalls).toBe(2);
    expect(recorded.drawCalls).toHaveLength(2);
    expect(materialBinds(recorded)).toEqual([bgA, bgB]);

    // 两个执行组的实例区起始必须不同，且第二个非零（否则会读到第一组的实例数据）。
    const offsets = recorded.renderBinds.filter((b) => b.group === 1).map((b) => b.offsets[0]!);
    expect(offsets).toHaveLength(2);
    expect(offsets[1]).toBeGreaterThan(0);
    expect(offsets[0]).not.toBe(offsets[1]);

    expect(recorded.gpuErrors).toEqual([]);
    store.dispose();
  });

  it('同 geometry + 同 pipeline + 同 bindGroup ⇒ 仍然合并为一个执行组', async () => {
    const { renderer, recorded, store, bgA, direct, geoA } = await setup();

    const stats = renderer.submit([item(geoA, direct, bgA, 0), item(geoA, direct, bgA, 1)]);

    expect(stats.batches).toBe(1);
    expect(recorded.drawCalls).toHaveLength(1);
    expect(recorded.drawCalls[0]!.instanceCount).toBe(2);
    expect(materialBinds(recorded)).toEqual([bgA]);

    store.dispose();
  });

  it('多 geometry × 多 material ⇒ 4 个独立执行组', async () => {
    const { renderer, recorded, store, bgA, bgB, direct, geoA, geoB } = await setup();

    const stats = renderer.submit([
      item(geoA, direct, bgA, 0),
      item(geoA, direct, bgB, 1),
      item(geoB, direct, bgA, 2),
      item(geoB, direct, bgB, 3),
    ]);

    expect(stats.batches).toBe(4);
    expect(recorded.drawCalls).toHaveLength(4);
    const binds = materialBinds(recorded);
    expect(binds).toHaveLength(4);
    expect(new Set(binds)).toEqual(new Set([bgA, bgB]));

    const offsets = recorded.renderBinds.filter((b) => b.group === 1).map((b) => b.offsets[0]!);
    expect(new Set(offsets).size).toBe(4);
    expect(recorded.gpuErrors).toEqual([]);

    store.dispose();
  });
});

describe('执行分组 — GPU Culled（submitCulled）', () => {
  it('同 geometry + 同 pipeline + 不同 bindGroup ⇒ 独立 indirect draw + 各自 group=2', async () => {
    const { renderer, recorded, store, bgA, bgB, culled, geoA } = await setup();

    const stats = renderer.submitCulled([item(geoA, culled, bgA, 0), item(geoA, culled, bgB, 1)], IDENTITY_VP);

    expect(stats.batches).toBe(2);
    expect(recorded.indirectDraws).toHaveLength(2);
    expect(materialBinds(recorded)).toEqual([bgA, bgB]);

    const offsets = recorded.renderBinds.filter((b) => b.group === 1).map((b) => b.offsets[0]!);
    expect(offsets).toHaveLength(2);
    expect(offsets[1]).toBeGreaterThan(0);
    expect(offsets[0]).not.toBe(offsets[1]);

    expect(recorded.gpuErrors).toEqual([]);
    store.dispose();
  });

  it('多 geometry × 多 material ⇒ 4 个 indirect draw', async () => {
    const { renderer, recorded, store, bgA, bgB, culled, geoA, geoB } = await setup();

    const stats = renderer.submitCulled(
      [
        item(geoA, culled, bgA, 0),
        item(geoA, culled, bgB, 1),
        item(geoB, culled, bgA, 2),
        item(geoB, culled, bgB, 3),
      ],
      IDENTITY_VP,
    );

    expect(stats.batches).toBe(4);
    expect(recorded.indirectDraws).toHaveLength(4);
    expect(materialBinds(recorded)).toHaveLength(4);
    expect(recorded.gpuErrors).toEqual([]);

    store.dispose();
  });
});
