/**
 * 管线描述符 / 实例记录布局的「声明必须等于行为」回归。
 *
 * 状态审计发现三处**声明了但被静默忽略**的执行边界契约（属于同一类问题：
 * 「不支持的东西必须明确报错，而不是悄悄渲染错」）：
 *
 *   1. `GlobalBinding.byteOffset` / `byteLength` —— 被忽略，永远绑整块 buffer；
 *      且管线缓存键只含 `binding:buffer.size`，同一 buffer 的不同切片会塌成一个管线。
 *   2. `sceneToRenderItems()` 的第 3 个参数 `globalBindings` —— 从未被使用
 *      （全局 uniform 由 `PipelineDesc.globalBindings` 在注册时声明）→ 已移除。
 *   3. `PipelineDesc.modelMatrixOffset` —— 被忽略，mat4 永远写在记录开头；
 *      着色器若把矩阵放在别的偏移会读到错位数据。
 *
 * 本文件把 1 和 3 的行为钉死（2 是纯签名修正，由编译期保证）。
 */
import { describe, it, expect } from 'vitest';

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { VS_INSTANCED, VS_INSTANCED_COMPACTION, FS_COLOR } from '../src/shaders/instance';
import { createFakeGPU } from './fake-gpu';
import type { Geometry, GlobalBinding, PipelineDesc, RenderItem, ResolvedPipeline } from '../src/types';

const FORMAT: GPUTextureFormat = 'bgra8unorm';
const DEPTH = { format: 'depth24plus' as GPUTextureFormat, depthWriteEnabled: true, depthCompare: 'less' as GPUCompareFunction };
const LAYOUT = [
  {
    arrayStride: 12,
    stepMode: 'vertex' as const,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
  },
];

function setup() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = Renderer.create({ device, context, format: FORMAT });
  return { renderer, device, recorded };
}

function makeGeometry(renderer: Renderer): Geometry {
  return renderer.createGeometry(new Float32Array(9), LAYOUT, new Uint16Array([0, 1, 2]));
}

function customInstanceLayoutShader(base: string, modelMatrixOffset: number): string {
  const marker = `struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};`;
  const replacement = modelMatrixOffset === 16
    ? `struct InstanceData {
    padding: vec4<f32>,
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};`
    : `struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
    padding: vec4<f32>,
};`;
  if (!base.includes(marker)) throw new Error('test shader does not contain InstanceData');
  return base.replace(marker, replacement);
}

function register(
  renderer: Renderer,
  device: GPUDevice,
  globalBindings: GlobalBinding[],
  extra: Partial<PipelineDesc> = {},
): ResolvedPipeline {
  const baseShader = extra.vsCode ?? (extra.compaction ? VS_INSTANCED_COMPACTION : VS_INSTANCED);
  const customLayout = extra.bytesPerInstance !== undefined || extra.modelMatrixOffset !== undefined;
  const vsCode = customLayout ? customInstanceLayoutShader(baseShader, extra.modelMatrixOffset ?? 0) : baseShader;
  return renderer.registerPipeline({
    label: 'descriptor-contract',
    vsCode,
    fsCode: FS_COLOR,
    vertexLayouts: LAYOUT,
    bindGroupLayouts: [uniformBindGroupLayout(device, [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
    ])],
    globalBindings,
    depth: DEPTH,
    targets: [{ format: FORMAT }],
    compactionContract: extra.compaction && customLayout ? 'hpg-compaction-v1' : extra.compactionContract,
    ...extra,
  });
}

function item(geometry: Geometry, pipeline: ResolvedPipeline, t?: Float32Array, data?: Float32Array): RenderItem {
  const it: RenderItem = { geometry, pipeline };
  if (t) it.transforms = t;
  if (data) it.instanceData = data;
  return it;
}

/** 取入队时恰好 size 字节的实例区写入。 */
function instanceWrite(recorded: ReturnType<typeof createFakeGPU>['recorded'], size: number) {
  const w = recorded.writes.find((x) => x.bytes.byteLength === size);
  if (!w) throw new Error(`没有找到 ${size} 字节的实例写入`);
  return new Float32Array(w.bytes.buffer, w.bytes.byteOffset, size / 4);
}

// ─── 1. GlobalBinding 切片 ──────────────────────────────────

describe('GlobalBinding 契约', () => {
  it('byteOffset / byteLength 真正进入 createBindGroup（不再静默绑整块 buffer）', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const sliced = register(renderer, device, [{ binding: 0, buffer, byteOffset: 256, byteLength: 64 }]);
    renderer.submit([item(makeGeometry(renderer), sliced)]);
    const slicedBG = recorded.bindGroups.find((b) => b.label.includes('global'));
    expect(slicedBG).toBeDefined();
    expect(slicedBG!.bindings[0]!.offset).toBe(256);
    expect(slicedBG!.bindings[0]!.size).toBe(64);
  });

  it('同一 buffer 的不同切片 ⇒ 不同管线 + 不同 bind group（缓存键含 offset/size）', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const lower = register(renderer, device, [{ binding: 0, buffer, byteOffset: 0, byteLength: 64 }]);
    const upper = register(renderer, device, [{ binding: 0, buffer, byteOffset: 256, byteLength: 64 }]);
    expect(lower.id).not.toBe(upper.id);

    const geo = makeGeometry(renderer);
    renderer.submit([item(geo, lower), item(geo, upper)]);

    const globals = recorded.bindGroups.filter((b) => b.label.includes('global'));
    expect(globals).toHaveLength(2);
    expect(globals.map((g) => g.bindings[0]!.offset).sort((a, b) => a - b)).toEqual([0, 256]);
  });

  it('拒绝未按设备 uniform 对齐的静态偏移', () => {
    const { renderer, device } = setup();
    const buffer = device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    expect(() => register(renderer, device, [{ binding: 0, buffer, byteOffset: 64, byteLength: 64 }])).toThrow(/aligned/);
  });

  it('未声明 offset/size 时保持原有行为（绑整块 buffer，无 static offset）', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = register(renderer, device, [{ binding: 0, buffer }]);
    renderer.submit([item(makeGeometry(renderer), p)]);

    const bg = recorded.bindGroups.find((b) => b.label.includes('global'));
    expect(bg!.bindings[0]!.offset).toBe(0);
    expect(bg!.bindings[0]!.size).toBeNull();
  });
});

// ─── 2. modelMatrixOffset ───────────────────────────────────

describe('modelMatrixOffset 契约', () => {
  const TRANSFORM = new Float32Array([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 7, 8, 9, 1]);
  const EXTRA = new Float32Array([0.1, 0.2, 0.3, 0.4]);

  it('submit：mat4 写在 modelMatrixOffset 处，预留区清零，额外数据紧随其后', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = register(renderer, device, [{ binding: 0, buffer }], {
      bytesPerInstance: 96, // 16(预留) + 64(mat4) + 16(额外)
      modelMatrixOffset: 16,
    });

    renderer.submit([item(makeGeometry(renderer), p, TRANSFORM, EXTRA)]);

    const f = instanceWrite(recorded, 96);
    // 预留区
    expect(Array.from(f.slice(0, 4))).toEqual([0, 0, 0, 0]);
    // mat4 落在 float 4..19
    expect(Array.from(f.slice(4, 20))).toEqual(Array.from(TRANSFORM));
    // 额外数据（实例颜色）落在 float 20..23
    for (let i = 0; i < 4; i++) expect(f[20 + i]!).toBeCloseTo(EXTRA[i]!, 6);
  });

  it('submitCulled：同一布局同样生效', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = register(renderer, device, [{ binding: 0, buffer }], {
      bytesPerInstance: 96,
      modelMatrixOffset: 16,
      compaction: true,
    });
    const geo = makeGeometry(renderer);
    renderer.submitCulled([item(geo, p, TRANSFORM, EXTRA)], new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));

    const f = instanceWrite(recorded, 96);
    expect(Array.from(f.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(f.slice(4, 20))).toEqual(Array.from(TRANSFORM));
    for (let i = 0; i < 4; i++) expect(f[20 + i]!).toBeCloseTo(EXTRA[i]!, 6);
  });

  it('默认（modelMatrixOffset 未声明）时 mat4 仍在记录开头', () => {
    const { renderer, device, recorded } = setup();
    const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const p = register(renderer, device, [{ binding: 0, buffer }], { bytesPerInstance: 80 });

    renderer.submit([item(makeGeometry(renderer), p, TRANSFORM, EXTRA)]);
    const f = instanceWrite(recorded, 80);
    expect(Array.from(f.slice(0, 16))).toEqual(Array.from(TRANSFORM));
    for (let i = 0; i < 4; i++) expect(f[16 + i]!).toBeCloseTo(EXTRA[i]!, 6);
  });

  it('非法实例布局在注册期显式报错（而不是静默写错位置）', () => {
    const { renderer, device } = setup();
    const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    expect(() => register(renderer, device, [{ binding: 0, buffer }], { modelMatrixOffset: 8 })).toThrow(/16 的倍数/);
    expect(() => register(renderer, device, [{ binding: 0, buffer }], { modelMatrixOffset: -16 })).toThrow(/非负/);
    expect(() => register(renderer, device, [{ binding: 0, buffer }], { bytesPerInstance: 65 })).toThrow(/16 字节/);
    expect(() =>
      register(renderer, device, [{ binding: 0, buffer }], { bytesPerInstance: 64, modelMatrixOffset: 16 }),
    ).toThrow(/bytesPerInstance/);
  });
});
