import { describe, it, expect } from 'vitest';
import { Renderer } from '../src/core/renderer';
import { VS_INSTANCED, VS_INSTANCED_COMPACTION, FS_COLOR } from '../src/shaders/instance';
import { uniformBindGroupLayout } from '../src/core/renderer';
import { createFakeGPU } from './fake-gpu';
import type { GlobalBinding } from '../src/types';

function setup() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = new Renderer(device, context, 'bgra8unorm');

  const layout = uniformBindGroupLayout(device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX },
  ]);
  const uniform = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const globalBindings: GlobalBinding[] = [{ binding: 0, buffer: uniform }];

  const pipeline = renderer.registerPipeline({
    label: 'cube',
    vsCode: VS_INSTANCED,
    fsCode: FS_COLOR,
    vertexLayouts: [
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { shaderLocation: 0, format: 'float32x3', offset: 0 },
          { shaderLocation: 1, format: 'float32x3', offset: 12 },
        ],
      },
    ],
    bindGroupLayouts: [layout],
    globalBindings,
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: 'bgra8unorm' }],
  });

  const cube = renderer.createGeometry(
    new Float32Array(24 * 4),
    [
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { shaderLocation: 0, format: 'float32x3', offset: 0 },
          { shaderLocation: 1, format: 'float32x3', offset: 12 },
        ],
      },
    ],
    new Uint16Array([0, 1, 2, 2, 3, 0]),
  );

  return { renderer, pipeline, cube, device, recorded };
}

describe('Renderer integration (recording GPU)', () => {
  it('renders a single instanced draw with instanceCount=1', () => {
    const { renderer, pipeline, cube, recorded } = setup();

    const stats = renderer.submit([{ geometry: cube, pipeline }]);

    expect(stats.drawCalls).toBe(1);
    expect(stats.instances).toBe(1);
    expect(stats.itemsSubmitted).toBe(1);
    expect(stats.batches).toBe(1);
    expect(recorded.drawCalls.length).toBe(1);
    expect(recorded.drawCalls[0]!.instanceCount).toBe(1);
    expect(recorded.drawCalls[0]!.indexCount).toBe(6);
    expect(recorded.writes.some((w) => w.bytes.byteLength >= 80)).toBe(true);
  });

  it('auto-batches N single-instance items into one instanced draw', () => {
    const { renderer, pipeline, cube, recorded } = setup();

    const stats = renderer.submit([
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
      { geometry: cube, pipeline, transforms: new Float32Array(32) },
    ]);

    expect(stats.drawCalls).toBe(1);
    expect(stats.instances).toBe(5);
    expect(stats.batches).toBe(1);
    expect(recorded.drawCalls[0]!.instanceCount).toBe(5);
    expect(stats.itemsSubmitted).toBe(4);
    expect(stats.itemsDrawn).toBe(4);
  });

  it('帧内异常不会污染下一次 ring allocation', () => {
    const { renderer, pipeline, cube, device, recorded } = setup();
    const original = device.createCommandEncoder.bind(device);
    let fail = true;
    device.createCommandEncoder = ((...args: unknown[]) => {
      if (fail) {
        fail = false;
        throw new Error('injected encoder failure');
      }
      return original(...args as []);
    }) as typeof device.createCommandEncoder;
    expect(() => renderer.submit([{ geometry: cube, pipeline }])).toThrow(/injected/);
    expect(() => renderer.submit([{ geometry: cube, pipeline }])).not.toThrow();
    const instanceWrites = recorded.writes.filter((w) => w.bytes.byteLength >= 80);
    expect(instanceWrites.at(-1)?.offset).toBe(0);
  });

  it('does not retain stale items after a larger frame', () => {
    const { renderer, pipeline, cube } = setup();
    renderer.submit([
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
    ]);
    const stats = renderer.submit([{ geometry: cube, pipeline }]);
    expect(stats.itemsSubmitted).toBe(1);
    expect(stats.instances).toBe(1);
    expect(stats.itemsDrawn).toBe(1);
  });

  it('respects pipeline boundaries when pipelines differ', () => {
    const { renderer, pipeline, cube, device } = setup();
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipelineB = renderer.registerPipeline({
      label: 'cube-b',
      vsCode: VS_INSTANCED + '// variant-b',
      fsCode: FS_COLOR,
      vertexLayouts: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [
            { shaderLocation: 0, format: 'float32x3', offset: 0 },
            { shaderLocation: 1, format: 'float32x3', offset: 12 },
          ],
        },
      ],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      targets: [{ format: 'bgra8unorm' }],
    });

    const stats = renderer.submit([
      { geometry: cube, pipeline },
      { geometry: cube, pipeline: pipelineB },
    ]);
    expect(stats.drawCalls).toBe(2);
    expect(stats.instances).toBe(2);
  });

  it('returns zeroed stats for empty submit', () => {
    const { renderer } = setup();
    const stats = renderer.submit([]);
    expect(stats.itemsSubmitted).toBe(0);
    expect(stats.drawCalls).toBe(0);
  });
});

describe('Renderer instance data assembly', () => {
  it('uploads instance data via queue.writeBuffer', () => {
    const { renderer, pipeline, cube, recorded } = setup();
    const stats = renderer.submit([{ geometry: cube, pipeline }]);
    expect(stats.itemsSubmitted).toBe(1);
    expect(stats.instances).toBe(1);
    // At least one writeBuffer call for the instance data (≥80 bytes for 1 instance with stride 80).
    const instWrites = recorded.writes.filter((w) => w.bytes.byteLength >= 80);
    expect(instWrites.length).toBeGreaterThanOrEqual(1);
  });

  it('uploads more data for multiple instances', () => {
    const { renderer, pipeline, cube, recorded } = setup();
    const stats = renderer.submit([
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
      { geometry: cube, pipeline },
    ]);
    expect(stats.instances).toBe(3);
    // Instance data should be ≥ 3*80 = 240 bytes total across writes.
    const totalInstBytes = recorded.writes
      .filter((w) => w.bytes.byteLength >= 16)
      .reduce((sum, w) => sum + w.bytes.byteLength, 0);
    expect(totalInstBytes).toBeGreaterThanOrEqual(240);
  });

  it('custom transforms affect instance count', () => {
    const { renderer, pipeline, cube } = setup();
    const customTransform = new Float32Array(32); // 2 transforms
    const stats = renderer.submit([{ geometry: cube, pipeline, transforms: customTransform }]);
    expect(stats.instances).toBe(2);
  });
});

describe('Renderer depth sorting', () => {
  it('sorts items by pipeline then depth', () => {
    const { renderer, pipeline, cube, device } = setup();
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipelineB = renderer.registerPipeline({
      label: 'depth-test',
      vsCode: VS_INSTANCED + '// depth-variant',
      fsCode: FS_COLOR,
      vertexLayouts: [
        {
          arrayStride: 24,
          stepMode: 'vertex',
          attributes: [
            { shaderLocation: 0, format: 'float32x3', offset: 0 },
            { shaderLocation: 1, format: 'float32x3', offset: 12 },
          ],
        },
      ],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      targets: [{ format: 'bgra8unorm' }],
    });

    const stats = renderer.submit([
      { geometry: cube, pipeline, depth: 0.9 },
      { geometry: cube, pipeline: pipelineB, depth: 0.1 },
      { geometry: cube, pipeline, depth: 0.1 },
    ]);
    // Different pipelines → at least 2 draw calls.
    expect(stats.drawCalls).toBeGreaterThanOrEqual(2);
  });
});

describe('RenderItem 输入校验', () => {
  it('拒绝非法 transform 长度、实例数量和非有限值', () => {
    const { renderer, pipeline, cube } = setup();

    expect(() => renderer.submit([{ geometry: cube, pipeline, transforms: new Float32Array(15) }])).toThrow(
      'multiple of 16',
    );
    expect(() => renderer.submit([{
      geometry: cube,
      pipeline,
      transforms: new Float32Array(16),
      instanceCount: 2,
    }])).toThrow('cannot exceed');
    expect(() => renderer.submit([{
      geometry: cube,
      pipeline,
      transforms: new Float32Array([NaN, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    }])).toThrow('finite');
    expect(() => renderer.submit([{ geometry: cube, pipeline, instanceCount: 0 }])).toThrow(/positive/);
  });

  it('instanceData 长度必须精确匹配管线额外区', () => {
    const { renderer, pipeline, cube } = setup();

    expect(() => renderer.submit([{ geometry: cube, pipeline, instanceData: new Float32Array(3) }])).toThrow(/length/);
    expect(() => renderer.submit([{ geometry: cube, pipeline, instanceData: new Float32Array(5) }])).toThrow(/length/);
  });

  it('拒绝非法 bounding 和 depth', () => {
    const { renderer, pipeline, cube } = setup();

    expect(() => renderer.submit([{
      geometry: cube,
      pipeline,
      bounding: { centerX: 0, centerY: 0, centerZ: 0, radius: -1 },
    }])).toThrow('radius');
    expect(() => renderer.submit([{ geometry: cube, pipeline, depth: NaN }])).toThrow('depth');
  });
});

describe('Renderer dispose', () => {
  it('dispose does not throw and can be called twice', () => {
    const { renderer } = setup();
    renderer.dispose();
    renderer.dispose();
  });

  it('dispose 后再提交给出可读错误（而不是空引用异常）', () => {
    const { renderer, pipeline, cube } = setup();
    renderer.dispose();
    expect(() => renderer.submit([{ geometry: cube, pipeline }])).toThrow(/已 dispose/);
    expect(() => renderer.submitDirect([{ geometry: cube, pipeline }])).toThrow(/已 dispose/);
    expect(() => renderer.submitCulled([{ geometry: cube, pipeline }], new Float32Array(16))).toThrow(/已 dispose/);
  });
});

describe('管线注册校验', () => {
  it('深度格式与 Renderer.depthFormat 不一致时在注册期抛错', () => {
    const { device, context } = createFakeGPU();
    const renderer = Renderer.create({ device, context, format: 'bgra8unorm' });
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    expect(() =>
      renderer.registerPipeline({
        label: 'bad-depth',
        vsCode: VS_INSTANCED,
        fsCode: FS_COLOR,
        vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
        bindGroupLayouts: [layout],
        globalBindings: [{ binding: 0, buffer: uniform }],
        depth: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
        targets: [{ format: 'bgra8unorm' }],
      }),
    ).toThrow(/depthFormat/);

    renderer.dispose();
  });
});

describe('执行契约边界', () => {
  it('拒绝没有 uniform/storage usage 的全局绑定', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST });
    expect(() => renderer.registerPipeline({
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer }],
      targets: [{ format: 'bgra8unorm' }],
    })).toThrow(/UNIFORM usage/);
    renderer.dispose();
  });

  it('内置实例着色器拒绝不匹配的实例布局', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const base = {
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex' as const, attributes: [{ shaderLocation: 0, format: 'float32x3' as GPUVertexFormat, offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'bgra8unorm' as GPUTextureFormat }],
    };
    expect(() => renderer.registerPipeline({ ...base, bytesPerInstance: 64 })).toThrow(/内置实例着色器/);
    expect(() => renderer.registerPipeline({ ...base, vsCode: `${VS_INSTANCED}// layout comment`, bytesPerInstance: 96 })).toThrow(/内置实例着色器/);
    expect(() => renderer.registerPipeline({ ...base, bytesPerInstance: 80, modelMatrixOffset: 16 })).toThrow(/内置实例着色器/);
    renderer.dispose();
  });

  it('普通提交入口拒绝 compaction 管线', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const culled = renderer.registerPipeline({
      vsCode: VS_INSTANCED_COMPACTION,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'bgra8unorm' }],
      compaction: true,
      compactionContract: 'hpg-compaction-v1',
    });
    const geometry = renderer.createGeometry(new Float32Array(9), [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }]);
    expect(() => renderer.submit([{ geometry, pipeline: culled }])).toThrow(/only be used with submitCulled/);
    expect(() => renderer.submitDirect([{ geometry, pipeline: culled }])).toThrow(/only be used with submitCulled/);
    renderer.dispose();
  });

  it('自定义 compaction shader 必须显式声明受信契约', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const base = {
      vsCode: VS_INSTANCED + '// custom',
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex' as const, attributes: [{ shaderLocation: 0, format: 'float32x3' as GPUVertexFormat, offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'bgra8unorm' as GPUTextureFormat }],
      compaction: true,
    };
    expect(() => renderer.registerPipeline(base)).toThrow(/compactionContract/);
    expect(() => renderer.registerPipeline({ ...base, compactionContract: 'hpg-compaction-v1' })).not.toThrow();
    renderer.dispose();
  });

  it('拒绝与 presentation format 不一致的管线 target', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    expect(() => renderer.registerPipeline({
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'rgba8unorm' }],
    })).toThrow(/presentation format/);
    renderer.dispose();
  });

  it('提交前拒绝 context 被重新配置为其他格式', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipeline = renderer.registerPipeline({
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'bgra8unorm' }],
    });
    const geometry = renderer.createGeometry(new Float32Array(9), [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }]);
    context.configure({ device, format: 'rgba8unorm' });
    expect(() => renderer.submit([{ geometry, pipeline }])).toThrow(/context format/);
    renderer.dispose();
  });

  it('拒绝 stencil 或 color depth format', () => {
    const { device, context } = createFakeGPU();
    expect(() => new Renderer(device, context, 'bgra8unorm', { depthFormat: 'depth24plus-stencil8' })).toThrow(/stencil/);
    expect(() => new Renderer(device, context, 'bgra8unorm', { depthFormat: 'rgba8unorm' })).toThrow(/depth-only/);
  });

  it('提交时要求 context 已配置且可用于 render attachment', () => {
    const { device, context } = createFakeGPU();
    const renderer = new Renderer(device, context, 'bgra8unorm');
    const layout = uniformBindGroupLayout(device, [{ binding: 0, visibility: GPUShaderStage.VERTEX }]);
    const uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipeline = renderer.registerPipeline({
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }],
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: 'bgra8unorm' }],
    });
    const geometry = renderer.createGeometry(new Float32Array(9), [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }] }]);
    context.unconfigure();
    expect(() => renderer.submit([{ geometry, pipeline }])).toThrow(/configured/);
     context.configure({ device, format: 'bgra8unorm', usage: GPUTextureUsage.COPY_SRC });
     expect(() => renderer.submit([{ geometry, pipeline }])).toThrow(/RENDER_ATTACHMENT/);
     const other = createFakeGPU();
     context.configure({ device: other.device, format: 'bgra8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
     expect(() => renderer.submit([{ geometry, pipeline }])).toThrow(/different GPUDevice/);
     renderer.dispose();
  });
});

describe('Renderer ownership', () => {
  it('拒绝已销毁或属于其他 Renderer 的 Geometry', () => {
    const first = setup();
    const foreign = first.renderer.createGeometry(
      new Float32Array(24),
      [{
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [{ shaderLocation: 0, format: 'float32x3', offset: 0 }, { shaderLocation: 1, format: 'float32x3', offset: 12 }],
      }],
    );
    first.renderer.geometryArena.destroyGeometry(first.cube);
    expect(() => first.renderer.submit([{ geometry: first.cube, pipeline: first.pipeline }])).toThrow(/destroyed/);

    const second = setup();
    expect(() => second.renderer.submit([{ geometry: foreign, pipeline: second.pipeline }])).toThrow(/not created/);
    first.renderer.dispose();
    second.renderer.dispose();
  });

  it('拒绝把另一个 Renderer 注册的 pipeline 混入当前 Renderer', () => {
    const first = setup();
    const secondDevice = createFakeGPU();
    const second = new Renderer(secondDevice.device, secondDevice.context, 'bgra8unorm');
    expect(() => second.submit([{ geometry: first.cube, pipeline: first.pipeline }])).toThrow(/not registered/);
    first.renderer.dispose();
    second.dispose();
  });
});

describe('Renderer.create()', () => {
  it('creates renderer from descriptor', () => {
    const { device, context } = createFakeGPU();
    const r = Renderer.create({ device, context, format: 'bgra8unorm' });
    expect(r.device).toBe(device);
    r.dispose();
  });

  it('geometryArena exposes arena stats', () => {
    const { device, context } = createFakeGPU();
    const r = Renderer.create({ device, context, format: 'bgra8unorm' });
    const s = r.geometryArena.stats();
    expect(s.geometries).toBe(0);
    expect(typeof s.vertexPoolBytes).toBe('number');
    r.dispose();
  });
});
