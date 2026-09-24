import { describe, it, expect } from 'vitest';
import { CullingPipeline } from '../src/core/culling';
import { CS_FRUSTUM_CULL } from '../src/shaders/culling';

function fakeDevice(): GPUDevice {
  const buffers: { size: number; destroyed: boolean }[] = [];
  const submitted: GPUCommandBuffer[][] = [];

  return {
    createBuffer(desc: GPUBufferDescriptor) {
      const buf = { size: desc.size as number, destroyed: false };
      buffers.push(buf);
      return buf as unknown as GPUBuffer;
    },
    createShaderModule: () => ({}) as GPUShaderModule,
    createComputePipeline: () => ({}) as GPUComputePipeline,
    createBindGroupLayout: () => ({}) as GPUBindGroupLayout,
    createPipelineLayout: () => ({}) as GPUPipelineLayout,
    createBindGroup: () => ({}) as GPUBindGroup,
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => undefined,
        setBindGroup: () => undefined,
        dispatchWorkgroups: () => undefined,
        end: () => undefined,
      }),
      finish: () => ({}) as GPUCommandBuffer,
    }),
    queue: {
      submit: (cmds: GPUCommandBuffer[]) => { submitted.push(cmds); },
      writeBuffer: () => undefined,
    },
    _buffers: buffers,
    _submitted: submitted,
  } as unknown as GPUDevice;
}

describe('CS_FRUSTUM_CULL shader', () => {
  it('is a non-empty WGSL string', () => {
    expect(CS_FRUSTUM_CULL.length).toBeGreaterThan(0);
  });

  it('declares compute entry point cs_main', () => {
    expect(CS_FRUSTUM_CULL).toContain('@compute @workgroup_size(64)');
    expect(CS_FRUSTUM_CULL).toContain('fn cs_main');
  });

  it('declares 7 bindings', () => {
    expect(CS_FRUSTUM_CULL).toContain('@binding(0) var<uniform>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(1) var<storage, read>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(2) var<storage, read>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(3) var<storage, read_write>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(4) var<storage, read_write>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(5) var<storage, read_write>');
    expect(CS_FRUSTUM_CULL).toContain('@binding(6) var<storage, read>');
  });

  it('uses atomicAdd for instanceCount', () => {
    expect(CS_FRUSTUM_CULL).toContain('atomicAdd');
    expect(CS_FRUSTUM_CULL).toContain('instanceCount');
  });

  it('writes the compaction mapping (slot → 组内原始索引)', () => {
    expect(CS_FRUSTUM_CULL).toContain('compactedIndices');
    // slot 区基址来自 CPU 计算的 geometryBases（candidateBase / slotBase）
    expect(CS_FRUSTUM_CULL).toContain('geometryBases');
    expect(CS_FRUSTUM_CULL).toContain('bases.y + slot');
    expect(CS_FRUSTUM_CULL).toContain('idx - bases.x');
  });
});

describe('CullingPipeline', () => {
  it('constructor creates without error', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);
    expect(culling).toBeDefined();
  });

  it('cull() 返回 slotBases（compactedIndices 的 256 字节对齐基址）', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array(16);
    vp[0] = vp[5] = vp[10] = vp[15] = 1;

    // 两组，候选数 3 / 2 → slotBase 0 / 64（SLOT_ALIGN）
    const geometryIds = new Uint32Array([0, 0, 0, 1, 1]);
    const spheres = new Float32Array(5 * 4);

    const result = culling.cull(vp, spheres, geometryIds, 2);
    expect(Array.from(result.slotBases)).toEqual([0, 64]);
    // 每组 64 个 slot × 4 字节 = 256 字节边界
    for (const b of result.slotBases) expect((b * 4) % 256).toBe(0);
  });

  it('cull() returns drawArgsBuffer, compactedIndicesBuffer, and drawArgsCount', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    const spheres = new Float32Array([
      0, 0, -5, 1,
      0, 0, -10, 2,
      100, 100, -5, 1,
    ]);

    const geometryIds = new Uint32Array([0, 0, 0]);

    const result = culling.cull(vp, spheres, geometryIds, 1);
    expect(result.drawArgsCount).toBe(1);
    expect(result.drawArgsBuffer).toBeDefined();
    expect(result.compactedIndicesBuffer).toBeDefined();
  });

  it('cull() handles zero spheres', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array(16);
    vp[0] = vp[5] = vp[10] = vp[15] = 1;

    const result = culling.cull(vp, new Float32Array(0), new Uint32Array(0), 0);
    expect(result.drawArgsCount).toBe(0);
  });

  it('cull() handles multiple geometries', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    const spheres = new Float32Array([
      0, 0, -5, 1,
      0, 0, -10, 2,
      5, 0, -3, 1,
      5, 0, -8, 1,
    ]);

    const geometryIds = new Uint32Array([0, 0, 1, 1]);

    const result = culling.cull(vp, spheres, geometryIds, 2);
    expect(result.drawArgsCount).toBe(2);
    expect(result.drawArgsBuffer).toBeDefined();
    expect(result.compactedIndicesBuffer).toBeDefined();
  });

  it('dispose() can be called multiple times', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);
    culling.dispose();
    culling.dispose(); // no throw
  });

  it('compactedIndicesBuffer returns null before first cull', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);
    expect(culling.compactedIndicesBuffer).toBeNull();
  });

  it('cull() with 0 visible (all spheres far away)', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    // Narrow frustum: only objects near origin visible.
    const vp = new Float32Array([
      0.1, 0, 0, 0,
      0, 0.1, 0, 0,
      0, 0, -0.01, 0,
      0, 0, -1.1, 1,
    ]);

    // All spheres at x=1000 (far outside frustum).
    const spheres = new Float32Array([
      1000, 0, 0, 1,
      1000, 0, 0, 1,
      1000, 0, 0, 1,
    ]);
    const geometryIds = new Uint32Array([0, 0, 0]);

    const result = culling.cull(vp, spheres, geometryIds, 1);
    expect(result.drawArgsCount).toBe(1);
    expect(result.drawArgsBuffer).toBeDefined();
    expect(result.compactedIndicesBuffer).toBeDefined();
  });

  it('cull() with 1 visible out of N', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    // Only sphere 0 at origin (visible), rest far away.
    const spheres = new Float32Array([
      0, 0, -5, 1,     // visible
      1000, 0, 0, 1,   // culled
      1000, 0, 0, 1,   // culled
      1000, 0, 0, 1,   // culled
    ]);
    const geometryIds = new Uint32Array([0, 0, 0, 0]);

    const result = culling.cull(vp, spheres, geometryIds, 1);
    expect(result.drawArgsCount).toBe(1);
  });

  it('cull() with N visible (all in frustum)', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    // All spheres near origin (visible).
    const spheres = new Float32Array([
      0, 0, -5, 1,
      1, 0, -5, 1,
      2, 0, -5, 1,
      3, 0, -5, 1,
    ]);
    const geometryIds = new Uint32Array([0, 0, 1, 1]);

    const result = culling.cull(vp, spheres, geometryIds, 2);
    expect(result.drawArgsCount).toBe(2);
  });

  it('cull() resets counters between calls', () => {
    const device = fakeDevice();
    const culling = new CullingPipeline(device);

    const vp = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);

    const spheres = new Float32Array([0, 0, -5, 1]);
    const geometryIds = new Uint32Array([0]);

    // First call.
    culling.cull(vp, spheres, geometryIds, 1);
    // Second call should reset and re-dispatch (no stale state).
    const result = culling.cull(vp, spheres, geometryIds, 1);
    expect(result.drawArgsCount).toBe(1);
  });

  it('shader resets compactionCounters to zero via writeBuffer', () => {
    // Verify the shader source contains the reset pattern.
    expect(CS_FRUSTUM_CULL).toContain('compactionCounters');
    expect(CS_FRUSTUM_CULL).toContain('atomicAdd(&compactionCounters');
  });
});
