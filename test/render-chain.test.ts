/**
 * 渲染链路回归测试 —— 覆盖「模型加载后不显示」类缺陷：
 *
 *   1. 世界空间包围盒（含节点变换 / 全局缩放）
 *   2. Geometry 局部 AABB
 *   3. 材质颜色进入 instanceData
 *   4. ring 扩容后实例 bind group 必须重建
 *   5. submitCulled 按 geometry 分组（而非 pipeline）
 *   6. compaction 管线注册到独立 group=1 布局
 *   7. 视锥平面提取的正确性（与 WGSL 镜像的 CPU 参考）
 */
import { describe, it, expect } from 'vitest';
import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { GeometryArena } from '../src/core/geometry';
import { importGltfAsset, sceneToRenderItems } from '../src/core/asset-importer';
import { extractFrustumPlanes, sphereInFrustum } from '../src/core/culling';
import { lookAt, perspective, multiply, invert } from '../src/core/math';
import { VS_INSTANCED, FS_COLOR } from '../src/shaders/instance';
import { createFakeGPU } from './fake-gpu';
import type { GltfAsset } from '../src/core/gltf';
import type { RenderItem } from '../src/types';

const FORMAT: GPUTextureFormat = 'bgra8unorm';

const LAYOUT = [
  {
    arrayStride: 12,
    stepMode: 'vertex' as const,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
  },
];

/** 单位立方体（8 顶点 + 12 三角形），用于构造确定性的包围盒。 */
function cubeGeometry(arena: GeometryArena, size = 1) {
  const h = size / 2;
  const corners: [number, number, number][] = [
    [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
    [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
  ];
  const verts = new Float32Array(corners.length * 3);
  corners.forEach((c, i) => verts.set(c, i * 3));
  const indices = new Uint16Array([
    0, 1, 2, 2, 3, 0,
    4, 6, 5, 6, 4, 7,
    0, 4, 5, 5, 1, 0,
    2, 6, 7, 7, 3, 2,
    0, 3, 7, 7, 4, 0,
    1, 5, 6, 6, 2, 1,
  ]);
  return arena.createGeometry(verts, LAYOUT, indices);
}

function makeRenderer() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = Renderer.create({ device, context, format: FORMAT });
  return { device, context, recorded, renderer };
}

function registerPipeline(
  renderer: Renderer,
  opts: { compaction?: boolean; label?: string; bytesPerInstance?: number } = {},
) {
  const layout = uniformBindGroupLayout(renderer.device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX },
  ]);
  const uniform = renderer.device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return renderer.registerPipeline({
    label: opts.label ?? 'chain',
    vsCode: VS_INSTANCED,
    fsCode: FS_COLOR,
    vertexLayouts: LAYOUT,
    bindGroupLayouts: [layout],
    globalBindings: [{ binding: 0, buffer: uniform }],
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: FORMAT }],
    compaction: opts.compaction,
    bytesPerInstance: opts.bytesPerInstance,
  });
}

// ─── 1/2/3. 导入链路 ────────────────────────────────────────

describe('glTF 导入链路', () => {
  /** 构造一个「顶点在本地大尺度 + 节点缩放到米级」的资产（类似 Khronos Duck.glb）。 */
  function scaledAsset(): GltfAsset {
    const verts = new Float32Array([
      -100, 0, -100,
      100, 0, -100,
      100, 200, 100,
      -100, 200, 100,
    ]);
    return {
      meshes: [
        {
          name: 'Quad',
          primitives: [
            {
              vertices: verts,
              vertexLayout: { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
              indices: new Uint16Array([0, 1, 2, 2, 3, 0]),
              indexFormat: 'uint16',
              materialIndex: 0,
            },
          ],
        },
      ],
      materials: [
        {
          name: 'M',
          baseColorFactor: [0.25, 0.5, 0.75, 1],
          metallicFactor: 0,
          roughnessFactor: 1,
          doubleSided: false,
          alphaMode: 'OPAQUE',
          alphaCutoff: 0.5,
        },
      ],
      images: [],
      nodes: [
        { name: 'Root', localMatrix: new Float32Array([0.01, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 1]), children: [1] },
        { name: 'Mesh', meshIndex: 0, localMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]), children: [] },
      ],
      rootNodes: [0],
      warnings: [],
    };
  }

  it('bounds 使用世界空间（含节点变换与缩放），而不是原始顶点坐标', () => {
    const { renderer } = makeRenderer();
    const scene = importGltfAsset(scaledAsset(), renderer);

    // 局部：x ∈ [-100, 100]、y ∈ [0, 200]
    // 根节点 0.01 缩放，子节点平移 5（平移会被父级缩放 → 世界平移 0.05）
    // → 世界：x ∈ [-1 + 0.05, 1 + 0.05]，y ∈ [0, 2]
    expect(scene.bounds[0]).toBeCloseTo(-0.95, 5);
    expect(scene.bounds[1]).toBeCloseTo(0, 5);
    expect(scene.bounds[3]).toBeCloseTo(1.05, 5);
    expect(scene.bounds[4]).toBeCloseTo(2, 5);

    // 半径（外接球）应在 ~1.5 量级，而不是 100+ —— 否则取景/远平面会算错。
    const radius = Math.hypot(
      scene.bounds[3] - scene.bounds[0],
      scene.bounds[4] - scene.bounds[1],
      scene.bounds[5] - scene.bounds[2],
    ) / 2;
    expect(radius).toBeLessThan(5);

    renderer.dispose();
  });

  it('importGltfAsset 的 scale 选项同样进入包围盒', () => {
    const { renderer } = makeRenderer();
    const scene = importGltfAsset(scaledAsset(), renderer, { scale: 0.5 });
    // 整体（含平移） × 0.5
    expect(scene.bounds[0]).toBeCloseTo(-0.475, 5);
    expect(scene.bounds[3]).toBeCloseTo(0.525, 5);
    expect(scene.bounds[4]).toBeCloseTo(1, 5);
    renderer.dispose();
  });

  it('sceneToRenderItems 把材质 baseColorFactor 写入 instanceData', () => {
    const { renderer } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const scene = importGltfAsset(scaledAsset(), renderer);
    const items = sceneToRenderItems(scene, pipeline);

    expect(items.length).toBe(1);
    const item = items[0]!;
    expect(item.instanceCount).toBe(1);
    expect(item.instanceData).toBeDefined();
    // instanceData 是 instanceData 区的首个 vec4（modelMatrix 之后）
    expect(Array.from(item.instanceData!.slice(0, 4))).toEqual([0.25, 0.5, 0.75, 1]);
    // 长度必须匹配管线跨步，否则 Renderer 会忽略（防止越界 NaN）
    const extraFloats = (pipeline.bytesPerInstance - 64) >> 2;
    expect(item.instanceData!.length).toBe(extraFloats);

    renderer.dispose();
  });

  it('同一 mesh 被多个节点引用时复用几何体，并被 Batcher 合并为一个 draw', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);

    const asset = scaledAsset();
    // 再加一个引用同一 mesh 的节点（Blender linked duplicate / instancing）
    asset.nodes[0]!.children = [1, 2];
    asset.nodes.push({
      name: 'Mesh2',
      meshIndex: 0,
      localMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -5, 0, 0, 1]),
      children: [],
    });

    const scene = importGltfAsset(asset, renderer);
    expect(scene.meshes.length).toBe(2);
    // 关键：两个节点共享同一 Geometry 引用（否则永远无法合批）
    expect(scene.meshes[0]!.geometry).toBe(scene.meshes[1]!.geometry);

    const items = sceneToRenderItems(scene, pipeline);
    const stats = renderer.submit(items);
    expect(stats.batches).toBe(1);
    expect(stats.drawCalls).toBe(1);
    expect(stats.instances).toBe(2);
    expect(recorded.drawCalls[0]!.instanceCount).toBe(2);

    renderer.dispose();
  });

  it('几何体携带局部 AABB（供 GPU 剔除使用）', () => {
    const { renderer } = makeRenderer();
    const geo = cubeGeometry(renderer.geometryArena, 2);
    expect(geo.bounds).toBeDefined();
    expect(geo.bounds!.min).toEqual([-1, -1, -1]);
    expect(geo.bounds!.max).toEqual([1, 1, 1]);
    renderer.dispose();
  });
});

// ─── 4. ring 扩容 ───────────────────────────────────────────

describe('瞬态实例缓冲（ring）扩容', () => {
  it('扩容后重建实例 bind group（否则读到已废弃 buffer → 无输出）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    const instanceBgs = () => recorded.bindGroups.filter((bg) => bg.label.endsWith(':instance-bg'));
    // 实例 bind group 惰性创建：先提交一帧拿到初始底层 buffer。
    renderer.submit([{ geometry: geo, pipeline }]);
    const before = instanceBgs().at(-1)!.bindings[0]!.buffer;
    expect(before).toBe(renderer.instanceBuffer);

    // 20k 实例 × 80B = 1.6 MB > 默认 1 MiB → 触发扩容，底层 GPUBuffer 被替换。
    const instanceCount = 20_000;
    const transforms = new Float32Array(instanceCount * 16);
    for (let i = 0; i < instanceCount; i++) {
      transforms[i * 16] = 1;
      transforms[i * 16 + 5] = 1;
      transforms[i * 16 + 10] = 1;
      transforms[i * 16 + 15] = 1;
    }
    const stats = renderer.submit([{ geometry: geo, pipeline, transforms, instanceCount }]);

    expect(stats.instances).toBe(instanceCount);
    expect(renderer.instanceBuffer).not.toBe(before);
    // 关键断言：最后一次实例 bind group 必须指向新的 ring buffer
    expect(instanceBgs().at(-1)!.bindings[0]!.buffer).toBe(renderer.instanceBuffer);
    expect(instanceBgs().at(-1)!.bindings[0]!.buffer).not.toBe(before);

    renderer.dispose();
  });

  it('多批（submit）：实例 bind group 显式声明绑定尺寸，dynamic offset 不越界', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geos = [
      cubeGeometry(renderer.geometryArena),
      cubeGeometry(renderer.geometryArena, 2),
      cubeGeometry(renderer.geometryArena, 3),
    ];
    const stats = renderer.submit(geos.map((geometry) => ({ geometry, pipeline })));
    expect(stats.batches).toBe(3);

    const bg = recorded.bindGroups.filter((b) => b.label.endsWith(':instance-bg')).at(-1)!;
    // 必须显式声明 size（绑定整块 buffer 时任何非零 dynamic offset 都越界）。
    expect(bg.bindings[0]!.size).not.toBeNull();
    expect(bg.bindings[0]!.size).toBe(80); // 覆盖最大批（1 实例 × 80B）
    // 第 2/3 个批确实用了非零 dynamic offset。
    const offsets = recorded.renderBinds.filter((b) => b.group === 1).map((b) => b.offsets[0]!);
    expect(offsets.some((o) => o > 0)).toBe(true);
    expect(recorded.gpuErrors).toEqual([]);
    renderer.dispose();
  });

  it('未扩容时不重建 bind group（热路径零分配）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    const count = recorded.bindGroups.length;
    renderer.submit([{ geometry: geo, pipeline }]);
    renderer.submit([{ geometry: geo, pipeline }]);
    // 首次提交会惰性创建 group=0 的 global bind group；实例 bind group 不应重复创建。
    const firstSubmit = recorded.bindGroups.length;
    renderer.submit([{ geometry: geo, pipeline }]);
    expect(recorded.bindGroups.length).toBe(firstSubmit);
    expect(firstSubmit).toBeLessThanOrEqual(count + 2);

    renderer.dispose();
  });
});

// ─── 4b. group=1 绑定范围（dynamic offset 越界防护） ─────────

/**
 * 回归：实例 bind group 曾绑定**整块** ring buffer（size = capacity），
 * 而 WebGPU 要求 `bindingOffset + dynamicOffset + bindingSize ≤ bufferSize` ——
 * 于是任何非零 dynamic offset 都必然越界。真实 Chrome/Dawn 的报错：
 *
 *   Dynamic Offset[0] (256) is out of bounds of [Buffer "hpg:ring"] with a size of 1048576
 *   and a bound range of (offset: 0, size: 1048576). Did you forget to specify the binding's size?
 *
 * 后果是整个 render pass 失效 → 整帧全黑。单 batch 帧 offset 恒为 0 所以看不出问题，
 * 多 mesh 模型（Duck / BrainStem / Lantern…）必黑。
 * fake-gpu 现在按 Dawn 的规则校验 dynamic offset 范围并记入 gpuErrors。
 */
describe('group=1 绑定范围（dynamic offset 越界防护）', () => {
  it('submitDirect 逐 item 偏移也不越界', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);
    const items: RenderItem[] = [
      { geometry: geo, pipeline, transforms: translate(0), instanceCount: 2 },
      { geometry: geo, pipeline, transforms: translate(5) },
      { geometry: geo, pipeline, transforms: translate(10) },
    ];
    renderer.submitDirect(items);

    const bg = recorded.bindGroups.filter((b) => b.label.endsWith(':instance-bg')).at(-1)!;
    // 绑定尺寸 = 最大 item 的实例字节数（2 实例 × 80B）
    expect(bg.bindings[0]!.size).toBe(160);
    expect(recorded.renderBinds.filter((b) => b.group === 1).length).toBe(3);
    expect(recorded.gpuErrors).toEqual([]);
    renderer.dispose();
  });

  it('submitCulled：实例区与 slot 区绑定尺寸都显式声明且不越界', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geos = [
      cubeGeometry(renderer.geometryArena),
      cubeGeometry(renderer.geometryArena, 2),
      cubeGeometry(renderer.geometryArena, 3),
    ];
    const items: RenderItem[] = geos.map((geometry, i) => ({
      geometry,
      pipeline,
      transforms: translate(i * 10),
    }));
    const vp = multiply(
      new Float32Array(16),
      perspective(Math.PI / 4, 1, 0.1, 100),
      lookAt(new Float32Array([0, 0, 40]), new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
    );
    renderer.submitCulled(items, vp);

    const bg = recorded.bindGroups.filter((b) => b.label.endsWith(':compaction-bg')).at(-1)!;
    expect(bg.bindings[0]!.size).toBe(80); // 实例绑定尺寸（最大组 1 实例 × 80B）
    expect(bg.bindings[1]!.size).not.toBeNull(); // slot 区绑定尺寸不得缺省
    expect(recorded.gpuErrors).toEqual([]);
    renderer.dispose();
  });
});

// ─── 5/6. submitCulled ─────────────────────────────────────

describe('submitCulled 分组与 draw args', () => {
  it('多个 geometry 共享同一 pipeline 时逐 geometry 生成 draw call', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });

    const geos = [
      cubeGeometry(renderer.geometryArena),
      cubeGeometry(renderer.geometryArena, 2),
      cubeGeometry(renderer.geometryArena, 3),
    ];
    const items: RenderItem[] = geos.map((geometry, i) => ({
      geometry,
      pipeline,
      transforms: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, i * 10, 0, 0, 1]),
      instanceCount: 1,
    }));

    const vp = multiply(new Float32Array(16), perspective(Math.PI / 4, 1, 0.1, 100), lookAt(
      new Float32Array([0, 0, 40]),
      new Float32Array([0, 0, 0]),
      new Float32Array([0, 1, 0]),
    ));

    const stats = renderer.submitCulled(items, vp);

    // 用 pipeline 分组时这里会变成 1（只画第一个 mesh）—— 这是模型显示不全的根因。
    expect(stats.drawCalls).toBe(3);
    expect(stats.batches).toBe(3);
    expect(recorded.indirectDraws.length).toBe(3);
    // 每个 geometry 一组 draw args（20 字节一组）
    expect(new Set(recorded.indirectDraws.map((d) => d.offset)).size).toBe(3);
    for (const d of recorded.indirectDraws) expect(d.indexed).toBe(true);

    // draw args 的 indexCount 必须来自各自的 geometry
    // （同尺寸的写有两次：compute 前的清零 + 这里的预写，取最后一次）
    const argsWrite = recorded.writes.filter((w) => w.bytes.byteLength === 3 * 20).at(-1)!;
    const argsU32 = new Uint32Array(argsWrite.bytes.buffer, argsWrite.bytes.byteOffset, 15);
    for (let g = 0; g < 3; g++) expect(argsU32[g * 5]).toBe(geos[g]!.indexCount);

    renderer.dispose();
  });

  it('compaction binding 逐 geometry 重绑两个 dynamic offset（实例区 + slot 区）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geos = [
      cubeGeometry(renderer.geometryArena),
      cubeGeometry(renderer.geometryArena, 2),
      cubeGeometry(renderer.geometryArena, 3),
    ];
    const items: RenderItem[] = geos.map((geometry) => ({
      geometry,
      pipeline,
      transforms: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    }));

    const vp = multiply(new Float32Array(16), perspective(Math.PI / 4, 1, 0.1, 100), lookAt(
      new Float32Array([0, 0, 30]),
      new Float32Array([0, 0, 0]),
      new Float32Array([0, 1, 0]),
    ));
    renderer.submitCulled(items, vp);

    const group1 = recorded.renderBinds.filter((b) => b.group === 1);
    expect(group1.length).toBe(3);
    group1.forEach((bind, i) => {
      expect(bind.offsets.length).toBe(2);
      // 实例区 offset：256 对齐
      expect(bind.offsets[0]! % 256).toBe(0);
      // slot 区 offset：第 i 组的 slotBase = i * SLOT_ALIGN(64) → 字节 256 对齐
      expect(bind.offsets[1]).toBe(i * 256);
    });

    renderer.dispose();
  });

  it('draw args 的 CPU 预写必须早于 compute dispatch（否则清零原子 instanceCount → 画 0 实例）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geos = [
      cubeGeometry(renderer.geometryArena),
      cubeGeometry(renderer.geometryArena, 2),
    ];
    const items: RenderItem[] = geos.map((geometry, i) => ({
      geometry,
      pipeline,
      transforms: translate(i * 10),
    }));
    const vp = multiply(
      new Float32Array(16),
      perspective(Math.PI / 4, 1, 0.1, 100),
      lookAt(new Float32Array([0, 0, 30]), new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
    );
    renderer.submitCulled(items, vp);

    const isIndirect = (b: unknown) =>
      (((b as { usage?: number } | undefined)?.usage ?? 0) & GPUBufferUsage.INDIRECT) !== 0;

    // queue.writeBuffer 按入队顺序执行：dispatch 之后再写 draw args 会把 GPU 原子填充的
    // instanceCount 覆盖回 0 → drawIndexedIndirect 画 0 个实例（无校验错误、整帧全黑）。
    const dispatchIdx = recorded.ops.findIndex((o) => o.kind === 'dispatch');
    expect(dispatchIdx).toBeGreaterThanOrEqual(0);
    const argsWriteIdxs = recorded.ops
      .map((o, i) => ({ o, i }))
      .filter(({ o }) => o.kind === 'write' && isIndirect(o.buffer))
      .map(({ i }) => i);
    expect(argsWriteIdxs.length).toBeGreaterThan(0);
    expect(Math.max(...argsWriteIdxs)).toBeLessThan(dispatchIdx);

    // 模板中 instanceCount 必须为 0（留给 compute 原子填充）。
    const argsWrite = recorded.writes.filter((w) => isIndirect(w.buffer)).at(-1)!;
    const u32 = new Uint32Array(argsWrite.bytes.buffer, argsWrite.bytes.byteOffset, 10);
    expect(u32[1]).toBe(0);
    expect(u32[6]).toBe(0);
    expect(u32[0]).toBe(geos[0]!.indexCount);

    renderer.dispose();
  });

  it('compaction bind group 跨帧复用（不逐帧 createBindGroup）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geo = cubeGeometry(renderer.geometryArena);
    const items: RenderItem[] = [
      { geometry: geo, pipeline, transforms: translate(0) },
    ];
    const vp = multiply(
      new Float32Array(16),
      perspective(Math.PI / 4, 1, 0.1, 100),
      lookAt(new Float32Array([0, 0, 20]), new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
    );

    const compactionBgs = () => recorded.bindGroups.filter((bg) => bg.label.endsWith(':compaction-bg'));
    renderer.submitCulled(items, vp);
    const afterFirst = compactionBgs().length;
    expect(afterFirst).toBe(1);

    renderer.submitCulled(items, vp);
    renderer.submitCulled(items, vp);
    // 底层 buffer 未变（无扩容 / 无容量增长）→ 不应重复创建
    expect(compactionBgs().length).toBe(afterFirst);

    renderer.dispose();
  });

  it('逐实例包围球由几何 AABB × 实例矩阵推导（并保持与实例数据同序）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geo = cubeGeometry(renderer.geometryArena, 2); // 半径 √3

    const items: RenderItem[] = [
      { geometry: geo, pipeline, transforms: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]) },
      { geometry: geo, pipeline, transforms: new Float32Array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 7, 0, 1]) },
    ];

    const vp = multiply(new Float32Array(16), perspective(Math.PI / 4, 1, 0.1, 100), lookAt(
      new Float32Array([0, 0, 60]),
      new Float32Array([0, 0, 0]),
      new Float32Array([0, 1, 0]),
    ));
    renderer.submitCulled(items, vp);

    // 2 个 instance → 2 × vec4 包围球（32 字节）
    const sphereWrite = recorded.writes.find((w) => w.bytes.byteLength === 32)!;
    const s = new Float32Array(sphereWrite.bytes.buffer, sphereWrite.bytes.byteOffset, 8);
    expect(Array.from(s.slice(0, 3))).toEqual([5, 0, 0]);
    expect(s[3]).toBeCloseTo(Math.sqrt(3), 5);          // 未缩放
    expect(Array.from(s.slice(4, 7))).toEqual([0, 7, 0]);
    expect(s[7]).toBeCloseTo(2 * Math.sqrt(3), 5);      // ×2 缩放

    renderer.dispose();
  });

  it('compaction 管线使用独立的 group=1 布局（与直接绘制管线区分）', () => {
    const { renderer } = makeRenderer();
    const direct = registerPipeline(renderer, { label: 'direct' });
    const culled = registerPipeline(renderer, { label: 'culled', compaction: true });

    expect(culled.id).not.toBe(direct.id);
    expect(direct.desc.compaction).toBeFalsy();
    expect(culled.desc.compaction).toBe(true);
    // group 0 = 调用方 layout，group 1 = 实例/compaction 布局
    expect(direct.bindGroupLayouts.length).toBe(2);
    expect(culled.bindGroupLayouts.length).toBe(2);
    expect(culled.bindGroupLayouts[1]).not.toBe(direct.bindGroupLayouts[1]);

    renderer.dispose();
  });

  it('未指定 compaction 的管线会被缓存区分（不会命中同一实例布局）', () => {
    const { renderer } = makeRenderer();
    const a = registerPipeline(renderer, { label: 'same' });
    const b = registerPipeline(renderer, { label: 'same', compaction: true });
    expect(a.id).not.toBe(b.id);
    renderer.dispose();
  });
});

// ─── 7. 视锥平面 ────────────────────────────────────────────

describe('视锥剔除平面提取（CS_FRUSTUM_CULL 的 CPU 参考）', () => {
  const eye = new Float32Array([0, 0, 5]);
  const vp = multiply(
    new Float32Array(16),
    perspective(Math.PI / 4, 1, 0.1, 100),
    lookAt(eye, new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
  );
  const planes = extractFrustumPlanes(vp);

  it('视锥内部的点全部被判为可见', () => {
    // 相机在 (0,0,5) 看向 -Z，fov 45°：深度 d 处半宽 = d*tan(22.5°)
    for (const d of [1, 10, 50]) {
      const h = d * Math.tan(Math.PI / 8);
      expect(sphereInFrustum(planes, 0, 0, 5 - d, 0)).toBe(true);
      expect(sphereInFrustum(planes, h * 0.5, 0, 5 - d, 0)).toBe(true);
      expect(sphereInFrustum(planes, 0, h * 0.5, 5 - d, 0)).toBe(true);
    }
  });

  it('视锥外部的点被判为不可见', () => {
    for (const d of [1, 10, 50]) {
      const h = d * Math.tan(Math.PI / 8);
      expect(sphereInFrustum(planes, h * 1.5, 0, 5 - d, 0)).toBe(false);
      expect(sphereInFrustum(planes, 0, h * 1.5, 5 - d, 0)).toBe(false);
      expect(sphereInFrustum(planes, 0, 0, 5 + d, 0)).toBe(false); // 相机身后
      expect(sphereInFrustum(planes, 0, 0, 5 - 200, 0)).toBe(false); // 超出远平面
    }
  });

  it('半径足够大的球体不会被误剔（保守可见）', () => {
    // 相机原点处的大球即使中心远离视锥也应通过（半径覆盖整个视锥）
    expect(sphereInFrustum(planes, 0, 0, 0, 1e30)).toBe(true);
  });

  it('矩阵行/列混淆的旧实现会剔除视锥内物体（回归防护）', () => {
    const col = (k: number) => [vp[k * 4]!, vp[k * 4 + 1]!, vp[k * 4 + 2]!, vp[k * 4 + 3]!];
    const oldPlane = (row: number[], sign: number) => {
      let p = [sign * row[0]! + row[3]!, sign * row[1]! + row[3]!, sign * row[2]! + row[3]!, row[3]!];
      const len = Math.hypot(p[0]!, p[1]!, p[2]!);
      p = len > 0 ? p.map((v) => v / len) : p;
      const out = new Float32Array(24);
      out.set(p, 0);
      return out;
    };
    const oldPlanes = new Float32Array(24);
    [0, 1, 2].forEach((c) => {
      oldPlanes.set(oldPlane(col(c), 1).subarray(0, 4), c * 8);
      oldPlanes.set(oldPlane(col(c), -1).subarray(0, 4), c * 8 + 4);
    });

    const inside = sphereInFrustum(planes, 0, 0, 5 - 10, 0);
    const insideOld = sphereInFrustum(oldPlanes, 0, 0, 5 - 10, 0);
    // 正确实现保留；旧实现（用行分量混合列索引）会误判
    expect(inside).toBe(true);
    expect(insideOld).toBe(false);
  });

  it('invert(vp) 反投影得到的 AABB 中心在视锥内', () => {
    const inv = new Float32Array(16);
    expect(invert(inv, vp)).toBe(true);
    const p = [
      inv[12]! / inv[15]!,
      inv[13]! / inv[15]!,
      inv[14]! / inv[15]!,
    ];
    // NDC 原点 → 近平面与远平面之间的正中，必然在视锥内
    expect(sphereInFrustum(planes, p[0]!, p[1]!, p[2]!, 0.001)).toBe(true);
  });
});

// ─── 8. 帧内混合实例跨步 ──────────────────────────────────────

/** 取本帧写入实例缓冲的字节（按底层 buffer 识别，避免与其他 writeBuffer 混淆）。 */
function instanceWrite(renderer: Renderer, recorded: { writes: { offset: number; bytes: Uint8Array; buffer?: unknown }[] }) {
  const buf = renderer.instanceBuffer;
  const w = recorded.writes.filter((x) => x.buffer === buf).at(-1);
  expect(w).toBeDefined();
  return new Float32Array(w!.bytes.buffer, w!.bytes.byteOffset, w!.bytes.byteLength / 4);
}

/** 平移矩阵（列主序）。 */
function translate(x: number, y = 0, z = 0) {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

describe('帧内混合实例跨步（bytesPerInstance）', () => {
  it('每个批按自己管线的跨步落位（着色器只按自己的结构体索引实例）', () => {
    const { renderer, recorded } = makeRenderer();
    const p80 = registerPipeline(renderer, { label: 'p80', bytesPerInstance: 80 });
    const p96 = registerPipeline(renderer, { label: 'p96', bytesPerInstance: 96 });
    const geo = cubeGeometry(renderer.geometryArena);

    // A：2 实例 × 80B → 占 160B；B：1 实例 × 96B → 批起始 256 对齐。
    const transformsA = new Float32Array(32);
    transformsA.set(translate(10), 0);
    transformsA.set(translate(11), 16);

    renderer.submit([
      { geometry: geo, pipeline: p80, transforms: transformsA, instanceCount: 2 },
      { geometry: geo, pipeline: p96, transforms: translate(99), instanceCount: 1 },
    ]);

    const f = instanceWrite(renderer, recorded);
    // A 的实例 1 必须落在 A 自己的跨步（+80B = float 20）处，而不是帧内最大跨步（+96B）。
    expect(f[20 + 12]).toBe(11);
    // B 的批起始仍为 256 字节对齐。
    expect(f[64 + 12]).toBe(99);

    renderer.dispose();
  });

  it('submitDirect 同样按各自管线跨步落位', () => {
    const { renderer, recorded } = makeRenderer();
    const p80 = registerPipeline(renderer, { label: 'p80', bytesPerInstance: 80 });
    const p96 = registerPipeline(renderer, { label: 'p96', bytesPerInstance: 96 });
    const geo = cubeGeometry(renderer.geometryArena);

    const transformsA = new Float32Array(32);
    transformsA.set(translate(7), 0);
    transformsA.set(translate(8), 16);

    renderer.submitDirect([
      { geometry: geo, pipeline: p80, transforms: transformsA, instanceCount: 2 },
      { geometry: geo, pipeline: p96, transforms: translate(77), instanceCount: 1 },
    ]);

    const f = instanceWrite(renderer, recorded);
    expect(f[12]).toBe(7);
    expect(f[20 + 12]).toBe(8);
    expect(f[64 + 12]).toBe(77);

    renderer.dispose();
  });

  it('submitCulled 的分组也按各自管线跨步落位', () => {
    const { renderer, recorded } = makeRenderer();
    const p80 = registerPipeline(renderer, { label: 'p80', bytesPerInstance: 80, compaction: true });
    const p96 = registerPipeline(renderer, { label: 'p96', bytesPerInstance: 96, compaction: true });
    const geoA = cubeGeometry(renderer.geometryArena);
    const geoB = cubeGeometry(renderer.geometryArena, 2);

    const transformsA = new Float32Array(32);
    transformsA.set(translate(3), 0);
    transformsA.set(translate(4), 16);

    const vp = multiply(
      new Float32Array(16),
      perspective(Math.PI / 4, 1, 0.1, 100),
      lookAt(new Float32Array([0, 0, 20]), new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
    );
    renderer.submitCulled(
      [
        { geometry: geoA, pipeline: p80, transforms: transformsA, instanceCount: 2 },
        { geometry: geoB, pipeline: p96, transforms: translate(88), instanceCount: 1 },
      ],
      vp,
    );

    const f = instanceWrite(renderer, recorded);
    expect(f[20 + 12]).toBe(4);
    expect(f[64 + 12]).toBe(88);

    renderer.dispose();
  });
});

// ─── 9. 相机深度排序基准点 ────────────────────────────────────

describe('相机深度排序使用世界空间基准点', () => {
  it('同一几何体在不同世界位置时按真实距离排序（近→远）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    // 先提交远的：若排序基准点用了局部中心（两者相同），就无法重排。
    renderer.submit(
      [
        { geometry: geo, pipeline, transforms: translate(1000) },
        { geometry: geo, pipeline, transforms: translate(1) },
      ],
      { camera: [0, 0, 0] },
    );

    // 两项同 geometry+pipeline → 合批成 1 个 batch，实例顺序即绘制顺序。
    const f = instanceWrite(renderer, recorded);
    expect(f[12]).toBe(1);      // 近的先画
    expect(f[20 + 12]).toBe(1000); // 远的后画

    renderer.dispose();
  });

  it('未提供相机时保持提交顺序（稳定计数排序）', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    renderer.submit([
      { geometry: geo, pipeline, transforms: translate(1000) },
      { geometry: geo, pipeline, transforms: translate(1) },
    ]);

    const f = instanceWrite(renderer, recorded);
    expect(f[12]).toBe(1000);
    expect(f[20 + 12]).toBe(1);

    renderer.dispose();
  });
});

// ─── 10. 实例存储容量必须先于填充确保 ─────────────────────────

describe('实例存储容量在填充前确保', () => {
  it('renderer 首帧走 submitDirect：实例矩阵不得写成零矩阵', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    renderer.submitDirect([{ geometry: geo, pipeline, transforms: translate(7) }]);

    const f = instanceWrite(renderer, recorded);
    expect(Array.from(f.slice(0, 16))).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 0, 0, 1]);

    renderer.dispose();
  });

  it('renderer 首帧走 submitCulled：实例矩阵不得写成零矩阵', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer, { compaction: true });
    const geo = cubeGeometry(renderer.geometryArena);

    const vp = multiply(
      new Float32Array(16),
      perspective(Math.PI / 4, 1, 0.1, 100),
      lookAt(new Float32Array([0, 0, 20]), new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0])),
    );
    renderer.submitCulled([{ geometry: geo, pipeline, transforms: translate(9) }], vp);

    const f = instanceWrite(renderer, recorded);
    expect(Array.from(f.slice(0, 16))).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 9, 0, 0, 1]);

    renderer.dispose();
  });

  it('实例数据超出当前存储容量（扩容那一帧）仍然是正确矩阵', () => {
    const { renderer, recorded } = makeRenderer();
    const pipeline = registerPipeline(renderer);
    const geo = cubeGeometry(renderer.geometryArena);

    // 20k 实例 × 80B = 1.6MB：同时超过初始 instanceStore（16384 浮点）与 ring（1MiB）。
    const count = 20_000;
    const transforms = new Float32Array(count * 16);
    for (let i = 0; i < count; i++) {
      transforms.set(translate(i), i * 16);
    }
    renderer.submitDirect([{ geometry: geo, pipeline, transforms, instanceCount: count }]);

    const f = instanceWrite(renderer, recorded);
    expect(f[12]).toBe(0);
    // 最后一个实例在 (count-1)*80 字节处
    const last = ((count - 1) * 80) / 4;
    expect(f[last + 12]).toBe(count - 1);

    renderer.dispose();
  });
});
