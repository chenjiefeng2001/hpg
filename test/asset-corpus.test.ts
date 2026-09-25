/**
 * Phase 11D/11E —— 真实资产回归语料。
 *
 * 用 benchmark/assets/models 下的 23 个**真实导出** GLB（Blender / glTF-Transform /
 * COLLADA2GLTF / 3ds Max / babylon.js 各自的导出结果，含多 mesh、多材质、层级、
 * 蒙皮、动画）跑通完整链路：
 *
 *   parseGltf → importGltfAsset(arena 上传) → sceneToRenderItems → submit / submitCulled
 *
 * 目的：把此前只在真实 Chrome 里暴露的两类 GPU-only bug 固化进 CI ——
 *   1. 实例 bind group 绑定尺寸 / dynamic offset 越界（第 2 个 batch 起整帧失效）
 *   2. GPU Culled 的 draw args CPU 预写晚于 compute dispatch（instanceCount 被清零）
 *
 * 这里所有断言都必须在 Node + fake GPU 下可跑（无需浏览器）；
 * 像素级 Direct/Culled 一致性由真实 Chrome harness 负责，不在此处。
 */
import { readFileSync } from 'node:fs';
import { describe, it, expect } from 'vitest';

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { importGltfAsset, sceneToRenderItems } from '../src/core/asset-importer';
import { parseGltf } from '../src/core/gltf';
import { lookAt, perspective, multiply } from '../src/core/math';
import { listCorpus, auditGlb, type AssetAudit } from '../benchmark/asset-compat';
import { createFakeGPU } from './fake-gpu';
import { VS_INSTANCED, VS_INSTANCED_COMPACTION, FS_COLOR } from '../src/shaders/instance';
import type { Geometry, RenderItem } from '../src/types';

const FORMAT: GPUTextureFormat = 'bgra8unorm';

/** asset-importer 的 canonical 顶点布局（pos3 + norm3 + uv2 + tan4，stride 48）。 */
const CANONICAL_LAYOUT = [
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

/**
 * 语料 golden —— 每个真实模型解析后的几何规模（由 `npm run audit` 生成）。
 * 导入器若开始丢 mesh / primitive / 顶点，这里会失败。
 *
 * `material.baseColorTexture` / `material.alphaMode=MASK` 已从 wrong 台账移除：
 * baseColorTexture 现在解析进 `AssetMaterial` + MaterialStore（group=2 材质路径），
 * MASK 由 FS_MATERIAL 按 alphaCutoff discard。BLEND（需要透明排序）仍然 wrong。
 */
const GOLDEN: Record<string, { meshes: number; prims: number; mats: number; verts: number; indices: number }> = {
  'feature/AlphaBlendModeTest.glb': { meshes: 9, prims: 9, mats: 6, verts: 137, indices: 183 },
  'feature/ClearCoatTest.glb': { meshes: 27, prims: 27, mats: 19, verts: 20106, indices: 111348 },
  'feature/EmissiveStrengthTest.glb': { meshes: 6, prims: 6, mats: 6, verts: 188, indices: 270 },
  'feature/IridescenceLamp.glb': { meshes: 3, prims: 3, mats: 3, verts: 10683, indices: 56598 },
  'feature/MetalRoughSpheres.glb': { meshes: 5, prims: 5, mats: 1, verts: 255914, indices: 1505328 },
  'feature/MorphPrimitivesTest.glb': { meshes: 1, prims: 2, mats: 2, verts: 30, indices: 96 },
  'feature/MultiUVTest.glb': { meshes: 1, prims: 1, mats: 1, verts: 24, indices: 36 },
  'feature/SheenChair.glb': { meshes: 4, prims: 4, mats: 6, verts: 22459, indices: 119808 },
  'feature/TextureEncodingTest.glb': { meshes: 14, prims: 14, mats: 14, verts: 2468, indices: 11532 },
  'feature/TextureLinearInterpolationTest.glb': { meshes: 3, prims: 3, mats: 3, verts: 414, indices: 1926 },
  'feature/TransmissionTest.glb': { meshes: 22, prims: 22, mats: 14, verts: 66902, indices: 386325 },
  'heavy/BrainStem.glb': { meshes: 1, prims: 59, mats: 59, verts: 34159, indices: 184998 },
  'heavy/CesiumMan.glb': { meshes: 1, prims: 1, mats: 1, verts: 3273, indices: 14016 },
  'heavy/CesiumMilkTruck.glb': { meshes: 2, prims: 4, mats: 4, verts: 3995, indices: 8568 },
  'heavy/Duck.glb': { meshes: 1, prims: 1, mats: 1, verts: 2399, indices: 12636 },
  'heavy/Fox.glb': { meshes: 1, prims: 1, mats: 1, verts: 1728, indices: 1728 },
  'light/Avocado.glb': { meshes: 1, prims: 1, mats: 1, verts: 406, indices: 2046 },
  'light/BoomBox.glb': { meshes: 1, prims: 1, mats: 1, verts: 3575, indices: 18108 },
  'light/BoxTextured.glb': { meshes: 1, prims: 1, mats: 1, verts: 24, indices: 36 },
  'light/Corset.glb': { meshes: 1, prims: 1, mats: 1, verts: 11505, indices: 54972 },
  'medium/DamagedHelmet.glb': { meshes: 1, prims: 1, mats: 1, verts: 14556, indices: 46356 },
  'medium/Lantern.glb': { meshes: 3, prims: 3, mats: 1, verts: 4145, indices: 16182 },
  'medium/WaterBottle.glb': { meshes: 1, prims: 1, mats: 1, verts: 2549, indices: 13530 },
};

/**
 * 已知限制台账（Phase 11B 审计结论）。
 *
 * 这些是真实资产命中、而 hpg 目前明确不支持的 feature（severity = wrong，即输出可见错误）。
 * 一旦某个改动新增/消除了其中一项，这里会失败 —— 请同步更新 benchmark 审计报告与 PLAN.md，
 * 而不是顺手把断言改掉。
 */
const KNOWN_WRONG_FEATURES = [
  'JOINTS_0',
  'WEIGHTS_0',
  'material.alphaMode=BLEND',
  'skins',
];

/** 多 mesh / 多 primitive 的代表性资产（单 mesh 模型无法暴露的 bug 必须有它们兜底）。 */
const MULTI_GEOMETRY_MODELS = [
  'feature/ClearCoatTest.glb',
  'feature/TransmissionTest.glb',
  'feature/TextureEncodingTest.glb',
  'heavy/BrainStem.glb',
];

interface CorpusCase {
  /** 相对 benchmark/assets/models 的 posix 路径（与审计报告一致）。 */
  path: string;
  bytes: ArrayBuffer;
  /** 模块加载时算一次，避免每个用例重复解析 11MB 模型。 */
  audit: AssetAudit;
}

const CORPUS: CorpusCase[] = listCorpus().map((file) => {
  const buf = readFileSync(file);
  const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  const audit = auditGlb(file);
  return { path: audit.path, bytes, audit };
});

function caseOf(path: string): CorpusCase {
  const found = CORPUS.find((c) => c.path === path);
  if (!found) throw new Error(`corpus 中不存在 ${path}`);
  return found;
}

function makeRenderer() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = Renderer.create({ device, context, format: FORMAT });
  return { recorded, renderer };
}

function registerPipeline(renderer: Renderer, compaction: boolean) {
  const layout = uniformBindGroupLayout(renderer.device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniform = renderer.device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return renderer.registerPipeline({
    label: compaction ? 'corpus-culled' : 'corpus-direct',
    vsCode: compaction ? VS_INSTANCED_COMPACTION : VS_INSTANCED,
    fsCode: FS_COLOR,
    vertexLayouts: CANONICAL_LAYOUT,
    bindGroupLayouts: [layout],
    globalBindings: [{ binding: 0, buffer: uniform }],
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: FORMAT }],
    compaction,
  });
}

/** 按 geometry 分组数 —— submitCulled 的 draw args 是逐 geometry 的。 */
function geometryGroupCount(items: RenderItem[]): number {
  const groups = new Set<Geometry>();
  for (const item of items) groups.add(item.geometry);
  return groups.size;
}

/** 从 bounds 推导一个能把整个模型框住的 VP 矩阵。 */
function frameVp(bounds: readonly number[]): Float32Array {
  const cx = ((bounds[0] as number) + (bounds[3] as number)) / 2;
  const cy = ((bounds[1] as number) + (bounds[4] as number)) / 2;
  const cz = ((bounds[2] as number) + (bounds[5] as number)) / 2;
  const r = Math.hypot(
    ((bounds[3] as number) - (bounds[0] as number)) / 2,
    ((bounds[4] as number) - (bounds[1] as number)) / 2,
    ((bounds[5] as number) - (bounds[2] as number)) / 2,
  );
  const radius = Math.max(r, 1e-3);
  const fov = Math.PI / 4;
  const dist = (radius / Math.sin(fov / 2)) * 1.25;
  return multiply(
    new Float32Array(16),
    perspective(fov, 1, 0.01, dist * 4),
    lookAt(new Float32Array([cx, cy, cz + dist]), new Float32Array([cx, cy, cz]), new Float32Array([0, 1, 0])),
  );
}

describe('Phase 11D — 真实资产语料完整性', () => {
  it('语料清单与 golden 完全一致（增删模型需要同步 golden 与审计报告）', () => {
    expect(CORPUS.map((c) => c.path)).toEqual(Object.keys(GOLDEN));
  });

  it('全部模型解析出的规模与 golden 一致', () => {
    const mismatches: string[] = [];
    for (const { path, audit } of CORPUS) {
      const want = GOLDEN[path]!;
      const got = {
        meshes: audit.hpg.meshes,
        prims: audit.hpg.primitives,
        mats: audit.materialCount,
        verts: audit.hpg.vertices,
        indices: audit.hpg.indices,
      };
      if (!audit.hpg.ok) mismatches.push(`${path}: parse failed — ${audit.hpg.error}`);
      else if (JSON.stringify(got) !== JSON.stringify(want)) {
        mismatches.push(`${path}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
      }
    }
    expect(mismatches).toEqual([]);
  });

  it('不支持 feature 台账与审计结论一致（wrong = 可见错误输出）', () => {
    const wrong = new Set<string>();
    for (const { audit } of CORPUS) {
      for (const issue of audit.issues) {
        if (issue.severity === 'wrong' || issue.severity === 'broken') wrong.add(issue.feature);
      }
    }
    expect([...wrong].sort()).toEqual(KNOWN_WRONG_FEATURES);
  });
});

describe('Phase 11D — Direct 路径（submit）跑通全部真实资产', () => {
  it('每个模型都能导入、上传并提交，无 GPU 校验错误', () => {
    const failures: string[] = [];

    for (const { path, bytes } of CORPUS) {
      const { recorded, renderer } = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(bytes), renderer, { flipV: true });
        const items = sceneToRenderItems(scene, registerPipeline(renderer, false));

        if (items.length === 0) failures.push(`${path}: 0 render items`);
        for (const item of items) {
          for (const v of item.transforms ?? []) if (!Number.isFinite(v)) failures.push(`${path}: 变换含非有限值`);
          for (const v of item.instanceData ?? []) if (!Number.isFinite(v)) failures.push(`${path}: instanceData 含非有限值`);
        }

        const stats = renderer.submit(items);
        if (stats.instances !== items.length) failures.push(`${path}: instances ${stats.instances} != items ${items.length}`);
        if (recorded.drawCalls.length === 0) failures.push(`${path}: 0 draw call`);
        const drawn = recorded.drawCalls.reduce((a, d) => a + d.instanceCount, 0);
        if (drawn !== items.length) failures.push(`${path}: 绘制实例 ${drawn} != ${items.length}`);
        if (recorded.gpuErrors.length > 0) failures.push(`${path}: GPU 校验错误 — ${recorded.gpuErrors[0]}`);
      } catch (e) {
        failures.push(`${path}: ${(e as Error).message}`);
      } finally {
        renderer.dispose();
      }
    }

    expect(failures).toEqual([]);
  });

  it('多 mesh 资产产生真正的多批 dynamic offset（此前第 2 批起整帧失效的 bug）', () => {
    let multiBatchModels = 0;
    const failures: string[] = [];

    for (const { path, bytes } of CORPUS) {
      const { recorded, renderer } = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(bytes), renderer, { flipV: true });
        const items = sceneToRenderItems(scene, registerPipeline(renderer, false));
        renderer.submit(items);

        const offsets = recorded.renderBinds.filter((b) => b.group === 1).flatMap((b) => b.offsets);
        if (recorded.drawCalls.length > 1) {
          multiBatchModels++;
          if (Math.max(...offsets) === 0) failures.push(`${path}: 多批但 dynamic offset 全为 0`);
        }
        if (recorded.gpuErrors.length > 0) failures.push(`${path}: ${recorded.gpuErrors[0]}`);
      } finally {
        renderer.dispose();
      }
    }

    // 语料里必须真的存在多批模型，否则这个测试什么都没验证。
    expect(multiBatchModels).toBeGreaterThanOrEqual(8);
    expect(failures).toEqual([]);
  });
});

describe('Phase 11D — GPU Culled 路径（submitCulled）跑通全部真实资产', () => {
  it('每个模型都产生 draw args、dispatch，且预写早于 dispatch、无 GPU 校验错误', () => {
    const failures: string[] = [];

    for (const { path, bytes } of CORPUS) {
      const { recorded, renderer } = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(bytes), renderer, { flipV: true });
        if (scene.meshes.length === 0) continue;
        const items = sceneToRenderItems(scene, registerPipeline(renderer, true));

        const stats = renderer.submitCulled(items, frameVp(scene.bounds));
        if (stats.instances !== items.length) failures.push(`${path}: instances ${stats.instances} != ${items.length}`);
        if (recorded.dispatches === 0) failures.push(`${path}: 没有 compute dispatch`);
        if (recorded.indirectDraws.length !== geometryGroupCount(items)) {
          failures.push(`${path}: indirect draw ${recorded.indirectDraws.length} != geometry 组 ${geometryGroupCount(items)}`);
        }
        if (recorded.gpuErrors.length > 0) failures.push(`${path}: GPU 校验错误 — ${recorded.gpuErrors[0]}`);

        // draw args 的 CPU 预写必须早于 dispatch，否则 GPU 原子累加的 instanceCount 会被清零。
        const dispatchIdx = recorded.ops.findIndex((o) => o.kind === 'dispatch');
        const argsWrites = recorded.ops
          .map((o, i) => ({ o, i }))
          .filter(({ o }) => o.kind === 'write' && recorded.indirectDraws.some((d) => d.buffer === o.buffer));
        if (dispatchIdx < 0 || argsWrites.length === 0) failures.push(`${path}: 找不到 draw args 预写`);
        else if (Math.max(...argsWrites.map(({ i }) => i)) > dispatchIdx) failures.push(`${path}: draw args 预写晚于 dispatch`);
      } catch (e) {
        failures.push(`${path}: ${(e as Error).message}`);
      } finally {
        renderer.dispose();
      }
    }

    expect(failures).toEqual([]);
  });

  it('Direct 与 Culled 的实例规模一致（分组不串）', () => {
    const failures: string[] = [];

    for (const { path, bytes } of CORPUS) {
      const direct = makeRenderer();
      const culled = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(bytes), direct.renderer, { flipV: true });
        if (scene.meshes.length === 0) continue;
        const dItems = sceneToRenderItems(scene, registerPipeline(direct.renderer, false));
        const cItems = sceneToRenderItems(scene, registerPipeline(culled.renderer, true));

        if (dItems.length !== cItems.length) failures.push(`${path}: item 数不一致`);
        const dInstances = dItems.reduce((a, i) => a + (i.instanceCount ?? 0), 0);
        const cInstances = cItems.reduce((a, i) => a + (i.instanceCount ?? 0), 0);
        if (dInstances !== cInstances) failures.push(`${path}: 实例数 ${dInstances} != ${cInstances}`);
        if (geometryGroupCount(dItems) !== geometryGroupCount(cItems)) failures.push(`${path}: geometry 组数不一致`);
      } finally {
        direct.renderer.dispose();
        culled.renderer.dispose();
      }
    }

    expect(failures).toEqual([]);
  });
});

describe('Phase 11D — 世界空间包围盒 / 层级', () => {
  it('每个模型的 bounds 都是有限且有实际尺寸的（节点变换 + 全局缩放都进入包围盒）', () => {
    const failures: string[] = [];

    for (const { path, bytes } of CORPUS) {
      const { renderer } = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(bytes), renderer, { flipV: true });
        const [minX, minY, minZ, maxX, maxY, maxZ] = scene.bounds;
        if (![minX, minY, minZ, maxX, maxY, maxZ].every(Number.isFinite)) {
          failures.push(`${path}: bounds 含非有限值`);
          continue;
        }
        // 允许单轴退化为 0（MorphPrimitivesTest 是平面），但整体必须有尺寸，
        // 且 max >= min —— 全零 bounds 说明节点变换 / 全局缩放没进包围盒。
        const extents = [maxX - minX, maxY - minY, maxZ - minZ];
        if (extents.some((e) => e < 0)) failures.push(`${path}: bounds 区间反转 [${scene.bounds}]`);
        else if (Math.max(...extents) <= 0) failures.push(`${path}: bounds 无尺寸 [${scene.bounds}]`);
      } finally {
        renderer.dispose();
      }
    }

    expect(failures).toEqual([]);
  });

  it('全局缩放同时作用于平移分量（否则层级模型相对位置会错）', () => {
    const bytes = caseOf('heavy/Fox.glb').bytes;
    const a = makeRenderer();
    const b = makeRenderer();
    try {
      const scaled = importGltfAsset(parseGltf(bytes), a.renderer, { scale: 10, flipV: true });
      const plain = importGltfAsset(parseGltf(bytes), b.renderer, { scale: 1, flipV: true });
      for (let i = 0; i < 6; i++) expect(scaled.bounds[i]!).toBeCloseTo(plain.bounds[i]! * 10, 3);
      const m0 = scaled.meshes[0]!.worldMatrix;
      const m1 = plain.meshes[0]!.worldMatrix;
      // 平移分量（列主序 12/13/14）也必须被缩放。
      for (const i of [12, 13, 14]) expect(m0[i]!).toBeCloseTo(m1[i]! * 10, 5);
    } finally {
      a.renderer.dispose();
      b.renderer.dispose();
    }
  });
});

describe('Phase 11D — 多几何资产生成多个 draw', () => {
  it('多 primitive / 多 mesh 资产不会被合并成单个 draw', () => {
    for (const path of MULTI_GEOMETRY_MODELS) {
      const { recorded, renderer } = makeRenderer();
      try {
        const scene = importGltfAsset(parseGltf(caseOf(path).bytes), renderer, { flipV: true });
        const items = sceneToRenderItems(scene, registerPipeline(renderer, false));
        renderer.submit(items);
        expect(items.length).toBeGreaterThan(1);
        expect(recorded.drawCalls.length).toBeGreaterThan(1);
        expect(geometryGroupCount(items)).toBeGreaterThan(1);
      } finally {
        renderer.dispose();
      }
    }
  });
});

/**
 * Phase 15F — alphaMode=BLEND 的需求边界。
 *
 * BLEND 不是「if (alpha < cutoff) discard」——它牵动 transparent classification → sorting →
 * depthWrite/depthTest → blend state，可能演变成独立的 render phase，属于 render execution
 * semantics 的改动。因此先把「真实需求有多大」固定下来，而不是因为 audit 还剩一项就自动开工。
 *
 * 实测：只有 2 个 Khronos feature-test 模型用到 BLEND，且各自只有 1 个材质。
 * 一旦出现第 3 个（尤其非 feature/ 的生产资产），这里会失败 → 强制一次显式决策。
 */
describe('Phase 15F — alphaMode=BLEND 需求边界', () => {
  it('只有 2 个 feature-test 模型使用 BLEND，且各只有 1 个材质', () => {
    const withBlend = CORPUS
      .map((c) => ({ path: c.path, materials: c.audit.materialFeatures['alphaMode:BLEND'] ?? 0 }))
      .filter((x) => x.materials > 0);

    expect(withBlend).toEqual([
      { path: 'feature/AlphaBlendModeTest.glb', materials: 1 },
      { path: 'feature/ClearCoatTest.glb', materials: 1 },
    ]);
    // 全部来自 feature/ 测试集 —— 没有生产资产命中 → 保持 warning，不实现透明排序。
    expect(withBlend.every((x) => x.path.startsWith('feature/'))).toBe(true);
  });
});
