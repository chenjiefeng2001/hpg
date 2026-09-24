/**
 * Phase 15A / 15B —— baseColorTexture 最小闭环 + TANGENT 保留。
 *
 * 真实语料里 22/23 个模型使用 baseColorTexture，而 hpg 此前只采样 baseColorFactor。
 * 本文件锁定这条路径的关键不变量（全部在 Node + fake GPU 下可跑）：
 *
 *   glTF image → AssetMaterial.baseColorTexture（解析，不解码）
 *        ↓  ImageDecoder（注入）
 *   GPUTexture（sRGB）+ GPUSampler
 *        ↓  MaterialStore
 *   material bind group（固定 group=2）
 *        ↓
 *   RenderItem.bindGroup → executor.setBindGroup(2, ...)
 *
 * 以及 TANGENT 不再在 canonical layout 中丢失。
 */
import { describe, it, expect, vi } from 'vitest';

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { parseGltf } from '../src/core/gltf';
import { importGltfAsset, sceneToRenderItems } from '../src/core/asset-importer';
import { MaterialStore, createBrowserImageDecoder, createMaterialBindGroupLayout } from '../src/core/texture';
import type { ImageDecoder } from '../src/core/texture';
import {
  VS_INSTANCED,
  VS_INSTANCED_MATERIAL,
  VS_INSTANCED_MATERIAL_COMPACTION,
  FS_COLOR,
  FS_MATERIAL,
} from '../src/shaders/instance';
import { createFakeGPU } from './fake-gpu';
import type { GltfAsset } from '../src/core/gltf';

const FORMAT: GPUTextureFormat = 'bgra8unorm';

/** canonical 顶点布局（pos3 + norm3 + uv2 + tan4，stride 48）。 */
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

// ─── 合成 GLB（含贴图 / MASK / 独立 TANGENT bufferView）──────

const TANGENTS = [0.6, 0.8, 0, 1, 0.8, 0.6, 0, -1, 1, 0, 0, 1];

function buildGlb(): ArrayBuffer {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  const uvs = new Float32Array([0, 0, 1, 0, 0, 1]);
  const tangents = new Float32Array(TANGENTS);
  const indices = new Uint16Array([0, 1, 2]);
  const image = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4, 5, 6, 7, 8]);

  const chunks: Uint8Array[] = [];
  const offsets: number[] = [];
  let cursor = 0;
  const push = (bytes: Uint8Array): void => {
    const pad = (4 - (cursor % 4)) % 4;
    if (pad > 0) {
      chunks.push(new Uint8Array(pad));
      cursor += pad;
    }
    offsets.push(cursor);
    chunks.push(bytes);
    cursor += bytes.byteLength;
  };
  push(new Uint8Array(positions.buffer));
  push(new Uint8Array(normals.buffer));
  push(new Uint8Array(uvs.buffer));
  push(new Uint8Array(tangents.buffer));
  push(new Uint8Array(indices.buffer));
  push(image);

  const bin = new Uint8Array(cursor);
  let o = 0;
  for (const c of chunks) {
    bin.set(c, o);
    o += c.byteLength;
  }

  const gltf = {
    asset: { version: '2.0', generator: 'hpg-material-test' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ name: 'Root', mesh: 0 }],
    meshes: [
      {
        name: 'Quad',
        primitives: [
          { attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2, TANGENT: 3 }, indices: 4, material: 0 },
          // 第二个 primitive 故意不带 TANGENT → 检查 canonical 默认值填充。
          { attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 4, material: 1 },
        ],
      },
    ],
    materials: [
      {
        name: 'Textured',
        pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], baseColorTexture: { index: 0 } },
      },
      {
        name: 'Masked',
        pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.6, 1] },
        alphaMode: 'MASK',
        alphaCutoff: 0.35,
      },
      { name: 'Blended', alphaMode: 'BLEND' },
    ],
    textures: [{ source: 0, sampler: 0 }],
    samplers: [{ magFilter: 9728, minFilter: 9987, wrapS: 33071, wrapT: 10497 }],
    images: [{ bufferView: 5, mimeType: 'image/png' }],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC2' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC4' },
      { bufferView: 4, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: offsets[0], byteLength: positions.byteLength, target: 34962 },
      { buffer: 0, byteOffset: offsets[1], byteLength: normals.byteLength, target: 34962 },
      { buffer: 0, byteOffset: offsets[2], byteLength: uvs.byteLength, target: 34962 },
      { buffer: 0, byteOffset: offsets[3], byteLength: tangents.byteLength, target: 34962 },
      { buffer: 0, byteOffset: offsets[4], byteLength: indices.byteLength, target: 34963 },
      { buffer: 0, byteOffset: offsets[5], byteLength: image.byteLength },
    ],
    buffers: [{ byteLength: bin.byteLength }],
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
  const jsonPadding = (4 - (jsonBytes.byteLength % 4)) % 4;
  const jsonChunkLength = jsonBytes.byteLength + jsonPadding;
  const binPadding = (4 - (bin.byteLength % 4)) % 4;
  const binChunkLength = bin.byteLength + binPadding;
  const total = 12 + 8 + jsonChunkLength + 8 + binChunkLength;
  const glb = new ArrayBuffer(total);
  const view = new DataView(glb);

  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);

  let offset = 12;
  view.setUint32(offset, jsonChunkLength, true);
  view.setUint32(offset + 4, 0x4e4f534a, true);
  const jsonDst = new Uint8Array(glb, offset + 8, jsonChunkLength);
  jsonDst.set(jsonBytes);
  for (let i = jsonBytes.byteLength; i < jsonChunkLength; i++) jsonDst[i] = 0x20;
  offset += 8 + jsonChunkLength;

  view.setUint32(offset, binChunkLength, true);
  view.setUint32(offset + 4, 0x004e4942, true);
  new Uint8Array(glb, offset + 8, binChunkLength).set(bin);

  return glb;
}

function makeRenderer() {
  const { device, context, recorded } = createFakeGPU();
  const renderer = Renderer.create({ device, context, format: FORMAT });
  return { renderer, recorded };
}

function registerMaterialPipeline(renderer: Renderer, materialLayout: GPUBindGroupLayout) {
  const layout = uniformBindGroupLayout(renderer.device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniform = renderer.device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return renderer.registerPipeline({
    label: 'material-pipeline',
    vsCode: VS_INSTANCED_MATERIAL,
    fsCode: FS_MATERIAL,
    vertexLayouts: CANONICAL_LAYOUT,
    bindGroupLayouts: [layout, materialLayout],
    globalBindings: [{ binding: 0, buffer: uniform }],
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: FORMAT }],
  });
}

function registerCulledMaterialPipeline(renderer: Renderer, materialLayout: GPUBindGroupLayout) {
  const layout = uniformBindGroupLayout(renderer.device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniform = renderer.device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  return renderer.registerPipeline({
    label: 'material-pipeline-culled',
    vsCode: VS_INSTANCED_MATERIAL_COMPACTION,
    fsCode: FS_MATERIAL,
    compaction: true,
    vertexLayouts: CANONICAL_LAYOUT,
    bindGroupLayouts: [layout, materialLayout],
    globalBindings: [{ binding: 0, buffer: uniform }],
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format: FORMAT }],
  });
}

/** 单位 VP（不关心剔除结果，只验证绑定与 indirect draw 形态）。 */
const IDENTITY_VP = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function uniformWrites(recorded: ReturnType<typeof createFakeGPU>['recorded']): Uint8Array[] {
  return recorded.writes.filter((w) => w.bytes.byteLength === 32).map((w) => w.bytes);
}

/** 32B material uniform 中的 alphaMode（u32 @ offset 20）。 */
function alphaModeOf(bytes: Uint8Array): number {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, 8)[5]!;
}

// ─── 解析层 ─────────────────────────────────────────────────

describe('Phase 15A — glTF 材质/贴图解析', () => {
  it('解析内嵌图片字节、sampler 状态与 baseColorTexture 引用（不解码）', () => {
    const asset = parseGltf(buildGlb());

    expect(asset.images).toHaveLength(1);
    expect(asset.images[0]!.mimeType).toBe('image/png');
    expect(asset.images[0]!.data.byteLength).toBe(12);
    expect(asset.images[0]!.data[0]).toBe(0x89);

    const ref = asset.materials[0]!.baseColorTexture;
    expect(ref).toBeDefined();
    expect(ref!.imageIndex).toBe(0);
    expect(ref!.texCoord).toBe(0);
    // glTF sampler 枚举 → WebGPU 描述（9987 → linear/linear）。
    expect(ref!.sampler).toEqual({
      magFilter: 'nearest',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'repeat',
    });
  });

  it('解析 alphaMode / alphaCutoff；MASK 不再产生 warning，BLEND 仍然提示', () => {
    const asset = parseGltf(buildGlb());
    expect(asset.materials[1]!.alphaMode).toBe('MASK');
    expect(asset.materials[1]!.alphaCutoff).toBe(0.35);
    expect(asset.materials[2]!.alphaMode).toBe('BLEND');

    expect(asset.warnings.some((w) => w.startsWith('baseColorTexture'))).toBe(false);
    expect(asset.warnings.some((w) => w.includes('MASK'))).toBe(false);
    expect(asset.warnings).toEqual(['alphaMode=BLEND 未实现：被当作不透明渲染，会露出被遮挡的面。']);
  });
});

// ─── 导入层：TANGENT 保留 ───────────────────────────────────

describe('Phase 15B — canonical layout 保留 TANGENT', () => {
  it('stride=48 且 4 个 location 全部声明；缺失 TANGENT 时填默认值', () => {
    const { renderer, recorded } = makeRenderer();
    const scene = importGltfAsset(parseGltf(buildGlb()), renderer, { flipV: true });

    expect(scene.meshes).toHaveLength(2);
    for (const mesh of scene.meshes) {
      const layout = mesh.geometry.vertexLayouts[0]!;
      expect(layout.arrayStride).toBe(48);
      expect(layout.attributes.map((a) => a.shaderLocation)).toEqual([0, 1, 2, 3]);
    }

    // 第一个 primitive 的顶点流：第三个 vec4 就是原始 TANGENT（未被丢弃）。
    const vertexWrite = recorded.writes.find((w) => w.bytes.byteLength === 3 * 48);
    expect(vertexWrite).toBeDefined();
    const floats = new Float32Array(vertexWrite!.bytes.buffer, vertexWrite!.bytes.byteOffset, 36);
    const tangent = Array.from(floats.slice(8, 12));
    for (let i = 0; i < 4; i++) expect(tangent[i]).toBeCloseTo(TANGENTS[i]!, 6);
    // UV V 翻转只影响 uv.y（index 7），不影响 tangent。
    expect(floats[7]).toBe(1); // 原 uv.y = 0 → 1
  });

  it('imported material 搬运 baseColorTexture / alphaMode', () => {
    const { renderer } = makeRenderer();
    const scene = importGltfAsset(parseGltf(buildGlb()), renderer, { flipV: true });

    expect(scene.materials).toHaveLength(3);
    expect(scene.materials[0]!.baseColorTexture?.imageIndex).toBe(0);
    expect(scene.materials[1]!.alphaMode).toBe('MASK');
    expect(scene.materials[1]!.alphaCutoff).toBe(0.35);
    expect(scene.materials[2]!.alphaMode).toBe('BLEND');
  });
});

// ─── 材质（GPU）层 ──────────────────────────────────────────

describe('Phase 15A — MaterialStore', () => {
  it('解码器可注入：图片只解码一次、上传为 sRGB 纹理并生成 group=2 bind group', async () => {
    const { renderer, recorded } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const decode = vi.fn(async () => ({ width: 2, height: 2, data: new Uint8Array(16).fill(200) }));
    const decoder: ImageDecoder = { decode };

    const store = await MaterialStore.create(renderer.device, asset, decoder, { label: 'test' });

    expect(decode).toHaveBeenCalledTimes(1);
    expect(decode).toHaveBeenCalledWith(expect.any(Uint8Array), 'image/png');
    expect(store.stats).toEqual({ materials: 3, textures: 1, skipped: [] });

    // 白色 fallback（1×1） + 材质贴图（2×2），一律 sRGB。
    const srgb = recorded.textures.filter((t) => t.format === 'rgba8unorm-srgb');
    expect(srgb).toHaveLength(2);
    expect(recorded.textureWrites).toEqual([
      { width: 1, height: 1, byteLength: 4 },
      { width: 2, height: 2, byteLength: 16 },
    ]);
    // 采样器来自 glTF sampler（nearest / clamp + repeat）。
    expect(recorded.samplers.at(-1)).toMatchObject({
      magFilter: 'nearest',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'repeat',
    });

    // bind group 布局：texture / sampler / material uniform(32B)。
    const materialBG = recorded.bindGroups.find((b) => b.label.includes('material-bg'));
    expect(materialBG).toBeDefined();
    expect(materialBG!.bindings.map((b) => b.binding)).toEqual([0, 1, 2]);
    // 材质 uniform buffer 固定 32 字节（vec4 factor + alphaCutoff + alphaMode + padding）。
    expect((materialBG!.bindings[2]!.buffer as { size?: number }).size).toBe(32);

    expect(store.has(0)).toBe(true);
    expect(store.has(2)).toBe(true);
    expect(store.has(9)).toBe(false);
    expect(store.bindGroupFor(9)).toBeUndefined();

    // 材质 uniform 的 alphaMode / alphaCutoff 正确编码（index 1 = Masked）。
    const uniforms = uniformWrites(recorded);
    expect(uniforms).toHaveLength(3);
    const masked = uniforms[1]!;
    const f = new Float32Array(masked.buffer, masked.byteOffset, 8);
    for (const [i, v] of [0.2, 0.4, 0.6, 1].entries()) expect(f[i]).toBeCloseTo(v, 6);
    expect(f[4]).toBeCloseTo(0.35, 6);
    expect(alphaModeOf(masked)).toBe(1);
    expect(alphaModeOf(uniforms[0]!)).toBe(0);
    expect(alphaModeOf(uniforms[2]!)).toBe(2);

    store.dispose();
  });

  it('解码失败时退回白色 fallback，记录 skipped，但材质仍可用', async () => {
    const { renderer, recorded } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const decoder: ImageDecoder = {
      decode: async () => {
        throw new Error('boom');
      },
    };

    const store = await MaterialStore.create(renderer.device, asset, decoder);
    expect(store.stats.materials).toBe(3);
    expect(store.stats.textures).toBe(0);
    expect(store.stats.skipped).toHaveLength(1);
    expect(store.stats.skipped[0]).toContain('boom');
    expect(store.has(0)).toBe(true);
    // 只有白色 fallback 被创建。
    expect(recorded.textures.filter((t) => t.format === 'rgba8unorm-srgb')).toHaveLength(1);

    store.dispose();
  });

  it('暴露浏览器解码器工厂与材质布局工具', () => {
    expect(typeof createBrowserImageDecoder).toBe('function');
    const { device } = createFakeGPU();
    const layout = createMaterialBindGroupLayout(device);
    expect(layout).toBeDefined();
  });
});

// ─── 渲染层：group=2 真正被绑定 ─────────────────────────────

describe('Phase 15A — 材质 bind group 落在 group 2', () => {
  it('registerPipeline([global, material]) → [global, instance, material]', async () => {
    const { renderer } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const store = await MaterialStore.create(renderer.device, asset, {
      decode: async () => ({ width: 1, height: 1, data: new Uint8Array(4) }),
    });
    const pipeline = registerMaterialPipeline(renderer, store.layout);
    expect(pipeline.bindGroupLayouts).toHaveLength(3);
    store.dispose();
  });

  it('sceneToRenderItems 附加 bindGroup；submit 时绑定到 group 2 且无校验错误', async () => {
    const { renderer, recorded } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const store = await MaterialStore.create(renderer.device, asset, {
      decode: async () => ({ width: 2, height: 2, data: new Uint8Array(16) }),
    });
    const pipeline = registerMaterialPipeline(renderer, store.layout);

    const scene = importGltfAsset(asset, renderer, { flipV: true });
    const items = sceneToRenderItems(scene, pipeline, store);

    expect(items).toHaveLength(2);
    expect(items[0]!.bindGroup).toBe(store.bindGroupFor(0));
    expect(items[1]!.bindGroup).toBe(store.bindGroupFor(1));
    // 附上材质时实例颜色是白色（baseColorFactor 由 material uniform 提供，避免乘两次）。
    expect(Array.from(items[0]!.instanceData!.slice(0, 4))).toEqual([1, 1, 1, 1]);

    const stats = renderer.submit(items);
    expect(stats.itemsDrawn).toBe(2);
    expect(recorded.gpuErrors).toEqual([]);

    const group2 = recorded.renderBinds.filter((b) => b.group === 2);
    expect(group2).toHaveLength(2);
    expect(group2[0]!.bindGroup).toBe(store.bindGroupFor(0));
    expect(group2[1]!.bindGroup).toBe(store.bindGroupFor(1));

    store.dispose();
  });

  it('submitDirect 逐 item 重绑 group 2', async () => {
    const { renderer, recorded } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const store = await MaterialStore.create(renderer.device, asset, {
      decode: async () => ({ width: 1, height: 1, data: new Uint8Array(4) }),
    });
    const pipeline = registerMaterialPipeline(renderer, store.layout);
    const scene = importGltfAsset(asset, renderer, { flipV: true });
    const items = sceneToRenderItems(scene, pipeline, store);

    renderer.submitDirect(items);
    expect(recorded.gpuErrors).toEqual([]);
    const group2 = recorded.renderBinds.filter((b) => b.group === 2);
    expect(group2).toHaveLength(2);
    expect(group2[0]!.bindGroup).toBe(store.bindGroupFor(0));
    expect(group2[1]!.bindGroup).toBe(store.bindGroupFor(1));

    store.dispose();
  });

  it('GPU Culled 路径按 (geometry, material) 拆组并绑定 group 2', async () => {
    const { renderer, recorded } = makeRenderer();
    const asset = parseGltf(buildGlb());
    const store = await MaterialStore.create(renderer.device, asset, {
      decode: async () => ({ width: 1, height: 1, data: new Uint8Array(4) }),
    });
    const pipeline = registerCulledMaterialPipeline(renderer, store.layout);
    const scene = importGltfAsset(asset, renderer, { flipV: true });
    const items = sceneToRenderItems(scene, pipeline, store);

    const stats = renderer.submitCulled(items, IDENTITY_VP);
    expect(stats.batches).toBe(2); // 两个 (geometry, material) 组
    expect(recorded.indirectDraws).toHaveLength(2);
    expect(recorded.gpuErrors).toEqual([]);

    const group2 = recorded.renderBinds.filter((b) => b.group === 2);
    expect(group2).toHaveLength(2);
    expect(group2[0]!.bindGroup).toBe(store.bindGroupFor(0));
    expect(group2[1]!.bindGroup).toBe(store.bindGroupFor(1));

    store.dispose();
  });

  it('不提供材质时保持旧的纯色路径（无 group 2 绑定）', () => {
    const { renderer, recorded } = makeRenderer();
    const asset: GltfAsset = parseGltf(buildGlb());
    const layout = uniformBindGroupLayout(renderer.device, [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
    ]);
    const uniform = renderer.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipeline = renderer.registerPipeline({
      label: 'plain',
      // 纯色管线（不声明 group 2）—— 验证无材质时不会绑 group 2。
      vsCode: VS_INSTANCED,
      fsCode: FS_COLOR,
      vertexLayouts: CANONICAL_LAYOUT,
      bindGroupLayouts: [layout],
      globalBindings: [{ binding: 0, buffer: uniform }],
      targets: [{ format: FORMAT }],
    });

    const scene = importGltfAsset(asset, renderer, { flipV: true });
    const items = sceneToRenderItems(scene, pipeline);
    expect(items[0]!.bindGroup).toBeUndefined();
    // 无材质 → 实例颜色回落到 baseColorFactor。
    expect(Array.from(items[0]!.instanceData!.slice(0, 4))).toEqual([1, 1, 1, 1]);

    renderer.submit(items);
    expect(recorded.renderBinds.some((b) => b.group === 2)).toBe(false);
  });
});
