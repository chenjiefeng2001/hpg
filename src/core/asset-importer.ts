/**
 * glTF Asset Importer — 将 GltfAsset 转换为 hpg Geometry + RenderItem。
 *
 * 职责分离：
 *   - gltf.ts: 解析 glTF 格式 → GltfAsset
 *   - asset-importer.ts: GltfAsset → hpg GeometryArena + RenderItem[]
 *
 * 不依赖任何 glTF 库；贴图解码/上传交由注入的 ImageDecoder + MaterialStore（见 texture.ts）。
 */

/// <reference types="@webgpu/types" />

import type { Aabb, Geometry, RenderItem, ResolvedPipeline, VertexLayoutDesc } from '../types';
import type { AssetMaterial, AssetPrimitive, AssetTextureRef, GltfAsset, FlattenedNode } from './gltf';
import { flattenScene } from './gltf';
import type { GeometryArena } from './geometry';
import type { Renderer } from './renderer';
import { identity } from './math';

// ─── Types ──────────────────────────────────────────────────

export interface ImportOptions {
  /** 全局缩放因子。默认 1。 */
  scale?: number;
  /** 是否翻转 V 坐标（glTF V=0 在底部，WebGPU V=0 在顶部）。默认 true。 */
  flipV?: boolean;
}

export interface ImportedMesh {
  name: string;
  geometry: Geometry;
  materialIndex: number;
  worldMatrix: Float32Array;
}

export interface ImportedMaterial {
  name: string;
  baseColor: [number, number, number, number];
  /** base color 贴图引用（图片字节在 GltfAsset.images 中，需外部解码）。 */
  baseColorTexture?: AssetTextureRef;
  alphaMode: 'OPAQUE' | 'MASK' | 'BLEND';
  alphaCutoff: number;
}

/**
 * 提供材质 bind group 的最小接口（MaterialStore 结构上即满足）。
 * `sceneToRenderItems` 只依赖这两个方法，不绑定具体实现。
 */
export interface MaterialBindingSource {
  has(materialIndex: number): boolean;
  bindGroupFor(materialIndex: number): GPUBindGroup | undefined;
}

export interface ImportedScene {
  meshes: ImportedMesh[];
  /** 逐 materialIndex 的规范材质数据。 */
  materials: ImportedMaterial[];
  /** flattenScene 的结果（供后续使用）。 */
  flatNodes: FlattenedNode[];
  /** Scene bounding box: [minX, minY, minZ, maxX, maxY, maxZ] */
  bounds: [number, number, number, number, number, number];
}

// ─── Geometry Creation ──────────────────────────────────────

/**
 * Canonical vertex layout for the viewer: position + normal + UV + tangent.
 * All models are converted to this layout so the pipeline doesn't need to change.
 *
 * TANGENT 始终保留（缺失时填默认值），因为顶点布局一旦公开就很难再改 ——
 * 布局漂移会牵动 interleaving / stride / shader input / pipeline / importer 全链路。
 */
const CANONICAL_STRIDE = 48; // 3+3+2+4 = 12 floats × 4 bytes
const CANONICAL_LAYOUT = {
  arrayStride: CANONICAL_STRIDE,
  attributes: [
    { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },   // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },  // normal
    { shaderLocation: 2, offset: 24, format: 'float32x2' as GPUVertexFormat },  // uv
    { shaderLocation: 3, offset: 32, format: 'float32x4' as GPUVertexFormat },  // tangent (xyz + handedness)
  ],
};

/** 缺失 TANGENT 时的默认值：+X 方向、右手系。 */
const DEFAULT_TANGENT = [1, 0, 0, 1] as const;

/**
 * Convert any vertex layout to the canonical layout (stride 48).
 * Missing attributes are filled with defaults (0 for UV, +X for tangent).
 */
function toCanonicalLayout(
  vertices: Float32Array,
  srcLayout: { arrayStride: number; attributes: { shaderLocation: number; offset: number; format: GPUVertexFormat }[] },
): Float32Array {
  const srcStride = srcLayout.arrayStride / 4; // floats per vertex
  const vertCount = vertices.length / srcStride;
  const out = new Float32Array(vertCount * 12); // 12 floats per output vertex (pos3 + norm3 + uv2 + tan4)

  // Find source attribute offsets (in floats)
  const getAttr = (loc: number) => srcLayout.attributes.find(a => a.shaderLocation === loc);

  const posAttr = getAttr(0); // position
  const normAttr = getAttr(1); // normal
  const uvAttr = getAttr(2); // uv
  const tangAttr = getAttr(3); // tangent

  const posOff = posAttr ? posAttr.offset / 4 : 0;
  const normOff = normAttr ? normAttr.offset / 4 : 3;
  const uvOff = uvAttr ? uvAttr.offset / 4 : 6;
  const tangOff = tangAttr ? tangAttr.offset / 4 : 0;

  for (let v = 0; v < vertCount; v++) {
    const src = v * srcStride;
    const dst = v * 12;

    // Position (required)
    if (posAttr) {
      out[dst] = vertices[src + posOff];
      out[dst + 1] = vertices[src + posOff + 1];
      out[dst + 2] = vertices[src + posOff + 2];
    }

    // Normal (required)
    if (normAttr) {
      out[dst + 3] = vertices[src + normOff];
      out[dst + 4] = vertices[src + normOff + 1];
      out[dst + 5] = vertices[src + normOff + 2];
    } else {
      out[dst + 5] = 1; // default normal Z
    }

    // UV (optional)
    if (uvAttr) {
      out[dst + 6] = vertices[src + uvOff];
      out[dst + 7] = vertices[src + uvOff + 1];
    }

    // Tangent (optional) — 不再丢弃；缺失时填默认值，保证 layout 恒定。
    if (tangAttr) {
      out[dst + 8] = vertices[src + tangOff];
      out[dst + 9] = vertices[src + tangOff + 1];
      out[dst + 10] = vertices[src + tangOff + 2];
      out[dst + 11] = vertices[src + tangOff + 3];
    } else {
      out[dst + 8] = DEFAULT_TANGENT[0];
      out[dst + 9] = DEFAULT_TANGENT[1];
      out[dst + 10] = DEFAULT_TANGENT[2];
      out[dst + 11] = DEFAULT_TANGENT[3];
    }
  }

  return out;
}

// ─── Bounds ─────────────────────────────────────────────────

/** 默认材质颜色（无材质时）。 */
const DEFAULT_BASE_COLOR: [number, number, number, number] = [1, 1, 1, 1];

/** 附上材质 bind group 时写入实例的白色 —— baseColorFactor 由 material uniform 提供，避免乘两次。 */
const WHITE: [number, number, number, number] = [1, 1, 1, 1];

/**
 * 单个 primitive 的局部空间 AABB（直接来自顶点流，未应用节点变换）。
 */
function primitiveLocalAabb(prim: AssetPrimitive): Aabb | null {
  const posAttr = prim.vertexLayout.attributes.find((a) => a.shaderLocation === 0);
  if (!posAttr) return null;
  const stride = prim.vertexLayout.arrayStride / 4;
  if (stride <= 0) return null;
  const posOff = posAttr.offset / 4;
  const vertCount = Math.floor(prim.vertices.length / stride);
  if (vertCount <= 0) return null;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let v = 0; v < vertCount; v++) {
    const o = v * stride + posOff;
    const x = prim.vertices[o] as number;
    const y = prim.vertices[o + 1] as number;
    const z = prim.vertices[o + 2] as number;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!isFinite(minX)) return null;
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

/** 把局部 AABB 的 8 个角点变换到世界空间后并入 min/max。 */
function expandWorldBounds(
  min: [number, number, number],
  max: [number, number, number],
  aabb: Aabb,
  m: Float32Array,
): void {
  for (let corner = 0; corner < 8; corner++) {
    const x = corner & 1 ? aabb.max[0] : aabb.min[0];
    const y = corner & 2 ? aabb.max[1] : aabb.min[1];
    const z = corner & 4 ? aabb.max[2] : aabb.min[2];
    const wx = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
    const wy = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
    const wz = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
    if (wx < min[0]) min[0] = wx;
    if (wy < min[1]) min[1] = wy;
    if (wz < min[2]) min[2] = wz;
    if (wx > max[0]) max[0] = wx;
    if (wy > max[1]) max[1] = wy;
    if (wz > max[2]) max[2] = wz;
  }
}

/**
 * 将 glTF primitive 转换为 hpg Geometry（通过 GeometryArena）。
 */
function createGeometryFromPrimitive(
  arena: GeometryArena,
  vertices: Float32Array,
  vertexLayout: { arrayStride: number; attributes: { shaderLocation: number; offset: number; format: GPUVertexFormat }[] },
  indices: Uint16Array | Uint32Array,
  indexFormat: GPUIndexFormat,
  flipV: boolean,
): Geometry {
  // Convert to canonical layout (stride 48: pos3 + norm3 + uv2 + tan4)
  let canonicalVerts = toCanonicalLayout(vertices, vertexLayout);

  // Flip V coordinate if needed
  if (flipV) {
    for (let v = 0; v < canonicalVerts.length / 12; v++) {
      canonicalVerts[v * 12 + 7] = 1 - canonicalVerts[v * 12 + 7]; // flip UV.y
    }
  }

  // Use canonical layout
  const hpgLayouts: VertexLayoutDesc[] = [{
    arrayStride: CANONICAL_STRIDE,
    stepMode: 'vertex',
    attributes: CANONICAL_LAYOUT.attributes.map((a) => ({
      shaderLocation: a.shaderLocation,
      offset: a.offset,
      format: a.format,
    })),
  }];

  return arena.createGeometry(canonicalVerts, hpgLayouts, indices, indexFormat);
}

// ─── Material Conversion ────────────────────────────────────

/**
 * 将 glTF material 转换为规范材质数据。
 *
 * 第一版只搬运真实语料已经证明需要的东西：baseColorFactor / baseColorTexture /
 * alphaMode / alphaCutoff —— 不把整个 PBR 材质搬进 runtime。
 */
function convertMaterial(gltfMat: AssetMaterial): ImportedMaterial {
  return {
    name: gltfMat.name,
    baseColor: gltfMat.baseColorFactor,
    baseColorTexture: gltfMat.baseColorTexture,
    alphaMode: gltfMat.alphaMode,
    alphaCutoff: gltfMat.alphaCutoff,
  };
}

// ─── Main Importer ──────────────────────────────────────────

/**
 * 将 GltfAsset 导入为 hpg 可用的几何体和渲染数据。
 *
 * @param asset - parseGltf() 返回的 GltfAsset
 * @param renderer - hpg Renderer 实例（用于 GeometryArena）
 * @param options - 导入选项
 * @returns ImportedScene — 包含 Geometry 引用 + worldMatrix
 */
export function importGltfAsset(
  asset: GltfAsset,
  renderer: Renderer,
  options: ImportOptions = {},
): ImportedScene {
  const { scale = 1, flipV = true } = options;
  const arena = renderer.geometryArena;

  // 展平场景树
  const flatNodes = flattenScene(asset);

  // 为每个有 mesh 的节点创建 Geometry
  const meshes: ImportedMesh[] = [];

  // 世界空间包围盒累加器（必须包含节点变换 + 全局缩放，否则取景 / 近远平面会算错）。
  const boundsMin: [number, number, number] = [Infinity, Infinity, Infinity];
  const boundsMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];

  // 同一 mesh 可能被多个节点复用（linked duplicate / instancing）：
  // 几何体与局部 AABB 都按 primitive 缓存 —— 共享同一 Geometry 引用的 item
  // 才能被 Batcher 合并成一个 instanced draw。
  const geometryCache = new Map<AssetPrimitive, Geometry>();
  const localAabbs = new Map<AssetPrimitive, Aabb | null>();

  for (const node of flatNodes) {
    if (node.meshIndex == null) continue;
    const mesh = asset.meshes[node.meshIndex];
    if (!mesh) continue;

    // 应用全局缩放：M' = S × M（列主序下标 0/1/2、4/5/6、8/9/10 是线性部分，
    // 12/13/14 是平移 —— 整体缩放必须连平移一起缩，否则"全局缩放"会改变模型相对位置）。
    const worldMatrix = new Float32Array(node.worldMatrix);
    if (scale !== 1) {
      for (const i of [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]) {
        worldMatrix[i] = worldMatrix[i]! * scale;
      }
    }

    for (const prim of mesh.primitives) {
      let geometry = geometryCache.get(prim);
      if (!geometry) {
        geometry = createGeometryFromPrimitive(
          arena,
          prim.vertices,
          prim.vertexLayout,
          prim.indices,
          prim.indexFormat,
          flipV,
        );
        geometryCache.set(prim, geometry);
      }

      meshes.push({
        name: `${node.name}/${mesh.name}`,
        geometry,
        materialIndex: prim.materialIndex,
        worldMatrix,
      });

      let local = localAabbs.get(prim);
      if (local === undefined) {
        local = primitiveLocalAabb(prim);
        localAabbs.set(prim, local);
      }
      if (local) expandWorldBounds(boundsMin, boundsMax, local, worldMatrix);
    }
  }

  // 转换材质
  const materials = asset.materials.map(convertMaterial);

  const bounds: [number, number, number, number, number, number] = isFinite(boundsMin[0])
    ? [boundsMin[0], boundsMin[1], boundsMin[2], boundsMax[0], boundsMax[1], boundsMax[2]]
    : [0, 0, 0, 0, 0, 0];

  return { meshes, materials, flatNodes, bounds };
}

/**
 * 从 ImportedScene 生成 RenderItem[]。
 *
 * 全局 uniform 由管线描述符（`PipelineDesc.globalBindings`）在注册时声明，渲染器按管线缓存并注入 group=0，
 * 因此本函数不接受 globalBindings —— 传了也不会生效的「静默参数」在这里被移除。
 *
 * @param scene - importGltfAsset 返回的场景
 * @param pipeline - 已注册的 ResolvedPipeline
 * @param materials - 可选材质提供者（MaterialStore）；提供后逐 mesh 附加 `bindGroup`（group=2）
 * @returns RenderItem[] — 可直接传入 renderer.submit() 或 renderer.submitCulled()
 */
export function sceneToRenderItems(
  scene: ImportedScene,
  pipeline: ResolvedPipeline,
  materials?: MaterialBindingSource,
): RenderItem[] {
  const items: RenderItem[] = [];

  // instanceData 的布局必须匹配管线的 [预留区 + mat4(64B) + 额外数据] 跨步，
  // 否则 Renderer 会忽略这段数据（防止越界读出 NaN）。
  const extraFloats = Math.max(
    4,
    (pipeline.bytesPerInstance - (pipeline.modelMatrixOffset || 0) - 64) >> 2,
  );

  for (const mesh of scene.meshes) {
    const mat = scene.materials[mesh.materialIndex];
    // 附上材质 bind group 的 mesh 使用白色实例颜色（baseColorFactor 在 material uniform 中，
    // 若实例颜色也带 factor 会被乘两次）。无材质时保持旧的纯色路径。
    const bindGroup = materials?.has(mesh.materialIndex)
      ? materials.bindGroupFor(mesh.materialIndex)
      : undefined;
    const baseColor = bindGroup ? WHITE : mat?.baseColor ?? DEFAULT_BASE_COLOR;

    const instanceCount = Math.max(1, Math.floor(mesh.worldMatrix.length / 16));
    // 目前 instanceData 只承载材质颜色（modelMatrix 之后的首个 vec4）。
    const instanceData = new Float32Array(instanceCount * extraFloats);
    for (let i = 0; i < instanceCount; i++) {
      instanceData[i * extraFloats] = baseColor[0];
      instanceData[i * extraFloats + 1] = baseColor[1];
      instanceData[i * extraFloats + 2] = baseColor[2];
      instanceData[i * extraFloats + 3] = baseColor[3];
    }

    const item: RenderItem = {
      geometry: mesh.geometry,
      pipeline,
      transforms: mesh.worldMatrix,
      instanceCount,
      instanceData,
    };
    if (bindGroup) item.bindGroup = bindGroup;
    items.push(item);
  }

  return items;
}
