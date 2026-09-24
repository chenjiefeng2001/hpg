/**
 * Real-asset compatibility audit（Phase 11B）。
 *
 * 读取 benchmark/assets/models 下真实导出的 GLB，**独立于 hpg 解析器**直接读 glTF JSON，
 * 枚举模型真正使用到的 feature，再与 hpg 实际解析/导入结果对照，输出：
 *
 *   feature 使用率 → hpg 状态 (supported / ignored / broken) → 症状
 *
 * 运行: npm run audit
 *
 * 这个工具只做测量与报告，不修改任何 runtime 行为 —— 是否要修某个缺失 feature，
 * 由报告里的「受影响模型数」决定优先级。
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import { parseGltf } from '../src/core/gltf';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const MODEL_ROOT = resolve(ROOT, 'benchmark', 'assets', 'models');

// ─── 原始 glTF JSON（独立解析，避免和 hpg 解析器共用同一套盲点）──

interface RawPrimitive {
  attributes?: Record<string, number>;
  indices?: number;
  material?: number;
  mode?: number;
  targets?: unknown[];
  extensions?: Record<string, unknown>;
}

interface RawMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: number[];
    baseColorTexture?: { index: number };
    metallicRoughnessTexture?: { index: number };
  };
  normalTexture?: { index: number };
  occlusionTexture?: { index: number };
  emissiveTexture?: { index: number };
  emissiveFactor?: number[];
  alphaMode?: string;
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: Record<string, unknown>;
}

interface RawAccessor {
  bufferView?: number;
  componentType?: number;
  count?: number;
  type?: string;
  normalized?: boolean;
  sparse?: { count?: number };
}

interface RawNode {
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

interface RawGltf {
  asset?: { version?: string; generator?: string };
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: RawNode[];
  meshes?: { primitives?: RawPrimitive[] }[];
  materials?: RawMaterial[];
  textures?: { source?: number; sampler?: number }[];
  images?: { uri?: string; mimeType?: string; bufferView?: number }[];
  samplers?: { magFilter?: number; minFilter?: number; wrapS?: number; wrapT?: number }[];
  accessors?: RawAccessor[];
  bufferViews?: unknown[];
  buffers?: { byteLength?: number; uri?: string }[];
  animations?: { channels?: unknown[]; samplers?: unknown[] }[];
  skins?: { joints?: number[]; skeleton?: number }[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
}

/** 只读 GLB 容器的 JSON / BIN chunk 边界，不解释内容。 */
export function readGlbContainer(buffer: ArrayBuffer): { json: RawGltf; jsonBytes: number; binBytes: number } {
  if (buffer.byteLength < 12) throw new Error('buffer too small for GLB header');
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB container');
  const total = dv.getUint32(8, true);

  let json: RawGltf | null = null;
  let jsonBytes = 0;
  let binBytes = 0;
  let off = 12;
  while (off + 8 <= total) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    if (type === 0x4e4f534a) {
      jsonBytes = len;
      json = JSON.parse(new TextDecoder().decode(buffer.slice(off + 8, off + 8 + len))) as RawGltf;
    } else if (type === 0x004e4942) {
      binBytes = len;
    }
    off += 8 + len;
  }
  if (!json) throw new Error('no JSON chunk');
  return { json, jsonBytes, binBytes };
}

// ─── Feature 采集 ───────────────────────────────────────────

/** hpg gltf.ts 目前真正读取的顶点属性（其余属性会被静默丢弃）。 */
const HPGL_ATTRIBUTES = new Set(['POSITION', 'NORMAL', 'TEXCOORD_0', 'TANGENT']);

/** asset-importer 的 canonical layout：pos/normal/uv/tangent（TANGENT 已保留）。 */
const HPGL_CANONICAL_ATTRIBUTES = new Set(['POSITION', 'NORMAL', 'TEXCOORD_0', 'TANGENT']);

export interface AssetAudit {
  /** 相对 benchmark/assets/models 的路径。 */
  path: string;
  fileBytes: number;
  jsonBytes: number;
  binBytes: number;
  generator: string;
  version: string;

  meshCount: number;
  meshNodeCount: number;
  primitiveCount: number;
  triangleCount: number;
  vertexCount: number;
  materialCount: number;

  /** 属性名 → 使用该属性的 primitive 数。 */
  attributeUse: Record<string, number>;
  primitiveModes: Record<string, number>;
  indexComponentTypes: Record<string, number>;
  primitivesWithoutIndices: number;
  sparseAccessors: number;
  morphTargetPrimitives: number;

  textureCount: number;
  imageCount: number;
  imageMimeTypes: Record<string, number>;
  imageSources: Record<string, number>;

  materialFeatures: Record<string, number>;
  materialExtensions: Record<string, number>;

  sceneCount: number;
  nodeCount: number;
  maxHierarchyDepth: number;
  matrixNodes: number;
  trsNodes: number;

  animationCount: number;
  skinCount: number;
  skinnedJointCount: number;

  extensionsUsed: string[];
  extensionsRequired: string[];

  /** hpg 解析结果。 */
  hpg: { ok: boolean; error?: string; meshes: number; primitives: number; vertices: number; indices: number; materials: number };

  /** 由上述数据推导出的兼容性问题。 */
  issues: CompatIssue[];
}

export type CompatSeverity = 'broken' | 'wrong' | 'lossy';
export type CompatStatus = 'supported' | 'ignored' | 'broken';

export interface CompatIssue {
  feature: string;
  status: CompatStatus;
  severity: CompatSeverity;
  detail: string;
}

const bump = (rec: Record<string, number>, key: string, by = 1): void => {
  rec[key] = (rec[key] ?? 0) + by;
};

export function auditGlb(absPath: string): AssetAudit {
  const bytes = readFileSync(absPath);
  const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const { json, jsonBytes, binBytes } = readGlbContainer(ab);
  const accessors = json.accessors ?? [];

  const attributeUse: Record<string, number> = {};
  const primitiveModes: Record<string, number> = {};
  const indexComponentTypes: Record<string, number> = {};
  let primitiveCount = 0;
  let triangleCount = 0;
  let vertexCount = 0;
  let primitivesWithoutIndices = 0;
  let morphTargetPrimitives = 0;

  for (const mesh of json.meshes ?? []) {
    for (const prim of mesh.primitives ?? []) {
      primitiveCount++;
      const mode = prim.mode ?? 4;
      bump(primitiveModes, String(mode));
      for (const name of Object.keys(prim.attributes ?? {})) bump(attributeUse, name);
      if (prim.targets && prim.targets.length > 0) morphTargetPrimitives++;
      if (prim.indices == null) primitivesWithoutIndices++;
      else {
        const acc = accessors[prim.indices];
        if (acc?.componentType != null) bump(indexComponentTypes, String(acc.componentType));
      }
      const pos = prim.attributes?.POSITION;
      if (pos != null) {
        const acc = accessors[pos];
        vertexCount += acc?.count ?? 0;
        const idxCount = prim.indices != null ? accessors[prim.indices]?.count ?? 0 : acc?.count ?? 0;
        if (mode === 4) triangleCount += Math.floor(idxCount / 3);
      }
    }
  }

  const sparseAccessors = accessors.filter((a) => a.sparse != null).length;

  const materialFeatures: Record<string, number> = {};
  const materialExtensions: Record<string, number> = {};
  for (const mat of json.materials ?? []) {
    const pbr = mat.pbrMetallicRoughness ?? {};
    if (pbr.baseColorFactor) bump(materialFeatures, 'baseColorFactor');
    if (pbr.baseColorTexture) bump(materialFeatures, 'baseColorTexture');
    if (pbr.metallicRoughnessTexture) bump(materialFeatures, 'metallicRoughnessTexture');
    if (mat.normalTexture) bump(materialFeatures, 'normalTexture');
    if (mat.occlusionTexture) bump(materialFeatures, 'occlusionTexture');
    if (mat.emissiveTexture) bump(materialFeatures, 'emissiveTexture');
    if (mat.emissiveFactor && mat.emissiveFactor.some((v) => v !== 0)) bump(materialFeatures, 'emissiveFactor');
    bump(materialFeatures, `alphaMode:${mat.alphaMode ?? 'OPAQUE'}`);
    if (mat.alphaCutoff != null) bump(materialFeatures, 'alphaCutoff');
    if (mat.doubleSided) bump(materialFeatures, 'doubleSided');
    for (const ext of Object.keys(mat.extensions ?? {})) bump(materialExtensions, ext);
  }

  const imageMimeTypes: Record<string, number> = {};
  const imageSources: Record<string, number> = {};
  for (const img of json.images ?? []) {
    bump(imageMimeTypes, img.mimeType ?? (img.uri ? 'uri' : 'unknown'));
    bump(imageSources, img.bufferView != null ? 'bufferView' : 'uri');
  }

  // 层级深度
  const nodes = json.nodes ?? [];
  let meshNodeCount = 0;
  let matrixNodes = 0;
  let trsNodes = 0;
  for (const n of nodes) {
    if (n.mesh != null) meshNodeCount++;
    if (n.matrix) matrixNodes++;
    else if (n.translation || n.rotation || n.scale) trsNodes++;
  }
  const depthMemo = new Map<number, number>();
  const depthOf = (idx: number, stack: Set<number>): number => {
    const cached = depthMemo.get(idx);
    if (cached != null) return cached;
    if (stack.has(idx)) return 0; // 环保护
    stack.add(idx);
    const children = nodes[idx]?.children ?? [];
    let d = 0;
    for (const c of children) d = Math.max(d, 1 + depthOf(c, stack));
    stack.delete(idx);
    depthMemo.set(idx, d);
    return d;
  };
  const roots = (json.scenes?.[json.scene ?? 0]?.nodes ?? []).concat(
    (json.scenes ?? []).every(() => true) ? nodes.map((_, i) => i).filter((i) => !nodes.some((n) => (n.children ?? []).includes(i))) : [],
  );
  let maxHierarchyDepth = 0;
  for (const r of new Set(roots)) maxHierarchyDepth = Math.max(maxHierarchyDepth, depthOf(r, new Set()));

  const skins = json.skins ?? [];

  // ── hpg 解析对照 ──
  let hpgOk = false;
  let hpgError: string | undefined;
  let hpgMeshes = 0;
  let hpgPrimitives = 0;
  let hpgVertices = 0;
  let hpgIndices = 0;
  let hpgMaterials = 0;
  try {
    const asset = parseGltf(ab);
    hpgOk = true;
    hpgMeshes = asset.meshes.length;
    hpgMaterials = asset.materials.length;
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        hpgPrimitives++;
        hpgVertices += prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
        hpgIndices += prim.indices.length;
      }
    }
  } catch (e) {
    hpgError = (e as Error).message;
  }

  const issues: CompatIssue[] = [];

  if (!hpgOk) {
    issues.push({ feature: 'GLB container', status: 'broken', severity: 'broken', detail: hpgError ?? 'parse failed' });
  }

  // 未支持 / 未使用的顶点属性
  for (const [name, count] of Object.entries(attributeUse)) {
    if (!HPGL_ATTRIBUTES.has(name)) {
      const skinned = name.startsWith('JOINTS_') || name.startsWith('WEIGHTS_');
      issues.push({
        feature: name,
        status: 'ignored',
        severity: skinned ? 'wrong' : 'lossy',
        detail: skinned
          ? `${count} 个 primitive 使用蒙皮属性 → hpg 渲染绑定姿势（T-pose），不是动画后的姿态`
          : `${count} 个 primitive 使用 ${name} → 该属性被静默丢弃`,
      });
    }
  }

  // 非 TRIANGLES
  for (const [mode, count] of Object.entries(primitiveModes)) {
    if (mode !== '4') {
      issues.push({ feature: `primitive mode ${mode}`, status: 'ignored', severity: 'lossy', detail: `${count} 个 primitive 被跳过（只支持 TRIANGLES）` });
    }
  }

  if (sparseAccessors > 0) {
    issues.push({
      feature: 'accessor.sparse',
      status: 'broken',
      severity: 'wrong',
      detail: `${sparseAccessors} 个 accessor 使用 sparse → hpg 读原始 bufferView，几何数据错误（静默）`,
    });
  }

  if (morphTargetPrimitives > 0) {
    issues.push({ feature: 'morph targets', status: 'ignored', severity: 'lossy', detail: `${morphTargetPrimitives} 个 primitive 带 morph target → 只渲染基础形状` });
  }

  if (materialFeatures['baseColorTexture']) {
    issues.push({
      feature: 'material.baseColorTexture',
      status: 'supported',
      severity: 'lossy',
      detail: `${materialFeatures['baseColorTexture']} 个材质带 baseColorTexture → 已解析进 AssetMaterial（需外部 ImageDecoder 上传，否则退回 baseColorFactor）`,
    });
  }
  if (materialFeatures['normalTexture']) {
    issues.push({ feature: 'material.normalTexture', status: 'ignored', severity: 'lossy', detail: `${materialFeatures['normalTexture']} 个材质带法线贴图 → 光照细节丢失` });
  }
  if (materialFeatures['emissiveTexture'] || materialFeatures['emissiveFactor']) {
    issues.push({ feature: 'material.emissive', status: 'ignored', severity: 'lossy', detail: `自发光未进入 shader` });
  }
  if (materialFeatures['alphaMode:MASK']) {
    issues.push({
      feature: 'material.alphaMode=MASK',
      status: 'supported',
      severity: 'lossy',
      detail: `${materialFeatures['alphaMode:MASK']} 个材质 alphaMode=MASK → 材质管线按 alphaCutoff discard（需要材质 bind group）`,
    });
  }
  if (materialFeatures['alphaMode:BLEND']) {
    issues.push({
      feature: 'material.alphaMode=BLEND',
      status: 'ignored',
      severity: 'wrong',
      detail: `${materialFeatures['alphaMode:BLEND']} 个材质需要 alpha BLEND → hpg 按不透明渲染（会露出被遮挡的面）`,
    });
  }
  if (materialFeatures['baseColorFactor'] === undefined && (json.materials ?? []).length > 0) {
    issues.push({ feature: 'material.baseColorFactor default', status: 'supported', severity: 'lossy', detail: '使用默认白色' });
  }

  const skippedPrimitives = primitiveCount - hpgPrimitives;
  if (skippedPrimitives > 0 && hpgOk) {
    issues.push({ feature: 'primitives dropped', status: 'ignored', severity: 'lossy', detail: `${skippedPrimitives} 个 primitive 未进入 hpg` });
  }
  if ((json.extensionsRequired ?? []).length > 0) {
    issues.push({
      feature: 'extensionsRequired',
      status: 'supported',
      severity: 'lossy',
      detail: (json.extensionsRequired ?? []).join(', '),
    });
  }
  for (const [ext, count] of Object.entries(materialExtensions)) {
    issues.push({ feature: `ext ${ext}`, status: 'ignored', severity: 'lossy', detail: `${count} 个材质使用该扩展 → 退化为标准 PBR` });
  }
  if (json.animations?.length) {
    issues.push({ feature: 'animations', status: 'ignored', severity: 'lossy', detail: `${json.animations.length} 个动画未被播放` });
  }
  if (skins.length) {
    issues.push({ feature: 'skins', status: 'ignored', severity: 'wrong', detail: `${skins.length} 个 skin（${skins.reduce((a, s) => a + (s.joints?.length ?? 0), 0)} joints）→ 渲染绑定姿势` });
  }

  return {
    path: relative(MODEL_ROOT, absPath).split(sep).join('/'),
    fileBytes: bytes.byteLength,
    jsonBytes,
    binBytes,
    generator: json.asset?.generator ?? '(none)',
    version: json.asset?.version ?? '?',

    meshCount: (json.meshes ?? []).length,
    meshNodeCount,
    primitiveCount,
    triangleCount,
    vertexCount,
    materialCount: (json.materials ?? []).length,

    attributeUse,
    primitiveModes,
    indexComponentTypes,
    primitivesWithoutIndices,
    sparseAccessors,
    morphTargetPrimitives,

    textureCount: (json.textures ?? []).length,
    imageCount: (json.images ?? []).length,
    imageMimeTypes,
    imageSources,

    materialFeatures,
    materialExtensions,

    sceneCount: (json.scenes ?? []).length,
    nodeCount: nodes.length,
    maxHierarchyDepth,
    matrixNodes,
    trsNodes,

    animationCount: json.animations?.length ?? 0,
    skinCount: skins.length,
    skinnedJointCount: skins.reduce((a, s) => a + (s.joints?.length ?? 0), 0),

    extensionsUsed: json.extensionsUsed ?? [],
    extensionsRequired: json.extensionsRequired ?? [],

    hpg: { ok: hpgOk, error: hpgError, meshes: hpgMeshes, primitives: hpgPrimitives, vertices: hpgVertices, indices: hpgIndices, materials: hpgMaterials },

    issues,
  };
}

// ─── 语料扫描 ───────────────────────────────────────────────

export function listCorpus(): string[] {
  if (!existsSync(MODEL_ROOT)) return [];
  const out: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const full = resolve(dir, entry);
      if (statSync(full).isDirectory()) walk(full);
      else if (entry.endsWith('.glb')) out.push(full);
    }
  };
  walk(MODEL_ROOT);
  return out.sort();
}

// ─── 报告 ──────────────────────────────────────────────────

const pad = (s: string, n: number): string => (s.length > n ? s.slice(0, n) : s.padEnd(n));
const fmtK = (n: number): string => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));

export function renderReport(audits: AssetAudit[]): string {
  const lines: string[] = [];
  const models = audits.length;

  lines.push('');
  lines.push('hpg — Real Asset Compatibility Audit (Phase 11B)');
  lines.push('='.repeat(112));
  lines.push(`corpus: ${models} GLB files under benchmark/assets/models`);
  lines.push('');

  // 1. 逐模型
  lines.push('── 1. Per-model feature sheet ' + '─'.repeat(84));
  lines.push(
    pad('model', 30) + pad('gen', 14) + pad('mesh', 6) + pad('prim', 6) + pad('tri', 8) + pad('mat', 5) +
      pad('tex', 5) + pad('anim', 6) + pad('skin', 5) + pad('hpg', 12) + 'issues',
  );
  for (const a of audits) {
    const gen = a.generator.replace(/^Khronos glTF Blender I\/O/, 'BlenderIO');
    const hpgState = a.hpg.ok ? `${a.hpg.meshes}mesh/${a.hpg.primitives}prim` : 'PARSE FAIL';
    const worst = a.issues.some((i) => i.severity === 'broken')
      ? 'BROKEN'
      : a.issues.some((i) => i.severity === 'wrong')
        ? 'wrong'
        : a.issues.some((i) => i.severity === 'lossy')
          ? 'lossy'
          : 'ok';
    lines.push(
      pad(a.path, 30) + pad(gen, 14) + pad(String(a.meshCount), 6) + pad(String(a.primitiveCount), 6) +
        pad(fmtK(a.triangleCount), 8) + pad(String(a.materialCount), 5) + pad(String(a.textureCount), 5) +
        pad(String(a.animationCount), 6) + pad(String(a.skinCount), 5) + pad(hpgState, 12) + worst,
    );
  }

  // 2. 属性使用率
  const attrTotals: Record<string, number> = {};
  const attrModels: Record<string, number> = {};
  for (const a of audits) {
    for (const [name, count] of Object.entries(a.attributeUse)) {
      bump(attrTotals, name, count);
      bump(attrModels, name);
    }
  }
  lines.push('');
  lines.push('── 2. Vertex attribute usage ' + '─'.repeat(86));
  lines.push(pad('attribute', 16) + pad('models', 8) + pad('primitives', 12) + 'hpg status');
  const attrNames = Object.keys(attrModels).sort((x, y) => attrModels[y]! - attrModels[x]!);
  for (const name of attrNames) {
    const status = !HPGL_ATTRIBUTES.has(name)
      ? 'IGNORED (silently dropped)'
      : !HPGL_CANONICAL_ATTRIBUTES.has(name)
        ? 'parsed then dropped in importer'
        : 'supported';
    lines.push(pad(name, 16) + pad(String(attrModels[name]), 8) + pad(String(attrTotals[name]), 12) + status);
  }

  // 3. 材质 feature
  const matTotals: Record<string, number> = {};
  const matModels: Record<string, number> = {};
  for (const a of audits) {
    for (const [name, count] of Object.entries(a.materialFeatures)) {
      bump(matTotals, name, count);
      bump(matModels, name);
    }
  }
  const totalMaterials = audits.reduce((acc, a) => acc + a.materialCount, 0);
  const totalTextures = audits.reduce((acc, a) => acc + a.textureCount, 0);
  lines.push('');
  lines.push('── 3. Material / texture feature usage ' + '─'.repeat(77));
  lines.push(pad('feature', 34) + pad('models', 8) + pad('materials', 11) + 'hpg status');
  const matNames = Object.keys(matModels).sort((x, y) => matTotals[y]! - matTotals[x]!);
  const matStatus = (name: string): string => {
    if (name === 'baseColorFactor') return 'supported';
    if (name === 'alphaMode:OPAQUE') return 'supported (default)';
    if (name === 'doubleSided') return 'parsed, not used by pipeline';
    if (name === 'baseColorTexture') return 'parsed (needs decoder)';
    if (name === 'alphaMode:MASK') return 'sample + discard';
    return 'IGNORED';
  };
  for (const name of matNames) {
    lines.push(pad(name, 34) + pad(String(matModels[name]), 8) + pad(String(matTotals[name]), 11) + matStatus(name));
  }
  lines.push(pad('(textures declared)', 34) + pad(String(audits.filter((a) => a.textureCount > 0).length), 8) + pad(String(totalTextures), 11) + `images: ${audits.reduce((a, x) => a + x.imageCount, 0)}`);
  lines.push(pad('(materials total)', 34) + pad('', 8) + pad(String(totalMaterials), 11));

  // 4. 场景 / 动画
  lines.push('');
  lines.push('── 4. Scene / animation ' + '─'.repeat(89));
  lines.push(pad('model', 30) + pad('nodes', 7) + pad('depth', 7) + pad('matrix/TRS', 12) + pad('scenes', 8) + pad('anim', 6) + 'skins(joints)');
  for (const a of audits) {
    lines.push(
      pad(a.path, 30) + pad(String(a.nodeCount), 7) + pad(String(a.maxHierarchyDepth), 7) +
        pad(`${a.matrixNodes}/${a.trsNodes}`, 12) + pad(String(a.sceneCount), 8) + pad(String(a.animationCount), 6) +
        (a.skinCount ? `${a.skinCount}(${a.skinnedJointCount})` : '-'),
    );
  }

  // 5. 支持矩阵
  interface IssueAggregate {
    models: Set<string>;
    status: CompatStatus;
    severity: CompatSeverity;
    detail: string;
    /** 逐模型 detail 里前导数字的和（如「6 个材质带贴图」→ 跨模型合计）。 */
    countSum: number;
  }
  const issueModels = new Map<string, IssueAggregate>();
  for (const a of audits) {
    for (const issue of a.issues) {
      const e =
        issueModels.get(issue.feature) ??
        ({ models: new Set(), status: issue.status, severity: issue.severity, detail: issue.detail, countSum: 0 } as IssueAggregate);
      e.models.add(a.path);
      const leading = /^(\d+)\s/.exec(issue.detail);
      if (leading) e.countSum += Number(leading[1]);
      if (issue.severity === 'broken') e.severity = 'broken';
      issueModels.set(issue.feature, e);
    }
  }
  lines.push('');
  lines.push('── 5. Compatibility matrix (feature → hpg) ' + '─'.repeat(72));
  lines.push(pad('feature', 34) + pad('models', 8) + pad('status', 10) + 'impact');
  const sevRank: Record<CompatSeverity, number> = { broken: 0, wrong: 1, lossy: 2 };
  const sorted = [...issueModels.entries()].sort(
    (x, y) => sevRank[x[1].severity] - sevRank[y[1].severity] || y[1].models.size - x[1].models.size,
  );
  for (const [feature, e] of sorted) {
    // 多模型命中时把逐模型的前导数字换成跨模型合计（避免「22 个模型 / 1 个材质」这类自相矛盾）。
    const detail =
      e.models.size > 1 && e.countSum > 0 ? e.detail.replace(/^\d+\s/, `${e.countSum} `) : e.detail;
    lines.push(pad(feature, 34) + pad(String(e.models.size), 8) + pad(e.severity.toUpperCase(), 10) + detail);
  }

  const broken = sorted.filter(([, e]) => e.severity === 'broken');
  const wrong = sorted.filter(([, e]) => e.severity === 'wrong');
  lines.push('');
  lines.push('── 6. Verdict ' + '─'.repeat(98));
  lines.push(`parse ok: ${audits.filter((a) => a.hpg.ok).length}/${models}`);
  lines.push(`BROKEN (silent wrong data): ${broken.length === 0 ? 'none' : broken.map(([f]) => f).join(', ')}`);
  lines.push(`WRONG (visible wrong output): ${wrong.length === 0 ? 'none' : wrong.map(([f]) => `${f} [${issueModels.get(f)!.models.size}]`).join(', ')}`);
  lines.push('');
  return lines.join('\n');
}

// ─── CLI ───────────────────────────────────────────────────

/**
 * CLI 入口（由 benchmark/asset-compat-cli.ts 调用 —— vite-node 会从 argv 中剥掉脚本路径，
 * 因此无法在本模块内可靠地判定「是否被直接运行」，故拆出独立入口文件）。
 *
 * @param filter 可选子串过滤（如 "heavy" / "DamagedHelmet"）。
 */
export function main(filter?: string): void {
  const files = listCorpus();
  if (files.length === 0) {
    console.log('no .glb found under benchmark/assets/models');
    return;
  }
  const target = filter ? files.filter((f) => f.includes(filter)) : files;
  const audits: AssetAudit[] = [];
  for (const file of target) {
    try {
      audits.push(auditGlb(file));
    } catch (e) {
      console.error(`FAILED to audit ${file}: ${(e as Error).message}`);
    }
  }
  console.log(renderReport(audits));
}
