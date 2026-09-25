/**
 * Phase 11C —— 「模型用到但 hpg 未实现」的 feature 必须显式告知。
 *
 * 真实资产（Blender / glTF-Transform / COLLADA2GLTF 导出）里大量 feature 属于
 * 「忽略后照样能画、但结果是错的」：贴图、alphaMode、蒙皮、morph、扩展材质 …
 * 这些以前是静默丢弃的 —— 页面上只表现为「模型看着不对，但没有任何提示」。
 *
 * 本文件锁定三件事：
 *   1. parseGltf 的 `asset.warnings` 覆盖全部真实语料，且文案来自已知集合；
 *   2. warnings 与 benchmark 审计（独立读原始 glTF JSON）的结论一致；
 *   3. 代表性模型的 warnings 精确可复现（新增/减少一条都会失败，需要显式更新）。
 */
import { readFileSync } from 'node:fs';
import { sep } from 'node:path';
import { describe, it, expect } from 'vitest';

import { parseGltf } from '../src/core/gltf';
import { listCorpus, auditGlb } from '../benchmark/asset-compat';

function load(relPath: string): ReturnType<typeof parseGltf> {
  const file = listCorpus().find((f) => f.endsWith(relPath.split('/').join(sep)));
  if (!file) throw new Error(`语料中缺少 ${relPath}`);
  const buf = readFileSync(file);
  return parseGltf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
}

/**
 * 已知警告类别 —— 新增一类提示时必须同时更新此表（防止无意的噪声/文案漂移）。
 */
const KNOWN_WARNING_PATTERNS = [
  /^metallic-roughness \/ normal \/ occlusion \/ emissive 贴图未采样/,
  // baseColorTexture / alphaMode=MASK 已实现（材质路径），不再产生 warning；
  // BLEND 仍需透明排序与混合状态，继续 warning。
  /^alphaMode=BLEND 未实现/,
  /^glTF 动画未播放/,
  /^JOINTS_0\/WEIGHTS_0 未实现/,
  /^morph target（变形目标）未实现/,
  /^COLOR_0（顶点色）被忽略/,
  /^TEXCOORD_\d 被忽略/,
  /^贴图 texCoord=\d+ 被忽略/,
  /^doubleSided=true 未实现/,
  /^顶点属性 [A-Z0-9_]+ 未实现/,
  /^模型声明了必需扩展/,
  /^glTF 扩展 [A-Za-z0-9_]+ 未实现/,
  /^外部图片文件/,
];

/** 审计 feature → 解析器必须给出的 warning 文案片段（两套独立实现的交叉校验）。 */
const AUDIT_FEATURE_TO_WARNING: Record<string, string> = {
  // baseColorTexture / MASK 已在材质路径实现 → 不再要求对应 warning。
  'material.alphaMode=BLEND': 'alphaMode=BLEND 未实现',
  doubleSided: 'doubleSided=true 未实现',
  'material.normalTexture': 'normal / occlusion / emissive 贴图未采样',
  JOINTS_0: 'JOINTS_0/WEIGHTS_0 未实现',
  WEIGHTS_0: 'JOINTS_0/WEIGHTS_0 未实现',
  skins: 'JOINTS_0/WEIGHTS_0 未实现',
  'morph targets': 'morph target（变形目标）未实现',
  animations: 'glTF 动画未播放',
  TEXCOORD_1: 'TEXCOORD_1 被忽略',
};

describe('Phase 11C — 未实现 feature 的结构化提示', () => {
  it('全部真实语料的 warnings 文案都属于已知类别', () => {
    const failures: string[] = [];
    let withWarnings = 0;
    for (const file of listCorpus()) {
      const buf = readFileSync(file);
      const asset = parseGltf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
      if (asset.warnings.length > 0) withWarnings++;
      for (const w of asset.warnings) {
        if (!KNOWN_WARNING_PATTERNS.some((p) => p.test(w))) failures.push(`${file}: 未登记的提示「${w}」`);
      }
    }
    // 语料仍以「未实现 feature」为主（贴图之外的 PBR 贴图、蒙皮、动画…），
    // 但不再强制每个模型都有 warning —— baseColorTexture 现在是实现而非缺失。
    expect(withWarnings).toBeGreaterThanOrEqual(20);
    expect(failures).toEqual([]);
  });

  it('审计（独立读 JSON）发现的 feature 必须都出现在解析器 warnings 中', () => {
    const failures: string[] = [];
    for (const file of listCorpus()) {
      const buf = readFileSync(file);
      const asset = parseGltf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
      const audit = auditGlb(file);
      for (const issue of audit.issues) {
        const expected = AUDIT_FEATURE_TO_WARNING[issue.feature];
        if (!expected) continue;
        if (!asset.warnings.some((w) => w.includes(expected))) {
          failures.push(`${audit.path}: 审计发现 ${issue.feature}，但 warnings 里没有「${expected}」`);
        }
      }
    }
    expect(failures).toEqual([]);
  });

  it('蒙皮 / 动画 / morph / 多 UV 的资产分别给出对应提示', () => {
    expect(load('heavy/Fox.glb').warnings).toEqual([
      'glTF 动画未播放（3 条）：只渲染静态场景。',
      'JOINTS_0/WEIGHTS_0 未实现：蒙皮网格按绑定姿势（A/T-pose）渲染，动画姿态不会生效。',
    ]);

    expect(load('heavy/BrainStem.glb').warnings).toEqual([
      'glTF 动画未播放（1 条）：只渲染静态场景。',
      'JOINTS_0/WEIGHTS_0 未实现：蒙皮网格按绑定姿势（A/T-pose）渲染，动画姿态不会生效。',
    ]);

    // BoxTextured 只用到 baseColorTexture —— 材质路径已实现，因此不再有任何 warning。
    expect(load('light/BoxTextured.glb').warnings).toEqual([]);

    const morph = load('feature/MorphPrimitivesTest.glb').warnings;
    expect(morph).toContain('morph target（变形目标）未实现：只渲染基础形状。');

    const multiUv = load('feature/MultiUVTest.glb').warnings;
    expect(multiUv).toContain('TEXCOORD_1 被忽略：只使用 TEXCOORD_0。');
  });

  it('非零 texture texCoord 给出一次结构化提示', () => {
    const warnings = load('feature/MultiUVTest.glb').warnings.filter((warning) => warning.startsWith('贴图 texCoord='));
    expect(warnings).toEqual(['贴图 texCoord=1 被忽略：只使用 TEXCOORD_0。']);
  });

  it('alphaMode 与材质扩展给出可读提示（同一扩展只报一次）', () => {
    const alpha = load('feature/AlphaBlendModeTest.glb').warnings;
    // MASK 已实现 → 只剩 BLEND 的提示。
    expect(alpha.filter((w) => w.startsWith('alphaMode='))).toEqual([
      'alphaMode=BLEND 未实现：被当作不透明渲染，会露出被遮挡的面。',
    ]);

    // SheenChair 同时用到 extensionsUsed 与 material.extensions 里的 KHR_materials_sheen
    // 以及必需的 KHR_texture_transform —— 每条只应出现一次。
    const chair = load('feature/SheenChair.glb').warnings;
    expect(new Set(chair).size).toBe(chair.length);
    expect(chair.filter((w) => w.includes('KHR_materials_sheen'))).toHaveLength(1);
    expect(chair.some((w) => w.startsWith('模型声明了必需扩展') && w.includes('KHR_texture_transform'))).toBe(true);
    expect(chair).toContain('glTF 扩展 KHR_texture_transform 未实现：UV 变换（offset/rotation/scale）被忽略，贴图坐标会错位。');
  });
});
