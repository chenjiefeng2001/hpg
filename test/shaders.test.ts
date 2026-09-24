import { describe, it, expect } from 'vitest';
import {
  VS_INSTANCED,
  FS_COLOR,
  VS_FLAT,
  FS_DEPTH_ONLY,
  INSTANCE_LAYOUT_TEMPLATE,
  MATERIAL_LAYOUT_TEMPLATE,
  VS_INSTANCED_MATERIAL,
  VS_INSTANCED_MATERIAL_COMPACTION,
  FS_MATERIAL,
} from '../src/shaders/instance';

describe('shader variants', () => {
  it('INSTANCE_LAYOUT_TEMPLATE declares InstanceData struct', () => {
    expect(INSTANCE_LAYOUT_TEMPLATE).toContain('struct InstanceData');
    expect(INSTANCE_LAYOUT_TEMPLATE).toContain('modelMatrix');
    expect(INSTANCE_LAYOUT_TEMPLATE).toContain('color');
    expect(INSTANCE_LAYOUT_TEMPLATE).toContain('var<storage, read>');
  });

  it('VS_INSTANCED includes instance_index and position/normal inputs', () => {
    expect(VS_INSTANCED).toContain('@builtin(instance_index)');
    expect(VS_INSTANCED).toContain('@location(0) position');
    expect(VS_INSTANCED).toContain('@location(1) normal');
    expect(VS_INSTANCED).toContain('fn vs_main');
  });

  it('FS_COLOR returns input color', () => {
    expect(FS_COLOR).toContain('fn fs_main');
    expect(FS_COLOR).toContain('@location(0) color');
    expect(FS_COLOR).toContain('return color');
  });

  it('VS_FLAT uses uniform color instead of instance color', () => {
    expect(VS_FLAT).toContain('struct Uniforms');
    expect(VS_FLAT).toContain('var<uniform> uniforms');
    expect(VS_FLAT).toContain('fn vs_main');
    expect(VS_FLAT).not.toContain('instances[');
  });

  it('FS_DEPTH_ONLY outputs zero color', () => {
    expect(FS_DEPTH_ONLY).toContain('fn fs_main');
    expect(FS_DEPTH_ONLY).toContain('vec4<f32>(0.0)');
  });

  it('all shaders are non-empty strings', () => {
    expect(VS_INSTANCED.length).toBeGreaterThan(0);
    expect(FS_COLOR.length).toBeGreaterThan(0);
    expect(VS_FLAT.length).toBeGreaterThan(0);
    expect(FS_DEPTH_ONLY.length).toBeGreaterThan(0);
    expect(INSTANCE_LAYOUT_TEMPLATE.length).toBeGreaterThan(0);
  });

  it('MATERIAL_LAYOUT_TEMPLATE binds texture/sampler/uniform at group 2', () => {
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('@group(2) @binding(0)');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('texture_2d<f32>');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('@group(2) @binding(1)');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('sampler');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('@group(2) @binding(2)');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('MaterialUniforms');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('alphaMode');
    expect(MATERIAL_LAYOUT_TEMPLATE).toContain('alphaCutoff');
  });

  it('VS_INSTANCED_MATERIAL outputs uv; FS_MATERIAL samples + MASK discard', () => {
    expect(VS_INSTANCED_MATERIAL).toContain('@location(2) uv: vec2<f32>');
    expect(VS_INSTANCED_MATERIAL).toContain('out.uv = uv');
    expect(FS_MATERIAL).toContain('textureSample(baseColorTexture, baseColorSampler, uv)');
    expect(FS_MATERIAL).toContain('discard');
    expect(FS_MATERIAL).toContain('material.alphaMode == 1u');
  });

  it('compaction material vertex shader indexes via compactedIndices', () => {
    expect(VS_INSTANCED_MATERIAL_COMPACTION).toContain('compactedIndices[instanceIdx]');
    expect(VS_INSTANCED_MATERIAL_COMPACTION).toContain('@group(1) @binding(1)');
  });
});
