/**
 * hpg —— WebGPU Rendering Runtime
 * 抽象中心是「GPU 工作」，不是 3D 对象。RenderItem in → GPU Command out。
 */

// Public types
export * from './types';

// Public math utilities
export * from './core/math';

// Core API
export { Renderer, uniformBindGroupLayout } from './core/renderer';
export type { RendererOptions, RendererDescriptor } from './core/renderer';
export { GeometryArena } from './core/geometry';
export type { GeometryArenaStats } from './core/geometry';
export { TimestampQuery } from './core/timestamp';

// Shaders（submitCulled 路径必须用 VS_INSTANCED_COMPACTION + compaction: true 注册）
export {
  INSTANCE_LAYOUT_TEMPLATE,
  MATERIAL_LAYOUT_TEMPLATE,
  VS_INSTANCED,
  VS_INSTANCED_COMPACTION,
  VS_INSTANCED_MATERIAL,
  VS_INSTANCED_MATERIAL_COMPACTION,
  VS_FLAT,
  FS_COLOR,
  FS_MATERIAL,
  FS_DEPTH_ONLY,
} from './shaders/instance';
export { CS_FRUSTUM_CULL } from './shaders/culling';

// GPU 剔除
// 参数含义：
//   const pipeline = renderer.registerPipeline({ compaction: true, vsCode: VS_INSTANCED_COMPACTION, ... });
//   renderer.submitCulled(items, vpMatrix);
export { CullingPipeline, extractFrustumPlanes, sphereInFrustum } from './core/culling';
export type { CullingBuffers } from './core/culling';

// Asset Pipeline
export { parseGltf, flattenScene } from './core/gltf';
export type {
  GltfAsset,
  AssetMesh,
  AssetPrimitive,
  AssetMaterial,
  AssetNode,
  AssetImage,
  AssetSampler,
  AssetTextureRef,
} from './core/gltf';
export { importGltfAsset, sceneToRenderItems } from './core/asset-importer';
export type {
  ImportedScene,
  ImportedMesh,
  ImportedMaterial,
  ImportOptions,
  MaterialBindingSource,
} from './core/asset-importer';

// Material / Texture 路径（group=2）：baseColorTexture + sampler + injectable decoder
export {
  MaterialStore,
  createBrowserImageDecoder,
  createMaterialBindGroupLayout,
  ALPHA_MODE_CODE,
  MATERIAL_UNIFORM_BYTES,
} from './core/texture';
export type { DecodedImage, ImageDecoder, MaterialStoreOptions, MaterialStoreStats } from './core/texture';
