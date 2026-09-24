/**
 * Material / Texture 路径 —— glTF baseColorTexture 的最小闭环。
 *
 *   GLB image (编码字节)
 *        ↓  ImageDecoder（可注入：浏览器 / Node / 测试替身）
 *   DecodedImage (RGBA8)
 *        ↓  uploadImageTextures
 *   GPUTexture (sRGB) + GPUSampler
 *        ↓  createMaterialBindGroup
 *   MaterialStore.bindGroupFor(idx)
 *        ↓  RenderItem.bindGroup → group(2)
 *   fragment shader textureSample()
 *
 * 设计边界（刻意保持窄）：
 *   - 只处理 base color texture；其余 PBR 贴图仍走 warning。
 *   - 解码器可注入，因此 `parseGltf()` 保持环境无关（Node 无 createImageBitmap）。
 *   - 每个材质一个 bind group（binding 0=texture, 1=sampler, 2=material uniform），
 *     固定占用 group=2；无贴图的材质绑定 1×1 白色 fallback，着色器无需分支。
 *   - OPAQUE / MASK（discard）；BLEND 需要透明排序与混合状态，不在此路径。
 */

/// <reference types="@webgpu/types" />

import type { AssetImage, AssetMaterial, AssetSampler, GltfAsset } from './gltf';

// ─── Decoding ───────────────────────────────────────────────

/** 解码后的像素：RGBA8、straight alpha、行优先。color 数据按 sRGB 编码值存放。 */
export interface DecodedImage {
  width: number;
  height: number;
  /** 长度 = width * height * 4。 */
  data: Uint8Array;
}

/**
 * 图片解码器 —— 唯一与环境耦合的一环，由调用方注入。
 *
 * 浏览器用 `createBrowserImageDecoder()`；Node / 资源管线可以预解码或返回固定像素。
 * 抛错的实现会让对应贴图退回 1×1 白色（记录在 `MaterialStore.stats.skipped`），
 * 不会让整份资产导入失败。
 */
export interface ImageDecoder {
  decode(data: Uint8Array, mimeType: string): Promise<DecodedImage>;
}

/**
 * 浏览器适配器：编码字节 → Blob → createImageBitmap → 2D canvas → RGBA8。
 *
 * `colorSpaceConversion: 'none'` + `premultiplyAlpha: 'none'` 保留原始 sRGB 编码值与
 * straight alpha —— 颜色空间转换交给 GPU 的 sRGB 纹理格式，避免 CPU/GPU 两次转换。
 */
export function createBrowserImageDecoder(): ImageDecoder {
  return {
    async decode(data: Uint8Array, mimeType: string): Promise<DecodedImage> {
      const blob = new Blob([data as BlobPart], { type: mimeType || 'application/octet-stream' });
      const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
      const width = bitmap.width;
      const height = bitmap.height;
      try {
        const canvas: OffscreenCanvas | HTMLCanvasElement =
          typeof OffscreenCanvas !== 'undefined'
            ? new OffscreenCanvas(width, height)
            : Object.assign(document.createElement('canvas'), { width, height });
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | null;
        if (!ctx) throw new Error('2D context unavailable for image decode');
        ctx.drawImage(bitmap as unknown as CanvasImageSource, 0, 0);
        const imageData = ctx.getImageData(0, 0, width, height);
        return { width, height, data: new Uint8Array(imageData.data) };
      } finally {
        bitmap.close?.();
      }
    },
  };
}

// ─── GPU resources ──────────────────────────────────────────

/** base color texture 的格式：glTF base color 是 color data，必须走 sRGB 采样。 */
const COLOR_FORMAT: GPUTextureFormat = 'rgba8unorm-srgb';

/**
 * base color texture 的 usage。
 *
 * 必须在**调用时**读取 `GPUTextureUsage` —— 该枚举只存在于 WebGPU 环境。
 * 若在模块顶层求值，任何非 WebGPU 环境（Node SSR / 构建期资产管线 / node 环境的单测）
 * 只要 `import` 本包入口就会在**导入阶段**直接 ReferenceError，即使它只用到
 * `parseGltf` 这类纯函数；同时也违反 package.json 声明的 `sideEffects: false`
 * （该字段告诉 bundler 本模块可在任意环境安全求值）。
 */
function textureUsage(): GPUTextureUsageFlags {
  return GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
}

/**
 * 材质 bind group 布局（固定 group=2）：
 *   binding 0 → baseColorTexture
 *   binding 1 → baseColorSampler
 *   binding 2 → material uniform（baseColorFactor / alphaCutoff / alphaMode）
 */
export function createMaterialBindGroupLayout(device: GPUDevice, label = 'hpg:material'): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
}

/** alphaMode → shader 常量（与 FS_MATERIAL 的约定一致）。 */
export const ALPHA_MODE_CODE = { OPAQUE: 0, MASK: 1, BLEND: 2 } as const;

/** 材质 uniform：vec4 baseColorFactor + f32 alphaCutoff + u32 alphaMode + padding = 32B。 */
export const MATERIAL_UNIFORM_BYTES = 32;

function createMaterialUniformBuffer(
  device: GPUDevice,
  material: AssetMaterial,
  label: string,
): GPUBuffer {
  const data = new Float32Array(MATERIAL_UNIFORM_BYTES / 4);
  const [r, g, b, a] = material.baseColorFactor;
  data[0] = r;
  data[1] = g;
  data[2] = b;
  data[3] = a;
  data[4] = material.alphaCutoff;
  new Uint32Array(data.buffer)[5] = ALPHA_MODE_CODE[material.alphaMode] ?? 0;
  const buffer = device.createBuffer({
    label,
    size: MATERIAL_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function samplerFromAsset(device: GPUDevice, s: AssetSampler, label: string): GPUSampler {
  return device.createSampler({
    label,
    magFilter: s.magFilter,
    minFilter: s.minFilter,
    mipmapFilter: s.mipmapFilter,
    addressModeU: s.addressModeU,
    addressModeV: s.addressModeV,
  });
}

/** 1×1 白色纹理 —— 让「无贴图材质」与「有贴图材质」共用同一条采样路径。 */
function createWhiteTexture(device: GPUDevice, label: string): GPUTexture {
  const texture = device.createTexture({
    label: `${label}:white`,
    size: { width: 1, height: 1, depthOrArrayLayers: 1 },
    format: COLOR_FORMAT,
    usage: textureUsage(),
  });
  device.queue.writeTexture(
    { texture },
    new Uint8Array([255, 255, 255, 255]),
    { bytesPerRow: 4, rowsPerImage: 1 },
    { width: 1, height: 1, depthOrArrayLayers: 1 },
  );
  return texture;
}

function uploadImage(device: GPUDevice, image: DecodedImage, label: string): GPUTexture {
  const texture = device.createTexture({
    label,
    size: { width: image.width, height: image.height, depthOrArrayLayers: 1 },
    format: COLOR_FORMAT,
    usage: textureUsage(),
  });
  device.queue.writeTexture(
    { texture },
    image.data,
    { bytesPerRow: image.width * 4, rowsPerImage: image.height },
    { width: image.width, height: image.height, depthOrArrayLayers: 1 },
  );
  return texture;
}

// ─── MaterialStore ──────────────────────────────────────────

export interface MaterialStoreStats {
  /** 生成的材质 bind group 数量。 */
  materials: number;
  /** 实际上传的贴图数量（不含 fallback）。 */
  textures: number;
  /** 跳过的贴图（无内嵌图片 / 解码失败）的人类可读原因。 */
  skipped: string[];
}

export interface MaterialStoreOptions {
  /** base color 贴图是否按 sRGB 解释。默认 true（glTF base color 是 color data）。 */
  srgb?: boolean;
  label?: string;
}

/**
 * 把 `GltfAsset` 的材质解析成 GPU 资源（贴图 + 采样器 + uniform + bind group）。
 *
 * 使用：
 *   const store = await MaterialStore.create(device, asset, createBrowserImageDecoder());
 *   renderer.registerPipeline({ bindGroupLayouts: [globalLayout, store.layout], ... });
 *   const items = sceneToRenderItems(scene, pipeline, globals, store);
 */
export class MaterialStore {
  /** 与管线 bindGroupLayouts[1]（→ group=2）配对的布局。 */
  readonly layout: GPUBindGroupLayout;
  readonly stats: MaterialStoreStats = { materials: 0, textures: 0, skipped: [] };

  private _bindGroups: (GPUBindGroup | undefined)[] = [];
  private _textures: GPUTexture[] = [];
  private _uniforms: GPUBuffer[] = [];
  private _white: GPUTexture;
  private _defaultSampler: GPUSampler;
  private _disposed = false;

  private constructor(device: GPUDevice, label: string) {
    this.layout = createMaterialBindGroupLayout(device, `${label}:material-layout`);
    this._white = createWhiteTexture(device, label);
    this._defaultSampler = device.createSampler({
      label: `${label}:default-sampler`,
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
  }

  /** 解码并上传全部材质贴图，返回可用作 `sceneToRenderItems` 第 4 个参数的 store。 */
  static async create(
    device: GPUDevice,
    asset: GltfAsset,
    decoder: ImageDecoder,
    opts: MaterialStoreOptions = {},
  ): Promise<MaterialStore> {
    const label = opts.label ?? 'hpg';
    const store = new MaterialStore(device, label);
    // 同一 image 可能被多个材质共用（texture atlas / 复用贴图）→ 只解码上传一次。
    const decodedCache = new Map<number, DecodedImage | null>();

    for (let i = 0; i < asset.materials.length; i++) {
      const material = asset.materials[i] as AssetMaterial;
      let texture = store._white;
      let sampler = store._defaultSampler;

      const ref = material.baseColorTexture;
      if (ref) {
        const image: AssetImage | undefined = asset.images[ref.imageIndex];
        if (!image) {
          store.stats.skipped.push(`material ${i} (${material.name}): image ${ref.imageIndex} 未内嵌`);
        } else {
          let decoded = decodedCache.get(ref.imageIndex);
          if (decoded === undefined) {
            try {
              decoded = await decoder.decode(image.data, image.mimeType);
            } catch (e) {
              store.stats.skipped.push(`material ${i} (${material.name}): 解码失败 — ${(e as Error).message}`);
              decoded = null;
            }
            decodedCache.set(ref.imageIndex, decoded);
          }
          if (decoded) {
            texture = uploadImage(device, decoded, `${label}:tex:${ref.imageIndex}`);
            sampler = samplerFromAsset(device, ref.sampler, `${label}:sampler:${ref.imageIndex}`);
            store._textures.push(texture);
          }
        }
      }

      const uniform = createMaterialUniformBuffer(device, material, `${label}:mat:${i}`);
      store._uniforms.push(uniform);
      store._bindGroups[i] = device.createBindGroup({
        label: `${label}:material-bg:${i}`,
        layout: store.layout,
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: uniform } },
        ],
      });
      store.stats.materials++;
      if (texture !== store._white) store.stats.textures++;
    }

    return store;
  }

  /** 材质是否存在对应的 bind group（越界 / 无材质时 false）。 */
  has(materialIndex: number): boolean {
    return this._bindGroups[materialIndex] !== undefined;
  }

  /** 取某材质的 bind group（供 `RenderItem.bindGroup`，固定绑定到 group=2）。 */
  bindGroupFor(materialIndex: number): GPUBindGroup | undefined {
    return this._bindGroups[materialIndex];
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    for (const t of this._textures) t.destroy();
    this._textures.length = 0;
    for (const u of this._uniforms) u.destroy();
    this._uniforms.length = 0;
    this._white.destroy();
    this._bindGroups = [];
  }
}
