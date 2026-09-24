/**
 * GLB Viewer — Real model validation.
 *
 * Features:
 *   1. Load GLB files (built-in test or user file)
 *   2. parseGltf → importGltfAsset → sceneToRenderItems
 *   3. Orbit camera
 *   4. submit() / submitCulled() toggle
 *   5. Detailed asset stats logging
 */

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { parseGltf } from '../src/core/gltf';
import { importGltfAsset, sceneToRenderItems } from '../src/core/asset-importer';
import { createBrowserImageDecoder, createMaterialBindGroupLayout, MaterialStore } from '../src/core/texture';
import { MATERIAL_LAYOUT_TEMPLATE } from '../src/shaders/instance';
import { identity, multiply, perspective, lookAt } from '../src/core/math';
import type { GlobalBinding, RenderItem, ResolvedPipeline } from '../src/types';

// ─── Shaders ────────────────────────────────────────────────

const VS_GLB = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

struct Uniforms {
    viewProj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
) -> VertexOutput {
    let inst = instances[instanceIdx];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    let worldNorm = (inst.modelMatrix * vec4<f32>(normal, 0.0)).xyz;
    var out: VertexOutput;
    out.clip = uniforms.viewProj * worldPos;
    out.color = inst.color;
    out.worldNormal = normalize(worldNorm);
    out.uv = uv;
    return out;
}
`;

/**
 * GPU Culled 路径：group(1) 额外绑定 compactedIndices，
 * 通过 compactedIndices[instance_index] 间接索引实例数据（只绘制可见实例）。
 *
 * 两个绑定都带 dynamic offset：instances 指向本 geometry 的候选实例区，
 * compactedIndices 指向本 geometry 的 slot 区，因此这里的下标都是“组内”下标。
 * 必须配合 registerPipeline({ compaction: true }) 使用，否则 group(1) 布局不匹配。
 */
const VS_GLB_CULLED = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(1) var<storage, read> compactedIndices: array<u32>;

struct Uniforms {
    viewProj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
) -> VertexOutput {
    let mapped = compactedIndices[instanceIdx];
    let inst = instances[mapped];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    let worldNorm = (inst.modelMatrix * vec4<f32>(normal, 0.0)).xyz;
    var out: VertexOutput;
    out.clip = uniforms.viewProj * worldPos;
    out.color = inst.color;
    out.worldNormal = normalize(worldNorm);
    out.uv = uv;
    return out;
}
`;

// group(2) 材质绑定（baseColorTexture / sampler / material uniform）—— 与 MaterialStore 对应。
const FS_GLB = /* wgsl */ `
${MATERIAL_LAYOUT_TEMPLATE}

@fragment
fn fs_main(
    @location(0) color: vec4<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(worldNormal);
    let diff = max(dot(n, lightDir), 0.15);
    let tex = textureSample(baseColorTexture, baseColorSampler, uv);
    let base = color * material.baseColorFactor * tex;
    if (material.alphaMode == 1u) {
        if (base.a < material.alphaCutoff) {
            discard;
        }
    }
    return vec4<f32>(base.rgb * diff, base.a);
}
`;

// ─── Test GLB Generator ─────────────────────────────────────

function generateTestGlb(): ArrayBuffer {
  // 3 cubes at different positions with different colors
  // Each cube: 8 vertices, 12 triangles (36 indices)

  interface CubeDef {
    name: string;
    translation: [number, number, number];
    color: [number, number, number, number];
  }

  const cubes: CubeDef[] = [
    { name: 'RedCube', translation: [-1.5, 0, 0], color: [0.9, 0.2, 0.2, 1] },
    { name: 'GreenCube', translation: [0, 0, 0], color: [0.2, 0.9, 0.2, 1] },
    { name: 'BlueCube', translation: [1.5, 0, 0], color: [0.2, 0.2, 0.9, 1] },
  ];

  const boxVerts = new Float32Array([
    // positions (x,y,z) + normals (nx,ny,nz)
    -0.5, -0.5, -0.5,  0, 0, -1,
     0.5, -0.5, -0.5,  0, 0, -1,
     0.5,  0.5, -0.5,  0, 0, -1,
    -0.5,  0.5, -0.5,  0, 0, -1,
    -0.5, -0.5,  0.5,  0, 0,  1,
     0.5, -0.5,  0.5,  0, 0,  1,
     0.5,  0.5,  0.5,  0, 0,  1,
    -0.5,  0.5,  0.5,  0, 0,  1,
    -0.5, -0.5, -0.5, -1, 0,  0,
    -0.5, -0.5,  0.5, -1, 0,  0,
    -0.5,  0.5,  0.5, -1, 0,  0,
    -0.5,  0.5, -0.5, -1, 0,  0,
     0.5, -0.5, -0.5,  1, 0,  0,
     0.5, -0.5,  0.5,  1, 0,  0,
     0.5,  0.5,  0.5,  1, 0,  0,
     0.5,  0.5, -0.5,  1, 0,  0,
    -0.5, -0.5, -0.5,  0,-1,  0,
     0.5, -0.5, -0.5,  0,-1,  0,
     0.5, -0.5,  0.5,  0,-1,  0,
    -0.5, -0.5,  0.5,  0,-1,  0,
    -0.5,  0.5, -0.5,  0, 1,  0,
     0.5,  0.5, -0.5,  0, 1,  0,
     0.5,  0.5,  0.5,  0, 1,  0,
    -0.5,  0.5,  0.5,  0, 1,  0,
  ]);
  const boxIndices = new Uint16Array([
    0,1,2, 2,3,0,  4,6,5, 6,4,7,
    8,10,9, 10,8,11, 12,13,14, 14,15,12,
    16,17,18, 18,19,16, 20,22,21, 22,20,23,
  ]);

  // Build per-cube mesh data
  const meshDataList = cubes.map((cube) => {
    // Interleave position + normal (skip UV/tangent for simplicity)
    const verts = new Float32Array(24 * 6); // 24 verts × 6 floats
    for (let v = 0; v < 24; v++) {
      verts[v * 6 + 0] = boxVerts[v * 6 + 0]; // px
      verts[v * 6 + 1] = boxVerts[v * 6 + 1]; // py
      verts[v * 6 + 2] = boxVerts[v * 6 + 2]; // pz
      verts[v * 6 + 3] = boxVerts[v * 6 + 3]; // nx
      verts[v * 6 + 4] = boxVerts[v * 6 + 4]; // ny
      verts[v * 6 + 5] = boxVerts[v * 6 + 5]; // nz
    }
    return { verts, indices: boxIndices, cube };
  });

  // Calculate total binary size
  let totalBinSize = 0;
  const meshBinOffsets: number[] = [];
  const idxBinOffsets: number[] = [];
  for (const md of meshDataList) {
    meshBinOffsets.push(totalBinSize);
    totalBinSize += md.verts.byteLength;
    // Pad to 4-byte
    while (totalBinSize % 4 !== 0) totalBinSize++;
    idxBinOffsets.push(totalBinSize);
    totalBinSize += md.indices.byteLength;
    while (totalBinSize % 4 !== 0) totalBinSize++;
  }

  const binData = new Uint8Array(totalBinSize);
  const meshAccessorIdxs: number[] = [];
  const idxAccessorIdxs: number[] = [];
  const meshBufferViewIdxs: number[] = [];
  const idxBufferViewIdxs: number[] = [];

  let accessorIdx = 0;
  let bufferViewIdx = 0;

  for (let i = 0; i < meshDataList.length; i++) {
    const md = meshDataList[i]!;

    // Write vertices
    binData.set(new Uint8Array(md.verts.buffer), meshBinOffsets[i]!);
    meshBufferViewIdxs.push(bufferViewIdx);
    meshAccessorIdxs.push(accessorIdx);
    bufferViewIdx++;
    accessorIdx++;

    // Write indices
    binData.set(new Uint8Array(md.indices.buffer), idxBinOffsets[i]!);
    idxBufferViewIdxs.push(bufferViewIdx);
    idxAccessorIdxs.push(accessorIdx);
    bufferViewIdx++;
    accessorIdx++;
  }

  // Build glTF JSON
  const nodes: any[] = cubes.map((cube, i) => ({
    name: cube.name,
    mesh: i,
    translation: cube.translation,
  }));

  const meshes = meshDataList.map((md, i) => ({
    name: md.cube.name + 'Mesh',
    primitives: [{
      attributes: {
        POSITION: meshAccessorIdxs[i],
        NORMAL: idxAccessorIdxs[i] - 1 + 1, // normals accessor = meshAccessor + 1
      },
      indices: idxAccessorIdxs[i],
      material: 0,
    }],
  }));

  // Wait — I need to fix this. For each cube, I need:
  // - 1 accessor for POSITION (vec3, float)
  // - 1 accessor for NORMAL (vec3, float)
  // - 1 accessor for INDICES (scalar, uint16)
  // But I only wrote 2 bufferViews per cube (vertices + indices).
  // The vertices are interleaved (pos + normal), so I need 3 accessors per cube.

  // Let me redo this properly.
  // Actually, for interleaved data, I can use the same bufferView for both POSITION and NORMAL
  // with different byteOffset and different accessor types.

  // Let me just simplify: 3 cubes, each with its own node/mesh/primitive.
  // Each primitive has interleaved [pos3, norm3] vertices.

  // Rebuild with correct accessor structure
  const accessors: any[] = [];
  const bufferViews: any[] = [];
  let accIdx = 0;
  let bvIdx = 0;

  const primAccessors: { posAcc: number; normAcc: number; idxAcc: number }[] = [];

  for (let i = 0; i < meshDataList.length; i++) {
    const md = meshDataList[i]!;
    const vertsByteLength = md.verts.byteLength;
    const indicesByteLength = md.indices.byteLength;

    // BufferView for interleaved vertices
    bufferViews.push({
      buffer: 0,
      byteOffset: meshBinOffsets[i],
      byteLength: vertsByteLength,
      byteStride: 24, // 6 floats × 4 bytes
      target: 34962, // ARRAY_BUFFER
    });
    const posBvIdx = bvIdx;
    bvIdx++;

    // BufferView for indices
    bufferViews.push({
      buffer: 0,
      byteOffset: idxBinOffsets[i],
      byteLength: indicesByteLength,
      target: 34963, // ELEMENT_ARRAY_BUFFER
    });
    const idxBvIdx = bvIdx;
    bvIdx++;

    // Accessor for POSITION
    accessors.push({
      bufferView: posBvIdx,
      byteOffset: 0,
      componentType: 5126, // FLOAT
      count: 24,
      type: 'VEC3',
    });
    const posAcc = accIdx;
    accIdx++;

    // Accessor for NORMAL
    accessors.push({
      bufferView: posBvIdx,
      byteOffset: 12, // after 3 floats of position
      componentType: 5126, // FLOAT
      count: 24,
      type: 'VEC3',
    });
    const normAcc = accIdx;
    accIdx++;

    // Accessor for INDICES
    accessors.push({
      bufferView: idxBvIdx,
      componentType: 5123, // UNSIGNED_SHORT
      count: 36,
      type: 'SCALAR',
    });
    const idxAcc = accIdx;
    accIdx++;

    primAccessors.push({ posAcc, normAcc, idxAcc });
  }

  const gltfJson = {
    asset: { version: '2.0', generator: 'hpg-glb-viewer' },
    scene: 0,
    scenes: [{ name: 'Scene', nodes: cubes.map((_, i) => i) }],
    nodes,
    meshes: meshDataList.map((md, i) => ({
      name: md.cube.name + 'Mesh',
      primitives: [{
        attributes: {
          POSITION: primAccessors[i]!.posAcc,
          NORMAL: primAccessors[i]!.normAcc,
        },
        indices: primAccessors[i]!.idxAcc,
        material: 0,
      }],
    })),
    materials: [{
      name: 'DefaultMaterial',
      pbrMetallicRoughness: {
        baseColorFactor: [1, 1, 1, 1] as [number, number, number, number],
        metallicFactor: 0.0,
        roughnessFactor: 0.8,
      },
    }],
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinSize }],
  };

  // Build GLB
  const jsonStr = JSON.stringify(gltfJson);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const jsonPadding = (4 - (jsonBytes.byteLength % 4)) % 4;
  const jsonChunkLength = jsonBytes.byteLength + jsonPadding;
  const binPadding = (4 - (totalBinSize % 4)) % 4;
  const binChunkLength = totalBinSize + binPadding;
  const totalLength = 12 + 8 + jsonChunkLength + 8 + binChunkLength;

  const glb = new ArrayBuffer(totalLength);
  const view = new DataView(glb);

  // Header
  view.setUint32(0, 0x46546C67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, totalLength, true);

  // JSON chunk
  let offset = 12;
  view.setUint32(offset, jsonChunkLength, true);
  view.setUint32(offset + 4, 0x4E4F534A, true);
  const jsonDst = new Uint8Array(glb, offset + 8, jsonChunkLength);
  jsonDst.set(jsonBytes);
  for (let i = jsonBytes.byteLength; i < jsonChunkLength; i++) jsonDst[i] = 0x20;
  offset += 8 + jsonChunkLength;

  // BIN chunk
  view.setUint32(offset, binChunkLength, true);
  view.setUint32(offset + 4, 0x004E4942, true);
  const binDst = new Uint8Array(glb, offset + 8, binChunkLength);
  binDst.set(binData);

  return glb;
}

// ─── Camera ─────────────────────────────────────────────────

class OrbitCamera {
  theta = 0.5;
  phi = 0.8;
  distance = 6;
  target: [number, number, number] = [0, 0, 0];
  /** 近/远平面随模型尺寸自适应（写死会被大模型整个裁掉）。 */
  near = 0.1;
  far = 100;

  getEye(): [number, number, number] {
    return [
      this.target[0] + this.distance * Math.sin(this.phi) * Math.cos(this.theta),
      this.target[1] + this.distance * Math.cos(this.phi),
      this.target[2] + this.distance * Math.sin(this.phi) * Math.sin(this.theta),
    ];
  }

  getViewMatrix(aspect: number): Float32Array {
    const eye = this.getEye();
    const eyeArr = new Float32Array(eye);
    const targetArr = new Float32Array(this.target);
    const upArr = new Float32Array([0, 1, 0]);
    const view = lookAt(eyeArr, targetArr, upArr);
    const proj = perspective(Math.PI / 4, aspect, this.near, this.far);
    const out = new Float32Array(16);
    return multiply(out, proj, view);
  }
}

// ─── Main ───────────────────────────────────────────────────

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const statsDiv = document.getElementById('stats')!;
  const btnDirect = document.getElementById('btn-direct')!;
  const btnCulled = document.getElementById('btn-culled')!;
  const btnFile = document.getElementById('btn-file')!;
  const fileInput = document.getElementById('file-input') as HTMLInputElement;

  // WebGPU init
  if (!navigator.gpu) {
    statsDiv.textContent = 'WebGPU not supported.';
    return;
  }

  // ── GPU 错误可见化 ────────────────────────────────────────
  // WebGPU 的校验错误默认只在控制台报警，页面看上去就是「什么都没画出来」。
  // 这里把未捕获错误 + 逐帧 validation 错误直接显示到 HUD 上。
  const errorsDiv = document.getElementById('errors')!;
  const noticesDiv = document.getElementById('notices')!;

  // ── 自动化验证钩子 ────────────────────────────────────────
  // `?harness=1`：渲染稳定后把「材质/贴图统计 + 校验错误 + 像素签名」写入 window.__hpgResult，
  // 供 benchmark/browser-material-check.mjs 通过 CDP 读取（真实 Chrome + WebGPU）。
  // `?mode=culled`：初始走 GPU Culled 路径（用于 Direct/Culled 一致性对比）。
  const params = new URLSearchParams(location.search);
  const harnessOn = params.get('harness') === '1';
  const harnessErrors: string[] = [];
  // 注意：必须在 loadGlbData() 调用之前声明 —— 它们在函数体顶部处于 TDZ，
  // 而 loadGlbData() 会在后面的语句执行之前就被 await 调用。
  let harnessTick = 0;
  let harnessDone = false;
  let harnessInfo: Record<string, unknown> | null = null;

  /**
   * 展示「模型用到但 hpg 未实现」的 feature（parseGltf 的结构化 warnings）。
   *
   * 以前这些信息只进 console，页面上表现为「模型加载成功、渲染却跟作者意图不符」而毫无提示——
   * 真实 Blender / Khronos 资产里贴图、alphaMode、蒙皮都属于这一类。
   */
  function showNotices(warnings: readonly string[]): void {
    if (warnings.length === 0) {
      noticesDiv.style.display = 'none';
      return;
    }
    noticesDiv.style.display = 'block';
    noticesDiv.textContent =
      `未实现的 glTF feature（${warnings.length}）：\n` +
      warnings.slice(0, 8).map((w) => `• ${w}`).join('\n') +
      (warnings.length > 8 ? `\n(+${warnings.length - 8} more — see console)` : '');
  }
  const seenErrors = new Set<string>();
  function showError(message: string, kind = 'error') {
    if (seenErrors.has(message)) return;
    seenErrors.add(message);
    console.error(`[hpg] GPU ${kind}: ${message}`);
    errorsDiv.style.display = 'block';
    errorsDiv.textContent =
      `GPU ${kind}: ${message}` + (seenErrors.size > 1 ? `\n(+${seenErrors.size - 1} more — see console)` : '');
    harnessErrors.push(`${kind}: ${message}`);
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    statsDiv.textContent = 'No GPU adapter.';
    return;
  }
  const device = await adapter.requestDevice();
  device.onuncapturederror = (e: GPUUncapturedErrorEvent) => {
    showError(e.error.message, 'uncaptured');
  };
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // Renderer
  const renderer = Renderer.create({ device, context, format });

  // Pipeline
  const layout = uniformBindGroupLayout(device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const vertexLayouts = [
    {
      arrayStride: 48, // canonical: pos(12) + norm(12) + uv(8) + tangent(16)
      stepMode: 'vertex' as const,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
        { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },
        { shaderLocation: 2, offset: 24, format: 'float32x2' as GPUVertexFormat },
        { shaderLocation: 3, offset: 32, format: 'float32x4' as GPUVertexFormat },
      ],
    },
  ];

  // 材质布局（group=2）。与 MaterialStore 内部创建的布局结构一致 → 互相兼容。
  const materialLayout = createMaterialBindGroupLayout(device, 'glb-viewer:material');

  const commonPipelineDesc = {
    vertexLayouts,
    bindGroupLayouts: [layout, materialLayout],
    globalBindings: [{ binding: 0, buffer: uniformBuffer }],
    depth: { format: 'depth24plus' as GPUTextureFormat, depthWriteEnabled: true, depthCompare: 'less' as GPUCompareFunction },
    targets: [{ format }],
  };

  // Direct 路径（submit）：group(1) = 单 instance buffer 绑定。
  const pipeline = renderer.registerPipeline({
    label: 'glb-pipeline',
    vsCode: VS_GLB,
    fsCode: FS_GLB,
    ...commonPipelineDesc,
  });

  // GPU Culled 路径（submitCulled）：group(1) = [instanceBuffer, compactedIndices]。
  // 两种模式的 group(1) 布局不同，必须分开注册（compaction 标记决定）。
  const pipelineCulled = renderer.registerPipeline({
    label: 'glb-pipeline-culled',
    vsCode: VS_GLB_CULLED,
    fsCode: FS_GLB,
    compaction: true,
    ...commonPipelineDesc,
  });

  // baseColorTexture 解码（浏览器适配器）。
  const imageDecoder = createBrowserImageDecoder();

  // Load GLB
  let useCulled = params.get('mode') === 'culled';
  let materialStore: MaterialStore | null = null;
  let renderItems: RenderItem[] = [];
  /** 与 renderItems 相同的几何/实例，但使用 compaction 管线（submitCulled 专用）。 */
  let culledItems: RenderItem[] = [];
  let meshCount = 0;
  let vertexCount = 0;
  let indexCount = 0;

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  async function loadGlbData(data: ArrayBuffer, fileName?: string) {
    const t0 = performance.now();

    // Parse
    const tParse0 = performance.now();
    const asset = parseGltf(data);
    const tParse1 = performance.now();

    // 未实现 feature 的可见提示（贴图 / alphaMode / 蒙皮 / 扩展 …）
    showNotices(asset.warnings);

    // Import (GPU upload)
    const tImport0 = performance.now();
    const scene = importGltfAsset(asset, renderer, { scale: 1, flipV: true });
    const tImport1 = performance.now();

    // 材质贴图（baseColorTexture → GPUTexture + sampler + group=2 bind group）
    materialStore?.dispose();
    materialStore = await MaterialStore.create(device, asset, imageDecoder, { label: 'glb-viewer' });

    // Scene → RenderItems（Direct 与 Culled 各一份，几何体共享；附上材质 bind group）
    renderItems = sceneToRenderItems(scene, pipeline, materialStore);
    culledItems = sceneToRenderItems(scene, pipelineCulled, materialStore);
    const t1 = performance.now();

    // 贴图统计（未解码 / 失败时给出可读原因）
    if (materialStore.stats.textures > 0 || materialStore.stats.skipped.length > 0) {
      console.log(
        `Materials: ${materialStore.stats.materials}, textures uploaded: ${materialStore.stats.textures}`,
        materialStore.stats.skipped.length > 0 ? materialStore.stats.skipped : '',
      );
    }

    // Auto-frame camera to fit model
    frameCameraToBounds(scene.bounds);

    console.log('Loaded:', fileName, 'renderItems:', renderItems.length, 'bounds:', scene.bounds);

    // Count stats
    meshCount = scene.meshes.length;
    vertexCount = 0;
    indexCount = 0;
    for (const m of scene.meshes) {
      vertexCount += m.geometry.vertexCount;
      indexCount += m.geometry.indexCount;
    }

    // Count materials used
    const materialSet = new Set<number>();
    for (const ri of renderItems) {
      materialSet.add(ri.pipeline ? 1 : 0);
    }

    // Console logging for benchmark
    const parseMs = (tParse1 - tParse0).toFixed(1);
    const importMs = (tImport1 - tImport0).toFixed(1);
    const totalMs = (t1 - t0).toFixed(1);

    console.group(`%c GLB Loaded: ${fileName || 'test.glb'}`, 'color: #4fc3f7; font-weight: bold');
    console.log(`File size: ${formatBytes(data.byteLength)}`);
    console.log(`Meshes: ${meshCount}`);
    console.log(`Vertices: ${vertexCount.toLocaleString()}`);
    console.log(`Indices: ${indexCount.toLocaleString()}`);
    console.log(`Materials: ${asset.materials.length}`);
    console.log(`RenderItems: ${renderItems.length}`);
    console.log(`Parse: ${parseMs} ms`);
    console.log(`Import: ${importMs} ms`);
    console.log(`Total: ${totalMs} ms`);
    console.groupEnd();

    // Log asset structure details
    console.group('Asset structure');
    console.log(`Nodes: ${asset.nodes.length}`);
    console.log(`Meshes: ${asset.meshes.length}`);
    console.log(`Materials: ${asset.materials.length}`);

    // Log primitive attribute summary
    const attrSet = new Set<string>();
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        for (const attr of prim.vertexLayout.attributes) {
          attrSet.add(`loc${attr.shaderLocation}`);
        }
      }
    }
    console.log(`Vertex attributes: ${[...attrSet].join(', ')}`);

    // Log hierarchy
    const hasHierarchy = asset.nodes.some(n => n.children && n.children.length > 0);
    console.log(`Hierarchy: ${hasHierarchy ? 'yes' : 'flat'}`);
    if (hasHierarchy) {
      const maxDepth = computeMaxDepth(asset, 0);
      console.log(`Max depth: ${maxDepth}`);
    }
    console.groupEnd();

    // Update UI
    statsDiv.innerHTML = [
      `<div class="row"><span class="label">File:</span><span>${fileName || 'test.glb'}</span></div>`,
      `<div class="row"><span class="label">Size:</span><span>${formatBytes(data.byteLength)}</span></div>`,
      `<div class="row"><span class="label">Parse:</span><span>${parseMs} ms</span></div>`,
      `<div class="row"><span class="label">Import:</span><span>${importMs} ms</span></div>`,
      `<div class="row"><span class="label">Total:</span><span>${totalMs} ms</span></div>`,
      '<hr style="border-color:#333; margin:6px 0">',
      `<div class="row"><span class="label">Meshes:</span><span>${meshCount}</span></div>`,
      `<div class="row"><span class="label">Vertices:</span><span>${vertexCount.toLocaleString()}</span></div>`,
      `<div class="row"><span class="label">Indices:</span><span>${indexCount.toLocaleString()}</span></div>`,
      `<div class="row"><span class="label">Materials:</span><span>${materialSet.size}</span></div>`,
      `<div class="row"><span class="label">RenderItems:</span><span>${renderItems.length}</span></div>`,
      (asset.warnings.length > 0
        ? `<div class="row"><span class="label">Unsupported:</span><span style="color:#d9a521">${asset.warnings.length} feature(s)</span></div>`
        : ''),
      '<hr style="border-color:#333; margin:6px 0">',
      `<div class="row"><span class="label">Mode:</span><span id="mode-label">${useCulled ? 'GPU Culled' : 'Direct'}</span></div>`,
      '<div class="row"><span class="label">CPU submit:</span><span id="cpu-ms">—</span></div>',
    ].join('');

    // Harness 统计（真实 Chrome 验证用）。
    harnessInfo = {
      asset: fileName ?? '',
      meshes: meshCount,
      items: renderItems.length,
      materials: asset.materials.length,
      warnings: asset.warnings.length,
      textures: materialStore?.stats.textures ?? 0,
      skipped: materialStore?.stats.skipped ?? [],
      itemsWithMaterial: renderItems.filter((i) => i.bindGroup).length,
    };
  }

  function computeMaxDepth(asset: { nodes: { children: number[] }[] }, nodeIdx: number): number {
    const node = asset.nodes[nodeIdx];
    if (!node || !node.children || node.children.length === 0) return 0;
    let max = 0;
    for (const ci of node.children) {
      const d = computeMaxDepth(asset, ci) + 1;
      if (d > max) max = d;
    }
    return max;
  }

  // Camera
  const camera = new OrbitCamera();
  let dragging = false;
  let lastX = 0, lastY = 0;

  /**
   * 按世界空间包围盒取景。
   *
   * bounds 是 importGltfAsset 返回的**世界空间** AABB（已含节点变换与缩放），
   * 因此 target 是真中心。near/far 必须随模型尺寸缩放：写死的 0.1/100 会把
   * 尺度大的模型（Duck≈317、Fox≈219 单位）整个裁到远平面之外 —— 屏幕全空。
   */
  function frameCameraToBounds(bounds: [number, number, number, number, number, number]) {
    const cx = (bounds[0] + bounds[3]) / 2;
    const cy = (bounds[1] + bounds[4]) / 2;
    const cz = (bounds[2] + bounds[5]) / 2;
    const dx = bounds[3] - bounds[0];
    const dy = bounds[4] - bounds[1];
    const dz = bounds[5] - bounds[2];
    // 用外接球半径（而非对角线/2）避免退化模型得到 0 尺寸。
    const radius = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) / 2, 1e-3);
    camera.target = [cx, cy, cz];
    camera.distance = radius * 2.5;
    camera.near = Math.max(radius * 0.01, 1e-3);
    camera.far = Math.max(radius * 20, 1);
  }

  // 默认加载内置测试模型；`?asset=<url>` 可直接加载指定 GLB（便于复现与自动化验证）。
  const assetUrl = new URLSearchParams(location.search).get('asset');
  if (assetUrl) {
    try {
      const res = await fetch(assetUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadGlbData(await res.arrayBuffer(), assetUrl.split('/').pop());
    } catch (e) {
      showError(`加载 ${assetUrl} 失败：${(e as Error).message}`, 'load');
      await loadGlbData(generateTestGlb());
    }
  } else {
    await loadGlbData(generateTestGlb());
  }

  canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    camera.theta -= (e.clientX - lastX) * 0.01;
    camera.phi -= (e.clientY - lastY) * 0.01;
    camera.phi = Math.max(0.1, Math.min(Math.PI - 0.1, camera.phi));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });
  canvas.addEventListener('wheel', (e) => {
    camera.distance *= 1 + e.deltaY * 0.001;
    // 限制在近/远平面之间，避免缩放到看不见任何东西。
    const minD = camera.near * 2;
    const maxD = camera.far * 0.5;
    camera.distance = Math.max(minD, Math.min(maxD, camera.distance));
  });

  // 初始模式按钮状态（?mode=culled 时默认走剔除路径）。
  if (useCulled) {
    btnCulled.classList.add('active');
    btnDirect.classList.remove('active');
  }

  // Mode buttons
  btnDirect.addEventListener('click', () => {
    useCulled = false;
    btnDirect.classList.add('active');
    btnCulled.classList.remove('active');
    const label = document.getElementById('mode-label');
    if (label) label.textContent = 'Direct';
  });
  btnCulled.addEventListener('click', () => {
    useCulled = true;
    btnCulled.classList.add('active');
    btnDirect.classList.remove('active');
    const label = document.getElementById('mode-label');
    if (label) label.textContent = 'GPU Culled';
  });

  // File input
  btnFile.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    await loadGlbData(buf, file.name);
  });

  // ── Harness：渲染稳定后采集一次结果 ────────────────────────
  /** 把 WebGPU canvas 拷到 2D canvas 后统计亮度（真实像素验证）。 */
  function capturePixels(): Record<string, unknown> | null {
    const probe = document.createElement('canvas');
    probe.width = canvas.width;
    probe.height = canvas.height;
    const ctx = probe.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(canvas, 0, 0);
    const d = ctx.getImageData(0, 0, probe.width, probe.height).data;
    const total = d.length / 4;
    let lit = 0;
    let sum = 0;
    const G = 8;
    const grid = new Array<number>(G * G).fill(0);
    const cnt = new Array<number>(G * G).fill(0);
    for (let y = 0; y < probe.height; y++) {
      const gy = Math.min(G - 1, Math.floor((y * G) / probe.height));
      for (let x = 0; x < probe.width; x++) {
        const i = (y * probe.width + x) * 4;
        const l = 0.2126 * (d[i] as number) + 0.7152 * (d[i + 1] as number) + 0.0722 * (d[i + 2] as number);
        sum += l;
        if (l > 8) lit++;
        const gx = Math.min(G - 1, Math.floor((x * G) / probe.width));
        grid[gy * G + gx] += l;
        cnt[gy * G + gx]++;
      }
    }
    return {
      litPixels: lit,
      totalPixels: total,
      avgLuma: Number((sum / Math.max(1, total)).toFixed(2)),
      grid: grid.map((s, i) => Number((s / Math.max(1, cnt[i] as number)).toFixed(1))),
    };
  }

  // Render loop
  function frame() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    const aspect = w / h;
    const vpMatrix = camera.getViewMatrix(aspect);

    // Update uniform
    device.queue.writeBuffer(uniformBuffer, 0, vpMatrix);

    // Submit（用 validation error scope 抓取逐帧校验错误）
    device.pushErrorScope('validation');
    const t0 = performance.now();
    if (culledItems.length > 0) {
      if (useCulled) {
        // 注意：必须用 compaction 管线注册的 items。
        renderer.submitCulled(culledItems, vpMatrix);
      } else {
        // 传入相机位置 → 未指定 depth 的物体按距离近→远排序（early-z 友好）。
        renderer.submit(renderItems, { camera: camera.getEye(), depthRange: [camera.near, camera.far] });
      }
    }
    const t1 = performance.now();
    void device.popErrorScope().then((err) => {
      if (err) showError(err.message, 'validation');
    });

    const cpuMs = document.getElementById('cpu-ms');
    if (cpuMs) cpuMs.textContent = `${(t1 - t0).toFixed(2)} ms`;

    // Harness：等几帧让贴图/绑定稳定，再采集一次结果。
    if (harnessOn && harnessInfo && !harnessDone && ++harnessTick >= 4) {
      harnessDone = true;
      const result = {
        ...harnessInfo,
        mode: useCulled ? 'culled' : 'direct',
        validationErrors: harnessErrors,
        pixels: capturePixels(),
      };
      (window as unknown as Record<string, unknown>).__hpgResult = result;
      const pre = document.createElement('pre');
      pre.id = 'harness-result';
      pre.textContent = JSON.stringify(result);
      document.body.appendChild(pre);
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((e: any) => {
  console.error('Fatal error:', e);
  const statsDiv = document.getElementById('stats');
  if (statsDiv) {
    statsDiv.innerHTML = `<div style="color:#f44336">Error: ${e.message || e}</div>`;
  }
});
