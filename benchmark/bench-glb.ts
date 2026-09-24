/**
 * Real-Workload Benchmark — 合成 GLB 模型，模拟 Blender 真实特征。
 *
 * 生成 3 种规模的模型:
 *   Small:  ~50 meshes, ~50K vertices
 *   Medium: ~200 meshes, ~500K vertices
 *   Large:  ~500 meshes, ~2M vertices
 *
 * 测量:
 *   - GLB parse time
 *   - Asset import time (GeometryArena upload)
 *   - CPU submit time (Direct / Batcher / GPU Culling)
 *   - GPU execution time (via TimestampQuery)
 */

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { parseGltf } from '../src/core/gltf';
import { importGltfAsset, sceneToRenderItems } from '../src/core/asset-importer';
import { TimestampQuery } from '../src/core/timestamp';
import { multiply, perspective, lookAt } from '../src/core/math';
import type { GlobalBinding, RenderItem, ResolvedPipeline } from '../src/types';

// ─── Shaders ────────────────────────────────────────────────

const VS_BENCH = /* wgsl */ `
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
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    let inst = instances[instanceIdx];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip = uniforms.viewProj * worldPos;
    out.color = inst.color;
    return out;
}
`;

/**
 * GPU Culled 路径专用顶点着色器：group(1) 多一个 compactedIndices 绑定，
 * 用 compactedIndices[instance_index] 间接索引实例数据。
 */
const VS_BENCH_CULLED = /* wgsl */ `
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
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    let mapped = compactedIndices[instanceIdx];
    let inst = instances[mapped];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip = uniforms.viewProj * worldPos;
    out.color = inst.color;
    return out;
}
`;

const FS_BENCH = /* wgsl */ `
@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
`;

// ─── GLB Generator ──────────────────────────────────────────

interface ModelConfig {
  name: string;
  meshCount: number;
  /** 每个 mesh 的平均顶点数（实际会变化 ±50%）。 */
  avgVerticesPerMesh: number;
  /** 每个 mesh 的平均三角形数。 */
  avgTrianglesPerMesh: number;
  /** 材质数量。 */
  materialCount: number;
  /** 层级深度（0=扁平，3=三层嵌套）。 */
  hierarchyDepth: number;
}

function generateGlb(config: ModelConfig): ArrayBuffer {
  const { meshCount, avgVerticesPerMesh, avgTrianglesPerMesh, materialCount, hierarchyDepth } = config;

  // Seeded random for reproducibility
  let seed = 42;
  function rand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  // Generate meshes with varying complexity
  const meshDescs: { verts: Float32Array; indices: Uint16Array; materialIdx: number }[] = [];

  for (let m = 0; m < meshCount; m++) {
    // Vary vertex count ±50%
    const vertCount = Math.max(4, Math.round(avgVerticesPerMesh * (0.5 + rand())));
    const triCount = Math.max(1, Math.round(avgTrianglesPerMesh * (0.5 + rand())));
    const idxCount = triCount * 3;

    // Generate sphere-like vertices (random positions on a small box)
    const verts = new Float32Array(vertCount * 6); // pos(3) + normal(3)
    for (let v = 0; v < vertCount; v++) {
      const x = (rand() - 0.5) * 0.5;
      const y = (rand() - 0.5) * 0.5;
      const z = (rand() - 0.5) * 0.5;
      verts[v * 6 + 0] = x;
      verts[v * 6 + 1] = y;
      verts[v * 6 + 2] = z;
      // Simple normal (normalize after)
      const len = Math.sqrt(x * x + y * y + z * z) || 1;
      verts[v * 6 + 3] = x / len;
      verts[v * 6 + 4] = y / len;
      verts[v * 6 + 5] = z / len;
    }

    // Generate indices (triangles from vertex soup)
    const indices = new Uint16Array(idxCount);
    for (let i = 0; i < idxCount; i++) {
      indices[i] = Math.floor(rand() * vertCount);
    }

    meshDescs.push({
      verts,
      indices,
      materialIdx: m % materialCount,
    });
  }

  // Calculate binary layout
  let totalBinSize = 0;
  const vertOffsets: number[] = [];
  const idxOffsets: number[] = [];

  for (const md of meshDescs) {
    vertOffsets.push(totalBinSize);
    totalBinSize += md.verts.byteLength;
    while (totalBinSize % 4 !== 0) totalBinSize++;
    idxOffsets.push(totalBinSize);
    totalBinSize += md.indices.byteLength;
    while (totalBinSize % 4 !== 0) totalBinSize++;
  }

  const binData = new Uint8Array(totalBinSize);
  for (let i = 0; i < meshDescs.length; i++) {
    binData.set(new Uint8Array(meshDescs[i]!.verts.buffer), vertOffsets[i]!);
    binData.set(new Uint8Array(meshDescs[i]!.indices.buffer), idxOffsets[i]!);
  }

  // Build glTF JSON
  const accessors: any[] = [];
  const bufferViews: any[] = [];
  let accIdx = 0;
  let bvIdx = 0;

  const primAccessors: { posAcc: number; normAcc: number; idxAcc: number }[] = [];

  for (let i = 0; i < meshDescs.length; i++) {
    const md = meshDescs[i]!;
    const vertByteLength = md.verts.byteLength;
    const idxByteLength = md.indices.byteLength;

    // BV: vertices (interleaved)
    bufferViews.push({
      buffer: 0, byteOffset: vertOffsets[i], byteLength: vertByteLength,
      byteStride: 24, target: 34962,
    });
    const vertBv = bvIdx++;

    // BV: indices
    bufferViews.push({
      buffer: 0, byteOffset: idxOffsets[i], byteLength: idxByteLength,
      target: 34963,
    });
    const idxBv = bvIdx++;

    // Acc: POSITION
    accessors.push({
      bufferView: vertBv, byteOffset: 0, componentType: 5126,
      count: md.verts.length / 6, type: 'VEC3',
    });
    const posAcc = accIdx++;

    // Acc: NORMAL
    accessors.push({
      bufferView: vertBv, byteOffset: 12, componentType: 5126,
      count: md.verts.length / 6, type: 'VEC3',
    });
    const normAcc = accIdx++;

    // Acc: INDICES
    accessors.push({
      bufferView: idxBv, componentType: 5123,
      count: md.indices.length, type: 'SCALAR',
    });
    const idxAcc = accIdx++;

    primAccessors.push({ posAcc, normAcc, idxAcc });
  }

  // Build hierarchy — flat glTF node array with integer children indices
  const gltfNodes: any[] = [];
  const meshNodeIndices: number[] = [];

  if (hierarchyDepth === 0) {
    // Flat: each mesh is a root node
    for (let i = 0; i < meshCount; i++) {
      meshNodeIndices.push(gltfNodes.length);
      gltfNodes.push({
        name: `Mesh_${i}`,
        mesh: i,
        translation: [(rand() - 0.5) * 10, (rand() - 0.5) * 10, (rand() - 0.5) * 10],
      });
    }
  } else {
    // Create mesh nodes first
    for (let i = 0; i < meshCount; i++) {
      meshNodeIndices.push(i);
      gltfNodes.push({
        name: `Mesh_${i}`,
        mesh: i,
        translation: [(rand() - 0.5) * 10, (rand() - 0.5) * 10, (rand() - 0.5) * 10],
      });
    }

    // Build hierarchy bottom-up: group nodes reference children by index
    let currentLevel = [...meshNodeIndices];
    for (let depth = 0; depth < hierarchyDepth && currentLevel.length > 1; depth++) {
      const groupSize = Math.max(2, Math.ceil(currentLevel.length / 3));
      const newLevel: number[] = [];
      for (let g = 0; g < currentLevel.length; g += groupSize) {
        const children = currentLevel.slice(g, g + groupSize);
        const groupIdx = gltfNodes.length;
        gltfNodes.push({
          name: `Group_D${depth}_G${newLevel.length}`,
          children,
          translation: [(rand() - 0.5) * 5, (rand() - 0.5) * 5, (rand() - 0.5) * 5],
        });
        newLevel.push(groupIdx);
      }
      currentLevel = newLevel;
    }
    // Wrap remaining groups in a single root node
    if (currentLevel.length > 1) {
      gltfNodes.push({
        name: 'Root',
        children: currentLevel,
        translation: [0, 0, 0],
      });
    }
  }

  const rootNodes = hierarchyDepth === 0
    ? meshNodeIndices
    : [gltfNodes.length - 1];

  // Materials
  const materials = [];
  for (let m = 0; m < materialCount; m++) {
    materials.push({
      name: `Material_${m}`,
      pbrMetallicRoughness: {
        baseColorFactor: [
          0.3 + rand() * 0.7,
          0.3 + rand() * 0.7,
          0.3 + rand() * 0.7,
          1.0,
        ],
        metallicFactor: rand(),
        roughnessFactor: 0.3 + rand() * 0.7,
      },
    });
  }

  const gltfJson = {
    asset: { version: '2.0', generator: 'hpg-bench' },
    scene: 0,
    scenes: [{ name: 'BenchScene', nodes: rootNodes }],
    nodes: gltfNodes,
    meshes: meshDescs.map((md, i) => ({
      name: `Mesh_${i}`,
      primitives: [{
        attributes: {
          POSITION: primAccessors[i]!.posAcc,
          NORMAL: primAccessors[i]!.normAcc,
        },
        indices: primAccessors[i]!.idxAcc,
        material: md.materialIdx,
      }],
    })),
    materials,
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinSize }],
  };

  // Build GLB
  const jsonStr = JSON.stringify(gltfJson);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const jsonPad = (4 - (jsonBytes.byteLength % 4)) % 4;
  const jsonChunkLen = jsonBytes.byteLength + jsonPad;
  const binPad = (4 - (totalBinSize % 4)) % 4;
  const binChunkLen = totalBinSize + binPad;
  const totalLen = 12 + 8 + jsonChunkLen + 8 + binChunkLen;

  const glb = new ArrayBuffer(totalLen);
  const dv = new DataView(glb);

  dv.setUint32(0, 0x46546C67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, totalLen, true);

  let off = 12;
  dv.setUint32(off, jsonChunkLen, true);
  dv.setUint32(off + 4, 0x4E4F534A, true);
  const jd = new Uint8Array(glb, off + 8, jsonChunkLen);
  jd.set(jsonBytes);
  for (let i = jsonBytes.byteLength; i < jsonChunkLen; i++) jd[i] = 0x20;
  off += 8 + jsonChunkLen;

  dv.setUint32(off, binChunkLen, true);
  dv.setUint32(off + 4, 0x004E4942, true);
  new Uint8Array(glb, off + 8, binChunkLen).set(binData);

  return glb;
}

// ─── Benchmark Runner ───────────────────────────────────────

interface BenchResult {
  model: string;
  glbSize: number;
  parseMs: number;
  importMs: number;
  meshCount: number;
  vertexCount: number;
  indexCount: number;
  materialCount: number;
  renderItemcount: number;
  direct: { cpuMs: number; gpuMs: number | null; draws: number; batches: number };
  culled: { cpuMs: number; gpuMs: number | null; draws: number; batches: number };
}

async function benchModel(
  config: ModelConfig,
  renderer: Renderer,
  makePipeline: () => ResolvedPipeline,
  makeCulledPipeline: () => ResolvedPipeline,
  vpMatrix: Float32Array,
  samples: number,
): Promise<BenchResult> {
  // 1. Generate GLB
  const tGen0 = performance.now();
  const glb = generateGlb(config);
  const tGen1 = performance.now();

  // 2. Parse GLB
  const tParse0 = performance.now();
  const asset = parseGltf(glb);
  const tParse1 = performance.now();

  // 3. Import (GeometryArena upload)
  const tImport0 = performance.now();
  const scene = importGltfAsset(asset, renderer, { scale: 1, flipV: true });
  const tImport1 = performance.now();

  // 4. Generate RenderItems。submitCulled 按 geometry 分组，无需逐 mesh 单独管线；
  //    但 group(1) 布局不同 —— culled 路径必须用 compaction 管线（几何体共享）。
  const renderItems = sceneToRenderItems(scene, makePipeline());
  const culledItems = sceneToRenderItems(scene, makeCulledPipeline());

  // Count stats
  let vertexCount = 0;
  let indexCount = 0;
  for (const m of scene.meshes) {
    vertexCount += m.geometry.vertexCount;
    indexCount += m.geometry.indexCount;
  }

  // 5. Benchmark submit paths — try timestamps, skip if unsupported
  let hasTimestampQuery = false;
  try {
    const tq = new TimestampQuery(renderer.device, 2);
    tq.destroy();
    hasTimestampQuery = true;
  } catch {
    hasTimestampQuery = false;
  }
  const statusSuffix = hasTimestampQuery ? '' : ' (GPU timing unavailable)';

  // Warmup
  for (let w = 0; w < 5; w++) {
    renderer.submit(renderItems);
  }
  // GPU sync
  await renderer.device.queue.onSubmittedWorkDone();

  // Direct
  const directCpuTimes: number[] = [];
  const directGpuTimes: number[] = [];
  for (let i = 0; i < samples; i++) {
    const tq = hasTimestampQuery ? new TimestampQuery(renderer.device, 2) : null;
    const t0 = performance.now();
    // Direct = 逐 item 独立 draw call（submit() 的第二个参数现在是 SubmitOptions，不是时间戳查询）。
    renderer.submitDirect(renderItems, tq ?? undefined);
    const t1 = performance.now();
    directCpuTimes.push(t1 - t0);
    if (tq) {
      const ts = await tq.readback(renderer.device);
      directGpuTimes.push((ts[1]! - ts[0]!) / 1e6);
      tq.destroy();
    }
  }

  // GPU Culling
  const culledCpuTimes: number[] = [];
  const culledGpuTimes: number[] = [];
  for (let i = 0; i < samples; i++) {
    const tq = hasTimestampQuery ? new TimestampQuery(renderer.device, 2) : null;
    const t0 = performance.now();
    renderer.submitCulled(culledItems, vpMatrix, tq ?? undefined);
    const t1 = performance.now();
    culledCpuTimes.push(t1 - t0);
    if (tq) {
      const ts = await tq.readback(renderer.device);
      culledGpuTimes.push((ts[1]! - ts[0]!) / 1e6);
      tq.destroy();
    }
  }

  // Stats
  const stats = renderer.submit(renderItems);

  function median(arr: number[]): number {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)] ?? 0;
  }

  return {
    model: config.name,
    glbSize: glb.byteLength,
    parseMs: tParse1 - tParse0,
    importMs: tImport1 - tImport0,
    meshCount: scene.meshes.length,
    vertexCount,
    indexCount,
    materialCount: scene.materials.length,
    renderItemcount: renderItems.length,
    direct: {
      cpuMs: median(directCpuTimes),
      gpuMs: median(directGpuTimes),
      draws: stats.drawCalls,
      batches: stats.batches,
    },
    culled: {
      cpuMs: median(culledCpuTimes),
      gpuMs: median(culledGpuTimes),
      draws: stats.drawCalls,
      batches: stats.batches,
    },
  };
}

// ─── Report ─────────────────────────────────────────────────

function renderReport(results: BenchResult[]): string {
  const lines: string[] = [];

  lines.push('<div class="section">');
  lines.push('<h2>Asset Statistics</h2>');
  lines.push('<table>');
  lines.push('<tr><th>Model</th><th>GLB Size</th><th class="num">Meshes</th><th class="num">Vertices</th><th class="num">Indices</th><th class="num">Materials</th><th class="num">RenderItems</th><th class="num">Parse</th><th class="num">Import</th></tr>');
  for (const r of results) {
    lines.push(`<tr>
      <td>${r.model}</td>
      <td>${(r.glbSize / 1024).toFixed(0)} KB</td>
      <td class="num">${r.meshCount}</td>
      <td class="num">${r.vertexCount.toLocaleString()}</td>
      <td class="num">${r.indexCount.toLocaleString()}</td>
      <td class="num">${r.materialCount}</td>
      <td class="num">${r.renderItemcount}</td>
      <td class="num">${r.parseMs.toFixed(1)} ms</td>
      <td class="num">${r.importMs.toFixed(1)} ms</td>
    </tr>`);
  }
  lines.push('</table>');
  lines.push('</div>');

  // CPU submit comparison
  lines.push('<div class="section">');
  lines.push('<h2>CPU Submit Time (median)</h2>');
  lines.push('<table>');
  lines.push('<tr><th>Model</th><th class="num">Direct</th><th class="num">GPU Culled</th><th class="num">Ratio</th></tr>');
  for (const r of results) {
    const ratio = r.culled.cpuMs > 0 ? (r.direct.cpuMs / r.culled.cpuMs).toFixed(2) : '—';
    const ratioClass = r.direct.cpuMs > r.culled.cpuMs ? 'good' : r.direct.cpuMs < r.culled.cpuMs ? 'warn' : '';
    lines.push(`<tr>
      <td>${r.model}</td>
      <td class="num">${r.direct.cpuMs.toFixed(2)} ms</td>
      <td class="num">${r.culled.cpuMs.toFixed(2)} ms</td>
      <td class="num ${ratioClass}">${ratio}x</td>
    </tr>`);
  }
  lines.push('</table>');
  lines.push('</div>');

  // GPU execution time
  lines.push('<div class="section">');
  lines.push('<h2>GPU Execution Time (median)</h2>');
  lines.push('<table>');
  lines.push('<tr><th>Model</th><th class="num">Direct</th><th class="num">GPU Culled</th><th class="num">Ratio</th><th class="num">Draw Calls</th></tr>');
  for (const r of results) {
    const gpuDirect = r.direct.gpuMs?.toFixed(2) ?? '—';
    const gpuCulled = r.culled.gpuMs?.toFixed(2) ?? '—';
    const ratio = (r.direct.gpuMs && r.culled.gpuMs)
      ? (r.culled.gpuMs / r.direct.gpuMs).toFixed(2) + 'x'
      : '—';
    const ratioClass = r.culled.gpuMs && r.direct.gpuMs && r.culled.gpuMs < r.direct.gpuMs ? 'good' : '';
    lines.push(`<tr>
      <td>${r.model}</td>
      <td class="num">${gpuDirect} ms</td>
      <td class="num">${gpuCulled} ms</td>
      <td class="num ${ratioClass}">${ratio}</td>
      <td class="num">${r.direct.draws}</td>
    </tr>`);
  }
  lines.push('</table>');
  lines.push('</div>');

  // Time breakdown (waterfall)
  lines.push('<div class="section">');
  lines.push('<h2>Time Breakdown — Where Does Time Go?</h2>');
  for (const r of results) {
    const total = r.parseMs + r.importMs + r.direct.cpuMs;
    const parsePct = (r.parseMs / total * 100).toFixed(0);
    const importPct = (r.importMs / total * 100).toFixed(0);
    const submitPct = (r.direct.cpuMs / total * 100).toFixed(0);

    lines.push(`<div style="margin-bottom:12px">`);
    lines.push(`<div style="font-size:12px; margin-bottom:4px;"><strong>${r.model}</strong> — Total CPU: ${total.toFixed(1)} ms</div>`);
    lines.push(`<div class="bar-container">`);
    lines.push(`<div class="bar bar-parse" style="width:${parsePct}%" title="Parse: ${r.parseMs.toFixed(1)}ms"></div>`);
    lines.push(`</div>`);
    lines.push(`<div class="bar-container" style="margin-top:2px">`);
    lines.push(`<div class="bar bar-upload" style="width:${importPct}%" title="Import: ${r.importMs.toFixed(1)}ms"></div>`);
    lines.push(`</div>`);
    lines.push(`<div class="bar-container" style="margin-top:2px">`);
    lines.push(`<div class="bar bar-cpu" style="width:${submitPct}%" title="Submit: ${r.direct.cpuMs.toFixed(1)}ms"></div>`);
    lines.push(`</div>`);
    lines.push(`<div style="font-size:11px; color:#8b949e; margin-top:4px;">`);
    lines.push(`<span style="color:#7ee787">■</span> Parse ${r.parseMs.toFixed(1)}ms (${parsePct}%) · `);
    lines.push(`<span style="color:#d2a8ff">■</span> Import ${r.importMs.toFixed(1)}ms (${importPct}%) · `);
    lines.push(`<span style="color:#58a6ff">■</span> Submit ${r.direct.cpuMs.toFixed(1)}ms (${submitPct}%)`);
    lines.push(`</div></div>`);
  }
  lines.push('</div>');

  // Bottleneck analysis
  lines.push('<div class="section">');
  lines.push('<h2>Bottleneck Analysis</h2>');
  lines.push('<table>');
  lines.push('<tr><th>Model</th><th>Dominant Phase</th><th>GPU Culling Benefit</th><th>Recommendation</th></tr>');
  for (const r of results) {
    const phases = [
      { name: 'Parse', time: r.parseMs },
      { name: 'Import', time: r.importMs },
      { name: 'Submit', time: r.direct.cpuMs },
    ].sort((a, b) => b.time - a.time);

    const dominant = phases[0]!;
    const gpuBenefit = (r.direct.gpuMs && r.culled.gpuMs)
      ? ((r.direct.gpuMs - r.culled.gpuMs) / r.direct.gpuMs * 100).toFixed(0) + '%'
      : '—';

    let rec = '';
    if (dominant.name === 'Parse') rec = 'Optimize GLB parser or cache parsed assets';
    else if (dominant.name === 'Import') rec = 'Optimize GeometryArena upload / pooling';
    else if (r.direct.draws > 1000) rec = 'Many draw calls — batching or indirect may help';
    else rec = 'Balanced workload';

    lines.push(`<tr>
      <td>${r.model}</td>
      <td class="highlight">${dominant.name} (${dominant.time.toFixed(1)} ms)</td>
      <td>${gpuBenefit}</td>
      <td>${rec}</td>
    </tr>`);
  }
  lines.push('</table>');
  lines.push('</div>');

  return lines.join('\n');
}

// ─── Entry Point ────────────────────────────────────────────

async function runBenchmarkInternal() {
  const statusEl = document.getElementById('status')!;
  const resultsEl = document.getElementById('results')!;

  if (!navigator.gpu) {
    statusEl.textContent = 'WebGPU not supported.';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    statusEl.textContent = 'No GPU adapter.';
    return;
  }
  const device = await adapter.requestDevice();
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const renderer = Renderer.create({ device, context, format });

  const layout = uniformBindGroupLayout(device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT },
  ]);
  const uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // 与 asset-importer 的 canonical 布局一致（stride 48：pos3 + norm3 + uv2 + tan4）。
  const vertexLayouts = [{
    arrayStride: 48, stepMode: 'vertex' as const,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
      { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },
      { shaderLocation: 2, offset: 24, format: 'float32x2' as GPUVertexFormat },
      { shaderLocation: 3, offset: 32, format: 'float32x4' as GPUVertexFormat },
    ],
  }];
  const commonDesc = {
    vertexLayouts,
    bindGroupLayouts: [layout],
    globalBindings: [{ binding: 0, buffer: uniformBuffer }],
    depth: { format: 'depth24plus' as GPUTextureFormat, depthWriteEnabled: true, depthCompare: 'less' as GPUCompareFunction },
    targets: [{ format }],
  };

  /** 普通实例管线（submit / submitDirect 路径）。 */
  function makePipeline(): ResolvedPipeline {
    return renderer.registerPipeline({
      label: 'bench-pipeline',
      vsCode: VS_BENCH,
      fsCode: FS_BENCH,
      ...commonDesc,
    });
  }

  /** 剔除管线（submitCulled 路径）：group(1) = [instances, compactedIndices]。 */
  function makeCulledPipeline(): ResolvedPipeline {
    return renderer.registerPipeline({
      label: 'bench-pipeline-culled',
      vsCode: VS_BENCH_CULLED,
      fsCode: FS_BENCH,
      compaction: true,
      ...commonDesc,
    });
  }

  const vpMatrix = new Float32Array(16);
  // Proper VP matrix: camera at (0, 0, 15) looking at origin.
  const aspect = 1;
  const fov = Math.PI / 4;
  const near = 0.1, far = 100;
  const f = 1 / Math.tan(fov / 2);
  // View matrix (lookAt)
  const eye = new Float32Array([0, 0, 15]);
  const target = new Float32Array([0, 0, 0]);
  const up = new Float32Array([0, 1, 0]);
  const view = lookAt(eye, target, up);
  // Projection matrix
  const proj = perspective(fov, aspect, near, far);
  multiply(vpMatrix, proj, view);

  const configs: ModelConfig[] = [
    { name: 'Small (50 meshes, ~50K verts)', meshCount: 50, avgVerticesPerMesh: 1000, avgTrianglesPerMesh: 300, materialCount: 5, hierarchyDepth: 2 },
    { name: 'Medium (200 meshes, ~500K verts)', meshCount: 200, avgVerticesPerMesh: 2500, avgTrianglesPerMesh: 800, materialCount: 10, hierarchyDepth: 3 },
    { name: 'Large (500 meshes, ~2M verts)', meshCount: 500, avgVerticesPerMesh: 4000, avgTrianglesPerMesh: 1200, materialCount: 15, hierarchyDepth: 3 },
  ];

  const SAMPLES = 10;
  const results: BenchResult[] = [];

  for (const config of configs) {
    statusEl.textContent = `Running: ${config.name}...`;
    await new Promise((r) => setTimeout(r, 0));

    const result = await benchModel(config, renderer, makePipeline, makeCulledPipeline, vpMatrix, SAMPLES);
    results.push(result);
  }

  statusEl.textContent = `Done — ${results.length} models benchmarked.`;
  resultsEl.innerHTML = renderReport(results);

  renderer.dispose();
}

// Expose to HTML
(window as any).runBenchmark = runBenchmarkInternal;
