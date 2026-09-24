/**
 * Node.js GLB Asset Pipeline Benchmark
 *
 * Measures parse time + memory for synthetic GLB models.
 * No GPU required — pure asset pipeline data.
 *
 * Usage: npx tsx benchmark/asset-bench.ts
 */

import { parseGltf } from '../src/core/gltf';

// ─── GLB Generator ──────────────────────────────────────────

interface ModelConfig {
  name: string;
  meshCount: number;
  avgVerticesPerMesh: number;
  avgTrianglesPerMesh: number;
  materialCount: number;
  hierarchyDepth: number;
}

function generateGlb(config: ModelConfig): ArrayBuffer {
  const { meshCount, avgVerticesPerMesh, avgTrianglesPerMesh, materialCount, hierarchyDepth } = config;

  let seed = 42;
  function rand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  const meshDescs: { verts: Float32Array; indices: Uint16Array; materialIdx: number }[] = [];

  for (let m = 0; m < meshCount; m++) {
    const vertCount = Math.max(4, Math.round(avgVerticesPerMesh * (0.5 + rand())));
    const triCount = Math.max(1, Math.round(avgTrianglesPerMesh * (0.5 + rand())));
    const idxCount = triCount * 3;

    const verts = new Float32Array(vertCount * 6);
    for (let v = 0; v < vertCount; v++) {
      const x = (rand() - 0.5) * 0.5;
      const y = (rand() - 0.5) * 0.5;
      const z = (rand() - 0.5) * 0.5;
      verts[v * 6 + 0] = x;
      verts[v * 6 + 1] = y;
      verts[v * 6 + 2] = z;
      const len = Math.sqrt(x * x + y * y + z * z) || 1;
      verts[v * 6 + 3] = x / len;
      verts[v * 6 + 4] = y / len;
      verts[v * 6 + 5] = z / len;
    }

    const indices = new Uint16Array(idxCount);
    for (let i = 0; i < idxCount; i++) {
      indices[i] = Math.floor(rand() * vertCount);
    }

    meshDescs.push({ verts, indices, materialIdx: m % materialCount });
  }

  // Binary layout
  let totalBinSize = 0;
  const vertOffsets: number[] = [];
  const idxOffsets: number[] = [];

  for (const md of meshDescs) {
    vertOffsets.push(totalBinSize);
    totalBinSize += md.verts.byteLength;
    totalBinSize = (totalBinSize + 3) & ~3;
    idxOffsets.push(totalBinSize);
    totalBinSize += md.indices.byteLength;
    totalBinSize = (totalBinSize + 3) & ~3;
  }

  const binData = new Uint8Array(totalBinSize);
  for (let i = 0; i < meshDescs.length; i++) {
    binData.set(new Uint8Array(meshDescs[i]!.verts.buffer), vertOffsets[i]!);
    binData.set(new Uint8Array(meshDescs[i]!.indices.buffer), idxOffsets[i]!);
  }

  // glTF JSON
  const accessors: any[] = [];
  const bufferViews: any[] = [];
  let accIdx = 0;
  let bvIdx = 0;
  const primAccessors: { posAcc: number; normAcc: number; idxAcc: number }[] = [];

  for (let i = 0; i < meshDescs.length; i++) {
    const md = meshDescs[i]!;

    bufferViews.push({ buffer: 0, byteOffset: vertOffsets[i], byteLength: md.verts.byteLength, byteStride: 24, target: 34962 });
    const vertBv = bvIdx++;
    bufferViews.push({ buffer: 0, byteOffset: idxOffsets[i], byteLength: md.indices.byteLength, target: 34963 });
    const idxBv = bvIdx++;

    accessors.push({ bufferView: vertBv, byteOffset: 0, componentType: 5126, count: md.verts.length / 6, type: 'VEC3' });
    const posAcc = accIdx++;
    accessors.push({ bufferView: vertBv, byteOffset: 12, componentType: 5126, count: md.verts.length / 6, type: 'VEC3' });
    const normAcc = accIdx++;
    accessors.push({ bufferView: idxBv, componentType: 5123, count: md.indices.length, type: 'SCALAR' });
    const idxAcc = accIdx++;

    primAccessors.push({ posAcc, normAcc, idxAcc });
  }

  // Flat node list (hierarchy)
  const allNodes: any[] = [];
  for (let i = 0; i < meshCount; i++) {
    allNodes.push({
      name: `Mesh_${i}`,
      mesh: i,
      translation: [(rand() - 0.5) * 10, (rand() - 0.5) * 10, (rand() - 0.5) * 10],
    });
  }

  const materials = [];
  for (let m = 0; m < materialCount; m++) {
    materials.push({
      name: `Material_${m}`,
      pbrMetallicRoughness: {
        baseColorFactor: [0.3 + rand() * 0.7, 0.3 + rand() * 0.7, 0.3 + rand() * 0.7, 1.0],
        metallicFactor: rand(),
        roughnessFactor: 0.3 + rand() * 0.7,
      },
    });
  }

  const gltfJson = {
    asset: { version: '2.0', generator: 'hpg-bench' },
    scene: 0,
    scenes: [{ name: 'BenchScene', nodes: Array.from({ length: meshCount }, (_, i) => i) }],
    nodes: allNodes,
    meshes: meshDescs.map((md, i) => ({
      name: `Mesh_${i}`,
      primitives: [{
        attributes: { POSITION: primAccessors[i]!.posAcc, NORMAL: primAccessors[i]!.normAcc },
        indices: primAccessors[i]!.idxAcc,
        material: md.materialIdx,
      }],
    })),
    materials,
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinSize }],
  };

  // GLB
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

// ─── Benchmark ──────────────────────────────────────────────

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
}

function percentile(arr: number[], p: number): number {
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil(s.length * p / 100) - 1;
  return s[Math.max(0, idx)] ?? 0;
}

const configs: ModelConfig[] = [
  { name: 'XS   (10 meshes, ~10K verts)',    meshCount: 10,  avgVerticesPerMesh: 1000,  avgTrianglesPerMesh: 300,   materialCount: 3,  hierarchyDepth: 0 },
  { name: 'S    (50 meshes, ~50K verts)',     meshCount: 50,  avgVerticesPerMesh: 1000,  avgTrianglesPerMesh: 300,   materialCount: 5,  hierarchyDepth: 2 },
  { name: 'M    (200 meshes, ~500K verts)',   meshCount: 200, avgVerticesPerMesh: 2500,  avgTrianglesPerMesh: 800,   materialCount: 10, hierarchyDepth: 3 },
  { name: 'L    (500 meshes, ~2M verts)',     meshCount: 500, avgVerticesPerMesh: 4000,  avgTrianglesPerMesh: 1200,  materialCount: 15, hierarchyDepth: 3 },
  { name: 'XL   (1000 meshes, ~5M verts)',    meshCount: 1000, avgVerticesPerMesh: 5000, avgTrianglesPerMesh: 1500,  materialCount: 20, hierarchyDepth: 4 },
];

const PARSE_SAMPLES = 20;

console.log('hpg — Asset Pipeline Benchmark (Node.js)\n');
console.log('='.repeat(100));

// Table header
console.log(
  'Model'.padEnd(28) +
  'GLB Size'.padStart(10) +
  'JSON Size'.padStart(10) +
  'Bin Size'.padStart(10) +
  'Meshes'.padStart(8) +
  'Verts'.padStart(10) +
  'Indices'.padStart(10) +
  'Mats'.padStart(6) +
  'Parse(ms)'.padStart(12) +
  'p95(ms)'.padStart(10) +
  'PeakMem'.padStart(10)
);
console.log('-'.repeat(100));

for (const config of configs) {
  // Generate
  const glb = generateGlb(config);

  // Measure JSON vs binary split
  const dv2 = new DataView(glb);
  const jsonChunkLen = dv2.getUint32(12, true);
  const jsonSize = jsonChunkLen;
  const binSize = glb.byteLength - 12 - 8 - jsonChunkLen - 8;

  // Count actual vertices/indices from parsed result
  let totalVerts = 0;
  let totalIndices = 0;

  // Warmup
  for (let w = 0; w < 3; w++) parseGltf(glb);

  // Benchmark parse
  const parseTimes: number[] = [];
  const memBefore = process.memoryUsage().heapUsed;
  let peakMem = memBefore;

  for (let s = 0; s < PARSE_SAMPLES; s++) {
    const t0 = performance.now();
    const asset = parseGltf(glb);
    const t1 = performance.now();
    parseTimes.push(t1 - t0);

    // Count on first sample
    if (s === 0) {
      for (const mesh of asset.meshes) {
        for (const prim of mesh.primitives) {
          totalVerts += prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
          totalIndices += prim.indices.length;
        }
      }
    }

    const memNow = process.memoryUsage().heapUsed;
    if (memNow > peakMem) peakMem = memNow;
  }

  const med = median(parseTimes);
  const p95 = percentile(parseTimes, 95);

  console.log(
    config.name.padEnd(28) +
    formatSize(glb.byteLength).padStart(10) +
    formatSize(jsonSize).padStart(10) +
    formatSize(binSize).padStart(10) +
    String(config.meshCount).padStart(8) +
    String(totalVerts).padStart(10) +
    String(totalIndices).padStart(10) +
    String(config.materialCount).padStart(6) +
    `${med.toFixed(2)}`.padStart(12) +
    `${p95.toFixed(2)}`.padStart(10) +
    formatSize(peakMem - memBefore).padStart(10)
  );
}

console.log('-'.repeat(100));
console.log(`\nSamples: ${PARSE_SAMPLES} per model | All times median unless noted`);
console.log('\nKey question: Does parse time scale linearly with model size, or super-linearly?');
