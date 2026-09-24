/**
 * Phase 10E — Integration tests for realistic GLB models.
 *
 * Validates: GLB generation → parseGltf → flattenScene → scene structure
 * with varying mesh counts, materials, hierarchy depths, and vertex sizes.
 */

import { describe, it, expect } from 'vitest';
import { parseGltf, flattenScene } from '../src/core/gltf';

// ─── GLB Generator (proper glTF 2.0 format) ────────────────

interface ModelConfig {
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

  // Generate meshes
  const meshDescs: { verts: Float32Array; indices: Uint16Array; matIdx: number }[] = [];
  for (let m = 0; m < meshCount; m++) {
    const vertCount = Math.max(4, Math.round(avgVerticesPerMesh * (0.5 + rand())));
    const triCount = Math.max(1, Math.round(avgTrianglesPerMesh * (0.5 + rand())));
    const idxCount = triCount * 3;

    const verts = new Float32Array(vertCount * 6);
    for (let v = 0; v < vertCount; v++) {
      const x = (rand() - 0.5) * 0.5;
      const y = (rand() - 0.5) * 0.5;
      const z = (rand() - 0.5) * 0.5;
      verts[v * 6] = x; verts[v * 6 + 1] = y; verts[v * 6 + 2] = z;
      const len = Math.sqrt(x * x + y * y + z * z) || 1;
      verts[v * 6 + 3] = x / len; verts[v * 6 + 4] = y / len; verts[v * 6 + 5] = z / len;
    }

    const indices = new Uint16Array(idxCount);
    for (let i = 0; i < idxCount; i++) indices[i] = Math.floor(rand() * vertCount);

    meshDescs.push({ verts, indices, matIdx: m % materialCount });
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

  // Accessors + buffer views
  const accessors: any[] = [];
  const bufferViews: any[] = [];
  let accIdx = 0;
  let bvIdx = 0;
  const primAccessors: { pos: number; norm: number; idx: number }[] = [];

  for (let i = 0; i < meshDescs.length; i++) {
    const md = meshDescs[i]!;
    bufferViews.push({ buffer: 0, byteOffset: vertOffsets[i], byteLength: md.verts.byteLength, byteStride: 24, target: 34962 });
    const vbv = bvIdx++;
    bufferViews.push({ buffer: 0, byteOffset: idxOffsets[i], byteLength: md.indices.byteLength, target: 34963 });
    const ibv = bvIdx++;
    accessors.push({ bufferView: vbv, byteOffset: 0, componentType: 5126, count: md.verts.length / 6, type: 'VEC3' });
    const pa = accIdx++;
    accessors.push({ bufferView: vbv, byteOffset: 12, componentType: 5126, count: md.verts.length / 6, type: 'VEC3' });
    const na = accIdx++;
    accessors.push({ bufferView: ibv, componentType: 5123, count: md.indices.length, type: 'SCALAR' });
    const ia = accIdx++;
    primAccessors.push({ pos: pa, norm: na, idx: ia });
  }

  // Build glTF node tree with integer children indices
  // Node layout: [mesh_0, mesh_1, ..., mesh_N] for flat
  // Or grouped into parent nodes for hierarchy
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
    // Hierarchical: create parent nodes, distribute meshes among them
    // Simple approach: create a tree of group nodes, meshes are leaves
    const meshNodesStart = 0;
    const groupNodesStart = meshCount;

    // First meshCount nodes are mesh nodes
    for (let i = 0; i < meshCount; i++) {
      gltfNodes.push({
        name: `Mesh_${i}`,
        mesh: i,
        translation: [(rand() - 0.5) * 10, (rand() - 0.5) * 10, (rand() - 0.5) * 10],
      });
      meshNodeIndices.push(i);
    }

    // Create hierarchy levels (bottom-up)
    let currentLevel = meshNodeIndices;
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
    // currentLevel now has 1 root node
  }

  // Materials
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

  const rootNodes = hierarchyDepth === 0
    ? meshNodeIndices
    : [gltfNodes.length - 1]; // Last node is the root group

  const gltfJson = {
    asset: { version: '2.0', generator: 'hpg-test' },
    scene: 0,
    scenes: [{ name: 'TestScene', nodes: rootNodes }],
    nodes: gltfNodes,
    meshes: meshDescs.map((md, i) => ({
      name: `Mesh_${i}`,
      primitives: [{
        attributes: { POSITION: primAccessors[i]!.pos, NORMAL: primAccessors[i]!.norm },
        indices: primAccessors[i]!.idx,
        material: md.matIdx,
      }],
    })),
    materials,
    accessors,
    bufferViews,
    buffers: [{ byteLength: totalBinSize }],
  };

  // GLB assembly
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

// ─── Tests ──────────────────────────────────────────────────

describe('10E: Realistic GLB parsing', () => {
  it('parses XS model (10 meshes, ~10K verts)', () => {
    const glb = generateGlb({
      meshCount: 10, avgVerticesPerMesh: 1000, avgTrianglesPerMesh: 300,
      materialCount: 3, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    expect(asset.meshes.length).toBe(10);
    expect(asset.materials.length).toBe(3);

    let totalVerts = 0;
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        expect(prim.vertices.length).toBeGreaterThan(0);
        expect(prim.indices.length).toBeGreaterThan(0);
        expect(prim.vertexLayout.arrayStride).toBe(24);
        totalVerts += prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
      }
    }
    expect(totalVerts).toBeGreaterThan(5000);
  });

  it('parses Medium model (200 meshes, ~500K verts)', () => {
    const glb = generateGlb({
      meshCount: 200, avgVerticesPerMesh: 2500, avgTrianglesPerMesh: 800,
      materialCount: 10, hierarchyDepth: 3,
    });
    const asset = parseGltf(glb);

    expect(asset.meshes.length).toBe(200);
    expect(asset.materials.length).toBe(10);

    let totalVerts = 0;
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        totalVerts += prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
      }
    }
    expect(totalVerts).toBeGreaterThan(200000);
  });

  it('parses Large model (500 meshes, ~2M verts)', () => {
    const glb = generateGlb({
      meshCount: 500, avgVerticesPerMesh: 4000, avgTrianglesPerMesh: 1200,
      materialCount: 15, hierarchyDepth: 3,
    });
    const asset = parseGltf(glb);

    expect(asset.meshes.length).toBe(500);
    expect(asset.materials.length).toBe(15);

    let totalVerts = 0;
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        totalVerts += prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
      }
    }
    expect(totalVerts).toBeGreaterThan(1000000);
  });

  it('vertex normals are unit length', () => {
    const glb = generateGlb({
      meshCount: 5, avgVerticesPerMesh: 100, avgTrianglesPerMesh: 30,
      materialCount: 2, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        const stride = prim.vertexLayout.arrayStride / 4;
        for (let v = 0; v < prim.vertices.length; v += stride) {
          const nx = prim.vertices[v + 3]!;
          const ny = prim.vertices[v + 4]!;
          const nz = prim.vertices[v + 5]!;
          const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
          expect(len).toBeCloseTo(1.0, 5);
        }
      }
    }
  });

  it('indices reference valid vertex positions', () => {
    const glb = generateGlb({
      meshCount: 10, avgVerticesPerMesh: 50, avgTrianglesPerMesh: 10,
      materialCount: 2, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        const vertexCount = prim.vertices.length / (prim.vertexLayout.arrayStride / 4);
        for (let i = 0; i < prim.indices.length; i++) {
          expect(prim.indices[i]).toBeLessThan(vertexCount);
          expect(prim.indices[i]).toBeGreaterThanOrEqual(0);
        }
      }
    }
  });

  it('materials have valid PBR properties', () => {
    const glb = generateGlb({
      meshCount: 5, avgVerticesPerMesh: 10, avgTrianglesPerMesh: 3,
      materialCount: 5, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    expect(asset.materials.length).toBe(5);
    for (const mat of asset.materials) {
      expect(mat.baseColorFactor.length).toBe(4);
      expect(mat.baseColorFactor[3]).toBe(1.0);
      expect(mat.metallicFactor).toBeGreaterThanOrEqual(0);
      expect(mat.metallicFactor).toBeLessThanOrEqual(1);
      expect(mat.roughnessFactor).toBeGreaterThanOrEqual(0);
      expect(mat.roughnessFactor).toBeLessThanOrEqual(1);
    }
  });

  it('flattenScene produces correct count for flat model', () => {
    const glb = generateGlb({
      meshCount: 20, avgVerticesPerMesh: 100, avgTrianglesPerMesh: 30,
      materialCount: 3, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);
    const flat = flattenScene(asset);

    expect(flat.length).toBe(20);
    for (const node of flat) {
      expect(node.meshIndex).toBeDefined();
      expect(node.worldMatrix).toBeDefined();
      expect(node.worldMatrix.length).toBe(16);
    }
  });

  it('flattenScene produces correct count for hierarchical model', () => {
    const glb = generateGlb({
      meshCount: 20, avgVerticesPerMesh: 100, avgTrianglesPerMesh: 30,
      materialCount: 3, hierarchyDepth: 3,
    });
    const asset = parseGltf(glb);
    const flat = flattenScene(asset);

    // All 20 meshes should still be present after flattening
    expect(flat.length).toBe(20);
    for (const node of flat) {
      expect(node.meshIndex).toBeDefined();
      expect(node.worldMatrix).toBeDefined();
    }
  });

  it('hierarchy transforms accumulate through flattenScene', () => {
    const glb = generateGlb({
      meshCount: 10, avgVerticesPerMesh: 10, avgTrianglesPerMesh: 3,
      materialCount: 1, hierarchyDepth: 3,
    });
    const asset = parseGltf(glb);
    const flat = flattenScene(asset);

    // With hierarchy depth 3, at least some nodes should have non-identity world matrices
    let hasNonIdentity = false;
    for (const node of flat) {
      const m = node.worldMatrix;
      const isIdentity = m[0] === 1 && m[5] === 1 && m[10] === 1 && m[15] === 1
        && m[1] === 0 && m[2] === 0 && m[3] === 0
        && m[4] === 0 && m[6] === 0 && m[7] === 0
        && m[8] === 0 && m[9] === 0 && m[11] === 0
        && m[12] === 0 && m[13] === 0 && m[14] === 0;
      if (!isIdentity) hasNonIdentity = true;
    }
    expect(hasNonIdentity).toBe(true);
  });

  it('handles single mesh, single material model', () => {
    const glb = generateGlb({
      meshCount: 1, avgVerticesPerMesh: 100, avgTrianglesPerMesh: 30,
      materialCount: 1, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    expect(asset.meshes.length).toBe(1);
    expect(asset.materials.length).toBe(1);
    expect(asset.meshes[0]!.primitives.length).toBe(1);
    expect(asset.meshes[0]!.primitives[0]!.materialIndex).toBe(0);
  });

  it('handles 1000 materials', () => {
    const glb = generateGlb({
      meshCount: 100, avgVerticesPerMesh: 10, avgTrianglesPerMesh: 3,
      materialCount: 1000, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    expect(asset.materials.length).toBe(1000);
    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        expect(prim.materialIndex).toBeGreaterThanOrEqual(0);
        expect(prim.materialIndex).toBeLessThan(1000);
      }
    }
  });

  it('index format is uint16 for models with < 65536 vertices per mesh', () => {
    const glb = generateGlb({
      meshCount: 10, avgVerticesPerMesh: 1000, avgTrianglesPerMesh: 300,
      materialCount: 3, hierarchyDepth: 0,
    });
    const asset = parseGltf(glb);

    for (const mesh of asset.meshes) {
      for (const prim of mesh.primitives) {
        expect(prim.indexFormat).toBe('uint16');
      }
    }
  });

  it('parse time scales linearly with model size', () => {
    const sizes = [10, 50, 200];
    const times: number[] = [];

    for (const meshCount of sizes) {
      const glb = generateGlb({
        meshCount, avgVerticesPerMesh: 1000, avgTrianglesPerMesh: 300,
        materialCount: 5, hierarchyDepth: 2,
      });
      // Warmup
      for (let w = 0; w < 3; w++) parseGltf(glb);

      const t0 = performance.now();
      for (let s = 0; s < 10; s++) parseGltf(glb);
      times.push((performance.now() - t0) / 10);
    }

    const ratio1 = times[1]! / times[0]!;
    const ratio2 = times[2]! / times[1]!;
    expect(ratio1).toBeGreaterThan(2);
    expect(ratio1).toBeLessThan(15);
    expect(ratio2).toBeGreaterThan(2);
    expect(ratio2).toBeLessThan(15);
  });
});
