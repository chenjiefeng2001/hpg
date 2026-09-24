import { describe, it, expect } from 'vitest';
import { parseGltf, flattenScene } from '../src/core/gltf';

// ─── GLB Generator (test helper) ────────────────────────────

function createTestGlb(): ArrayBuffer {
  // Simple scene: 1 mesh, 1 primitive, 3 vertices (triangle)
  // Attributes: POSITION (float32x3), NORMAL (float32x3)

  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const normals = new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]);
  const indices = new Uint16Array([0, 1, 2]);

  // Build BIN chunk — allocate generous space for padding
  const maxBinSize = positions.byteLength + normals.byteLength + indices.byteLength + 16;
  const binData = new Uint8Array(maxBinSize);
  const binView = new DataView(binData.buffer);
  let binOffset = 0;

  // Copy positions
  binData.set(new Uint8Array(positions.buffer), binOffset);
  const posBufferViewIdx = 0;
  const posAccessorIdx = 0;
  binOffset += positions.byteLength;

  // Pad indices to 4-byte boundary
  while (binOffset % 4 !== 0) binOffset++;
  const idxStart = binOffset;
  binData.set(new Uint8Array(indices.buffer), binOffset);
  const idxBufferViewIdx = 1;
  const idxAccessorIdx = 1;
  binOffset += indices.byteLength;

  // Copy normals
  while (binOffset % 4 !== 0) binOffset++;
  binData.set(new Uint8Array(normals.buffer), binOffset);
  const normBufferViewIdx = 2;
  const normAccessorIdx = 2;

  // Trim binData to actual used size
  const actualBinLength = binOffset + normals.byteLength;
  const trimmedBin = binData.slice(0, actualBinLength);

  // Build glTF JSON
  const gltfJson = {
    asset: { version: '2.0', generator: 'hpg-test' },
    scene: 0,
    scenes: [{ name: 'Test Scene', nodes: [0] }],
    nodes: [{ name: 'Triangle', mesh: 0 }],
    meshes: [{
      name: 'TriangleMesh',
      primitives: [{
        attributes: {
          POSITION: posAccessorIdx,
          NORMAL: normAccessorIdx,
        },
        indices: idxAccessorIdx,
        material: 0,
      }],
    }],
    materials: [{
      name: 'DefaultMaterial',
      pbrMetallicRoughness: {
        baseColorFactor: [1, 0, 0, 1] as [number, number, number, number],
        metallicFactor: 0.5,
        roughnessFactor: 0.8,
      },
    }],
    accessors: [
      {
        bufferView: posBufferViewIdx,
        componentType: 5126, // FLOAT
        count: 3,
        type: 'VEC3',
        min: [0, 0, 0],
        max: [1, 1, 0],
      },
      {
        bufferView: idxBufferViewIdx,
        componentType: 5123, // UNSIGNED_SHORT
        count: 3,
        type: 'SCALAR',
      },
      {
        bufferView: normBufferViewIdx,
        componentType: 5126, // FLOAT
        count: 3,
        type: 'VEC3',
        min: [0, 0, 1],
        max: [0, 0, 1],
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 },
      { buffer: 0, byteOffset: idxStart, byteLength: indices.byteLength, target: 34963 },
      { buffer: 0, byteOffset: idxStart + indices.byteLength + (4 - (indices.byteLength % 4)) % 4, byteLength: normals.byteLength, target: 34962 },
    ],
    buffers: [{ byteLength: trimmedBin.byteLength }],
  };

  // Build GLB
  const jsonStr = JSON.stringify(gltfJson);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  // Pad JSON to 4-byte boundary with spaces (per glTF spec)
  const jsonPadding = (4 - (jsonBytes.byteLength % 4)) % 4;
  const jsonChunkLength = jsonBytes.byteLength + jsonPadding;
  // Pad BIN to 4-byte boundary
  const binPadding = (4 - (trimmedBin.byteLength % 4)) % 4;
  const binChunkLength = trimmedBin.byteLength + binPadding;

  const totalLength = 12 + 8 + jsonChunkLength + 8 + binChunkLength;
  const glb = new ArrayBuffer(totalLength);
  const glbView = new DataView(glb);

  // Header
  glbView.setUint32(0, 0x46546C67, true); // magic "glTF"
  glbView.setUint32(4, 2, true);           // version 2
  glbView.setUint32(8, totalLength, true);  // total length

  // JSON chunk
  let offset = 12;
  glbView.setUint32(offset, jsonChunkLength, true);
  glbView.setUint32(offset + 4, 0x4E4F534A, true); // "JSON"
  const jsonDst = new Uint8Array(glb, offset + 8, jsonChunkLength);
  jsonDst.set(jsonBytes);
  // Pad with spaces (per glTF spec: 0x20)
  for (let i = jsonBytes.byteLength; i < jsonChunkLength; i++) {
    jsonDst[i] = 0x20;
  }
  offset += 8 + jsonChunkLength;

  // BIN chunk
  glbView.setUint32(offset, binChunkLength, true);
  glbView.setUint32(offset + 4, 0x004E4942, true); // "BIN\0"
  new Uint8Array(glb, offset + 8, binChunkLength).set(trimmedBin);

  return glb;
}

// ─── Tests ──────────────────────────────────────────────────

describe('glTF Loader', () => {
  it('parses GLB header', () => {
    const glb = createTestGlb();
    const asset = parseGltf(glb);

    expect(asset.meshes.length).toBe(1);
    expect(asset.materials.length).toBe(1);
    expect(asset.nodes.length).toBe(1);
  });

  it('extracts mesh primitives with correct vertex layout', () => {
    const glb = createTestGlb();
    const asset = parseGltf(glb);
    const mesh = asset.meshes[0]!;

    expect(mesh.name).toBe('TriangleMesh');
    expect(mesh.primitives.length).toBe(1);

    const prim = mesh.primitives[0]!;
    // interleaved: pos(3) + normal(3) = 6 floats = 24 bytes
    expect(prim.vertexLayout.arrayStride).toBe(24);
    expect(prim.vertexLayout.attributes.length).toBe(2);
    expect(prim.vertexLayout.attributes[0]!.shaderLocation).toBe(0); // POSITION
    expect(prim.vertexLayout.attributes[1]!.shaderLocation).toBe(1); // NORMAL
    expect(prim.indexFormat).toBe('uint16');
    expect(prim.indices.length).toBe(3);
  });

  it('extracts vertex positions', () => {
    const glb = createTestGlb();
    const asset = parseGltf(glb);
    const prim = asset.meshes[0]!.primitives[0]!;

    // Interleaved: [px, py, pz, nx, ny, nz, ...]
    const FLOATS_PER_VERT = 6; // 3 pos + 3 normal
    expect(prim.vertices.length).toBe(3 * FLOATS_PER_VERT);
    expect(prim.vertices[0]).toBe(0); // first vertex x
    expect(prim.vertices[1]).toBe(0); // first vertex y
    expect(prim.vertices[2]).toBe(0); // first vertex z
  });

  it('extracts material with PBR factors', () => {
    const glb = createTestGlb();
    const asset = parseGltf(glb);
    const mat = asset.materials[0]!;

    expect(mat.name).toBe('DefaultMaterial');
    expect(mat.baseColorFactor).toEqual([1, 0, 0, 1]);
    expect(mat.metallicFactor).toBe(0.5);
    expect(mat.roughnessFactor).toBe(0.8);
  });

  it('flattens scene tree with world matrices', () => {
    const glb = createTestGlb();
    const asset = parseGltf(glb);
    const flat = flattenScene(asset);

    expect(flat.length).toBe(1);
    expect(flat[0]!.meshIndex).toBe(0);
    // Identity matrix (no transform on node)
    expect(flat[0]!.worldMatrix[0]).toBe(1);
    expect(flat[0]!.worldMatrix[5]).toBe(1);
    expect(flat[0]!.worldMatrix[10]).toBe(1);
    expect(flat[0]!.worldMatrix[15]).toBe(1);
  });

  it('rejects non-GLB data', () => {
    const bad = new ArrayBuffer(16);
    expect(() => parseGltf(bad)).toThrow('Invalid GLB magic');
  });
});
