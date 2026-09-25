/**
 * glTF 解析健壮性回归测试。
 *
 * 覆盖三类真实文件导致的失败：
 *   1. 非 4 字节对齐的 UINT 索引 accessor（TypedArray 构造函数抛异常）
 *   2. 归一化整数属性（KHR_mesh_quantization / 压缩导出工具）
 *   3. 交错（byteStride）属性
 * 以及 benchmark/assets 下真实 Khronos 模型的解析冒烟测试。
 */
import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseGltf } from '../src/core/gltf';

// ─── 最小 GLB 构造器（可控制对齐 / 交错 / componentType）─────

interface BinSection {
  data: Uint8Array;
  byteOffset: number; // 在 BIN chunk 中的偏移（可人为制造非对齐）
  byteStride?: number;
  target?: number;
}

function buildGlb(opts: {
  sections: BinSection[];
  accessors: unknown[];
  bufferViews: unknown[];
  primitives: unknown[];
  materials?: unknown[];
  extensionsRequired?: string[];
  binFirst?: boolean;
}): ArrayBuffer {
  let binLength = 0;
  for (const s of opts.sections) binLength = Math.max(binLength, s.byteOffset + s.data.byteLength);
  binLength = (binLength + 3) & ~3;
  const bin = new Uint8Array(binLength);
  for (const s of opts.sections) bin.set(s.data, s.byteOffset);

  const json = {
    asset: { version: '2.0', generator: 'hpg-test' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: opts.primitives }],
    accessors: opts.accessors,
    bufferViews: opts.bufferViews,
    buffers: [{ byteLength: binLength }],
    ...(opts.materials ? { materials: opts.materials } : {}),
    ...(opts.extensionsRequired ? { extensionsRequired: opts.extensionsRequired } : {}),
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonBytes.byteLength % 4)) % 4;
  const jsonChunkLen = jsonBytes.byteLength + jsonPad;
  const total = 12 + 8 + jsonChunkLen + 8 + binLength;

  const glb = new ArrayBuffer(total);
  const dv = new DataView(glb);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  const chunks = opts.binFirst
    ? [
        { length: binLength, type: 0x004e4942, bytes: bin, pad: 0 },
        { length: jsonChunkLen, type: 0x4e4f534a, bytes: jsonBytes, pad: 0x20 },
      ]
    : [
        { length: jsonChunkLen, type: 0x4e4f534a, bytes: jsonBytes, pad: 0x20 },
        { length: binLength, type: 0x004e4942, bytes: bin, pad: 0 },
      ];
  let offset = 12;
  for (const chunk of chunks) {
    dv.setUint32(offset, chunk.length, true);
    dv.setUint32(offset + 4, chunk.type, true);
    const dst = new Uint8Array(glb, offset + 8, chunk.length);
    dst.set(chunk.bytes);
    for (let i = chunk.bytes.byteLength; i < chunk.length; i++) dst[i] = chunk.pad;
    offset += 8 + chunk.length;
  }
  return glb;
}

describe('glTF 解析健壮性', () => {
  it('接受非 4 字节对齐的 UINT32 索引 accessor（复制到对齐缓冲）', () => {
    // 顶点：3 × float32x3 = 36 字节；索引故意放在 byteOffset = 34（非 4 对齐）
    const verts = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indexOffset = 34;
    const indices = new Uint32Array([0, 1, 2]);
    const bin = new Uint8Array(indexOffset + 12);
    bin.set(new Uint8Array(verts.buffer), 0);
    bin.set(new Uint8Array(indices.buffer), indexOffset);

    const glb = buildGlb({
      sections: [{ data: bin, byteOffset: 0 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: verts.byteLength, target: 34962 },
        { buffer: 0, byteOffset: indexOffset, byteLength: 12, target: 34963 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5125, count: 3, type: 'SCALAR' },
      ],
      primitives: [{ attributes: { POSITION: 0 }, indices: 1 }],
    });

    const asset = parseGltf(glb);
    const prim = asset.meshes[0]!.primitives[0]!;
    expect(prim.indexFormat).toBe('uint32');
    expect(Array.from(prim.indices)).toEqual([0, 1, 2]);
  });

  it('接受带 byteStride 的索引 accessor 并按步长读取', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indexData = new Uint16Array([0, 0, 1, 0, 2, 0]);
    const bin = new Uint8Array(positions.byteLength + indexData.byteLength);
    bin.set(new Uint8Array(positions.buffer), 0);
    bin.set(new Uint8Array(indexData.buffer), positions.byteLength);
    const glb = buildGlb({
      sections: [{ data: bin, byteOffset: 0 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 },
        { buffer: 0, byteOffset: positions.byteLength, byteLength: indexData.byteLength, byteStride: 4, target: 34963 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
      ],
      primitives: [{ attributes: { POSITION: 0 }, indices: 1 }],
    });
    expect(Array.from(parseGltf(glb).meshes[0]!.primitives[0]!.indices)).toEqual([0, 1, 2]);
  });

  it('归一化整数属性被正确解码（1.0 表示最大值）', () => {
    // POSITION: float32x3；TEXCOORD_0: normalized uint8x2 → 0 / 255
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const uvOffset = positions.byteLength;
    const uvs = new Uint8Array([0, 255, 128, 64, 255, 0]);

    const bin = new Uint8Array(uvOffset + uvs.byteLength);
    bin.set(new Uint8Array(positions.buffer), 0);
    bin.set(uvs, uvOffset);

    const glb = buildGlb({
      sections: [{ data: bin, byteOffset: 0 }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 },
        { buffer: 0, byteOffset: uvOffset, byteLength: uvs.byteLength, target: 34962 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 1, componentType: 5121, count: 3, type: 'VEC2', normalized: true },
      ],
      primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: undefined }],
    });

    const asset = parseGltf(glb);
    const prim = asset.meshes[0]!.primitives[0]!;
    // 交错布局：pos(3) + uv(2) = 5 floats/顶点
    expect(prim.vertexLayout.arrayStride).toBe(20);
    expect(prim.vertices[3]).toBeCloseTo(0, 6);
    expect(prim.vertices[4]).toBeCloseTo(1, 6);
    expect(prim.vertices[8]).toBeCloseTo(128 / 255, 6);
    expect(prim.vertices[9]).toBeCloseTo(64 / 255, 6);
  });

  it('处理交错（byteStride）顶点缓冲', () => {
    // [px, py, pz, nx, ny, nz] 交错，stride = 24
    const interleaved = new Float32Array([
      0, 0, 0, 0, 0, 1,
      1, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 1, 0,
    ]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(interleaved.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: interleaved.byteLength, byteStride: 24, target: 34962 }],
      accessors: [
        { bufferView: 0, byteOffset: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { bufferView: 0, byteOffset: 12, componentType: 5126, count: 3, type: 'VEC3' },
      ],
      primitives: [{ attributes: { POSITION: 0, NORMAL: 1 } }],
    });

    const asset = parseGltf(glb);
    const prim = asset.meshes[0]!.primitives[0]!;
    expect(prim.vertexLayout.arrayStride).toBe(24);
    // 顶点 1：pos = (1,0,0)，normal = (1,0,0)
    expect(Array.from(prim.vertices.slice(6, 9))).toEqual([1, 0, 0]);
    expect(Array.from(prim.vertices.slice(9, 12))).toEqual([1, 0, 0]);
  });

  it('无索引 primitive 生成 uint32 顺序索引（顶点数超过 uint16 范围）', () => {
    const count = 70002;
    const positions = new Float32Array(count * 3);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 } }],
    });

    const asset = parseGltf(glb);
    const prim = asset.meshes[0]!.primitives[0]!;
    expect(prim.indexFormat).toBe('uint32');
    expect(prim.indices.length).toBe(count);
  });

  it('跳过非三角形拓扑（mode != 4）而不是产出错误几何', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [
        { attributes: { POSITION: 0 }, mode: 0 }, // POINTS
        { attributes: { POSITION: 0 }, mode: 4 }, // TRIANGLES
      ],
    });

    const asset = parseGltf(glb);
    expect(asset.meshes[0]!.primitives.length).toBe(1);
  });

  it('缺少 POSITION 的 primitive 被跳过', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [
        { attributes: { NORMAL: 0 } },
        { attributes: { POSITION: 0 } },
      ],
    });

    const asset = parseGltf(glb);
    expect(asset.meshes[0]!.primitives.length).toBe(1);
  });

  it('primitive 未指定 material 时保留默认材质语义', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 } }, { attributes: { POSITION: 0 }, material: 0 }],
      materials: [{ pbrMetallicRoughness: { baseColorFactor: [1, 0, 0, 1] } }],
    });
    const asset = parseGltf(glb);
    expect(asset.meshes[0]!.primitives[0]!.materialIndex).toBeUndefined();
    expect(asset.meshes[0]!.primitives[1]!.materialIndex).toBe(0);
  });

  it('截断 accessor、稀疏 accessor 和错误索引范围显式拒绝', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const truncated = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength - 4, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 } }],
    });
    expect(() => parseGltf(truncated)).toThrow(/bufferView/);

    const invalidIndex = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 }, indices: 9 }],
    });
    expect(() => parseGltf(invalidIndex)).toThrow(/index accessor/);

    const sparse = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', sparse: {} }],
      primitives: [{ attributes: { POSITION: 0 } }],
    });
    expect(() => parseGltf(sparse)).toThrow(/Sparse/);
  });

  it('不被渲染路径引用的 sparse accessor 不会阻断 GLB', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [
        { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
        { componentType: 5126, count: 1, type: 'SCALAR', sparse: {} },
      ],
      primitives: [{ attributes: { POSITION: 0 } }],
    });
    expect(() => parseGltf(glb)).not.toThrow();
  });

  it('压缩扩展（Draco）给出可读错误而非静默乱码', () => {
    const positions = new Float32Array([0, 0, 0]);
    const glb = buildGlb({
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 1, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 } }],
      extensionsRequired: ['KHR_draco_mesh_compression'],
    });

    expect(() => parseGltf(glb)).toThrow(/KHR_draco_mesh_compression/);
  });

  it('拒绝 BIN chunk 位于 JSON chunk 之前', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const glb = buildGlb({
      binFirst: true,
      sections: [{ data: new Uint8Array(positions.buffer), byteOffset: 0 }],
      bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength, target: 34962 }],
      accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
      primitives: [{ attributes: { POSITION: 0 } }],
    });
    expect(() => parseGltf(glb)).toThrow(/JSON chunk must be first/);
  });

  it('文本 .gltf / 非 GLB 输入给出可读错误', () => {
    const text = new TextEncoder().encode('{"asset":{"version":"2.0"}}');
    const buf = new Uint8Array(64);
    buf.set(text, 0);
    expect(() => parseGltf(buf.buffer as ArrayBuffer)).toThrow(/Invalid GLB magic/);
  });
});

// ─── 真实模型语料 ───────────────────────────────────────────

const MODEL_ROOT = resolve(__dirname, '..', 'benchmark', 'assets', 'models');

function listModels(): string[] {
  if (!existsSync(MODEL_ROOT)) return [];
  const out: string[] = [];
  for (const dir of readdirSync(MODEL_ROOT)) {
    const full = resolve(MODEL_ROOT, dir);
    for (const file of readdirSync(full)) {
      if (file.endsWith('.glb')) out.push(resolve(full, file));
    }
  }
  return out;
}

describe('真实 GLB 语料解析', () => {
  const models = listModels();

  it('至少存在一个测试模型', () => {
    expect(models.length).toBeGreaterThan(0);
  });

  it('全部模型都能解析出有限数值的顶点（含此前失败的 MultiUVTest）', () => {
    const failures: string[] = [];
    for (const file of models) {
      const data = readFileSync(file);
      const ab = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer;
      try {
        const asset = parseGltf(ab);
        expect(asset.meshes.length).toBeGreaterThanOrEqual(0);
        for (const mesh of asset.meshes) {
          for (const prim of mesh.primitives) {
            for (const v of prim.vertices) {
              if (!Number.isFinite(v)) throw new Error(`non-finite vertex in ${file}`);
            }
            for (const i of prim.indices) {
              if (!Number.isInteger(i) || i < 0) throw new Error(`invalid index in ${file}`);
            }
          }
        }
      } catch (e) {
        failures.push(`${file.replace(MODEL_ROOT, '')}: ${(e as Error).message}`);
      }
    }
    expect(failures).toEqual([]);
  });
});
