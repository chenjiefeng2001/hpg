import { describe, it, expect } from 'vitest';
import { GeometryArena } from '../src/core/geometry';
import type { VertexLayoutDesc } from '../src/types';
import { createFakeGPU } from './fake-gpu';

function fakeDevice(): GPUDevice {
  return {
    createBuffer(desc: GPUBufferDescriptor) {
      return { size: desc.size as number, destroy() {} } as unknown as GPUBuffer;
    },
    queue: {
      submit: () => undefined,
      writeBuffer() {},
    },
  } as unknown as GPUDevice;
}

const LAYOUT: VertexLayoutDesc[] = [
  {
    stepMode: 'vertex',
    arrayStride: 32,
    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
  },
];

describe('GeometryArena', () => {
  it('createGeometry returns valid geometry with vertex slice', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24); // 24 floats = 96 bytes, stride=32 → 3 vertices
    const geo = arena.createGeometry(verts, LAYOUT);
    expect(geo.vertexBuffer).toBeDefined();
    expect(geo.vertexCount).toBe(3);
    expect(geo.vertexSlice.byteLength).toBe(verts.byteLength);
    expect(geo.primitive).toBe('triangle-list');
  });

  it('非 float32x3 position 不推导错误的 AABB', () => {
    const arena = new GeometryArena(fakeDevice());
    const layout: VertexLayoutDesc[] = [{
      arrayStride: 8,
      stepMode: 'vertex',
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' as GPUVertexFormat }],
    }];
    const geometry = arena.createGeometry(new Float32Array([0, 0, 1, 0, 0, 1]), layout);
    expect(geometry.bounds).toBeUndefined();
  });

  it('createGeometry with index data returns indexed geometry', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24);
    const indices = new Uint16Array([0, 1, 2]);
    const geo = arena.createGeometry(verts, LAYOUT, indices);
    expect(geo.indexBuffer).toBeDefined();
    expect(geo.indexCount).toBe(3);
    expect(geo.indexFormat).toBe('uint16');
    expect(geo.indexSlice!.byteLength).toBeGreaterThanOrEqual(indices.byteLength);
    // writeBuffer requires 4-byte alignment, so index slice may be padded
    expect(geo.indexSlice!.byteLength % 4).toBe(0);
  });

  it('重复 create/destroy 保持 live geometry 为零且复用同一 pool', () => {
    const arena = new GeometryArena(fakeDevice());
    const layout: VertexLayoutDesc[] = [{
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
    }];
    const vertices = new Float32Array(18);
    for (let i = 0; i < 100; i++) {
      const geometry = arena.createGeometry(vertices, layout);
      arena.destroyGeometry(geometry);
    }
    const stats = arena.stats();
    expect(stats.geometries).toBe(0);
    expect(stats.vertexPools).toBe(1);
    expect(stats.vertexFreeBytes).toBe(80);
    expect(stats.vertexUsedBytes).toBe(80);
    arena.dispose();
  });

  it('回收非 16 对齐的 vertex payload 时保留 allocation padding', () => {
    const arena = new GeometryArena(fakeDevice());
    const layout: VertexLayoutDesc[] = [{
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }],
    }];
     const vertices = new Float32Array(18);
     const first = arena.createGeometry(vertices, layout);
    const offset = first.vertexSlice.byteOffset;
    arena.destroyGeometry(first);
    const second = arena.createGeometry(vertices, layout);
    expect(second.vertexSlice.byteOffset).toBe(offset);
  });

  it('destroyGeometry reclaims vertex slice for reuse', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(16);
    const geo = arena.createGeometry(verts, LAYOUT);
    const vBuf = geo.vertexBuffer;

    arena.destroyGeometry(geo);

    const geo2 = arena.createGeometry(verts, LAYOUT);
    expect(geo2.vertexBuffer).toBe(vBuf);
  });

  it('回收后精确匹配复用：返回非负且正确的 byteOffset（不得回退到块首之前）', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(16);
    const g0 = arena.createGeometry(verts, LAYOUT);
    const offset0 = g0.vertexSlice.byteOffset;
    expect(offset0).toBe(0);

    arena.destroyGeometry(g0);
    const g1 = arena.createGeometry(verts, LAYOUT); // 与释放块完全等长

    expect(g1.vertexSlice.byteOffset).toBe(offset0);
    expect(g1.vertexSlice.byteOffset).toBeGreaterThanOrEqual(0);

    // 索引侧同理。
    const idx = new Uint16Array([0, 1, 2]);
    const indexVerts = new Float32Array(24);
    const a0 = arena.createGeometry(indexVerts, LAYOUT, idx);
    const iOffset0 = a0.indexSlice!.byteOffset;
    arena.destroyGeometry(a0);
    const a1 = arena.createGeometry(indexVerts, LAYOUT, idx);
    expect(a1.indexSlice!.byteOffset).toBe(iOffset0);
    expect(a1.indexSlice!.byteOffset).toBeGreaterThanOrEqual(0);
  });

  it('回收后部分匹配复用：块首作为起始，剩余部分留在 free-list', () => {
    const arena = new GeometryArena(fakeDevice());
    const big = new Float32Array(64);   // 256B
    const small = new Float32Array(16); // 64B
    const g0 = arena.createGeometry(big, LAYOUT);
    const g1 = arena.createGeometry(small, LAYOUT);
    expect(g1.vertexSlice.byteOffset).toBe(256);

    arena.destroyGeometry(g0);
    expect(arena.stats().vertexFreeBytes).toBe(256);

    const reused = arena.createGeometry(small, LAYOUT);
    expect(reused.vertexSlice.byteOffset).toBe(0); // 从块首开始
    expect(arena.stats().vertexFreeBytes).toBe(192); // 剩余部分仍在 free-list
  });

  it('多次创建/回收后，存活几何体的字节区间互不重叠', () => {
    const arena = new GeometryArena(fakeDevice());
    const layout: VertexLayoutDesc[] = [
      { stepMode: 'vertex', arrayStride: 16, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat }] },
    ];
    const live: { offset: number; length: number }[] = [];
    const created: { entry: (typeof live)[number]; geo: ReturnType<GeometryArena['createGeometry']> }[] = [];

    for (let i = 0; i < 40; i++) {
      // 变长 + 交替释放，制造精确/部分匹配两种复用路径。
      const floats = 4 * ((i % 5) + 1) * 4;
      const geo = arena.createGeometry(new Float32Array(floats), layout);
      const slice = geo.vertexSlice;
      const entry = { offset: slice.byteOffset, length: slice.byteLength };
      live.push(entry);
      created.push({ entry, geo });

      if (i % 3 === 2) {
        const target = created[created.length - 3]!;
        arena.destroyGeometry(target.geo);
        live.splice(live.indexOf(target.entry), 1);
        created.splice(created.length - 3, 1);
      }
    }

    for (const r of live) expect(r.offset).toBeGreaterThanOrEqual(0);
    for (let i = 0; i < live.length; i++) {
      for (let j = i + 1; j < live.length; j++) {
        const a = live[i]!, b = live[j]!;
        const overlap = a.offset < b.offset + b.length && b.offset < a.offset + a.length;
        expect(overlap).toBe(false);
      }
    }
    arena.dispose();
  });

  it('destroyGeometry reclaims index slice for reuse', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24);
    const indices = new Uint16Array([0, 1, 2]);
    const geo = arena.createGeometry(verts, LAYOUT, indices);
    const iBuf = geo.indexBuffer!;

    arena.destroyGeometry(geo);

    const geo2 = arena.createGeometry(verts, LAYOUT, indices);
    expect(geo2.indexBuffer).toBe(iBuf);
  });

  it('pool grows when vertex capacity exhausted', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(16);
    const firstGeo = arena.createGeometry(verts, LAYOUT);
    const firstBuf = firstGeo.vertexBuffer;

    // Oversized data to force growth.
    const bigVerts = new Float32Array(2_500_000); // 10 MiB > 4 MiB
    const bigGeo = arena.createGeometry(bigVerts, LAYOUT);
    expect(bigGeo.vertexBuffer).not.toBe(firstBuf);
  });

  it('pool grows when index capacity exhausted', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24);
    const indices = new Uint16Array([0, 1, 2]);
    const firstGeo = arena.createGeometry(verts, LAYOUT, indices);
    const firstIdxBuf = firstGeo.indexBuffer;

    // Oversized index data to force growth.
    const bigIndices = new Uint16Array(600_000); // ~1.2 MiB > 1 MiB
    const bigGeo = arena.createGeometry(verts, LAYOUT, bigIndices);
    expect(bigGeo.indexBuffer).not.toBe(firstIdxBuf);
  });

  it('Uint32 index format is inferred and mismatches are rejected', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24);
    const geo = arena.createGeometry(verts, LAYOUT, new Uint32Array([0, 1, 2]));
    expect(geo.indexFormat).toBe('uint32');
    expect(() => arena.createGeometry(verts, LAYOUT, new Uint32Array([0, 1, 2]), 'uint16')).toThrow(/does not match/);
    expect(() => arena.createGeometry(verts, LAYOUT, new Uint16Array([0, 1, 3]))).toThrow(/exceeds vertexCount/);
  });

  it('rejects unsupported multi-slot or instance-step layouts', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(24);
    const second = { ...LAYOUT[0]! };
    expect(() => arena.createGeometry(verts, [LAYOUT[0]!, second])).toThrow(/exactly one/);
    expect(() => arena.createGeometry(verts, [{ ...LAYOUT[0]!, stepMode: 'instance' as const }])).toThrow(/vertex-step/);
  });

  it('dispose destroys all pools', () => {
    const arena = new GeometryArena(fakeDevice());
    arena.createGeometry(new Float32Array(16), LAYOUT);
    arena.dispose();
    expect(() => arena.createGeometry(new Float32Array(16), LAYOUT)).toThrow(/dispose/);
  });

  it('destroyGeometry is no-op for unknown geometry', () => {
    const arena = new GeometryArena(fakeDevice());
    const fakeGeo = { vertexSlice: {} } as unknown as import('../src/types').Geometry;
    arena.destroyGeometry(fakeGeo);
  });

  it('concurrent allocations share same pool when capacity permits', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(16);
    const g1 = arena.createGeometry(verts, LAYOUT);
    const g2 = arena.createGeometry(verts, LAYOUT);
    expect(g1.vertexBuffer).toBe(g2.vertexBuffer);
  });

  it('free-list merges adjacent blocks after destroy', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(16);
    const g1 = arena.createGeometry(verts, LAYOUT);
    const g2 = arena.createGeometry(verts, LAYOUT);
    const g3 = arena.createGeometry(verts, LAYOUT);

    arena.destroyGeometry(g2);
    arena.destroyGeometry(g1);
    arena.destroyGeometry(g3);

    const r1 = arena.createGeometry(verts, LAYOUT);
    const r2 = arena.createGeometry(verts, LAYOUT);
    const r3 = arena.createGeometry(verts, LAYOUT);
    expect(r1.vertexBuffer).toBe(r2.vertexBuffer);
    expect(r2.vertexBuffer).toBe(r3.vertexBuffer);
  });

  it('回收切片复用时写回**它所属的池**，不得落到最新池的同名偏移（会覆盖存活几何体）', () => {
    const { device, recorded } = createFakeGPU();
    const arena = new GeometryArena(device);

    const small = new Float32Array(16); // 64B
    const g0 = arena.createGeometry(small, LAYOUT);
    const pool0 = g0.vertexBuffer;

    // 5 MiB > VERTEX_CHUNK(4 MiB) → 触发第二个池，gBig 落在 pool1。
    const big = new Float32Array((5 * 1024 * 1024) / 4);
    const gBig = arena.createGeometry(big, LAYOUT);
    const pool1 = gBig.vertexBuffer;
    expect(pool1).not.toBe(pool0);

    // 释放 pool0 中的切片，再申请一个同样大小的几何体。
    arena.destroyGeometry(g0);
    const writesBefore = recorded.writes.length;
    const g1 = arena.createGeometry(small, LAYOUT);

    // 该切片属于 pool0 → 必须写回 pool0，并把 slice 指向 pool0。
    // 否则 g1 的数据会覆盖 pool1 中 gBig 的顶点区间。
    const vertexWrites = recorded.writes.slice(writesBefore).filter((w) => w.bytes.byteLength === small.byteLength);
    expect(vertexWrites.length).toBeGreaterThan(0);
    for (const w of vertexWrites) expect(w.buffer).toBe(pool0);
    expect(g1.vertexBuffer).toBe(pool0);
    expect(g1.vertexSlice.byteOffset).toBe(g0.vertexSlice.byteOffset);

    // 存活几何体之间不得重叠：g1 的切片不得与 gBig 落在同一个池。
    expect(g1.vertexSlice.buffer).not.toBe(gBig.vertexSlice.buffer);
    arena.dispose();
  });

  it('multiple pools coexist: old pool retained after growth', () => {
    const arena = new GeometryArena(fakeDevice());
    const smallVerts = new Float32Array(16);
    const g1 = arena.createGeometry(smallVerts, LAYOUT);
    const firstBuf = g1.vertexBuffer;

    const bigVerts = new Float32Array(2_500_000);
    const bigGeo = arena.createGeometry(bigVerts, LAYOUT);
    expect(bigGeo.vertexBuffer).not.toBe(firstBuf);

    expect(g1.vertexBuffer).toBe(firstBuf);
  });

  it('stats() reports pool usage and geometry count', () => {
    const arena = new GeometryArena(fakeDevice());
    const before = arena.stats();
    expect(before.geometries).toBe(0);
    expect(before.vertexPools).toBe(0);

    const verts = new Float32Array(64);
    const idx = new Uint16Array(3);
    const g1 = arena.createGeometry(verts, LAYOUT, idx);
    const g2 = arena.createGeometry(verts, LAYOUT);

    const s = arena.stats();
    expect(s.geometries).toBe(2);
    expect(s.vertexPools).toBeGreaterThanOrEqual(1);
    expect(s.vertexUsedBytes).toBeGreaterThan(0);
    expect(s.indexUsedBytes).toBeGreaterThan(0);
    expect(s.vertexFreeBytes).toBe(0);
    expect(s.indexFreeBytes).toBe(0);

    arena.destroyGeometry(g1);
    const after = arena.stats();
    expect(after.geometries).toBe(1);
    // Freed vertex slice should appear in free-list.
    expect(after.vertexFreeBytes).toBeGreaterThan(0);
  });

  it('geometry.vertexBuffers array matches vertexBuffer', () => {
    const arena = new GeometryArena(fakeDevice());
    const verts = new Float32Array(64);
    const geo = arena.createGeometry(verts, LAYOUT);
    expect(geo.vertexBuffers.length).toBe(1);
    expect(geo.vertexBuffers[0]!.buffer).toBe(geo.vertexBuffer);
    expect(geo.vertexBuffers[0]!.byteOffset).toBe(geo.vertexSlice.byteOffset);
  });
});
