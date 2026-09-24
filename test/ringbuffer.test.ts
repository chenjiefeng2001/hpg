import { describe, it, expect } from 'vitest';
import { RingBuffer, align16, align256 } from '../src/core/ringbuffer';

interface MockBuffer {
  size: number;
  _data: Uint8Array;
  unmap(): void;
  destroyed: boolean;
}

function fakeDevice(): GPUDevice {
  const buffers: MockBuffer[] = [];
  return {
    createBuffer(desc: GPUBufferDescriptor): GPUBuffer {
      const size = desc.size as number;
      const buf: MockBuffer = {
        size,
        _data: new Uint8Array(size),
        unmap() {},
        destroyed: false,
      };
      buffers.push(buf);
      return buf as unknown as GPUBuffer;
    },
    queue: {
      submit: () => undefined,
      writeBuffer(
        buffer: GPUBuffer,
        offset: number,
        data: ArrayBufferView,
        srcOffset?: number,
        srcLength?: number,
      ) {
        const buf = buffer as unknown as MockBuffer;
        const src = data as ArrayBufferView;
        const s = srcOffset ?? 0;
        const l = srcLength ?? src.byteLength;
        buf._data.set(
          new Uint8Array(src.buffer, src.byteOffset + s, l),
          offset,
        );
      },
    },
  } as unknown as GPUDevice;
}

describe('ringbuffer', () => {
  it('align16 / align256 enforce alignment', () => {
    expect(align16(15)).toBe(16);
    expect(align16(16)).toBe(16);
    expect(align256(255)).toBe(256);
    expect(align256(256)).toBe(256);
  });

  it('alloc returns 16-aligned offsets and advances head', () => {
    const ring = new RingBuffer(fakeDevice(), 1024);
    expect(ring.alloc(16)).toBe(0);
    expect(ring.alloc(80)).toBe(16);
    expect(ring.alloc(64)).toBe(align16(16 + 80));
  });

  it('push writes data and endFrame resets head (zero GC, no reset cost)', () => {
    const ring = new RingBuffer(fakeDevice(), 1024);
    const data = new Float32Array([1, 2, 3, 4]);
    const off = ring.push(data);
    expect(off).toBe(0);
    expect(ring.bytesUsed).toBe(16);

    const off2 = ring.push(new Float32Array([5, 6, 7, 8]));
    expect(off2).toBe(16);

    const stats = ring.endFrame();
    expect(stats.bytesUsed).toBe(32);
    expect(stats.bytesCapacity).toBe(1024);
    expect(ring.bytesUsed).toBe(0); // 归位
  });

  it('grows when capacity exhausted and preserves written data (2x doubling)', () => {
    const ring = new RingBuffer(fakeDevice(), 256);
    ring.push(new Float32Array(60)); // 240 bytes, offset 0
    const stats = ring.endFrame();
    expect(stats.bytesCapacity).toBe(256);

    // 超过容量 → 翻倍。
    const big = new Float32Array(80); // 320 bytes
    ring.push(big);
    expect(ring.bytesCapacity).toBe(512);
  });

  it('write writes data to buffer at correct offset', () => {
    const ring = new RingBuffer(fakeDevice(), 1024);
    const offset = ring.alloc(16);
    const data = new Float32Array([10, 20, 30, 40]);
    const written = ring.write(offset, data);
    expect(written).toBe(16);
    expect(ring.bytesUsed).toBe(16);
  });

  it('write with srcOffset and srcLength writes slice', () => {
    const ring = new RingBuffer(fakeDevice(), 1024);
    const offset = ring.alloc(8);
    const data = new Float32Array([1, 2, 3, 4]);
    const written = ring.write(offset, data, 2, 8); // write bytes 2..10 (2 floats)
    expect(written).toBe(8);
  });

  it('endFrame resets head to 0 and returns correct stats', () => {
    const ring = new RingBuffer(fakeDevice(), 2048);
    ring.push(new Float32Array(16)); // 64 bytes
    ring.push(new Float32Array(32)); // 128 bytes

    const stats = ring.endFrame();
    expect(stats.bytesUsed).toBe(192);
    expect(stats.bytesCapacity).toBe(2048);
    expect(stats.allocations).toBe(0); // no growth yet

    expect(ring.bytesUsed).toBe(0);

    // After reset, can allocate again from offset 0.
    const offset = ring.alloc(16);
    expect(offset).toBe(0);
  });

  it('allocations counter increments on grow', () => {
    const ring = new RingBuffer(fakeDevice(), 128);
    ring.push(new Float32Array(32)); // fits

    const big = new Float32Array(100); // 400 bytes > 128 → grow (may double multiple times)
    ring.push(big);
    expect(ring.allocations).toBeGreaterThanOrEqual(1);
  });

  it('扩容重试后仍保持请求的对齐（实例区必须 256 对齐）', () => {
    const ring = new RingBuffer(fakeDevice(), 256);
    // 占满一部分：head = 240（非 256 对齐）。
    expect(ring.alloc(240)).toBe(0);
    // 256 对齐后的偏移 256 + 32 > 256 → 触发扩容；返回的偏移必须仍是 256 对齐。
    const offset = ring.alloc(32, 256);
    expect(offset).toBe(256);
    expect(offset % 256).toBe(0);
    expect(ring.bytesCapacity).toBe(512);
  });

  it('dispose does not throw', () => {
    const ring = new RingBuffer(fakeDevice(), 1024);
    ring.dispose();
    // Should not throw on double dispose.
    ring.dispose();
  });
});
