import { describe, it, expect } from 'vitest';
import {
  identity, multiply, perspective, lookAt, rotationY,
  rotationQuat, fromTranslationRotationScale,
  orthographic, invert, translation, scaling, rotationX, rotationZ,
} from '../src/core/math';

describe('math', () => {
  it('identity returns 16-element identity', () => {
    const m = identity();
    expect(m.length).toBe(16);
    expect(Array.from(m)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  });

  it('perspective has column-major layout (f/aspect at [0])', () => {
    const p = perspective(Math.PI / 2, 2, 0.1, 100);
    expect(p[0]).toBeCloseTo(1 / 2, 6);
    expect(p[5]).toBeCloseTo(1, 6);
    expect(p[11]).toBe(-1);
  });

  it('multiply(C) = A*B satisfies C*v = A*(B*v)', () => {
    const a = rotationY(new Float32Array(16), 0.7);
    const b = rotationY(new Float32Array(16), 1.2);
    const c = multiply(new Float32Array(16), a, b);
    const expected = rotationY(new Float32Array(16), 1.9);
    for (let i = 0; i < 16; i++) {
      expect(c[i]).toBeCloseTo(expected[i] as number, 5);
    }
  });

  it('lookAt at origin looking -Z has identity-ish view (eye at 0, target -Z)', () => {
    const view = lookAt(new Float32Array([0, 0, 0]), new Float32Array([0, 0, -1]), new Float32Array([0, 1, 0]));
    expect(view[10]).toBeCloseTo(1, 5);
    expect(view[14]).toBeCloseTo(0, 5);
  });

  it('lookAt translates by -eye', () => {
    const eye = new Float32Array([1, 2, 3]);
    const view = lookAt(eye, new Float32Array([1, 2, 0]), new Float32Array([0, 1, 0]));
    expect(view[12]).toBeCloseTo(-1, 5);
    expect(view[13]).toBeCloseTo(-2, 5);
    expect(view[14]).toBeCloseTo(-3, 5);
  });

  it('rotationQuat: identity quaternion (0,0,0,1) gives identity rotation', () => {
    const out = new Float32Array(16);
    rotationQuat(out, 0, 0, 0, 1);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[5]).toBeCloseTo(1, 6);
    expect(out[10]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(0, 6);
    expect(out[4]).toBeCloseTo(0, 6);
    expect(out[8]).toBeCloseTo(0, 6);
  });

  it('rotationQuat: 180° around X flips Y and Z', () => {
    // 180° around X: quat = (1, 0, 0, 0)
    const out = new Float32Array(16);
    rotationQuat(out, 1, 0, 0, 0);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[5]).toBeCloseTo(-1, 6);
    expect(out[10]).toBeCloseTo(-1, 6);
  });

  it('rotationQuat: 180° around Y flips X and Z', () => {
    // 180° around Y: quat = (0, 1, 0, 0)
    const out = new Float32Array(16);
    rotationQuat(out, 0, 1, 0, 0);
    expect(out[0]).toBeCloseTo(-1, 6);
    expect(out[5]).toBeCloseTo(1, 6);
    expect(out[10]).toBeCloseTo(-1, 6);
  });

  it('rotationQuat: rotation preserves orthogonality (columns unit length)', () => {
    const out = new Float32Array(16);
    // Use unit quaternion: normalize (0.3, 0.5, 0.1, 0.8)
    const len = Math.hypot(0.3, 0.5, 0.1, 0.8);
    rotationQuat(out, 0.3 / len, 0.5 / len, 0.1 / len, 0.8 / len);
    // Column 0: out[0], out[1], out[2]
    const c0 = Math.hypot(out[0], out[1], out[2]);
    expect(c0).toBeCloseTo(1, 5);
    // Column 1: out[4], out[5], out[6]
    const c1 = Math.hypot(out[4], out[5], out[6]);
    expect(c1).toBeCloseTo(1, 5);
    // Column 2: out[8], out[9], out[10]
    const c2 = Math.hypot(out[8], out[9], out[10]);
    expect(c2).toBeCloseTo(1, 5);
  });

  it('rotationQuat preserves out[3,7,11,15] untouched', () => {
    const out = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    rotationQuat(out, 0, 0, 0, 1);
    expect(out[3]).toBe(4);
    expect(out[7]).toBe(8);
    expect(out[11]).toBe(12);
    expect(out[15]).toBe(16);
  });

  it('fromTranslationRotationScale: identity quat + unit scale = translation matrix', () => {
    const out = new Float32Array(16);
    fromTranslationRotationScale(out, 5, 6, 7, 0, 0, 0, 1, 1, 1, 1);
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[5]).toBeCloseTo(1, 6);
    expect(out[10]).toBeCloseTo(1, 6);
    expect(out[12]).toBeCloseTo(5, 6);
    expect(out[13]).toBeCloseTo(6, 6);
    expect(out[14]).toBeCloseTo(7, 6);
    expect(out[15]).toBeCloseTo(1, 6);
  });

  it('fromTranslationRotationScale: scale non-uniform affects rotation columns', () => {
    const out = new Float32Array(16);
    fromTranslationRotationScale(out, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4);
    expect(out[0]).toBeCloseTo(2, 6);
    expect(out[5]).toBeCloseTo(3, 6);
    expect(out[10]).toBeCloseTo(4, 6);
    expect(out[12]).toBeCloseTo(0, 6);
  });

  it('fromTranslationRotationScale: compose with TRS produces valid matrix', () => {
    const out = new Float32Array(16);
    // 90° around Y via quat, scale 2, translate (10, 0, 0).
    const s45 = Math.SQRT1_2;
    fromTranslationRotationScale(out, 10, 0, 0, 0, s45, 0, s45, 2, 2, 2);

    // Verify rotation columns are orthogonal and scaled.
    const c0 = Math.hypot(out[0], out[1], out[2]);
    expect(c0).toBeCloseTo(2, 4); // scale
    const c1 = Math.hypot(out[4], out[5], out[6]);
    expect(c1).toBeCloseTo(2, 4);
    const c2 = Math.hypot(out[8], out[9], out[10]);
    expect(c2).toBeCloseTo(2, 4);

    // Translation.
    expect(out[12]).toBeCloseTo(10, 6);
    expect(out[15]).toBeCloseTo(1, 6);
  });

  // ──────────────────────────────────────────────
  // Sprint 3A: 数学库扩展测试
  // ──────────────────────────────────────────────

  describe('orthographic', () => {
    it('maps (left,right,bottom,top,near,far) to [-1,1] NDC', () => {
      const out = new Float32Array(16);
      orthographic(out, 0, 800, 600, 0, 0.1, 100);
      // (0,600,0.1) → top-left near → should map to NDC (-1, -1, ~0)
      // (800,0,100) → bottom-right far → should map to NDC (1, 1, ~1)
      expect(out[0]).toBeCloseTo(2 / 800, 6); // 2/(right-left)
      expect(out[5]).toBeCloseTo(2 / -600, 6); // 2/(bottom-top) negative because top<bottom
      expect(out[15]).toBeCloseTo(1, 6);
    });

    it('identity-like for symmetric range [-1,1]', () => {
      const out = new Float32Array(16);
      orthographic(out, -1, 1, -1, 1, 0, 1);
      expect(out[0]).toBeCloseTo(1, 6);
      expect(out[5]).toBeCloseTo(1, 6);
      expect(out[10]).toBeCloseTo(-2, 6); // 2/(near-far) = 2/(0-1) = -2
    });

    it('preserves out[3,7,11] = 0', () => {
      const out = new Float32Array(16);
      orthographic(out, 0, 100, 0, 100, 0.1, 1000);
      expect(out[3]).toBe(0);
      expect(out[7]).toBe(0);
      expect(out[11]).toBe(0);
    });
  });

  describe('invert', () => {
    it('inverts identity → identity', () => {
      const out = new Float32Array(16);
      const ok = invert(out, identity());
      expect(ok).toBe(true);
      for (let i = 0; i < 16; i++) {
        expect(out[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 5);
      }
    });

    it('inverts rotationY(θ) → rotationY(-θ)', () => {
      const m = rotationY(new Float32Array(16), 0.7);
      const out = new Float32Array(16);
      const ok = invert(out, m);
      expect(ok).toBe(true);
      const ref = rotationY(new Float32Array(16), -0.7);
      for (let i = 0; i < 16; i++) {
        expect(out[i]).toBeCloseTo(ref[i] as number, 4);
      }
    });

    it('inverts translation(tx,ty,tz) → translation(-tx,-ty,-tz)', () => {
      const m = translation(new Float32Array(16), 5, 10, 15);
      const out = new Float32Array(16);
      const ok = invert(out, m);
      expect(ok).toBe(true);
      expect(out[12]).toBeCloseTo(-5, 6);
      expect(out[13]).toBeCloseTo(-10, 6);
      expect(out[14]).toBeCloseTo(-15, 6);
    });

    it('inverts scaling(2,3,4) → scaling(0.5,0.333,0.25)', () => {
      const m = scaling(new Float32Array(16), 2, 3, 4);
      const out = new Float32Array(16);
      const ok = invert(out, m);
      expect(ok).toBe(true);
      expect(out[0]).toBeCloseTo(0.5, 6);
      expect(out[5]).toBeCloseTo(1 / 3, 6);
      expect(out[10]).toBeCloseTo(0.25, 6);
    });

    it('returns false for singular matrix', () => {
      const singular = new Float32Array(16); // all zeros
      const out = new Float32Array(16);
      const ok = invert(out, singular);
      expect(ok).toBe(false);
    });

    it('A * A⁻¹ ≈ identity', () => {
      const m = new Float32Array([
        2, 1, 0, 0,
        1, 3, 1, 0,
        0, 1, 4, 1,
        0, 0, 1, 2,
      ]);
      const inv = new Float32Array(16);
      invert(inv, m);
      const product = multiply(new Float32Array(16), m, inv);
      for (let i = 0; i < 16; i++) {
        expect(product[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 4);
      }
    });
  });

  describe('translation', () => {
    it('sets diagonal to 1 and position', () => {
      const out = new Float32Array(16);
      translation(out, 3, 7, 11);
      expect(out[0]).toBe(1);
      expect(out[5]).toBe(1);
      expect(out[10]).toBe(1);
      expect(out[12]).toBe(3);
      expect(out[13]).toBe(7);
      expect(out[14]).toBe(11);
      expect(out[15]).toBe(1);
    });

    it('zeros non-diagonal elements', () => {
      const out = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      translation(out, 1, 2, 3);
      expect(out[1]).toBe(0);
      expect(out[4]).toBe(0);
      expect(out[8]).toBe(0);
    });
  });

  describe('scaling', () => {
    it('sets diagonal to scale factors', () => {
      const out = new Float32Array(16);
      scaling(out, 2, 3, 4);
      expect(out[0]).toBe(2);
      expect(out[5]).toBe(3);
      expect(out[10]).toBe(4);
      expect(out[15]).toBe(1);
    });

    it('zeros non-diagonal elements', () => {
      const out = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
      scaling(out, 2, 3, 4);
      expect(out[1]).toBe(0);
      expect(out[4]).toBe(0);
      expect(out[8]).toBe(0);
      expect(out[12]).toBe(0);
    });
  });

  describe('rotationX', () => {
    it('identity at θ=0', () => {
      const out = new Float32Array(16);
      rotationX(out, 0);
      expect(out[0]).toBeCloseTo(1, 6);
      expect(out[5]).toBeCloseTo(1, 6);
      expect(out[10]).toBeCloseTo(1, 6);
    });

    it('180° flips Y and Z', () => {
      const out = new Float32Array(16);
      rotationX(out, Math.PI);
      expect(out[0]).toBeCloseTo(1, 6);
      expect(out[5]).toBeCloseTo(-1, 6);
      expect(out[10]).toBeCloseTo(-1, 6);
    });

    it('90° rotates Y→Z', () => {
      const out = new Float32Array(16);
      rotationX(out, Math.PI / 2);
      expect(out[5]).toBeCloseTo(0, 5); // cos(90°) ≈ 0
      expect(out[6]).toBeCloseTo(1, 5); // sin(90°) ≈ 1
      expect(out[9]).toBeCloseTo(-1, 5); // -sin(90°) ≈ -1
      expect(out[10]).toBeCloseTo(0, 5);
    });
  });

  describe('rotationZ', () => {
    it('identity at θ=0', () => {
      const out = new Float32Array(16);
      rotationZ(out, 0);
      expect(out[0]).toBeCloseTo(1, 6);
      expect(out[5]).toBeCloseTo(1, 6);
      expect(out[10]).toBeCloseTo(1, 6);
    });

    it('180° flips X and Y', () => {
      const out = new Float32Array(16);
      rotationZ(out, Math.PI);
      expect(out[0]).toBeCloseTo(-1, 6);
      expect(out[5]).toBeCloseTo(-1, 6);
      expect(out[10]).toBeCloseTo(1, 6);
    });

    it('90° rotates X→Y', () => {
      const out = new Float32Array(16);
      rotationZ(out, Math.PI / 2);
      expect(out[0]).toBeCloseTo(0, 5);
      expect(out[1]).toBeCloseTo(1, 5);
      expect(out[4]).toBeCloseTo(-1, 5);
      expect(out[5]).toBeCloseTo(0, 5);
    });

    it('columns are unit length', () => {
      const out = new Float32Array(16);
      rotationZ(out, 1.23);
      expect(Math.hypot(out[0], out[1])).toBeCloseTo(1, 5);
      expect(Math.hypot(out[4], out[5])).toBeCloseTo(1, 5);
    });
  });
});
