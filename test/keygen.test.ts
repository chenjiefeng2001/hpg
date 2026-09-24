import { describe, it, expect } from 'vitest';
import {
  packKey,
  packKeyValue,
  unpackKey,
  countingSortKeys,
  quantizeDepth,
  DEPTH_BITS,
  DEPTH_MASK,
  PIPELINE_BITS,
  BINDGROUP_BITS,
  LAYER_BITS,
  KEY_BITS,
} from '../src/core/keygen';

describe('keygen', () => {
  it('bit budget fits in 53-bit safe integer', () => {
    expect(KEY_BITS).toBeLessThan(53);
    expect(DEPTH_BITS).toBe(17);
    expect(PIPELINE_BITS).toBe(16);
    expect(BINDGROUP_BITS).toBe(12);
    expect(LAYER_BITS).toBe(2);
  });

  it('pack/unpack round-trips all components', () => {
    const { key } = packKey(1, 0x1234, 0xabc, 0.5);
    const u = unpackKey(key);
    expect(u.layer).toBe(1);
    expect(u.pipeline).toBe(0x1234);
    expect(u.bindGroup).toBe(0xabc);
    expect(u.depth).toBe(quantizeDepth(0.5));
  });

  it('packKeyValue 与 packKey 数学一致（零分配热路径版本）', () => {
    for (const [layer, pipeline, bg, depth] of [
      [0, 0, 0, 0],
      [1, 0x1234, 0xabc, 0.5],
      [2, 0xffff, 0xfff, 1],
      [3, 7, 3, -1],
    ] as const) {
      expect(packKeyValue(layer, pipeline, bg, depth)).toBe(packKey(layer, pipeline, bg, depth).key);
    }
  });

  it('quantizeDepth clamps to [0, DEPTH_MASK]', () => {
    expect(quantizeDepth(-1)).toBe(0);
    expect(quantizeDepth(2)).toBe(DEPTH_MASK);
    expect(quantizeDepth(0.5)).toBe(Math.floor(0.5 * DEPTH_MASK));
  });

  it('layer dominates pipeline dominates bindGroup dominates depth', () => {
    const smallLayer = packKey(0, 0xffff, 0xfff, 1).key;
    const bigLayer = packKey(1, 0, 0, 0).key;
    expect(bigLayer).toBeGreaterThan(smallLayer);

    const smallPipeline = packKey(0, 0, 0xfff, 1).key;
    const bigPipeline = packKey(0, 1, 0, 0).key;
    expect(bigPipeline).toBeGreaterThan(smallPipeline);

    const smallBG = packKey(0, 0, 0, 1).key;
    const bigBG = packKey(0, 0, 1, 0).key;
    expect(bigBG).toBeGreaterThan(smallBG);

    const smallDepth = packKey(0, 0, 0, 0).key;
    const bigDepth = packKey(0, 0, 0, 1).key;
    expect(bigDepth).toBeGreaterThan(smallDepth);
  });

  it('countingSortKeys is stable and ascending', () => {
    const keys = [3, 1, 2, 1, 3, 0];
    const idx = [0, 1, 2, 3, 4, 5];
    const scratch = new Array(6);
    const out = countingSortKeys(idx, keys, scratch);
    const sortedKeys = out.map((i) => keys[i]);
    expect(sortedKeys).toEqual([0, 1, 1, 2, 3, 3]);
    // 稳定：值相等时保持原始相对顺序。
    expect(out).toEqual([5, 1, 3, 2, 0, 4]);
  });

  it('countingSortKeys handles large keys and single element', () => {
    const keys = [DEPTH_MASK, 0, Math.floor(DEPTH_MASK / 2)];
    const idx = [0, 1, 2];
    const out = countingSortKeys(idx, keys, new Array(3));
    expect(out).toEqual([1, 2, 0]);
    expect(countingSortKeys([0], [5], new Array(1))).toEqual([0]);
  });

  it('sorting respects key priority (layer before pipeline before depth)', () => {
    // 构造混合输入，验证排序后按 layer → pipeline → depth。
    const items = [
      packKey(1, 5, 0, 0).key,
      packKey(0, 9, 0, 1).key,
      packKey(0, 3, 0, 0.2).key,
      packKey(1, 1, 0, 0.9).key,
    ];
    const idx = [0, 1, 2, 3];
    const out = countingSortKeys(idx, items, new Array(4));
    const sorted = out.map((i) => items[i]);
    expect(sorted).toEqual([items[2], items[1], items[3], items[0]]);
  });
});