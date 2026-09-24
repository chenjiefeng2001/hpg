/**
 * 排序键编码 + 三路计数排序（Radix / Counting Sort）。
 *
 * 键布局（53 位整数，number 可精确表示 ≤ 2^53）：
 *   [ layer:8 | pipeline:16 | bindGroup:12 | depth:17 ]
 * 排序优先级：layer → pipeline（状态切换）→ bindGroup（材质）→ depth（深度）。
 *
 * depth 采用 17 位量化，供排序阶段做 early-z 时按相机近→远（0 → 32767）排序。
 */

/** 每层占 2 位，共 8 层（层级 > 8 则 clamp）。 */
export const LAYER_BITS = 2;
export const LAYER_MASK = (1 << LAYER_BITS) - 1;
/** 最大管线数 = 2^16。 */
export const PIPELINE_BITS = 16;
export const PIPELINE_MASK = (1 << PIPELINE_BITS) - 1;
/** 每管线 bindGroup 最大数 = 2^12。 */
export const BINDGROUP_BITS = 12;
export const BINDGROUP_MASK = (1 << BINDGROUP_BITS) - 1;
/** depth 量化位数。 */
export const DEPTH_BITS = 17;
export const DEPTH_MASK = (1 << DEPTH_BITS) - 1;

export const SHIFT_BINDGROUP = DEPTH_BITS;
export const SHIFT_PIPELINE = SHIFT_BINDGROUP + BINDGROUP_BITS;
export const SHIFT_LAYER = SHIFT_PIPELINE + PIPELINE_BITS;
/** 总位数 = 2+16+12+17 = 47 < 53。 */
export const KEY_BITS = SHIFT_LAYER + LAYER_BITS;

/** 乘法常量（避免 JS 位运算 32 位截断）。 */
const LAYER_MULT = 2 ** SHIFT_LAYER;
const PIPELINE_MULT = 2 ** SHIFT_PIPELINE;
const BINDGROUP_MULT = 2 ** SHIFT_BINDGROUP;

export const DEFAULT_LAYER = 0;
/** 顶层不透明层（透贴物体等）。 */
export const LAYER_OPAQUE = 0;
/** 透明层：depth 反转（远→近）以便混合。 */
export const LAYER_TRANSLUCENT = 1;

/** 将[0,1)世界距离量化为[0, DEPTH_MASK]。 */
export function quantizeDepth(d: number): number {
  if (d <= 0) return 0;
  if (d >= 1) return DEPTH_MASK;
  return (d * DEPTH_MASK) | 0;
}

/** 从 3D 坐标得到默认排序深度（基于相机视线分量的简单启发）。 */
export function spatialKey(cx: number, cy: number, cz: number): number {
  return Math.abs(cx) + Math.abs(cy) + Math.abs(cz);
}

export interface SortKeyResult {
  key: number;
  layer: number;
  quantizedDepth: number;
}

/**
 * 组装 47 位排序键，只返回键值。
 *
 * 与 `packKey` 数学完全一致，但**不产生每个 item 一个的对象** ——
 * submit 的排序键循环是逐实例热路径，用它避免每帧 GC 抖动。
 */
export function packKeyValue(layer: number, pipelineId: number, bindGroupId: number, depth01: number): number {
  return (
    (layer & LAYER_MASK) * LAYER_MULT +
    (pipelineId & PIPELINE_MASK) * PIPELINE_MULT +
    (bindGroupId & BINDGROUP_MASK) * BINDGROUP_MULT +
    quantizeDepth(depth01)
  );
}

/** 组装 47 位排序键。layer/clamp + 量化 depth。 */
export function packKey(
  layer: number,
  pipelineId: number,
  bindGroupId: number,
  depth01: number,
): SortKeyResult {
  return {
    key: packKeyValue(layer, pipelineId, bindGroupId, depth01),
    layer: layer & LAYER_MASK,
    quantizedDepth: quantizeDepth(depth01),
  };
}

/** 从键还原组件（供测试/调试）。 */
export function unpackKey(key: number): { layer: number; pipeline: number; bindGroup: number; depth: number } {
  return {
    layer: Math.floor(key / LAYER_MULT) & LAYER_MASK,
    pipeline: Math.floor(key / PIPELINE_MULT) & PIPELINE_MASK,
    bindGroup: Math.floor(key / BINDGROUP_MULT) & BINDGROUP_MASK,
    depth: key & DEPTH_MASK,
  };
}

/** 向上取整到 8 位。 */
export const nextBlockAlign = (x: number): number => (x + 7) & ~7;

/**
 * 三路计数排序 —— 对 keys 数组的索引做稳定排序（升序）。
 * 采用 8 位分块计数，避免 47 位直桶的开销；原地不分配新数组。
 * @returns 排序后的索引数组（复用传入的 scratch，避免 GC）。
 */
/** 计数排序桶（模块级复用：热路径零分配；本函数不可重入，内部亦无重入调用）。 */
const SORT_BUCKET = new Int32Array(256);

export function countingSortKeys(
  items: number[],
  keys: number[],
  scratch: number[],
): number[] {
  const n = items.length;
  if (n <= 1) return items;

  const blockShift = 8;
  const blockMask = 0xff;
  const bucket = SORT_BUCKET;
  const rounds = Math.ceil(KEY_BITS / blockShift);

  // src/dst 两块缓冲交替；最终总是写回 items，保证调用方可直接复用 items。
  let src = items;
  let dst = scratch;
  let inScratch = false;

  for (let round = 0; round < rounds; round++) {
    const shift = round * blockShift;
    const divisor = 2 ** shift;
    bucket.fill(0);
    for (let i = 0; i < n; i++) {
      const k = keys[src[i] as number] as number;
      bucket[Math.floor(k / divisor) & blockMask]++;
    }
    let total = 0;
    for (let i = 0; i < 256; i++) {
      const c = bucket[i] as number;
      bucket[i] = total;
      total += c;
    }
    for (let i = 0; i < n; i++) {
      const idx = src[i] as number;
      const k = keys[idx] as number;
      const b = Math.floor(k / divisor) & blockMask;
      const pos = bucket[b] as number;
      dst[pos] = idx;
      bucket[b] = pos + 1;
    }
    const tmp = src;
    src = dst;
    dst = tmp;
    inScratch = !inScratch;
  }

  if (inScratch) {
    for (let i = 0; i < n; i++) items[i] = scratch[i] as number;
    return items;
  }
  return items;
}