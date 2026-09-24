/**
 * WebGPU 全局 polyfill —— 集中定义 Node.js 测试环境缺少的 GPU 常量与类型。
 * 仅影响 test/ 下的 vitest 运行，不污染生产代码。
 */

const g = globalThis as Record<string, unknown>;

/* ------------------------------------------------------------------ */
/* GPUBufferUsage 位掩码（与 WebGPU 规范对齐）                         */
/* ------------------------------------------------------------------ */
if (typeof g.GPUBufferUsage === 'undefined') {
  g.GPUBufferUsage = {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    INDEX: 0x04,
    INDIRECT: 0x08,
    MAP_READ: 0x10,
    MAP_WRITE: 0x20,
    QUERY_RESOLVE: 0x40,
    STORAGE: 0x80,
    UNIFORM: 0x100,
    VERTEX: 0x200,
  };
}

/* ------------------------------------------------------------------ */
/* GPUColorWrite 位掩码                                                */
/* ------------------------------------------------------------------ */
if (typeof g.GPUColorWrite === 'undefined') {
  g.GPUColorWrite = {
    RED: 0x1,
    GREEN: 0x2,
    BLUE: 0x4,
    ALPHA: 0x8,
    ALL: 0xf,
  };
}

/* ------------------------------------------------------------------ */
/* GPUMapMode 位掩码                                                   */
/* ------------------------------------------------------------------ */
if (typeof g.GPUMapMode === 'undefined') {
  g.GPUMapMode = {
    READ: 0x1,
    WRITE: 0x2,
  };
}

/* ------------------------------------------------------------------ */
/* GPUTextureUsage 位掩码                                              */
/* ------------------------------------------------------------------ */
if (typeof g.GPUTextureUsage === 'undefined') {
  g.GPUTextureUsage = {
    COPY_SRC: 0x01,
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
    STORAGE_BINDING: 0x08,
    RENDER_ATTACHMENT: 0x10,
    // 历史别名（旧代码用 SAMPLED）—— 与 TEXTURE_BINDING 同值。
    SAMPLED: 0x04,
  };
}

/* ------------------------------------------------------------------ */
/* GPUCullMode / GPUIndexFormat / GPUPrimitiveTopology 枚举            */
/* ------------------------------------------------------------------ */
if (typeof g.GPUCullMode === 'undefined') {
  g.GPUCullMode = {
    none: 'none',
    front: 'front',
    back: 'back',
  };
}

if (typeof g.GPUIndexFormat === 'undefined') {
  g.GPUIndexFormat = {
    uint16: 'uint16',
    uint32: 'uint32',
  };
}

if (typeof g.GPUPrimitiveTopology === 'undefined') {
  g.GPUPrimitiveTopology = {
    'triangle-list': 'triangle-list',
    'triangle-strip': 'triangle-strip',
    'line-list': 'line-list',
    'line-strip': 'line-strip',
    'point-list': 'point-list',
  };
}

/* ------------------------------------------------------------------ */
/* GPUShaderStage 位掩码                                               */
/* ------------------------------------------------------------------ */
if (typeof g.GPUShaderStage === 'undefined') {
  g.GPUShaderStage = {
    VERTEX: 0x01,
    FRAGMENT: 0x02,
    COMPUTE: 0x04,
  };
}
