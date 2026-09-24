/**
 * PipelineCache —— 以内容为哈希键的管线缓存。
 *
 * 规避浏览器 createRenderPipeline 时高昂的校验 + 编译惩罚：
 * 相同 desc 只创建一次，通过哈希命中直接返回 ResolvedPipeline。
 */

import type { PipelineDesc, ResolvedPipeline } from '../types';

/** FNV-1a 64 位散列（BigInt 实现）。 */
export function fnv1a64(input: string): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let i = 0; i < input.length; i++) {
    h ^= BigInt(input.charCodeAt(i));
    h = (h * prime) & 0xffffffffffffffffn;
  }
  return h;
}

export function hashPipelineDesc(desc: PipelineDesc): bigint {
  const canonical = canonicalize(desc);
  return fnv1a64(canonical);
}

export function canonicalize(desc: PipelineDesc): string {
  return JSON.stringify({
    vs: desc.vsCode,
    fs: desc.fsCode,
    bpi: desc.bytesPerInstance ?? 80,
    mmo: desc.modelMatrixOffset ?? 0,
    // compaction 决定 group=1 的绑定布局（単实例 vs 实例+compaction mapping）。
    cmp: desc.compaction === true,
    v: desc.vertexLayouts,
    bgl: desc.bindGroupLayouts.map((l) => l.label ?? ''),
    // 包含 byteOffset / byteLength：否则「同一 buffer 的不同切片」会塌到同一个 key，
    // 拿到错误的 global bind group。
    g: desc.globalBindings.map((b) => `${b.binding}:${b.buffer.size}:${b.byteOffset ?? 0}:${b.byteLength ?? 0}`),
    d: desc.depth,
    t: desc.targets,
    p: desc.primitive,
  });
}

export class PipelineCache {
  private _map = new Map<string, ResolvedPipeline>();
  private _nextId = 0;

  get size(): number {
    return this._map.size;
  }

  /** 尝试命中；未命中则创建。返回 [pipeline, created]。使用 canonical string 作 key 避免哈希碰撞。 */
  getOrCreate(
    device: GPUDevice,
    desc: PipelineDesc,
    onStats?: (created: boolean) => void,
  ): ResolvedPipeline {
    const canonical = canonicalize(desc);
    const existing = this._map.get(canonical);
    if (existing) {
      onStats?.(false);
      return existing;
    }

    const resolved = createResolved(device, desc, this._nextId);
    this._nextId++;
    this._map.set(canonical, resolved);
    onStats?.(true);
    return resolved;
  }
}

export function createResolved(device: GPUDevice, desc: PipelineDesc, id: number): ResolvedPipeline {
  const bgls = desc.bindGroupLayouts;
  const pipelineLayout = device.createPipelineLayout({
    label: `hpg:pl:${desc.label ?? id}`,
    bindGroupLayouts: bgls,
  });

  const bytesPerInstance = desc.bytesPerInstance ?? 80;
  const modelMatrixOffset = desc.modelMatrixOffset ?? 0;
  const label = desc.compaction ? `${desc.label ?? `pipeline-${id}`}:compaction` : desc.label ?? `pipeline-${id}`;

  const pipeline = device.createRenderPipeline({
    label: `hpg:rp:${label}`,
    layout: pipelineLayout,
    vertex: {
      module: createShaderModule(device, desc.vsCode, `${label}:vs`),
      entryPoint: 'vs_main',
      buffers: desc.vertexLayouts.map((l) => ({
        arrayStride: l.arrayStride,
        stepMode: l.stepMode,
        attributes: l.attributes,
      })),
    },
    fragment: {
      module: createShaderModule(device, desc.fsCode, `${label}:fs`),
      entryPoint: 'fs_main',
      targets: desc.targets,
    },
    primitive: desc.primitive,
    depthStencil: desc.depth as GPUDepthStencilState | undefined,
  });

  return {
    id,
    desc,
    pipeline,
    layout: bgls[0],
    bindGroupLayouts: bgls,
    bytesPerInstance,
    modelMatrixOffset,
    label,
  };
}

function createShaderModule(device: GPUDevice, code: string, label: string): GPUShaderModule {
  return device.createShaderModule({ label, code });
}