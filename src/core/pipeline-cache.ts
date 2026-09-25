/**
 * PipelineCache —— 以内容为哈希键的管线缓存。
 *
 * 规避浏览器 createRenderPipeline 时高昂的校验 + 编译惩罚：
 * 相同 desc 只创建一次，通过哈希命中直接返回 ResolvedPipeline。
 */

import type { PipelineDesc, ResolvedPipeline } from '../types';

const objectIds = new WeakMap<object, number>();
let nextObjectId = 1;

function objectId(value: object): number {
  const existing = objectIds.get(value);
  if (existing !== undefined) return existing;
  const id = nextObjectId++;
  objectIds.set(value, id);
  return id;
}

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
     cc: desc.compactionContract ?? '',
     v: desc.vertexLayouts,
    bgl: desc.bindGroupLayouts.map((l) => `${objectId(l)}:${l.label ?? ''}`),
     g: desc.globalBindings.map((b) => `${b.binding}:${objectId(b.buffer)}:${b.buffer.size}:${b.byteOffset ?? 0}:${b.byteLength ?? 0}`),
    d: desc.depth,
    t: desc.targets,
    p: desc.primitive,
  });
}

export class PipelineCache {
  private _map = new Map<string, ResolvedPipeline>();
  private _nextId = 0;
  private _device: GPUDevice | null = null;

  get size(): number {
    return this._map.size;
  }

  clear(): void {
    this._map.clear();
    this._device = null;
  }

  /** 尝试命中；未命中则创建。返回 [pipeline, created]。使用 canonical string 作 key 避免哈希碰撞。 */
  getOrCreate(
    device: GPUDevice,
    desc: PipelineDesc,
    onStats?: (created: boolean) => void,
  ): ResolvedPipeline {
    if (this._device && this._device !== device) {
      throw new Error('PipelineCache cannot be reused across GPU devices.');
    }
    this._device = device;
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

function clonePipelineDesc(desc: PipelineDesc): PipelineDesc {
  return {
    ...desc,
    vertexLayouts: desc.vertexLayouts.map((layout) => ({
      ...layout,
      attributes: layout.attributes.map((attribute) => ({ ...attribute })),
    })),
    bindGroupLayouts: [...desc.bindGroupLayouts],
    globalBindings: desc.globalBindings.map((binding) => ({ ...binding })),
    targets: desc.targets.map((target) => ({
      ...target,
      blend: target.blend ? { ...target.blend } : undefined,
      writeMask: target.writeMask,
    })),
    depth: desc.depth ? { ...desc.depth } : undefined,
    primitive: desc.primitive ? { ...desc.primitive } : undefined,
  };
}

function freezePipelineDesc(desc: PipelineDesc): PipelineDesc {
  const frozen = {
    ...desc,
    vertexLayouts: desc.vertexLayouts.map((layout) => Object.freeze({
      ...layout,
      attributes: Object.freeze(layout.attributes.map((attribute) => Object.freeze({ ...attribute }))),
    })),
    bindGroupLayouts: [...desc.bindGroupLayouts],
    globalBindings: desc.globalBindings.map((binding) => Object.freeze({ ...binding })),
    targets: desc.targets.map((target) => Object.freeze({
      ...target,
      blend: target.blend ? Object.freeze({ ...target.blend }) : undefined,
    })),
    depth: desc.depth ? Object.freeze({ ...desc.depth }) : undefined,
    primitive: desc.primitive ? Object.freeze({ ...desc.primitive }) : undefined,
  };
  Object.freeze(frozen.vertexLayouts);
  Object.freeze(frozen.bindGroupLayouts);
  Object.freeze(frozen.globalBindings);
  Object.freeze(frozen.targets);
  return Object.freeze(frozen) as PipelineDesc;
}

export function createResolved(device: GPUDevice, input: PipelineDesc, id: number): ResolvedPipeline {
  const desc = freezePipelineDesc(clonePipelineDesc(input));
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

  return Object.freeze({
    id,
    device,
    desc,
    pipeline,
    layout: bgls[0],
    bindGroupLayouts: bgls,
    bytesPerInstance,
    modelMatrixOffset,
    label,
  });
}

function createShaderModule(device: GPUDevice, code: string, label: string): GPUShaderModule {
  return device.createShaderModule({ label, code });
}