/**
 * Phase 6.5 — Measurement Benchmark。
 *
 * 3 paths × N workload cases，统计 CPU submit 时间 + draw/instance 计数。
 *
 * Paths:
 *   A. Direct      — renderer.submit()（无合批，逐 draw call）
 *   B. Batcher     — renderer.submit()（自动合批，instanced draw）
 *   C. GPU Culling — renderer.submitCulled()（compute culling + indirect）
 *
 * 每个 case:
 *   - 10 warm-up iterations
 *   - 50 measured iterations
 *   - median / p95 / min / max
 *
 * 用法: 浏览器打开 benchmark/index.html
 */

import { Renderer, uniformBindGroupLayout } from '../src/core/renderer';
import { VS_INSTANCED, VS_INSTANCED_COMPACTION, FS_COLOR } from '../src/shaders/instance';
import { TimestampQuery } from '../src/core/timestamp';
import { identity } from '../src/core/math';
import type { GlobalBinding, RenderItem, BoundingSphere, ResolvedPipeline } from '../src/types';

// ─── Types ───────────────────────────────────────────────────

interface BenchCase {
  name: string;
  count: number;
  visibilityRatio: number;
  path: 'direct' | 'batcher' | 'gpu-cull';
}

interface Stats {
  median: number;
  p95: number;
  min: number;
  max: number;
}

interface BenchResult {
  caseName: string;
  count: number;
  visible: string;
  path: string;
  cpuMs: Stats;
  gpuMs: Stats | null;
  drawCalls: number;
  instances: number;
  batches: number;
  itemsDrawn: number;
}

// ─── Helpers ─────────────────────────────────────────────────

function computeStats(times: number[]): Stats {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const median = sorted[Math.floor(n * 0.5)]!;
  const p95 = sorted[Math.floor(n * 0.95)]!;
  return {
    median: +median.toFixed(3),
    p95: +p95.toFixed(3),
    min: +sorted[0]!.toFixed(3),
    max: +sorted[n - 1]!.toFixed(3),
  };
}

function detectGPU(): string {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) return 'unknown (no WebGL2)';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  if (!ext) return 'unknown (no debug info)';
  const renderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL);
  const vendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL);
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  return `${vendor} / ${renderer}`;
}

function detectLimits(device: GPUDevice): string {
  try {
    const limits = (device as any).limits;
    if (!limits) return 'limits: not available';
    return [
      `maxBufferSize       : ${(limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`,
      `maxStorageBuffer    : ${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`,
      `maxUniformBuffer    : ${(limits.maxUniformBufferBindingSize / 1024).toFixed(0)} KB`,
      `maxComputeWorkgrps  : ${limits.maxComputeWorkgroupsPerDimension}`,
      `maxStorageBuffers   : ${limits.maxStorageBuffersPerShaderStage}`,
    ].join('\n');
  } catch {
    return 'limits: not available';
  }
}

// ─── Scene Generation ────────────────────────────────────────

interface BenchItem {
  x: number;
  y: number;
  z: number;
  color: Float32Array;
  bounding: BoundingSphere;
}

function generateGrid(count: number, visibilityRatio: number): BenchItem[] {
  const items: BenchItem[] = [];
  const gridSize = Math.ceil(Math.sqrt(count));
  const spacing = Math.min(2.0, 56 / Math.max(1, gridSize - 1));
  const halfGrid = (Math.max(0, gridSize - 1) * spacing) / 2;

  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / gridSize);
    const col = i % gridSize;
    const x = col * spacing - halfGrid;
    const z = row * spacing - halfGrid;
    const y = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 0.5;

    const inFrustum = i / count < visibilityRatio;
    const cx = inFrustum ? x : x + 1000;

    const r = ((i * 137 + 50) % 256) / 256;
    const g = ((i * 251 + 100) % 256) / 256;
    const b = ((i * 359 + 150) % 256) / 256;

    items.push({
      x: cx, y, z,
      color: new Float32Array([r, g, b, 1]),
      bounding: { centerX: cx, centerY: y, centerZ: z, radius: 0.6 },
    });
  }
  return items;
}

// ─── VP Matrix ───────────────────────────────────────────────

function makeVP(): Float32Array {
  const l = -30, r = 30, b = -30, t = 30, n = 0.1, f = 200;
  const out = new Float32Array(16);
  out[0] = 2 / (r - l);
  out[5] = 2 / (t - b);
  out[10] = -f / (f - n);
  out[12] = -(r + l) / (r - l);
  out[13] = -(t + b) / (t - b);
  out[14] = -(f * n) / (f - n);
  out[15] = 1;
  return out;
}

// ─── Benchmark Runner ────────────────────────────────────────

const WARMUP = 10;
const SAMPLES = 50;

function buildItems(
  scene: BenchItem[],
  transforms: Float32Array[],
  geometry: ReturnType<Renderer['createGeometry']>,
  pipeline: ResolvedPipeline,
): RenderItem[] {
  return scene.map((cfg, i) => ({
    geometry,
    pipeline,
    transforms: transforms[i]!,
    instanceData: cfg.color,
    bounding: cfg.bounding,
  }));
}

async function runCase(
  renderer: Renderer,
  scene: BenchItem[],
  transforms: Float32Array[],
  geometry: ReturnType<Renderer['createGeometry']>,
  pipeline: ResolvedPipeline,
  culledPipeline: ResolvedPipeline,
  vpMatrix: Float32Array,
  c: BenchCase,
  timestampQueryAvailable: boolean,
): Promise<BenchResult> {
  // 每条路径的 group(1) 布局不同：culled 用 compaction 管线，其余用普通实例管线。
  const activePipeline = c.path === 'gpu-cull' ? culledPipeline : pipeline;
  const items = buildItems(scene, transforms, geometry, activePipeline);

  // Warm-up (no timestamps).
  for (let w = 0; w < WARMUP; w++) {
    if (c.path === 'gpu-cull') {
      renderer.submitCulled(items, vpMatrix);
    } else if (c.path === 'direct') {
      renderer.submitDirect(items);
    } else {
      renderer.submit(items);
    }
  }

  // Measured samples — GPU timestamps for direct and gpu-cull paths.
  const cpuTimes: number[] = [];
  const gpuTimes: number[] = [];
  let lastStats;
  const useTimestamps = timestampQueryAvailable && (c.path === 'direct' || c.path === 'gpu-cull');

  for (let i = 0; i < SAMPLES; i++) {
    let tq: TimestampQuery | null = null;
    try {
      if (useTimestamps) {
        tq = new TimestampQuery(renderer.device, 2);
      }
      const t0 = performance.now();
      try {
        lastStats = c.path === 'gpu-cull'
          ? renderer.submitCulled(items, vpMatrix, tq ?? undefined)
          : c.path === 'direct'
            ? renderer.submitDirect(items, tq ?? undefined)
            : renderer.submit(items);
      } catch (e: any) {
        console.error(`[hpg:bench] FAIL case=${c.name} count=${c.count} vis=${c.visibilityRatio} path=${c.path}`, e);
        throw e;
      }
      const t1 = performance.now();
      cpuTimes.push(t1 - t0);

      if (tq) {
        const timestamps = await tq.readback(renderer.device);
        const gpuNs = timestamps[1]! - timestamps[0]!;
        gpuTimes.push(gpuNs / 1e6); // ns → ms
      }
    } finally {
      tq?.destroy();
    }
  }

  const cpuStats = computeStats(cpuTimes);
  const gpuStats = gpuTimes.length > 0 ? computeStats(gpuTimes) : null;
  const visiblePct = Math.round(c.visibilityRatio * 100);

  return {
    caseName: c.name,
    count: c.count,
    visible: `${visiblePct}%`,
    path: c.path === 'gpu-cull' ? 'GPU Culling' : c.path === 'batcher' ? 'Batcher' : 'Direct',
    cpuMs: cpuStats,
    gpuMs: gpuStats,
    drawCalls: lastStats!.drawCalls,
    instances: lastStats!.instances,
    batches: lastStats!.batches,
    itemsDrawn: lastStats!.itemsDrawn,
  };
}

// ─── Output Formatting ───────────────────────────────────────

function formatEnv(device: GPUDevice): string {
  return [
    `Browser:  ${navigator.userAgent}`,
    `GPU:      ${detectGPU()}`,
    `OS:       ${navigator.platform}`,
    `Cores:    ${navigator.hardwareConcurrency ?? '?'}`,
    `Samples:  ${SAMPLES} (+ ${WARMUP} warm-up)`,
    '',
    'Device Limits:',
    detectLimits(device),
  ].join('\n');
}

function formatTable(results: BenchResult[]): string {
  const lines: string[] = [];

  lines.push('Case    │ Objects │ Visible │ Path         │ CPU ms (median) │ GPU render ms │ CPU p95  │ GPU render p95 │ Draws │ Candidates│ Batches');
  lines.push('────────┼─────────┼─────────┼──────────────┼───────────────┼───────────────┼─────────────────┼───────┼───────────┼────────');

  for (const r of results) {
    lines.push([
      r.caseName.padEnd(7),
      String(r.count).padStart(7),
      r.visible.padStart(7),
      r.path.padEnd(12),
      r.cpuMs.median.toFixed(3).padStart(15),
      (r.gpuMs ? r.gpuMs.median.toFixed(3) : '—').padStart(15),
      r.cpuMs.p95.toFixed(3).padStart(8),
      (r.gpuMs ? r.gpuMs.p95.toFixed(3) : '—').padStart(8),
      String(r.drawCalls).padStart(5),
      String(r.instances).padStart(9),
      String(r.batches).padStart(8),
    ].join(' │ '));
  }

  return lines.join('\n');
}

function formatBreakEven(results: BenchResult[]): string {
  const lines: string[] = ['', '─── CPU Submission Crossover ───', ''];

  // Group by count.
  const byCount = new Map<number, BenchResult[]>();
  for (const r of results) {
    const arr = byCount.get(r.count) ?? [];
    arr.push(r);
    byCount.set(r.count, arr);
  }

  for (const [count, cases] of byCount) {
    const direct = cases.find((c) => c.path === 'Direct');
    const batcher = cases.find((c) => c.path === 'Batcher');
    const gpuCull = cases.find((c) => c.path === 'GPU Culling');

    if (!direct && !batcher && !gpuCull) continue;

    lines.push(`${count} candidates:`);

    if (direct) {
       const gpu = direct.gpuMs ? ` + render ${direct.gpuMs.median.toFixed(3)}` : '';
       lines.push(`  Direct       : CPU ${direct.cpuMs.median.toFixed(3)}${gpu} ms (p95 ${direct.cpuMs.p95.toFixed(3)})`);
    }
    if (batcher) {
      lines.push(`  Batcher      : CPU ${batcher.cpuMs.median.toFixed(3)} ms (p95 ${batcher.cpuMs.p95.toFixed(3)})`);
    }
    if (gpuCull) {
       const gpu = gpuCull.gpuMs ? ` + render ${gpuCull.gpuMs.median.toFixed(3)}` : '';
       lines.push(`  GPU Culling  : CPU ${gpuCull.cpuMs.median.toFixed(3)}${gpu} ms (p95 ${gpuCull.cpuMs.p95.toFixed(3)})`);
    }

    if (direct && gpuCull) {
      const cpuRatio = gpuCull.cpuMs.median / direct.cpuMs.median;
      lines.push(`  → GPU Culling CPU submit ${cpuRatio < 1 ? '<' : '>'} Direct (${cpuRatio.toFixed(2)}x)`);
    }
    if (batcher && gpuCull) {
      const cpuRatio = gpuCull.cpuMs.median / batcher.cpuMs.median;
      lines.push(`  → GPU Culling CPU submit ${cpuRatio < 1 ? '<' : '>'} Batcher (${cpuRatio.toFixed(2)}x)`);
    }
    if (direct?.gpuMs && gpuCull?.gpuMs) {
      const gpuRatio = gpuCull.gpuMs.median / direct.gpuMs.median;
       lines.push(`  → GPU Culling render-pass time ${gpuRatio < 1 ? '<' : '>'} Direct (${gpuRatio.toFixed(2)}x)`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

// ─── Cases ───────────────────────────────────────────────────

function buildCases(): BenchCase[] {
  const cases: BenchCase[] = [];

  // Baseline at 100% visibility (Direct + Batcher).
  cases.push({ name: 'A1', count: 500, visibilityRatio: 1.0, path: 'direct' });
  cases.push({ name: 'A2', count: 500, visibilityRatio: 1.0, path: 'batcher' });
  cases.push({ name: 'A3', count: 500, visibilityRatio: 1.0, path: 'gpu-cull' });

  // Scale sweep at 100% visibility.
  for (const count of [1000, 5000, 50000]) {
    const tag = count >= 1000 ? `${count / 1000}K` : `${count}`;
    cases.push({ name: `B-${tag}-D`, count, visibilityRatio: 1.0, path: 'direct' });
    cases.push({ name: `B-${tag}-B`, count, visibilityRatio: 1.0, path: 'batcher' });
    cases.push({ name: `B-${tag}-G`, count, visibilityRatio: 1.0, path: 'gpu-cull' });
  }

  // Visibility sweep at 50K (all 3 paths).
  for (const vis of [0.5, 0.1, 0.01, 0.0]) {
    const pct = Math.round(vis * 100);
    cases.push({ name: `C-${pct}%-D`, count: 50000, visibilityRatio: vis, path: 'direct' });
    cases.push({ name: `C-${pct}%-B`, count: 50000, visibilityRatio: vis, path: 'batcher' });
    cases.push({ name: `C-${pct}%-G`, count: 50000, visibilityRatio: vis, path: 'gpu-cull' });
  }

  // Scale sweep at 10% (GPU culling only, for scaling analysis).
  for (const count of [1000, 5000, 10000, 50000, 100000, 500000]) {
    const tag = count >= 1000 ? `${count / 1000}K` : `${count}`;
    cases.push({ name: `D-${tag}`, count, visibilityRatio: 0.1, path: 'gpu-cull' });
  }

  // Extreme at 500K.
  cases.push({ name: 'E-1%', count: 500000, visibilityRatio: 0.01, path: 'gpu-cull' });
  cases.push({ name: 'E-0.1%', count: 500000, visibilityRatio: 0.001, path: 'gpu-cull' });

  return cases;
}

// ─── Init ────────────────────────────────────────────────────

async function main() {
  const output = document.getElementById('results')!;
  output.textContent = 'Initializing WebGPU…';

  if (!navigator.gpu) {
    output.textContent = 'WebGPU not supported. Use Chrome 113+ / Edge 113+.';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    output.textContent = 'No GPU adapter found.';
    return;
  }

  const timestampQueryAvailable = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: timestampQueryAvailable ? ['timestamp-query'] : [],
  });
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const renderer = new Renderer(device, context, format, {
    clearColor: [0.05, 0.06, 0.09, 1],
  });

  const layout = uniformBindGroupLayout(device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX },
  ]);
  const dummyUniform = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const globalBindings: GlobalBinding[] = [{ binding: 0, buffer: dummyUniform }];

  const vertexLayouts = [{
    arrayStride: 24,
    stepMode: 'vertex' as const,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
      { shaderLocation: 1, offset: 12, format: 'float32x3' as GPUVertexFormat },
    ],
  }];

  const commonDesc = {
    vertexLayouts,
    bindGroupLayouts: [layout],
    globalBindings,
    depth: { format: 'depth24plus' as GPUTextureFormat, depthWriteEnabled: true, depthCompare: 'less' as GPUCompareFunction },
    targets: [{ format }],
  };

  // Direct / Batcher 路径：group(1) = 单实例绑定（VS_INSTANCED 读 instances[instanceIdx]）。
  const pipeline = renderer.registerPipeline({
    label: 'bench-quad',
    vsCode: VS_INSTANCED,
    fsCode: FS_COLOR,
    ...commonDesc,
  });

  // GPU Culling 路径：group(1) = [instances, compactedIndices]（必须 compaction: true）。
  const culledPipeline = renderer.registerPipeline({
    label: 'bench-quad-culled',
    vsCode: VS_INSTANCED_COMPACTION,
    fsCode: FS_COLOR,
    compaction: true,
    ...commonDesc,
  });

  // Quad geometry.
  const vertices = new Float32Array([
    -0.4, -0.4, 0,  0, 0, 1,
     0.4, -0.4, 0,  0, 0, 1,
     0.4,  0.4, 0,  0, 0, 1,
    -0.4,  0.4, 0,  0, 0, 1,
  ]);
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);
  const geometry = renderer.createGeometry(
    vertices,
    [{
      arrayStride: 24,
      stepMode: 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
      ],
    }],
    indices,
  );

  const vpMatrix = makeVP();
  const cases = buildCases();

  // Group cases by (count, visibility) to avoid regenerating scenes.
  const sceneCache = new Map<string, { scene: BenchItem[]; transforms: Float32Array[] }>();
  function getScene(count: number, vis: number) {
    const key = `${count}-${vis}`;
    if (!sceneCache.has(key)) {
      const scene = generateGrid(count, vis);
      const transforms = scene.map(({ x, y, z }) => {
        const transform = identity();
        transform[12] = x;
        transform[13] = y;
        transform[14] = z;
        return transform;
      });
      sceneCache.set(key, { scene, transforms });
    }
    return sceneCache.get(key)!;
  }

  const totalCases = cases.length;
  output.textContent = `Running ${totalCases} cases × ${SAMPLES} samples…\n\n${formatEnv(device)}\n\n`;

  const results: BenchResult[] = [];
  let i = 0;
  for (const c of cases) {
    i++;
    output.textContent = `Running ${i}/${totalCases}: ${c.name} (${c.count} obj, ${Math.round(c.visibilityRatio * 100)}% vis, ${c.path})…`;
    await new Promise((r) => setTimeout(r, 0)); // yield to browser.

    const { scene, transforms } = getScene(c.count, c.visibilityRatio);
    const result = await runCase(renderer, scene, transforms, geometry, pipeline, culledPipeline, vpMatrix, c, timestampQueryAvailable);
    results.push(result);
  }

  // Output.
  output.textContent = [
    '═'.repeat(100),
    'hpg Phase 6.5/7B — Measurement Benchmark',
    '═'.repeat(100),
    '',
    formatEnv(device),
    '',
    '═'.repeat(100),
    'Results',
    '═'.repeat(100),
    '',
    formatTable(results),
    formatBreakEven(results),
    '═'.repeat(100),
    'Cost Model',
    '═'.repeat(100),
    '',
    '                    Direct       Batcher       GPU Culling',
    '──────────────────────────────────────────────────────────',
    'CPU preparation      O(N)         O(N)           O(N)',
    'Draw submission      O(N)         O(B)           O(G)',
    'GPU culling           —             —             O(N)',
    'Rasterization        O(N)         O(N)           O(M)',
    '',
    'N = candidate instances, B = batches, G = indirect draw groups, M = visible instances',
    '',
    'GPU Culling replaces GPU rasterization O(N) with O(M).',
    'When M << N the savings outweigh the O(N) culling + upload overhead.',
    '',
    '═'.repeat(100),
    'Notes',
    '═'.repeat(100),
    '',
    '  - CPU ms = time from performance.now() around submit() call.',
     '  - GPU render ms = render-pass time via timestamp query (timestampWrites → resolve → readback; culling compute excluded).',
    '  - "Direct" = renderer.submitDirect() — per-item draw calls, no batching.',
    '  - "Batcher" = renderer.submit() — automatic batching, instanced draw.',
    '  - "GPU Culling" = renderer.submitCulled() — compute frustum culling + indirect draw.',
    '  - "Candidates" = total instances submitted for processing (not necessarily drawn).',
    '  - Batcher path does not report GPU time (no timestamp integration for submit()).',
    '  - CPU Submission Crossover: GPU Culling CPU submit < Batcher CPU submit.',
    '',
  ].join('\n');

  renderer.dispose();
}

main().catch((e) => {
  const output = document.getElementById('results')!;
  output.textContent = `Error: ${String(e)}`;
  console.error(e);
});
