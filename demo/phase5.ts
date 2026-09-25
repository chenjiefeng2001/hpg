/**
 * Phase 5 Demo — GPU 视锥剔除 + Indirect Drawing。
 *
 * 展示：500 个对象分布在场景中，使用 GPU Compute Shader 剔除视锥外的对象，
 * 剩余对象通过 Indirect Draw 提交。
 *
 * 对比：按 C 键切换 CPU/GPU 路径，观察 drawCalls 和 instances 变化。
 */

import { Renderer } from '../src/core/renderer';
import { VS_INSTANCED, VS_INSTANCED_COMPACTION, FS_COLOR } from '../src/shaders/instance';
import { uniformBindGroupLayout } from '../src/core/renderer';
import { identity, multiply, rotationY, perspective, lookAt } from '../src/core/math';
import type { GlobalBinding, RenderItem, BoundingSphere } from '../src/types';

// ─── WebGPU 初始化 ───────────────────────────────────────────

async function initWebGPU(): Promise<{ device: GPUDevice; context: GPUCanvasContext; format: GPUTextureFormat }> {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported. Use Chrome 113+ / Edge 113+.');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No GPU adapter found.');

  const device = await adapter.requestDevice();
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  return { device, context, format };
}

// ─── 几何体：一个正方形 ──────────────────────────────────────

function createQuadGeometry(renderer: Renderer) {
  const vertices = new Float32Array([
    -0.4, -0.4, 0,  0, 0, 1,
     0.4, -0.4, 0,  0, 0, 1,
     0.4,  0.4, 0,  0, 0, 1,
    -0.4,  0.4, 0,  0, 0, 1,
  ]);
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);

  return renderer.createGeometry(
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
}

// ─── 场景生成 ─────────────────────────────────────────────────

const COUNT = 500;
const GRID_SIZE = Math.ceil(Math.sqrt(COUNT)); // ~23 × 23

interface InstanceConfig {
  x: number;
  y: number;
  z: number;
  color: Float32Array;
  bounding: BoundingSphere;
}

function generateScene(): InstanceConfig[] {
  const configs: InstanceConfig[] = [];
  const spacing = 2.0;
  const halfGrid = (GRID_SIZE * spacing) / 2;

  for (let i = 0; i < COUNT; i++) {
    const row = Math.floor(i / GRID_SIZE);
    const col = i % GRID_SIZE;
    const x = col * spacing - halfGrid;
    const z = row * spacing - halfGrid;
    const y = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 0.5;

    // 随机颜色（基于索引的伪随机）。
    const r = ((i * 137 + 50) % 256) / 256;
    const g = ((i * 251 + 100) % 256) / 256;
    const b = ((i * 359 + 150) % 256) / 256;

    configs.push({
      x, y, z,
      color: new Float32Array([r, g, b, 1]),
      bounding: { centerX: x, centerY: y, centerZ: z, radius: 0.6 },
    });
  }
  return configs;
}

// ─── 主流程 ──────────────────────────────────────────────────

async function main() {
  const { device, context, format } = await initWebGPU();
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

  // CPU 路径（submit）：group(1) = 单实例绑定
  const pipeline = renderer.registerPipeline({
    label: 'phase5-quad',
    vsCode: VS_INSTANCED,
    fsCode: FS_COLOR,
    ...commonDesc,
  });

  // GPU 剔除路径（submitCulled）：group(1) = [instances, compactedIndices]
  // 拓扑上两种路径产生相同视觉结果，但 group(1) 布局不同，必须分别注册。
  const culledPipeline = renderer.registerPipeline({
    label: 'phase5-quad-culled',
    vsCode: VS_INSTANCED_COMPACTION,
    fsCode: FS_COLOR,
    compaction: true,
    ...commonDesc,
  });

  const quad = createQuadGeometry(renderer);
  const scene = generateScene();
  const transforms = scene.map(() => identity());

  const statsEl = document.getElementById('stats')!;
  const modeEl = document.getElementById('mode')!;
  let frame = 0;
  let useGPU = true;

  // C 键切换模式。
  document.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') {
      useGPU = !useGPU;
    }
  });

  function render() {
    const t = frame * 0.01;

    // 相机：围绕场景旋转。
    const camAngle = t * 0.5;
    const camRadius = 30;
    const eye = [
      Math.sin(camAngle) * camRadius,
      15,
      Math.cos(camAngle) * camRadius,
    ] as [number, number, number];
    const target = [0, 0, 0] as [number, number, number];
    const up = [0, 1, 0] as [number, number, number];

    const aspect = (context.canvas as HTMLCanvasElement).width / (context.canvas as HTMLCanvasElement).height;
    const proj = perspective(Math.PI / 3, aspect, 0.1, 200);
    const view = lookAt(new Float32Array(eye), new Float32Array(target), new Float32Array(up));
    const vp = multiply(new Float32Array(16), proj, view);

    // 更新变换。
    for (let i = 0; i < COUNT; i++) {
      const cfg = scene[i]!;
      const m = transforms[i]!;
      const angle = t + i * 0.01;
      const rot = rotationY(new Float32Array(16), angle);
      const translate = new Float32Array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        cfg.x, cfg.y, cfg.z, 1,
      ]);
      multiply(m, translate, rot);
    }

    // 构建 RenderItem[]（按模式选择对应管线的 group(1) 布局）。
    const activePipeline = useGPU ? culledPipeline : pipeline;
    const items: RenderItem[] = scene.map((cfg, i) => ({
      geometry: quad,
      pipeline: activePipeline,
      transforms: transforms[i]!,
      instanceData: cfg.color,
      bounding: cfg.bounding,
    }));

    // 根据模式选择路径。
    const stats = useGPU
      ? renderer.submitCulled(items, vp)
      : renderer.submit(items);

    modeEl.textContent = useGPU ? 'GPU Culling + Indirect' : 'CPU Direct Draw';
    statsEl.textContent =
      `drawCalls: ${stats.drawCalls} | instances: ${stats.instances} | batches: ${stats.batches} | items: ${stats.itemsDrawn}/${stats.itemsSubmitted} | [C] 切换模式`;

    frame++;
    requestAnimationFrame(render);
  }

  render();
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="color:red;padding:2rem">${String(e)}</pre>`;
  console.error(e);
});
