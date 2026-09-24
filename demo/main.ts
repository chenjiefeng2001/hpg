/**
 * hpg Demo — 最小可运行 WebGPU 渲染示例。
 *
 * 展示：5 个彩色方块自动合批为 1 个 Instanced Draw。
 */

import { Renderer } from '../src/core/renderer';
import { VS_INSTANCED, FS_COLOR } from '../src/shaders/instance';
import { uniformBindGroupLayout } from '../src/core/renderer';
import { identity, multiply, rotationY } from '../src/core/math';
import type { GlobalBinding, RenderItem } from '../src/types';

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

// ─── 几何体：一个正方形（两个三角形）──────────────────────────

function createQuadGeometry(renderer: Renderer) {
  // 顶点：position(3) + normal(3)，stride = 24 字节
  // 正方形：-0.4 ~ 0.4，法线统一朝 +Z
  const vertices = new Float32Array([
    // position        // normal
    -0.4, -0.4, 0,     0, 0, 1,
     0.4, -0.4, 0,     0, 0, 1,
     0.4,  0.4, 0,     0, 0, 1,
    -0.4,  0.4, 0,     0, 0, 1,
  ]);
  const indices = new Uint16Array([0, 1, 2, 2, 3, 0]);

  return renderer.createGeometry(
    vertices,
    [
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
        ],
      },
    ],
    indices,
  );
}

// ─── 主流程 ──────────────────────────────────────────────────

async function main() {
  const { device, context, format } = await initWebGPU();
  const renderer = new Renderer(device, context, format, {
    clearColor: [0.05, 0.06, 0.09, 1],
  });

  // 创建管线：使用 VS_INSTANCED + FS_COLOR
  const layout = uniformBindGroupLayout(device, [
    { binding: 0, visibility: GPUShaderStage.VERTEX },
  ]);
  // 伪 uniform（占位，本 demo 未使用 VP 矩阵）
  const dummyUniform = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const globalBindings: GlobalBinding[] = [{ binding: 0, buffer: dummyUniform }];

  const pipeline = renderer.registerPipeline({
    label: 'demo-quad',
    vsCode: VS_INSTANCED,
    fsCode: FS_COLOR,
    vertexLayouts: [
      {
        arrayStride: 24,
        stepMode: 'vertex',
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
        ],
      },
    ],
    bindGroupLayouts: [layout],
    globalBindings,
    depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    targets: [{ format }],
  });

  // 创建几何体
  const quad = createQuadGeometry(renderer);

  // 5 个实例配置：环形排列 + 不同颜色
  const COUNT = 5;
  const configs = [
    { x: 0,    y: 0,    color: [1.0, 0.3, 0.3, 1] },  // 红
    { x: 0.5,  y: 0.3,  color: [0.3, 1.0, 0.3, 1] },  // 绿
    { x: -0.5, y: 0.3,  color: [0.3, 0.3, 1.0, 1] },  // 蓝
    { x: 0.35, y: -0.4, color: [1.0, 1.0, 0.3, 1] },  // 黄
    { x: -0.35,y: -0.4, color: [1.0, 0.3, 1.0, 1] },  // 紫
  ];

  // 预分配变换矩阵
  const transforms = configs.map(() => identity());

  const statsEl = document.getElementById('stats')!;
  let frame = 0;

  function render() {
    const t = frame * 0.02;

    // 更新每个实例的变换矩阵
    for (let i = 0; i < COUNT; i++) {
      const cfg = configs[i]!;
      const angle = t + (i * Math.PI * 2) / COUNT;
      const ox = cfg.x + Math.sin(angle) * 0.05;
      const oy = cfg.y + Math.cos(angle) * 0.05;

      const m = transforms[i]!;
      // 重置为单位矩阵
      m[0] = 1; m[1] = 0; m[2] = 0; m[3] = 0;
      m[4] = 0; m[5] = 1; m[6] = 0; m[7] = 0;
      m[8] = 0; m[9] = 0; m[10] = 1; m[11] = 0;
      m[12] = ox; m[13] = oy; m[14] = 0; m[15] = 1;

      // 绕 Z 轴旋转
      const rot = rotationY(new Float32Array(16), angle * 0.5);
      // 与平移矩阵相乘：M = T * R
      multiply(m, rot, m);
    }

    // 构建 RenderItem[] — 全部使用相同 pipeline + geometry → 自动合批
    const items: RenderItem[] = configs.map((cfg, i) => ({
      geometry: quad,
      pipeline,
      transforms: transforms[i]!,
      instanceData: new Float32Array(cfg.color),
    }));

    const stats = renderer.submit(items);

    statsEl.textContent =
      `drawCalls: ${stats.drawCalls} | instances: ${stats.instances} | batches: ${stats.batches} | items: ${stats.itemsDrawn}/${stats.itemsSubmitted}`;

    frame++;
    requestAnimationFrame(render);
  }

  render();
}

main().catch((e) => {
  document.body.innerHTML = `<pre style="color:red;padding:2rem">${String(e)}</pre>`;
  console.error(e);
});
