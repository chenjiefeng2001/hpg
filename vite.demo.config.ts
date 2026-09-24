import { defineConfig } from 'vite';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

/**
 * Demo / benchmark 多页面构建。
 *
 * 注意：Vite 默认只把根目录的 index.html 作为入口 —— 不显式声明 rollupOptions.input 时
 * demo/glb-viewer.html 等页面不会进入产物（`npm run dev` 能访问，build 后丢失）。
 */
export default defineConfig({
  build: {
    target: 'es2022',
    outDir: 'dist/demo',
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'),
        phase5: resolve(__dirname, 'phase5.html'),
        glbViewer: resolve(__dirname, 'demo/glb-viewer.html'),
        benchmark: resolve(__dirname, 'benchmark/index.html'),
        glbBenchmark: resolve(__dirname, 'benchmark/glb-bench.html'),
      },
    },
  },
});
