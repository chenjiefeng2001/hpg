/**
 * Phase 15E — 真实 Chrome 材质验证（Direct / GPU Culled）。
 *
 * 用本机 Chrome（headless=new + WebGPU）加载 demo/glb-viewer.html?harness=1，
 * 逐模型采集：
 *   - 材质/贴图统计（上传的纹理数、跳过原因、带材质的 RenderItem 数）
 *   - WebGPU validation / uncaptured error
 *   - canvas 像素签名（litPixels / avgLuma / 8×8 亮度网格）
 * 并对比 Direct 与 GPU Culled 两条路径是否视觉一致。
 *
 * 无第三方依赖：Vite dev server（子进程）+ Chrome 远程调试（Node 内置 WebSocket + CDP）。
 *
 * 用法:
 *   npm run verify:browser                 # 全部语料 × direct,culled
 *   npm run verify:browser -- medium       # 路径子串过滤
 *   node benchmark/browser-material-check.mjs --modes=direct --limit=3 --smoke
 */

import { spawn } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const MODEL_ROOT = join(ROOT, 'benchmark', 'assets', 'models');
const VITE_PORT = 5199;
const BASE = `http://127.0.0.1:${VITE_PORT}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const childProcesses = new Set();
const terminateChild = (child) => {
  childProcesses.delete(child);
  child.kill();
};
process.on('exit', () => {
  for (const child of childProcesses) child.kill();
});

// ─── CLI ────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : dflt;
};
const FILTER = flag('filter', argv.find((a) => !a.startsWith('--')) ?? '');
const MODES = flag('modes', 'direct,culled').split(',').map((s) => s.trim()).filter(Boolean);
if (MODES.length === 0 || MODES.some((mode) => mode !== 'direct' && mode !== 'culled')) throw new Error(`Unsupported mode in --modes: ${MODES.join(',')}`);
const SMOKE = argv.includes('--smoke');
if (!SMOKE && (!MODES.includes('direct') || !MODES.includes('culled'))) throw new Error('Full gate requires both direct and culled modes; use --smoke for a single-mode smoke test.');
const LIMIT = Number(flag('limit', '0')) || 0;
const PARTIAL_RUN = FILTER !== '' || LIMIT > 0;

// ─── Chrome 定位 ────────────────────────────────────────────

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);

function findChrome() {
  for (const p of CHROME_CANDIDATES) if (existsSync(p)) return p;
  throw new Error('Chrome 未找到：请设置 CHROME_PATH 环境变量。');
}

// ─── 语料 ───────────────────────────────────────────────────

function listGlb(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...listGlb(full));
    else if (entry.endsWith('.glb')) out.push(full);
  }
  return out.sort();
}

// ─── Vite dev server ────────────────────────────────────────

async function startVite() {
  // 直接用 node 跑 vite 的入口（不经 shell）—— Windows 上 shell 包装会让 kill 只杀掉外壳，
  // 真正的 vite 子进程会留下来占住端口。
  const viteBin = join(ROOT, 'node_modules', 'vite', 'bin', 'vite.js');
  const child = spawn(process.execPath, [viteBin, '--port', String(VITE_PORT), '--strictPort', '--host', '127.0.0.1'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  childProcesses.add(child);
  child.stderr.on('data', () => {});
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      childProcesses.delete(child);
      throw new Error(`vite 退出（code ${child.exitCode}）`);
    }
    try {
      const res = await fetch(BASE, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return child;
    } catch {
      /* not up yet */
    }
    await sleep(400);
  }
  terminateChild(child);
  throw new Error(`vite dev server 未在 60s 内就绪（${BASE}）`);
}

// ─── CDP ────────────────────────────────────────────────────

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(typeof ev.data === 'string' ? ev.data : ev.data.toString());
      if (msg.id != null && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }
      }, 30000);
    });
  }

  close() {
    try {
      this.ws.close();
    } catch {
      /* ignore */
    }
  }
}

async function launchChrome(exe) {
  const userDataDir = join(tmpdir(), `hpg-cdp-${Date.now()}`);
  const child = spawn(
    exe,
    [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--no-first-run',
      '--no-default-browser-check',
      '--window-size=800,600',
      // rAF 在后台/被遮挡的页面会被节流 —— headless 下会直接停掉。
      '--disable-background-timer-throttling',
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      `--user-data-dir=${userDataDir}`,
      '--remote-debugging-port=0',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  childProcesses.add(child);
  let launchError;
  child.once('error', (error) => { launchError = error; });

  try {
    const portFile = join(userDataDir, 'DevToolsActivePort');
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      if (launchError) throw launchError;
      if (child.exitCode !== null) throw new Error(`Chrome 退出（code ${child.exitCode}）`);
      if (existsSync(portFile)) {
        const port = Number(readFileSync(portFile, 'utf8').split('\n')[0]);
        if (port > 0) return { child, userDataDir, port };
      }
      await sleep(200);
    }
    throw new Error('Chrome DevTools 端口未就绪');
  } catch (error) {
    terminateChild(child);
    try {
      rmSync(userDataDir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
    throw error;
  }
}

async function connect(port) {
  const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  const ws = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((res, rej) => {
    ws.addEventListener('open', res, { once: true });
    ws.addEventListener('error', () => rej(new Error('WebSocket 连接失败')), { once: true });
  });
  return new Cdp(ws);
}

async function assertWebGpuAdapter(cdp) {
  const { targetId } = await cdp.send('Target.createTarget', { url: `${BASE}/demo/glb-viewer.html?probe=1` });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  try {
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Page.bringToFront', {}, sessionId).catch(() => {});
    const deadline = Date.now() + 10000;
    let value;
    while (Date.now() < deadline) {
      const result = await cdp.send('Runtime.evaluate', {
        expression: '(async () => ({ gpu: !!navigator.gpu, adapter: !!(navigator.gpu && await navigator.gpu.requestAdapter()) }))()',
        awaitPromise: true,
        returnByValue: true,
      }, sessionId);
      value = result?.result?.value;
      if (value?.gpu && value?.adapter) return;
      await sleep(250);
    }
    throw new Error(`Chrome 没有可用的 WebGPU adapter（gpu=${value?.gpu}, adapter=${value?.adapter}）；请使用具备 WebGPU 能力的 runner。`);
  } finally {
    cdp.send('Target.closeTarget', { targetId }).catch(() => {});
  }
}

/** 打开一个页面，轮询 window.__hpgResult 直到出现或超时。 */
async function runPage(cdp, url, timeoutMs = 30000) {
  const { targetId } = await cdp.send('Target.createTarget', { url });
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
  const evaluate = async (expression) => {
    const r = await cdp.send('Runtime.evaluate', { expression, returnByValue: true }, sessionId);
    return r?.result?.value;
  };
  try {
    await cdp.send('Runtime.enable', {}, sessionId);
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Page.bringToFront', {}, sessionId).catch(() => {});
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const value = await evaluate('window.__hpgResult ? JSON.stringify(window.__hpgResult) : null');
      if (value) return { result: JSON.parse(value) };
      await sleep(250);
    }
    // 超时诊断：页面到底卡在哪一步。
    const diag = await evaluate(
      `JSON.stringify({
         gpu: !!navigator.gpu,
         stats: (document.getElementById('stats')||{}).textContent || '',
         errors: (document.getElementById('errors')||{}).textContent || '',
         title: document.title || '',
          hasCanvas: !!document.querySelector('canvas'),
       })`,
    );
    return { result: null, diag: diag ? JSON.parse(diag) : null };
  } finally {
    cdp.send('Target.closeTarget', { targetId }).catch(() => {});
  }
}

// ─── 比较 ───────────────────────────────────────────────────

function gridDelta(a, b) {
  if (!a?.grid || !b?.grid || a.grid.length !== b.grid.length) return Infinity;
  let max = 0;
  for (let i = 0; i < a.grid.length; i++) max = Math.max(max, Math.abs(a.grid[i] - b.grid[i]));
  return Number(max.toFixed(2));
}

// ─── 主流程 ─────────────────────────────────────────────────

async function main() {
  const chromeExe = findChrome();
  const assets = listGlb(MODEL_ROOT)
    .filter((f) => !FILTER || f.includes(FILTER))
    .slice(0, LIMIT > 0 ? LIMIT : undefined);

  if (assets.length === 0) {
    throw new Error('没有匹配的 .glb。');
  }
  if (!FILTER && LIMIT === 0 && assets.length !== 23) {
    throw new Error(`完整语料 gate 期望 23 个 .glb，实际发现 ${assets.length} 个。`);
  }

  console.log(`hpg — Real Chrome material validation (Phase 15E)`);
  console.log(`chrome : ${chromeExe}`);
  console.log(`assets : ${assets.length} (${MODES.join(' / ')}${PARTIAL_RUN ? ' / partial' : ''})`);
  console.log('');

  const vite = await startVite();
  let chrome;
  let cdp;
  const failures = [];
  const rows = [];

  try {
     chrome = await launchChrome(chromeExe);
     cdp = await connect(chrome.port);
     await assertWebGpuAdapter(cdp);

     for (const abs of assets) {
      const rel = relative(MODEL_ROOT, abs).split(sep).join('/');
      const urlPath = `/benchmark/assets/models/${rel}`;
      const perMode = {};

      for (const mode of MODES) {
        const url = `${BASE}/demo/glb-viewer.html?harness=1&mode=${mode}&asset=${encodeURIComponent(urlPath)}`;
        let run = null;
        try {
          run = await runPage(cdp, url);
        } catch (e) {
          failures.push(`${rel} [${mode}]: CDP 错误 — ${e.message}`);
          continue;
        }
        const result = run?.result ?? null;
        if (!result) {
          failures.push(`${rel} [${mode}]: 30s 内没有结果 — diag=${JSON.stringify(run?.diag)}`);
          continue;
        }
        if (result.mode !== mode) failures.push(`${rel} [${mode}]: page reported mode ${result.mode}`);
        if (!Number.isFinite(result.items) || result.items <= 0) failures.push(`${rel} [${mode}]: no render items`);
        if (!Number.isFinite(result.drawCalls) || result.drawCalls <= 0) failures.push(`${rel} [${mode}]: no draw calls`);
        if (!Number.isFinite(result.instances) || result.instances <= 0) failures.push(`${rel} [${mode}]: no instances`);
        if (!result.pixels || !Number.isFinite(result.pixels.totalPixels) || result.pixels.totalPixels <= 0 || !Number.isFinite(result.pixels.foregroundPixels) || result.pixels.foregroundPixels <= 0) {
          failures.push(`${rel} [${mode}]: empty pixel signature — ${JSON.stringify(result.pixels)}`);
        }
        perMode[mode] = result;

        const err = result.validationErrors ?? [];
        if (err.length > 0) failures.push(`${rel} [${mode}]: validation error — ${err[0]}`);
        if (result.skipped?.length > 0) failures.push(`${rel} [${mode}]: 贴图跳过 — ${result.skipped.join('; ')}`);
      }

      const direct = perMode.direct;
      const culled = perMode.culled;
      let parity = '—';
      if (MODES.includes('direct') && MODES.includes('culled')) {
        if (!direct?.pixels || !culled?.pixels) {
          failures.push(`${rel}: Direct/Culled 缺少像素签名`);
        } else {
          const dMax = gridDelta(direct.pixels, culled.pixels);
          const litDelta = Math.abs(direct.pixels.litPixels - culled.pixels.litPixels);
          parity = `gridΔ${dMax} litΔ${litDelta}`;
          if (!(dMax <= 12 && litDelta <= Math.max(40, direct.pixels.litPixels * 0.02))) {
            failures.push(`${rel}: Direct/Culled 亮度网格不一致（${parity}）`);
          }
        }
      }

      const ref = direct ?? culled;
      if (!ref) continue;
      rows.push({
        model: rel,
        items: ref.items,
        mats: ref.materials,
        tex: ref.textures,
        withMat: ref.itemsWithMaterial,
        lit: ref.pixels?.litPixels ?? -1,
        luma: ref.pixels?.avgLuma ?? -1,
        errs: (ref.validationErrors ?? []).length,
        parity,
      });
    }
  } finally {
    cdp?.close();
    if (chrome) terminateChild(chrome.child);
    terminateChild(vite);
    await sleep(300);
    if (chrome) {
      try {
        rmSync(chrome.userDataDir, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  }

  const pad = (s, n) => String(s).padEnd(n);
  console.log(pad('model', 34) + pad('items', 7) + pad('mats', 6) + pad('tex', 5) + pad('withMat', 9) + pad('lit', 9) + pad('luma', 8) + pad('err', 5) + 'parity');
  for (const r of rows) {
    console.log(
      pad(r.model, 34) + pad(r.items, 7) + pad(r.mats, 6) + pad(r.tex, 5) + pad(r.withMat, 9) +
        pad(r.lit, 9) + pad(r.luma, 8) + pad(r.errs, 5) + r.parity,
    );
  }

  console.log('');
  if (failures.length === 0) {
    const parityText = MODES.includes('direct') && MODES.includes('culled') ? '、Direct/Culled 亮度网格近似一致' : '';
    const scopeText = PARTIAL_RUN ? (SMOKE ? '（smoke，部分语料）' : '（部分语料）') : SMOKE ? '（smoke）' : '';
    console.log(`通过 ${rows.length} 个模型：无 validation error、无贴图跳过${parityText}${scopeText}`);
  } else {
    console.log(`❌ ${failures.length} 个问题：`);
    for (const f of failures) console.log(`  - ${f}`);
    process.exitCode = 1;
  }
}

// 全局看门狗：任何环节卡死都要能退出（否则后台子进程会挂住整个进程）。
const watchdog = setTimeout(() => {
  console.error('✖ 超时（45 分钟）—— 强制退出');
  process.exit(1);
}, 45 * 60 * 1000);
watchdog.unref?.();

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => {
    clearTimeout(watchdog);
    // 显式退出：子进程（vite / chrome）可能仍持有 handle。
    process.exit(process.exitCode ?? 0);
  });
