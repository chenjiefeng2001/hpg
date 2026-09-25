# @hpg/runtime

> WebGPU rendering runtime: **RenderItem[] in → GPU commands out.**
> Auto-batching, GPU-driven culling, indirect drawing — no scene graph, no ECS, no opinions.

**hpg does not own your scene graph, ECS, lights, or general material and asset systems.**
It takes a flat list of render work and turns it into efficient GPU commands. The package includes
a narrow glTF and material helper path for producing render work, not a complete asset engine.

After the npm publication workflow has run, install the default technical-baseline channel with:

```bash
npm install @hpg/runtime@next
```

If a release is explicitly published with the `latest` distribution tag, use:

```bash
npm install @hpg/runtime
```

Runtime and build requirements: Node `>=18` and a WebGPU-capable browser (Chrome 113+ / Edge 113+).
The package is ESM-only and exposes an `import` entry; it does not provide a CommonJS `require` entry.

TypeScript consumers must also have the WebGPU ambient types available, because the public API is
typed in terms of `GPUDevice` / `GPUCanvasContext` / `GPUBuffer`. The shipped `.d.ts` files do not
self-reference those types:

```
npm install -D @webgpu/types
```

A strict consumer configuration should use the same module resolution mode as the project:

```jsonc
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "types": ["@webgpu/types"],
    "skipLibCheck": false
  }
}
```

`@webgpu/types` is declared as an **optional** peer dependency, so npm will not install it for you.
Without it, `tsc` reports `Cannot find name 'GPUDevice'` from both application code and the package
declarations when `skipLibCheck` is false. Setting `skipLibCheck` to true only suppresses those
errors; it does not restore the missing types.

> **Known limitation:** under `moduleResolution: "node16"` / `"nodenext"` the emitted declarations
> are not resolvable because relative declaration imports do not include a `.js` extension. This
> produces `TS2834`; use `bundler` or `node` resolution until the declaration extension fix is made.
> Runtime is unaffected.

The real Chrome validation command is a separate WebGPU gate. It requires Node 22 or newer, a local
Chrome or Chromium installation, and WebGPU support. Set `CHROME_PATH` when the browser is not in a
standard location. The same check is exposed as a reusable GitHub workflow and is required by the
release and npm publication workflows; hosted WebGPU availability still depends on the runner.

## Release Status

**`0.2.0` is a stable `0.x` technical baseline — not a production-ready release.**

What it commits to: a WebGPU-first **Render Work → GPU Execution** runtime (batching, sorting,
GPU culling, indirect execution, geometry management) with textured rendering validated against a
23-model real GLB corpus in real Chrome. Unimplemented asset features are **surfaced explicitly**
(`GltfAsset.warnings`, registration-time errors) rather than silently producing wrong output.

What it does **not** commit to: being a production-ready general 3D rendering runtime. There is no
workload evidence for long-running behaviour, resource pressure, target-hardware performance, or
real project integration — those stay **UNKNOWN**, not TODO. See `PLAN.md` →
“Phase 15.5 Finalize — Baseline Freeze”.

Frozen verification is a historical, reproducible baseline snapshot:

```text
npm run typecheck      0 error
npm test               19 files / 282 passed
npm run audit          parse ok 23/23 · BROKEN = 0
npm run verify:browser 23/23 · validation errors 0 · Direct/Culled brightness-grid parity
npm run verify:culling-matrix 0% / 10% / 50% / 100% · drawArgs + mapping valid
npm run build          dist/lib/index.js + declarations + source maps
npm run build:demo     all five page entries emitted
```

The package boundary is verified from outside the repository with `npm run verify:package`:
`npm pack` installs the tarball in a fresh directory, imports it as bare Node ESM, and runs a
strict TypeScript consumer check with `@webgpu/types`. The published file allowlist is `dist/lib`,
`LICENSE`, `CHANGELOG.md`, and `README.md`, plus npm's generated `package.json` metadata. It does not contain `src`, `test`, `demo`, or `benchmark`. The build size is a toolchain snapshot, not a package
contract; source maps remain part of the published `dist/lib` artifact.

## Quick Start

The following is a structural example. WGSL, the global bind-group layout, and the vertex buffers
are application-owned; the example shows the hpg contract rather than a complete render loop.

```ts
import { Renderer, identity, translation, type VertexLayoutDesc } from '@hpg/runtime';

declare const canvas: HTMLCanvasElement;
declare const globalLayout: GPUBindGroupLayout;
declare const globalBuffer: GPUBuffer;
declare const vertices: Float32Array;
declare const indices: Uint16Array;

const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
const adapter = (await navigator.gpu.requestAdapter())!;
const device = await adapter.requestDevice();
context.configure({ device, format, alphaMode: 'premultiplied' });

const renderer = Renderer.create({ device, context, format });
const vertexLayouts: VertexLayoutDesc[] = [{
  arrayStride: 24,
  stepMode: 'vertex',
  attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
}];

const pipeline = renderer.registerPipeline({
  vsCode: 'application vertex shader',
  fsCode: 'application fragment shader',
  vertexLayouts,
  bindGroupLayouts: [globalLayout],
  globalBindings: [{ binding: 0, buffer: globalBuffer }],
  targets: [{ format }],
});

const geometry = renderer.createGeometry(vertices, pipeline.desc.vertexLayouts, indices);
const transform = translation(new Float32Array(16), 1, 0, 0);

renderer.submit([{
  geometry,
  pipeline,
  transforms: new Float32Array([...identity(), ...transform]),
}]);

renderer.dispose();
```

### GPU Culling Path

For large scenes, skip CPU-side batching and let the GPU cull. The pipeline **must** be
registered with `compaction: true` and a compaction vertex shader — group 1 then binds
`[instanceBuffer, compactedIndices]` instead of the instance buffer alone:

```ts
import { VS_INSTANCED_COMPACTION, FS_COLOR } from '@hpg/runtime';

const culled = renderer.registerPipeline({
  label: 'culled',
  compaction: true,
  // Built-in compaction shaders are trusted by the runtime.
  vsCode: VS_INSTANCED_COMPACTION,
  fsCode: FS_COLOR,
  vertexLayouts: [...],
  bindGroupLayouts: [globalLayout],
  globalBindings: [{ binding: 0, buffer: uniformBuffer }],
  depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  targets: [{ format }],
});

const stats = renderer.submitCulled(items, viewProjectionMatrix);
```

`globalLayout`, `uniformBuffer`, `items`, and `viewProjectionMatrix` are application-owned values in
this fragment. The GPU computes visibility and draws only visible instances through indirect draw.
`submitCulled()` only accepts affine transforms; custom compaction WGSL must explicitly set
`compactionContract: 'hpg-compaction-v1'`.

Per-instance bounding spheres are derived from `Geometry.bounds` (local AABB) times each
instance matrix; provide `RenderItem.bounding` to override. Items sharing one pipeline are
grouped per geometry — each geometry gets its own indirect draw args.

## API Reference

### Core

| Symbol | Description |
|--------|-------------|
| `Renderer.create(desc)` | Create renderer from descriptor |
| `renderer.submit(items, opts?)` | Submit `RenderItem[]` (auto-batch, CPU-side). `opts.camera` enables camera-relative depth sorting |
| `renderer.submitCulled(items, vpMatrix)` | Submit with GPU frustum culling + indirect draw (requires `compaction: true` pipelines) |
| `renderer.geometryArena` | Geometry pool manager |
| `renderer.createGeometry(vtx, layouts, idx?, indexFormat?, primitive?)` | Convenience wrapper over `geometryArena.createGeometry` |
| `renderer.instanceBuffer` | Current transient instance buffer (diagnostics; rebuilt on growth) |
| `renderer.dispose()` | Release all GPU resources |

### Geometry

| Symbol | Description |
|--------|-------------|
| `geometryArena.createGeometry(vtx, layouts, idx?, indexFormat?)` | Upload one vertex-step layout + optional typed indices; format is inferred for `Uint32Array`/`Uint16Array` and mismatches are rejected |
| `geometryArena.destroyGeometry(geo)` | Free arena allocation (GPU memory recycled) |
| `geometryArena.stats()` | Pool usage: capacity, used bytes, fragmentation, live count |
| `geometry.bounds` | Local-space AABB derived from the position attribute (used by `submitCulled`) |

The current Arena supports one vertex-step layout; multi-slot and instance-step layouts are rejected explicitly.
`RenderItem.instanceData`, when provided, must exactly match the pipeline's post-matrix float count.

### Asset Pipeline

| Symbol | Description |
|--------|-------------|
| `parseGltf(arrayBuffer)` | Parse a self-contained `.glb` (POSITION/NORMAL/UV/TANGENT + materials + embedded image bytes) |
| `importGltfAsset(asset, renderer, opts)` | Upload to `GeometryArena`; returns meshes, materials and **world-space** bounds |
| `sceneToRenderItems(scene, pipeline, materials?)` | Build `RenderItem[]` (node world matrices + material color/`bindGroup`) |
| `MaterialStore.create(device, asset, decoder)` | Decode + upload `baseColorTexture` (sRGB) → per-material bind group at **group 2** |
| `createBrowserImageDecoder()` | Default `ImageDecoder` (GLB image bytes → `createImageBitmap` → RGBA8) |
| `createMaterialBindGroupLayout(device)` | `group 2` layout: `texture` / `sampler` / `material uniform` |
| `ImageDecoder` | Inject your own decoder (Node tests / asset pipelines) — the parser never decodes |

`ImportedScene.dispose()` releases the scene's arena geometry; any `RenderItem` created from that scene is invalid afterward and must not be submitted.

### Material path (group 2)

Bind groups are slotted by the renderer: **group 0 = global, group 1 = instances (internal), group 2+ = your own layouts**.
A pipeline that declares a group 2 layout requires every `RenderItem` using it to provide `bindGroup`; missing state is rejected before recording instead of inheriting a previous material.
To sample `baseColorTexture`:

The following fragment assumes that `asset`, `scene`, `globalLayout`, `uniformBuffer`, and the
canonical `vertexLayouts` are application-owned. The imported geometry uses the canonical stride-48
layout: position, normal, UV, and tangent.

```ts
import {
  FS_MATERIAL,
  MaterialStore,
  VS_INSTANCED_MATERIAL,
  createBrowserImageDecoder,
  sceneToRenderItems,
} from '@hpg/runtime';

const store = await MaterialStore.create(device, asset, createBrowserImageDecoder());

const pipeline = renderer.registerPipeline({
  vsCode: VS_INSTANCED_MATERIAL,
  fsCode: FS_MATERIAL,
  vertexLayouts,
  bindGroupLayouts: [globalLayout, store.layout],
  globalBindings: [{ binding: 0, buffer: uniformBuffer }],
  depth: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  targets: [{ format }],
});

const items = sceneToRenderItems(scene, pipeline, store);
renderer.submit(items);
```

`OPAQUE` and `MASK` (`alphaCutoff` + `discard`) are supported; `BLEND` still reports a warning
(it needs transparent ordering + blend state).

### GPU Culling Helpers

| Symbol | Description |
|--------|-------------|
| `extractFrustumPlanes(vpColumnMajor)` | 6 normalized frustum planes (CPU mirror of `CS_FRUSTUM_CULL`) |
| `sphereInFrustum(planes, x, y, z, r)` | Sphere/plane visibility test |
| `CullingPipeline` | Compute culling pipeline (used internally by `submitCulled`) |

### Instrumentation

| Symbol | Description |
|--------|-------------|
| `new TimestampQuery(device, 2)` | GPU timestamp query |
| `tq.timestampWrites(begin, end)` | Standard pass-descriptor timestamp query slots |
| `tq.resolve(encoder)` | Resolve query set |
| `await tq.readback(device)` | Async GPU time in nanoseconds |
| `tq.destroy()` | Release query resources; call only after readback settles |

### Lifecycle

```
Renderer
  └── GeometryArena
        └── Geometry   ←── createGeometry() / destroyGeometry()

submit(RenderItem[])
      ↓
  sort → batch → upload → GPU execution

renderer.dispose()
      ↓
  all GPU resources released
```

### Ownership

| Resource | Owner | Lifetime |
|----------|-------|----------|
| `Renderer` | User | Until `dispose()` |
| `Geometry` | `GeometryArena` | Until `destroyGeometry()` |
| `RenderItem.transforms` | User | Must outlive `submit()` call |
| GPU buffers | `GeometryArena` / `RingBuffer` | Managed internally |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Application / ECS / SceneGraph            │
│                                                 │
│  Produce: RenderItem[] per frame                │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  @hpg/runtime                                   │
│                                                 │
│  Sort → Batch → Bind → Execute                  │
│                                                 │
│  • 47-bit spatial key encoding                  │
│  • Auto-batching (instanced draw)               │
│  • Pipeline cache (content hash)                │
│  • Transient memory (ring buffer)               │
│  • GPU frustum culling (compute shader)         │
│  • Compaction mapping + indirect draw           │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
              WebGPU API
```

**hpg is not a general rendering engine.** It does not own the application scene graph, ECS, camera
model, lighting system, post-processing pipeline, or complete asset and material system. It does
provide a narrow glTF and material helper path for producing `RenderItem[]`. Your application decides
*what* to draw; hpg decides how to turn that render work into GPU commands.

## Benchmark

### Phase 6.5 — Synthetic Instance Benchmark

```text
Benchmark: 3 paths / 32 cases, 500–500K instances
GPU:       RTX 3070 Laptop / D3D11
Browser:   Edge 154
```

| Path | What it does | CPU submit | GPU execution |
|------|-------------|-----------|---------------|
| Direct | Per-item draw calls, no batching | Measured | Measured |
| Batcher | Auto-batching, instanced draw | Measured | — |
| GPU Culling | Compute frustum culling + indirect draw | Measured | Measured |

**CPU Submission Crossover**: GPU Culling CPU submit crosses Batcher between 1K–5K candidates.

**Key insight**: GPU Culling CPU cost is invariant to visibility ratio — it always processes all N candidates. The value is in replacing GPU rasterization O(N) with O(M) where M = visible instances.

### Phase 10E — Real-Workload Parse Benchmark (Node.js)

Synthetic GLB models with Blender-like characteristics: hierarchy, varying mesh sizes, multiple materials.

```text
Model                         GLB Size   Meshes     Verts   Indices   Mats   Parse(ms)
────────────────────────────────────────────────────────────────────────────────────────
XS   (10 meshes, ~10K verts)    279 KB      10    10,837     9,486     3       0.28
S    (50 meshes, ~50K verts)    1.3 MB      50    49,625    45,543     5       1.02
M    (200 meshes, ~500K verts)  11.9 MB    200   474,142   483,114    10       8.36
L    (500 meshes, ~2M verts)    49.3 MB    500  1,995,789 1,731,243   15      33.86
XL   (1000 meshes, ~5M verts)  122.9 MB   1000  4,964,886 4,574,274   20      75.86
```

**Parse scales linearly: ~0.62 ms/MB.** Material count has negligible effect. 50 MB GLB ≈ 34ms.

The synthetic parser benchmark lives in `benchmark/asset-bench.ts`. It requires a separately
provisioned TypeScript runner and is not a package script or a release gate.

### Phase 10E — Browser Full-Pipeline Benchmark (WebGPU)

`benchmark/glb-bench.html` — measures the full pipeline: Parse → Import → CPU Submit → GPU Execution.

```bash
npm run dev
```

Open `benchmark/glb-bench.html` in the browser.

> Benchmark results are hardware- and browser-dependent. Do not treat specific numbers as hpg performance characteristics.

## Design Principles

1. **No scene model** — `RenderItem[]` is the only input. No Object3D, no Scene Graph.
2. **Progressive enhancement** — 1 object and 10,000 objects use the same code path and shader.
3. **500-object rule** — No optimization should make a 500-object scene slower. The normal submit path reuses sorting and batching scratch; allocation behavior is not assumed for every GPU-culling path.
4. **Batching is an optimization, not a requirement** — When batch conditions aren't met, falls back to direct draw with identical output.

## Auto-Batch Conditions

| Condition | Rule |
|-----------|------|
| `pipeline` object | Same registered object identity (not merely numeric id) |
| `geometry` | Same reference (same arena allocation) |
| `bindGroup` | Same reference |
| Instance data layout | Unified across compatible items |

## Phase History

- **Phase 1–4** — Core renderer, math library, geometry arena, pipeline cache, auto-batching
- **Phase 5** — GPU frustum culling (Gribb-Hartmann), indirect drawing
- **Phase 6** — Full GPU-driven: compute culling → compaction mapping → indirect draw, zero readback
- **Phase 6.5** — 3-path benchmark, CPU submission crossover analysis
- **Phase 7A** — API stabilization: `Renderer.create()`, `GeometryArena.stats()`, multi vertex buffer
- **Phase 7B** — GPU timestamp query: `TimestampQuery` class, benchmark GPU timing
- **Phase 8A** — Package boundary: Vite library build, compiled JS output, clean public API surface
- **Phase 8B** — Documentation: Quick Start, API Reference, Architecture diagram
- **Phase 8C** — Release readiness: LICENSE (MIT), CHANGELOG.md, CI gate
- **Phase 9** — Release: v0.1.0 release record; historical npm publication is not part of the current GitHub Release workflow
- **Phase 10A** — glTF/GLB parser: POSITION/NORMAL/UV/TANGENT, PBR metallic roughness, scene hierarchy
- **Phase 10B** — Asset importer: glTF → hpg adapter, interleaved vertices, UV V-flip, sceneToRenderItems
- **Phase 10C** — Tests: 6 glTF tests, 122/122 total
- **Phase 10D** — Browser GLB viewer: orbit camera, Direct/Culled toggle, file input for external GLB
- **Phase 10E** — Real-workload benchmark: Node parse benchmark (0.62ms/MB linear), browser full-pipeline benchmark
- **Phase 11A–11C** — Render-chain correctness: world-space asset bounds + adaptive near/far, geometry-keyed indirect grouping, fixed frustum plane extraction, ring-buffer bind group refresh, material colors, glTF accessor/alignment robustness
- **Phase 12** — Renderer audit: per-batch instance stride, world-space depth sorting, compaction bind-group reuse, GeometryArena cross-pool free-list aliasing
- **Phase 13** — Browser-verified fixes: instance binding size (dynamic-offset range) and draw-args write ordering — the two reasons multi-mesh models rendered nothing in a real browser while the headless tests were green
- **Phase 14** — Real-asset compatibility: `npm run audit` feature/support matrix over 23 real GLBs, structured `GltfAsset.warnings` for unimplemented features, and the corpus turned into a regression suite (`test/asset-corpus.test.ts`)
- **Phase 15** — Material/texture path: `baseColorTexture` + sampler + `MaterialStore` (group 2, sRGB) with an injectable `ImageDecoder`; `alphaMode=MASK`; TANGENT preserved in the canonical layout; execution-grouping regression lock; real-Chrome validation of all 23 corpus models (`npm run verify:browser`)
- **Phase 15.5** — State audit of the execution boundary: no evidence-backed capability gap; fixed 4 declared-but-silently-ignored contracts (`GlobalBinding` slice, pipeline cache key, `modelMatrixOffset`, a dead `sceneToRenderItems` parameter)
- **Phase 15.5 Finalize — Baseline Freeze** — development stops here: `0.2.0` frozen as a stable `0.x` technical baseline (six local verification gates recorded), unknowns registered as UNKNOWN instead of TODO, public API reviewed without changes. The next development action is triggered by new workload evidence, not by roadmap completeness

## Real-Asset Compatibility

`npm run audit` reads the raw glTF JSON of every GLB under `benchmark/assets/models` (independently of the parser) and reports what real exporters actually use, plus how hpg currently handles it. Status over 23 real assets:

| Status | Feature | Models |
|---|---|---|
| **broken** (silently wrong data) | — none: 23/23 parse, no sparse accessors, no compressed meshes | 0 |
| **wrong** (visibly wrong output) | `alphaMode=BLEND` (rendered opaque) | 2 |
| | skinning (`JOINTS_0`/`WEIGHTS_0`, rendered in bind pose) | 3 |
| **supported now** | `baseColorTexture` (needs an `ImageDecoder`), `alphaMode=MASK`, `TANGENT` | 22 / 4 / 8 |
| **lossy** | other PBR textures, `TEXCOORD_1`, morph targets, animations, `KHR_materials_*` extensions | 1–13 |

The first real blocker was the **material/texture path**, not parsing — Phase 15A closed it. `parseGltf` still reports every unimplemented feature through `asset.warnings` (also shown in the viewer's notice panel) so a loaded model never silently looks wrong for no stated reason.

```bash
npm run audit            # full report over the whole corpus
npm run audit heavy      # filter by path substring
```

## CI and GitHub Release

The repository uses four GitHub Actions workflows. All of them require the source, tests, benchmark
assets, and `package-lock.json` to be committed; a local untracked file is not available to a clean
runner.

| Workflow | Trigger | Automated checks |
|----------|---------|------------------|
| `.github/workflows/ci.yml` | Push to `main`, pull request, or manual dispatch | Node 18/20/22 typecheck and tests; Node 22 asset audit, library/demo builds, package-boundary inspection, external consumer verification, and tarball artifact upload |
| `.github/workflows/release.yml` | Push a `vX.Y.Z` tag | Version consistency, browser gate, typecheck, tests, asset audit, builds, package-boundary inspection, external consumer verification, and GitHub Release creation with the final `.tgz` |
| `.github/workflows/publish-npm.yml` | Manual dispatch for an existing release tag | Re-runs the browser and release gates, verifies the exact package tarball consumer, and publishes `@hpg/runtime` to npm with provenance |
| `.github/workflows/browser-gate.yml` | Manual dispatch or reusable workflow call | Runs the real Chrome/WebGPU asset validation separately from the Node matrix; release and npm publication depend on it |

The browser WebGPU gate is intentionally separate. For a local run:

```bash
npm run verify:browser
```

The repository also provides `.github/workflows/browser-gate.yml` for a manual GitHub-hosted-runner
check and for the release/publish dependency chain. If the runner does not expose WebGPU, the gate
fails closed; use a machine with a supported browser and run the local command instead.

The release workflow creates a GitHub Release and attaches the npm tarball. It does not run
`npm publish`; publication is a separate protected operation through `publish-npm.yml`.

### npm publication

1. Create an npm automation token with publish permission for `@hpg/runtime`.
2. Add it to the repository as the `NPM_TOKEN` GitHub Actions secret.
3. Push the version tag and wait for the GitHub Release checks to pass.
4. Run the `Publish npm package` workflow with that tag and choose `next` or `latest`.

The workflow validates the tag, reruns the browser and package gates, runs the external consumer check, and
publishes with npm provenance. The default `next` channel reflects that `0.2.0` is a technical
baseline rather than a production-ready release. Use `latest` only when that channel should expose
the version to unqualified `npm install @hpg/runtime` commands.

Until publication, other projects can consume the `.tgz` attached to the corresponding GitHub
Release.

Before creating a tag:

1. Update `package.json`, `package-lock.json`, and `CHANGELOG.md` together.
2. Run the local and browser gates listed below on the release commit.
3. Commit the complete source tree and benchmark assets.
4. Create and push the version tag:

```bash
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

The tag name must match the version in both `package.json` and `package-lock.json`.

## Development

| Command | Description |
|---------|-------------|
| `npm ci` | Clean install from the committed lockfile |
| `npm run dev` | Start the Vite development server |
| `npm run bench` | Start the browser benchmark |
| `npm run audit [filter]` | Audit the real GLB corpus; an optional path filter is supported |
| `npm run verify:browser [filter]` | Run real Chrome and WebGPU validation; requires Node 22+, Chrome, and WebGPU |
| `npm test` | Unit and integration tests; no GPU is required |
| `npm run test:watch` | Run Vitest in watch mode |
| `npm run typecheck` | Run the strict TypeScript check |
| `npm run build` | Build the library JavaScript, declarations, and source maps |
| `npm run build:demo` | Build all five demo and benchmark pages |
| `npm run verify:package` | Pack, install in a fresh directory, import, and typecheck the consumer boundary |
| `npm pack --dry-run` | Preview the package file list and lifecycle build |
| `npm publish --dry-run --access public` | Validate the publish lifecycle without publishing |
| `npm run clean` | Remove generated `dist` output |

Coverage is configured in `vitest.config.ts`, but `@vitest/coverage-v8` and a coverage script are
not currently part of the package contract. Do not use coverage as a release gate until both are
added and verified.

### Demo Files

| File | Content |
|------|---------|
| `index.html` | Basic demo (5 instances, auto-batch) |
| `phase5.html` | 500 objects, GPU culling + indirect draw |
| `demo/glb-viewer.html` | GLB viewer — load a `.glb` via file picker or `?asset=<url>` (e.g. `?asset=/benchmark/assets/models/medium/DamagedHelmet.glb`), orbit camera, Direct/Culled toggle. GPU validation errors and unimplemented glTF features (`asset.warnings`) are surfaced in on-page overlays |
| `benchmark/index.html` | 3-path benchmark (500–500K instances) |
| `benchmark/glb-bench.html` | Real-workload benchmark (synthetic GLB, full pipeline timing) |

The `?asset=<url>` form is intended for the Vite development server, where files under
`benchmark/assets/models` are available. The production demo build is not guaranteed to copy those
external benchmark assets.
