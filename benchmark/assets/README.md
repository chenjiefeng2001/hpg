# Benchmark Assets

This directory contains the GLB fixtures used by the asset parser regression tests, the compatibility
audit, and the browser material validation harness.

## Corpus

The supported corpus is the 23 `.glb` files under `benchmark/assets/models`. The regression tests
lock the model set and the expected geometry results. The CI workflows perform a non-empty corpus
check before running the audit; the full list remains a test contract.

`benchmark/assets/models/medium/FlightHelmet.gltf` is a separate non-GLB fixture. It is not included
in the 23-model GLB corpus, and `npm run audit` scans `.glb` files only.

## Categories

### Light

Basic single-mesh compatibility fixtures.

| File | Purpose |
|------|---------|
| `light/BoxTextured.glb` | Small textured model |
| `light/Avocado.glb` | Large single-mesh model |
| `light/BoomBox.glb` | Multi-material exporter output |
| `light/Corset.glb` | Dense single-mesh geometry |

### Medium

Normal rendering and importer fixtures.

| File | Purpose |
|------|---------|
| `medium/DamagedHelmet.glb` | PBR material and texture path |
| `medium/Lantern.glb` | Multiple meshes and materials |
| `medium/WaterBottle.glb` | Packed index and vertex data |

### Heavy

Multi-mesh, animation, and skinning-related inputs.

| File | Purpose |
|------|---------|
| `heavy/BrainStem.glb` | Many meshes and materials |
| `heavy/CesiumMilkTruck.glb` | Multi-mesh scene |
| `heavy/CesiumMan.glb` | Skeleton and animation attributes |
| `heavy/Fox.glb` | Small skinned model |
| `heavy/Duck.glb` | Small multi-material model |

### Feature

Exercised glTF feature combinations. Unsupported features are reported through structured
`GltfAsset.warnings`; they are not silently treated as supported behavior.

| File | Feature focus |
|------|---------------|
| `feature/AlphaBlendModeTest.glb` | Alpha modes |
| `feature/ClearCoatTest.glb` | Clearcoat extension |
| `feature/EmissiveStrengthTest.glb` | Emissive strength |
| `feature/IridescenceLamp.glb` | Iridescence extension |
| `feature/MetalRoughSpheres.glb` | Metallic-roughness workflow |
| `feature/MorphPrimitivesTest.glb` | Morph targets |
| `feature/MultiUVTest.glb` | Non-aligned and multi-UV parsing |
| `feature/SheenChair.glb` | Sheen extension |
| `feature/TextureEncodingTest.glb` | Texture encoding |
| `feature/TextureLinearInterpolationTest.glb` | Texture filtering |
| `feature/TransmissionTest.glb` | Transmission extension |

`MultiUVTest.glb` is parseable with the current accessor path. `TEXCOORD_1` is currently ignored by
the importer and reported as a warning; the behavior is covered by the warning regression tests.

## Usage

Start the development server to inspect a model through the viewer:

```bash
npm run dev
```

Open `demo/glb-viewer.html`, then use the file picker or the `?asset=<url>` parameter. Asset URLs
under `benchmark/assets/models` are available in the development server; they are not guaranteed to
be copied into the production demo build.

Run the compatibility audit over the complete corpus or a path filter:

```bash
npm run audit
npm run audit heavy
```

Run the real Chrome and WebGPU validation with a local browser installation:

```bash
npm run verify:browser
npm run verify:browser -- heavy
```

`benchmark/asset-bench.ts` is a separate synthetic-model parse benchmark. It is not a package script
and is not part of the CI or release gate.

## Sources and licenses

The fixtures are collected from upstream model collections, including the Khronos glTF Sample Assets:

- https://github.com/KhronosGroup/glTF-Sample-Assets

Do not assume that every file has the same license. Check the upstream repository and the per-model
license before redistributing a fixture.
