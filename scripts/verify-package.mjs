import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const runNpm = (args, options = {}) => {
  if (process.env.npm_execpath) {
    return execFileSync(process.execPath, [process.env.npm_execpath, ...args], options);
  }
  return execFileSync(npm, args, {
    ...options,
    shell: process.platform === 'win32',
  });
};
const requestedTarball = process.argv[2] ? resolve(process.argv[2]) : undefined;
const lockfile = JSON.parse(readFileSync(new URL('../package-lock.json', import.meta.url), 'utf8'));
const webgpuTypesVersion = lockfile.packages['node_modules/@webgpu/types'].version;
const typescriptVersion = lockfile.packages['node_modules/typescript'].version;
const tempRoot = mkdtempSync(join(tmpdir(), 'hpg-consumer-'));
const consumerDir = join(tempRoot, 'consumer');
let tarball = requestedTarball;

try {
  mkdirSync(consumerDir);

  if (!tarball) {
    runNpm(['pack', '--pack-destination', tempRoot], { stdio: 'inherit' });
    const tarballs = readdirSync(tempRoot).filter((file) => file.endsWith('.tgz'));
    if (tarballs.length !== 1) {
      throw new Error(`Expected one package tarball, found ${tarballs.length}`);
    }
    tarball = join(tempRoot, tarballs[0]);
  }

  runNpm(['init', '--yes'], { cwd: consumerDir, stdio: 'ignore' });
  runNpm(
    [
      'install',
      '--ignore-scripts',
      '--no-package-lock',
      '--no-save',
      tarball,
      `@webgpu/types@${webgpuTypesVersion}`,
      `typescript@${typescriptVersion}`,
    ],
    { cwd: consumerDir, stdio: 'inherit' },
  );

  writeFileSync(
    join(consumerDir, 'consumer.mjs'),
    [
      "import * as runtime from '@hpg/runtime';",
      "for (const name of ['Renderer', 'GeometryArena', 'CullingPipeline', 'TimestampQuery', 'parseGltf', 'flattenScene', 'extractFrustumPlanes', 'sphereInFrustum', 'importGltfAsset', 'sceneToRenderItems', 'MaterialStore', 'VS_INSTANCED_COMPACTION']) {",
      "  if (!(name in runtime)) throw new Error(`${name} export is missing`);",
      '}',
      "if (Object.keys(runtime).length !== 43) throw new Error(`expected 43 runtime exports, received ${Object.keys(runtime).length}`);",
      "if (typeof runtime.identity !== 'function') throw new Error('identity export is missing');",
      'console.log(`consumer import ok: ${Object.keys(runtime).length} exports`);',
    ].join('\n'),
  );
  writeFileSync(
    join(consumerDir, 'consumer.ts'),
    [
      "import { identity, type FlattenedNode, type SubmitOptions } from '@hpg/runtime';",
      'const matrix = identity();',
      'const submitOptions: SubmitOptions = {};',
      'const node: FlattenedNode = { name: "node", worldMatrix: matrix };',
      'void matrix; void submitOptions; void node;',
    ].join('\n'),
  );
  writeFileSync(
    join(consumerDir, 'tsconfig.json'),
    JSON.stringify(
      {
        compilerOptions: {
          target: 'ES2022',
          module: 'ESNext',
          moduleResolution: 'bundler',
          strict: true,
          skipLibCheck: false,
          types: ['@webgpu/types'],
          noEmit: true,
        },
        files: ['consumer.ts'],
      },
      null,
      2,
    ),
  );

  execFileSync(process.execPath, ['consumer.mjs'], { cwd: consumerDir, stdio: 'inherit' });
  execFileSync(process.execPath, [join(consumerDir, 'node_modules', 'typescript', 'bin', 'tsc'), '-p', 'tsconfig.json'], {
    cwd: consumerDir,
    stdio: 'inherit',
  });
  console.log('consumer typecheck ok');
} finally {
  rmSync(tempRoot, { recursive: true, force: true });
}
