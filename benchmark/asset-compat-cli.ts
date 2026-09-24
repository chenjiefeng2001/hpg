/**
 * CLI 入口：`npm run audit [filter]`
 *
 * 与审计逻辑（benchmark/asset-compat.ts）分开，因为 vite-node 会把脚本路径从
 * process.argv 中剥掉，模块内无法判断是否被直接运行。
 */

import { main } from './asset-compat';

main(process.argv[2]);
