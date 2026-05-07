// esbuild driver — bundles src/jobs.ts into the Python package's templates dir.
// The Python wheel ships only the built artifact, so end users never need Node.
import { build, context } from 'esbuild';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outFile = resolve(here, '../src/pgw/templates/jobs.js');
mkdirSync(dirname(outFile), { recursive: true });

const watch = process.argv.includes('--watch');

/** @type {import('esbuild').BuildOptions} */
const options = {
  entryPoints: [resolve(here, 'src/jobs.ts')],
  bundle: true,
  format: 'iife',
  target: ['es2022'],
  outfile: outFile,
  // Inline source maps in dev (`--watch`) only — keeps the prod bundle
  // ~30 % smaller for end users who never need them.
  sourcemap: watch ? 'inline' : false,
  minify: !watch,
  legalComments: 'none',
  banner: { js: '// Generated from frontend/src — do not edit by hand.' },
  logLevel: 'info',
};

if (watch) {
  const ctx = await context(options);
  await ctx.watch();
  console.log(`watching for changes → ${outFile}`);
} else {
  await build(options);
  if (!existsSync(outFile)) throw new Error(`Build produced no output at ${outFile}`);
}
