# pgw frontend

TypeScript bundle for the `pgw serve` library page — handles the new-job
dialog, in-flight job cards, live progress streaming, cancellation, and
reattach across browser refresh.

## Build

```bash
npm install
npm run build      # → ../src/pgw/templates/jobs.js
npm run watch      # rebuild on changes during dev
npm run typecheck  # tsc --noEmit
```

The Python wheel ships only the built artifact, so end users do not need
Node. Contributors editing TypeScript run `npm run build` and commit the
generated `src/pgw/templates/jobs.js`.

## Layout

- `src/types.ts` — wire-format types mirroring `src/pgw/server/jobs.py`
- `src/api.ts` — typed `fetch` wrappers (`/jobs`, `/uploads`, NDJSON stream)
- `src/dom.ts` — small typed DOM helpers
- `src/jobs.ts` — entry point: form, card lifecycle, reattach
- `build.mjs` — esbuild driver (bundles to IIFE, inline source maps in dev)
