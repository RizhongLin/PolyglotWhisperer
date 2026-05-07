# pgw frontend

React SPA for the `pgw serve` web UI. Pages: **Library** (workspace
grid), **Studio** (new-job form + live progress), **Player** (video +
transcript + vocab + downloads).

Stack: React 19, Vite 6, TypeScript, TanStack Router (file-based) +
TanStack Query, Tailwind v4, shadcn-style primitives, lucide-react.

## Build / dev

```bash
npm ci
npm run build      # → ../src/pgw/templates/dist/
npm run typecheck  # tsc -b
npm run dev        # Vite dev server :5173 with HMR + API proxy
```

The Python wheel ships only the built `dist/` tree, so end users do
not need Node. `docker build` runs `npm run build` in a Node stage.

## Tight dev loop

```bash
# terminal 1 — backend
pgw serve --no-open --port 8321

# terminal 2 — Vite with HMR; proxies /api /jobs /uploads /ws → :8321
cd frontend && npm run dev
# open http://127.0.0.1:5173
```

## Layout

```
src/
├── routes/                    TanStack Router file-based routes
│   ├── __root.tsx             header + footer + <Outlet />
│   ├── index.tsx              redirect → /library
│   ├── library.tsx            workspace grid
│   ├── library.$slug.$ts.tsx  player
│   └── studio.tsx             new-job form + live jobs strip
├── components/
│   ├── header.tsx             nav + theme toggle
│   └── ui/                    Button, Card, Dialog, Input, Label,
│                              Badge, Progress, Select, Checkbox
├── api/
│   ├── client.ts              typed fetch + openJobStream()
│   └── types.ts               wire-format types (mirrors server)
├── lib/
│   ├── cn.ts                  tailwind-merge wrapper
│   ├── theme.ts               useTheme hook (light/dark + persist)
│   ├── vtt.ts                 minimal WebVTT parser
│   └── format.ts              duration / bytes / stage formatters
├── index.css                  Tailwind + design tokens
└── main.tsx                   QueryClient + RouterProvider
```

## Wire format

Backend types live in `src/api/types.ts` and mirror
`src/pgw/server/jobs.py` (JobRequest, JobRecord, JobEvent) and
`src/pgw/server/app.py` (WorkspaceSummary, WorkspaceDetail,
SubtitleTrack, VocabSummary). Keep them in sync when adding fields.

For OpenAPI codegen (auto-derive `schema.d.ts` from FastAPI):

```bash
# backend running on :8321
npm run openapi
```
