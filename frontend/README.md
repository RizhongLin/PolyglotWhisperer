# pgw frontend

React SPA for the `pgw serve` web UI. Pages: **Library** (workspace
grid), **Studio** (new-job form + live progress), **Player** (video +
transcript + vocab + downloads), **Login / Setup** (auth bootstrap).

Stack: React 19, Vite 6, TypeScript, TanStack Router (file-based) +
TanStack Query, Tailwind v4, shadcn-style primitives, lucide-react.

## Auth

`__root.tsx` reads `/api/auth/state` once on mount and redirects:

- DB has no users → `/setup` (admin bootstrap form).
- Has admin, not authenticated → `/login`.
- Otherwise renders the requested route.

`api/client.ts` sets `credentials: 'include'` so the session cookie
travels on every request, and reads the `pgw_csrf` cookie to attach
`X-CSRF-Token` on all non-`GET` requests. POST `/api/auth/login`,
`/api/auth/setup`, `/api/auth/logout` use these helpers under the hood.

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

When the backend runs in Docker instead of locally:

```bash
# terminal 1 — Docker backend
docker run --rm -it -p 8321:8321 -v "$PWD:/data" pgw serve --no-open

# terminal 2 — Vite with HMR proxying to Docker host
cd frontend && PGW_DEV_BACKEND=http://localhost:8321 npm run dev
```

Set `PGW_DEV_BACKEND` to point the Vite proxy at a different backend host.

### Env vars for dev

| Var               | Purpose                    | Default                 |
| ----------------- | -------------------------- | ----------------------- |
| `PGW_DEV_BACKEND` | Backend URL for Vite proxy | `http://127.0.0.1:8321` |

## Layout

```
src/
├── routes/                    TanStack Router file-based routes
│   ├── __root.tsx             header + footer + auth gate + <Outlet />
│   ├── index.tsx              redirect → /library
│   ├── library.tsx            parent layout (Outlet for nested routes)
│   ├── library.index.tsx      workspace grid
│   ├── library.$slug.$ts.tsx  player
│   ├── studio.tsx             new-job form (language dropdowns, executor selector, live jobs strip)
│   ├── login.tsx              email + password sign-in
│   └── setup.tsx              first-time admin creation form
├── components/
│   ├── header.tsx             nav + theme toggle
│   └── ui/                    Button, Card, Dialog, Input, Label,
│                              Badge, Progress, Select, Checkbox
├── api/
│   ├── client.ts              typed fetch + auth + CSRF + openJobStream()
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

Backend types live in `src/api/types.ts` and mirror:

- `src/pgw/server/jobs.py` — `JobRequest`, `JobRecord`, `JobEvent`
- `src/pgw/server/app.py` — `WorkspaceSummary`, `WorkspaceDetail`, `SubtitleTrack`, `VocabSummary`
- `src/pgw/server/routes/auth.py` — `AuthState`, `MeResponse`
- `src/pgw/server/routes/workers.py` — `WorkerSummary`

Keep them in sync when adding fields. The CI assertion that hand-written
types match server-side Pydantic models is on the P9 hardening list.

For OpenAPI codegen (auto-derive `schema.d.ts` from FastAPI):

```bash
# backend running on :8321
npm run openapi
```
