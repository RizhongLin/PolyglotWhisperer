<p align="center">
  <img src="src/pgw/templates/logo.png" width="400" alt="PolyglotWhisperer logo">
</p>

<h1 align="center">PolyglotWhisperer</h1>

<p align="center">Video transcription and translation CLI for language learners.<br>Transcribe, refine, translate, and study — all in one pipeline.</p>

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/)
(optional: [mpv](https://mpv.io/), [Ollama](https://ollama.com/))

```bash
git clone https://github.com/RizhongLin/PolyglotWhisperer.git
cd PolyglotWhisperer
uv sync --all-extras

# API keys (optional — only for cloud providers)
cp .env.example .env
```

spaCy language models download automatically on first use.

For local Postgres-backed development of `pgw serve`:

```bash
docker compose -f docker-compose.dev.yml up -d   # Postgres on :5432
export PGW_DATABASE_URL=postgresql+psycopg://pgw:pgw@localhost:5432/pgw_dev
uv run pgw maintenance migrate                   # alembic upgrade head, with legacy-stamp fallback
```

Without `PGW_DATABASE_URL`, the server falls back to a local SQLite file under the workspace dir — fine for single-user use.

For full-stack production-style testing (pgw + Postgres in containers), the included `docker-compose.yml` boots both services together:

```bash
docker compose up -d            # builds image, starts Postgres + pgw
docker compose logs -f pgw      # entrypoint runs `pgw maintenance migrate` before serving
```

The container's entrypoint (`docker/entrypoint.sh`) runs migrations automatically whenever `PGW_DATABASE_URL` is set, so you never need to invoke alembic by hand on the production path.

## Basic Usage

```bash
# Full pipeline: download → transcribe → (refine + translate combined)
pgw run video.mp4 -l fr --refine --translate en

# From a URL (requires yt-dlp)
pgw run "https://youtube.com/watch?v=..." -l fr --translate en

# Cloud backends (no GPU needed)
pgw run video.mp4 --backend api --llm-backend api --translate en

# Skip transcription — use existing subtitles from the video page
pgw run "https://youtube.com/watch?v=..." --subs --translate en
```

All pipeline output lands in `pgw_workspace/<slug>/<timestamp>/` alongside a `metadata.json`.

## Commands

| Command              | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `pgw run`            | Full pipeline: download, transcribe, refine, translate |
| `pgw transcribe`     | Whisper transcription only (local or cloud API)        |
| `pgw translate`      | Translate existing subtitle files                      |
| `pgw vocab`          | Vocabulary analysis (difficulty tiers, rare words)     |
| `pgw export`         | Export vocabulary as CSV for Anki/spreadsheet          |
| `pgw play`           | Play video with dual subtitles via mpv                 |
| `pgw serve`          | Launch web player for a workspace (or library view)    |
| `pgw clean`          | Clear cached files (downloads, audio, transcriptions)  |
| `pgw languages`      | List all supported languages                           |
| `pgw worker connect` | Run as a remote worker against a `pgw serve` instance  |

## Backends

| Component                | Local                    | Cloud API (default)                            |
| ------------------------ | ------------------------ | ---------------------------------------------- |
| Transcription            | stable-ts (MLX/CUDA/CPU) | OpenAI SDK → Groq, OpenAI, custom servers      |
| Translation / Refinement | Ollama (via OpenAI SDK)  | OpenAI SDK → DeepSeek, Groq, OpenAI, Claude, … |

Any OpenAI SDK-compatible server works too — set `api_base`, `api_key`, and `api_model`:

```toml
# pgw.toml
[whisper]
backend = "api"
api_base = "https://your-whisper-server/v1"
api_key = "sk-..."
api_model = "openai/whisper-1"

[llm]
backend = "api"
api_base = "https://your-llm-server/v1"
api_key = "sk-..."
api_model = "openai/meta-llama-3.1-8b-instruct"
```

```bash
# Per-run overrides
pgw run video.mp4 -l fr --backend api --whisper-model groq/whisper-large-v3-turbo
pgw run video.mp4 -l fr --llm-backend api --llm-model groq/openai/gpt-oss-120b
```

For local LLM, pull a model first: `ollama pull qwen3:8b`

## Configuration

Four layers, lowest to highest: **packaged defaults** → `~/.config/pgw/config.toml` → `./pgw.toml` → `.env` / env vars → CLI flags.

```toml
# pgw.toml
[whisper]
backend = "api"
language = "fr"

[llm]
backend = "api"
target_language = "en"
```

Env vars use `PGW_` prefix: `PGW_WHISPER__BACKEND=api`, `PGW_LLM__API_MODEL=groq/...`. See `.env.example` for all options.

## Web UI

```bash
pgw serve                      # library + studio + player (multi-page SPA)
pgw serve <workspace-dir>       # single-workspace player
```

The web UI is a **React SPA** built from `frontend/` (Vite + TypeScript + TanStack Router/Query + Tailwind v4 + shadcn-style components). The bundle is shipped as static assets inside the Python wheel, so end users never need Node.

Pages:

- **Library** (`/library`) — workspace grid with thumbnails, language pair, difficulty, dates. Click any card to open the player.
- **Studio** (`/studio`) — paste a URL or drop a file, pick source + target language from dropdowns, choose where to run (auto / worker / server), hit _Start_. Live progress cards stream events from the backend; cancel any time, close the tab and come back without losing state. Advanced flags (backends, models, chunk size, ffmpeg start/duration, refine, subs) are tucked behind a disclosure.
- **Player** (`/library/<slug>/<ts>`) — HTML5 video, click-to-seek transcript with anticipate/linger windows, track switcher (bilingual / original / translation), vocab card (top rare words + difficulty), downloads card, re-download fallback for missing video files.

Backend is **FastAPI + uvicorn** serving JSON over `/api/...` and raw workspace files over `/ws/<slug>/<ts>/<file>`. Job state is persisted as append-only JSONL under `<workspace>/.jobs/`, so an in-flight job survives a browser refresh and the server's restart marks orphaned jobs as `interrupted` rather than leaving them stuck.

Knobs:

- `PGW_SERVE_HOST` — bind address (default `127.0.0.1`; Docker sets `0.0.0.0`).
- `PGW_SERVE_MAX_JOBS` — concurrent pipeline workers (default `1`, keeps Whisper warm).
- `PGW_JOBS_RETENTION` — how many finished job logs to keep (default `200`).
- `PGW_DATABASE_URL` — DB connection string (default: SQLite under workspace dir; production: `postgresql+psycopg://...`).
- `PGW_DB_POOL_SIZE` — Postgres connection pool size (default `5`).
- `PGW_ADMIN_EMAIL`, `PGW_ADMIN_PASSWORD` — non-interactive admin bootstrap on first start. Without them, the SPA's `/setup` flow handles it.
- `PGW_SECRET_KEY` — signs CSRF cookies and signed URLs. Required in production.

- `PGW_SPA_DIR` — override the built-in SPA bundle path (useful during Docker-based frontend dev with a host-mounted `dist/` volume).
- `PGW_DEV_BACKEND` — backend URL for the Vite dev server proxy (default `http://127.0.0.1:8321`; set when Docker hosts the backend).

### Auth

When the DB has no users, `pgw serve` runs in **bootstrap mode** — the SPA serves `/setup` on first visit so you can create the admin. After that, `/login` is required for `/api/*` and `/jobs/*`. CSRF protection is double-submit cookie + `X-CSRF-Token` header on every state-changing request.

### Workers

`pgw worker connect --server <url> --token <t>` runs the pipeline on the user's machine using their own IP, GPU, and API keys. The remote `pgw serve` becomes a thin orchestrator + library surface; videos and big artifacts stay local. On the server side, manage tokens with `POST /api/workers`, `GET /api/workers`, `DELETE /api/workers/{id}`. In the Studio, select where to run each job — Auto (prefer connected worker), This machine (explicit worker), or Server (admin-only). When a worker disconnects, its in-flight jobs are marked `interrupted` and any open NDJSON stream reflects it.

## Vocabulary

Each run generates a `vocabulary.<lang>.json` in the workspace. Difficulty tiers (A1–C2) are estimated from word frequency — approximations, not official CEFR levels.

```bash
pgw vocab <workspace> --top 50          # terminal view
pgw export <workspace>                   # → vocabulary.csv for Anki
```

## Docker

The image is multi-stage: a Node stage builds the TypeScript frontend, then a uv stage installs the Python wheel with all extras (transcribe, llm, vocab, export, **serve**). End users never need Node installed.

```bash
docker build -t pgw .
```

All commands work inside Docker — mount your project at `/data` and ensure `.env` has your API keys:

```bash
# Web UI (library + end-to-end pipeline)
docker run --rm -it -p 8321:8321 -v "$PWD:/data" pgw serve --no-open

# Full pipeline (CLI)
docker run --rm -it -v "$PWD:/data" pgw run /data/video.mp4 -l fr \
  --translate en --backend api --llm-backend api --no-play
```

The mounted `/data` is also where `pgw_workspace/` and `pgw_workspace/.jobs/<id>.jsonl` live — keep the volume mount stable across restarts so in-flight jobs reattach cleanly.

**Dev mode** — mount `src/` to iterate on Python without rebuilding:

```bash
docker run --rm -it -p 8321:8321 \
  -v "$PWD:/data" -v "$PWD/src:/app/src" \
  pgw serve --no-open
```

`docker build` always rebuilds the React SPA from the TypeScript source via the `js-builder` stage, so you don't need Node locally to ship a Docker image. The host-side `npm run build` step is only needed when running `pgw serve` directly against your working tree (no Docker), since `pgw serve` reads `src/pgw/templates/dist/` from disk.

## Architecture

```text
src/pgw/
├── auth/         Argon2 passwords, sessions, CSRF, FastAPI deps, env-bootstrap
├── cli/          Typer commands (run, transcribe, translate, serve, worker, …)
├── core/         Config (Pydantic), pipeline orchestrator, events, JobContext
├── db/           SQLAlchemy 2.0 engine + ORM models (users, workspaces, vocab, workers)
├── downloader/   yt-dlp wrapper, URL resolver
├── llm/          OpenAI SDK client, translation, refinement, prompts
├── server/       FastAPI app + JobManager + routes/ (auth, workers, …)
├── subtitles/    Format conversion (VTT/SRT), PDF/EPUB export
├── transcriber/  Whisper backends (stable-ts local + API), segmentation
├── templates/    Built React SPA (templates/dist/) + favicon + brand mark
├── utils/        Audio extraction, cache, logging, spaCy, paths
├── vocab/        Vocabulary analysis + CEFR estimation
└── worker/       `pgw worker connect` agent + protocol (WebSocket to remote server)

frontend/        React SPA source (compiles → src/pgw/templates/dist/)
├── src/
│   ├── routes/         TanStack Router file-based routes (library, studio, player)
│   ├── components/ui/  shadcn-style primitives (Button, Card, Dialog, …)
│   ├── api/            typed fetch client + wire-format types
│   ├── lib/            cn(), VTT parser, formatters, theme hook
│   └── main.tsx        entry: QueryClient + RouterProvider
├── vite.config.ts
└── tsconfig*.json
```

Build the frontend bundle:

```bash
cd frontend && npm ci && npm run build            # → ../src/pgw/templates/dist/
cd frontend && npm run typecheck                  # tsc -b
cd frontend && npm run dev                        # Vite dev server on :5173 (proxies /api → :8321)
```

Tight TypeScript dev loop:

```bash
# terminal 1 — backend
pgw serve --no-open --port 8321

# terminal 2 — Vite dev server (HMR + API proxy)
cd frontend && npm run dev
# open http://127.0.0.1:5173
```

## Tech Stack

| Role            | Library                                                          |
| --------------- | ---------------------------------------------------------------- |
| Transcription   | stable-ts (local), OpenAI SDK (cloud / OpenAI-compatible)        |
| LLMs            | OpenAI SDK → Ollama / DeepSeek / Groq / OpenAI / Claude / custom |
| NLP             | spaCy (POS, lemmatizer), wordfreq (frequency)                    |
| Subtitles       | pysubs2                                                          |
| Download        | yt-dlp                                                           |
| Export          | WeasyPrint (PDF), ebooklib (EPUB)                                |
| CLI             | Typer + Rich                                                     |
| Web UI backend  | FastAPI + uvicorn                                                |
| Web UI frontend | React 19 + Vite + TanStack Router/Query + Tailwind v4            |
| Playback        | mpv (CLI), browser `<video>` (web UI)                            |

## License

[MIT](LICENSE)
