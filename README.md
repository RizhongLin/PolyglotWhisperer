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

## Basic Usage

```bash
# Full pipeline: download → transcribe → refine → translate
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

| Command          | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| `pgw run`        | Full pipeline: download, transcribe, refine, translate |
| `pgw transcribe` | Whisper transcription only (local or cloud API)        |
| `pgw translate`  | Translate existing subtitle files                      |
| `pgw refine`     | Fix ASR errors in subtitles with an LLM                |
| `pgw vocab`      | Vocabulary analysis (difficulty tiers, rare words)     |
| `pgw export`     | Export vocabulary as CSV for Anki/spreadsheet          |
| `pgw play`       | Play video with dual subtitles via mpv                 |
| `pgw serve`      | Launch web player for a workspace (or library view)    |
| `pgw clean`      | Clear cached files (downloads, audio, transcriptions)  |
| `pgw languages`  | List all supported languages                           |

## Backends

| Component                | Local (default)          | Cloud API                            |
| ------------------------ | ------------------------ | ------------------------------------ |
| Transcription            | stable-ts (MLX/CUDA/CPU) | LiteLLM → Groq, OpenAI, etc.         |
| Translation / Refinement | Ollama                   | LiteLLM → Groq, OpenAI, Claude, etc. |

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

## Web Player

```bash
pgw serve                      # library view of all workspaces
pgw serve <workspace-dir>       # single video player
```

The player includes keyboard shortcuts (← → ↑ ↓ Space f 1-3), click-to-seek in transcript, and click-any-word to reveal difficulty + translation when vocabulary data is available.

## Vocabulary

Each run generates a `vocabulary.<lang>.json` in the workspace. Difficulty tiers (A1–C2) are estimated from word frequency — approximations, not official CEFR levels.

```bash
pgw vocab <workspace> --top 50          # terminal view
pgw export <workspace>                   # → vocabulary.csv for Anki
```

## Docker

```bash
docker build -t pgw .
```

All commands work inside Docker — mount your project at `/data` and ensure `.env` has your API keys:

```bash
# Web player
docker run --rm -it -p 8321:8321 -v "$PWD:/data" pgw serve --no-open

# Full pipeline
docker run --rm -it -v "$PWD:/data" pgw run /data/video.mp4 -l fr \
  --translate en --backend api --llm-backend api --no-play
```

**Dev mode** — mount `src/` to skip rebuilds on code changes:

```bash
docker run --rm -it -p 8321:8321 \
  -v "$PWD:/data" -v "$PWD/src:/app/src" \
  pgw serve --no-open
```

Only `pyproject.toml` or `uv.lock` changes require a rebuild.

## Architecture

```
src/pgw/
├── cli/          Typer commands (run, transcribe, translate, etc.)
├── core/         Config (Pydantic), pipeline orchestrator, events
├── downloader/   yt-dlp wrapper, URL resolver
├── llm/          LiteLLM client, translation, refinement, prompts
├── server/       Web player HTTP handlers + HTML templates
├── subtitles/    Format conversion (VTT/SRT), PDF/EPUB export
├── transcriber/  Whisper backends (stable-ts local + API), segmentation
├── templates/    HTML/CSS/JS for web player and library
├── utils/        Audio extraction, cache, logging, spaCy, paths
└── vocab/        Vocabulary analysis + CEFR estimation
```

## Tech Stack

| Role          | Library                                       |
| ------------- | --------------------------------------------- |
| Transcription | stable-ts (local), LiteLLM (cloud)            |
| LLMs          | LiteLLM + Ollama                              |
| NLP           | spaCy (POS, lemmatizer), wordfreq (frequency) |
| Subtitles     | pysubs2                                       |
| Download      | yt-dlp                                        |
| Export        | WeasyPrint (PDF), ebooklib (EPUB)             |
| CLI           | Typer + Rich                                  |
| Playback      | mpv, built-in HTTP server                     |

## License

[MIT](LICENSE)
