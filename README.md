# PolyglotWhisperer

Video and audio transcription and translation tool for language learners. Transcribe media with word-level accuracy using Whisper (locally or via cloud APIs), translate subtitles with LLMs, and play with dual-language subtitles.

Built for watching foreign-language media with accurate, word-for-word subtitles and their translations side by side.

## Features

- **Accurate transcription** — Word-level timestamps via Whisper with two backends:
  - **Local**: stable-ts with MLX on Apple Silicon, CUDA/CPU elsewhere
  - **Cloud API**: Groq, OpenAI, and other providers via LiteLLM (fast, no GPU needed)
- **Smart subtitle segmentation** — Custom regrouping by punctuation, gaps, and length; spaCy POS tagging moves dangling articles/prepositions across 24 languages
- **Romance clitic handling** — Apostrophe-ending tokens (l', d', qu') in French, Italian, Catalan, etc. are kept with the next word, not left dangling
- **LLM translation** — Translate subtitles to any language using local (Ollama) or cloud LLMs
- **Dual subtitle playback** — Watch videos with original + translated subtitles simultaneously
- **Bilingual subtitles** — Single VTT file with original at bottom + translation at top, works in any player
- **Audio extraction cache** — Shared cache across workspaces avoids redundant ffmpeg extraction
- **Multiple output formats** — VTT (default), SRT, ASS, and plain text
- **URL support** — Download and process video or audio from YouTube and other sites via yt-dlp
- **Local-first** — Runs entirely offline with Ollama + Whisper, no cloud APIs required

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [ffmpeg](https://ffmpeg.org/) (audio extraction)
- [mpv](https://mpv.io/) (video playback, optional)
- [Ollama](https://ollama.com/) (local LLM, optional — for translation)

```bash
# macOS
brew install uv ffmpeg mpv
brew install --cask ollama   # optional

# Ubuntu/Debian
sudo apt install ffmpeg mpv
curl -fsSL https://astral.sh/uv/install.sh | sh           # uv
curl -fsSL https://ollama.com/install.sh | sh             # optional
```

### Installation

```bash
git clone https://github.com/RizhongLin/PolyglotWhisperer.git
cd PolyglotWhisperer
uv sync --all-extras

# Pull a local LLM for translation (optional)
ollama pull qwen3:8b
```

> **Note:** spaCy language models (for subtitle segmentation) are downloaded automatically on first use.

### API Keys (for cloud providers)

If using cloud APIs (Groq, OpenAI, etc.) for transcription or LLM processing:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Or export directly in your shell:

```bash
export GROQ_API_KEY=gsk_...
export OPENAI_API_KEY=sk-...
```

The `.env` file is loaded automatically on startup. Shell exports take precedence over `.env` values.

### Usage

```bash
# Full pipeline: download, transcribe, translate
pgw run "https://example.com/video" --translate en --no-play

# Use cloud API for transcription (no local GPU needed)
pgw run "https://example.com/video" --backend api --translate en --no-play

# Override the API model
pgw run video.mp4 --backend api --model openai/whisper-1 -l fr

# Play from workspace (auto-detects video + bilingual subtitles)
pgw play pgw_workspace/my-video/20260217_164802/

# Transcribe a local video or audio file
pgw transcribe ~/Videos/news.mp4 --language fr

# Transcribe via cloud API
pgw transcribe audio.wav --backend api -l fr

# Translate existing subtitles
pgw translate subtitles.fr.vtt --to en

# Play with explicit subtitle file
pgw play video.mp4 --subs transcription.fr.vtt
pgw play video.mp4 --bilingual bilingual.fr-en.vtt

# Web player (opens browser, no mpv needed)
pgw serve pgw_workspace/my-video/20260217_164802/
```

### Configuration

Config is loaded in layers (lowest to highest priority):

1. `config/default.toml` — shipped defaults
2. `~/.config/pgw/config.toml` — user-level
3. `./pgw.toml` — project-level
4. `.env` file + environment variables
5. CLI flags

Example `pgw.toml`:

```toml
[whisper]
backend = "api"                            # "local" or "api"
local_model = "large-v3-turbo"             # model for local backend
api_model = "groq/whisper-large-v3-turbo"  # model for API backend
language = "fr"

[llm]
model = "ollama_chat/qwen3:8b"             # or "groq/llama-3.3-70b-versatile"
translation_enabled = true
target_language = "en"
```

Environment variables use the `PGW_` prefix with `__` as delimiter:

```bash
PGW_WHISPER__BACKEND=api
PGW_WHISPER__LANGUAGE=de
PGW_LLM__MODEL=groq/llama-3.3-70b-versatile
```

### Workspace Output

Each processed file gets a workspace directory. All cached data lives under `.cache/` — downloaded media and extracted audio are shared across workspaces. Files are symlinked into workspaces (with copy fallback) to avoid duplication.

```plaintext
pgw_workspace/
├── .cache/                              # Shared media cache (cross-workspace)
│   ├── audio/                           # Extracted audio keyed by video metadata hash
│   │   └── a1b2c3d4e5f6g7h8.wav
│   └── downloads/                       # yt-dlp downloaded media (video/audio, URL-keyed)
│       ├── .downloads.jsonl
│       └── My Video_abc123.mp4
└── my-video/
    └── 20260217_164802/
        ├── video.mp4                    # Symlinked from source (via shared cache utils)
        ├── audio.wav                    # Symlinked from .cache/audio/
        ├── transcription.fr.vtt         # Original language subtitles
        ├── transcription.fr.txt         # Plain text transcript
        ├── translation.en.vtt           # Translated subtitles
        ├── translation.en.txt           # Plain text translation
        ├── bilingual.fr-en.vtt          # Both languages in one file
        ├── transcription.json           # Full Whisper result (local backend only)
        └── metadata.json               # Processing parameters and file inventory
```

Both video and audio linking use the same `link_or_copy()` utility from the shared cache module (`utils/cache.py`).

## Transcription Backends

### Local (default)

Uses [stable-ts](https://github.com/jianfch/stable-ts) to run Whisper locally. Best quality with word-level timestamps and custom regrouping. Requires GPU/large model downloads.

```bash
pgw transcribe audio.wav -l fr                        # default: large-v3-turbo on MLX
pgw transcribe audio.wav -l fr --model medium          # smaller model
pgw transcribe audio.wav -l fr --device cpu             # force CPU
```

### Cloud API

Uses [LiteLLM](https://github.com/BerriAI/litellm) to call cloud Whisper APIs. Fast, cheap, no GPU needed. Requires API key. 25 MB file size limit per request.

```bash
pgw transcribe audio.wav --backend api -l fr                             # default: groq/whisper-large-v3-turbo
pgw transcribe audio.wav --backend api --model openai/whisper-1 -l fr    # OpenAI
```

Supported API providers (any LiteLLM-compatible transcription endpoint):

- **Groq**: `groq/whisper-large-v3-turbo`, `groq/whisper-large-v3` — fast, free tier available
- **OpenAI**: `openai/whisper-1` — supports word-level timestamps natively

For long audio files exceeding 25 MB, use `--start` and `--duration` to clip, or use the local backend.

## Supported Languages

Whisper supports **100 languages** for transcription with word-level timestamps via stable-ts. Run `pgw languages` to see the full list.

<details>
<summary>Common language codes</summary>

| Code | Language   | Alignment |
| ---- | ---------- | --------- |
| `fr` | French     | yes       |
| `en` | English    | yes       |
| `de` | German     | yes       |
| `es` | Spanish    | yes       |
| `it` | Italian    | yes       |
| `pt` | Portuguese | yes       |
| `nl` | Dutch      | yes       |
| `zh` | Chinese    | yes       |
| `ja` | Japanese   | yes       |
| `ko` | Korean     | yes       |
| `ar` | Arabic     | yes       |
| `ru` | Russian    | yes       |
| `hi` | Hindi      | yes       |
| `tr` | Turkish    | yes       |
| `pl` | Polish     | yes       |
| `sv` | Swedish    | yes       |
| `da` | Danish     | yes       |
| `fi` | Finnish    | yes       |
| `uk` | Ukrainian  | yes       |
| `vi` | Vietnamese | yes       |

</details>

## How It Works

```plaintext
Video/Audio/URL
  → Download (yt-dlp, cached)
  → Extract Audio (ffmpeg, cached across workspaces)
  → Transcription
      ├─ Local: stable-ts Whisper → regroup + spaCy function word fix
      └─ API:   LiteLLM → word regrouping → spaCy clitic fix
  → LLM Translation (optional)
  → Save VTT/TXT + bilingual VTT
  → Play with dual subtitles in mpv
```

## Tech Stack

| Component       | Technology                                                                         |
| --------------- | ---------------------------------------------------------------------------------- |
| Transcription   | [stable-ts](https://github.com/jianfch/stable-ts) (local, MLX/CUDA/CPU)            |
| Cloud Whisper   | [LiteLLM](https://github.com/BerriAI/litellm) (Groq, OpenAI, etc.)                 |
| LLM Integration | [LiteLLM](https://github.com/BerriAI/litellm) (Ollama, OpenAI, Claude, Groq, etc.) |
| Local LLM       | [Ollama](https://ollama.com/) with Qwen 3 (default)                                |
| Video Download  | [yt-dlp](https://github.com/yt-dlp/yt-dlp)                                         |
| NLP             | [spaCy](https://spacy.io/) (POS tagging for subtitle segmentation, 24 languages)   |
| Subtitle I/O    | [pysubs2](https://github.com/tkarabela/pysubs2)                                    |
| Video Playback  | [mpv](https://mpv.io/) (system binary, no Python binding needed)                   |
| CLI             | [Typer](https://typer.tiangolo.com/) + [Rich](https://github.com/Textualize/rich)  |

## Roadmap

- [x] Project setup
- [x] stable-ts transcription with word-level timestamps
- [x] LLM subtitle translation
- [x] yt-dlp video download
- [x] mpv dual-subtitle playback
- [x] Full pipeline CLI (`pgw run`)
- [x] Web-based player alternative (`pgw serve`)
- [x] spaCy-based subtitle segmentation (dangling function word fix, 24 languages)
- [x] Cloud API transcription backend (Groq, OpenAI via LiteLLM)
- [x] Audio extraction cache (shared across workspaces)
- [x] Unified clitic handling for Romance languages
- [ ] Vocabulary extraction for language learners
- [ ] Anki card generation from subtitle pairs

## License

[MIT](LICENSE)
