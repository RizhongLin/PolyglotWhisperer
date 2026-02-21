# PolyglotWhisperer

Video transcription and translation CLI for language learners. Transcribe with Whisper (local or cloud API), translate with LLMs, play with dual subtitles — all in one pipeline.

## Features

- **Whisper transcription** with word-level timestamps — local (stable-ts, MLX/CUDA/CPU) or cloud API (Groq, OpenAI via LiteLLM)
- **Smart subtitle segmentation** — spaCy POS tagging fixes dangling articles, prepositions, and Romance clitics (l', d', qu') across 24 languages
- **LLM translation** — any language via Ollama (local) or cloud LLMs (OpenAI, Groq, Claude, etc.)
- **Vocabulary analysis** — CEFR difficulty estimation (A1–C2), rare word extraction with context and translations
- **Parallel text export** — side-by-side PDF/EPUB for printing or e-readers
- **Batch processing** — multiple files, glob patterns, URL lists, with error-continue
- **Dual subtitle playback** — original + translation simultaneously in mpv
- **Audio cache** — shared across workspaces, avoids redundant ffmpeg extraction
- **Multiple formats** — VTT, SRT, ASS, plain text, bilingual VTT, parallel PDF
- **URL support** — YouTube and other sites via yt-dlp

## Quick Start

### Prerequisites

- Python 3.12+, [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/)
- Optional: [mpv](https://mpv.io/) (playback), [Ollama](https://ollama.com/) (local LLM)

```bash
# macOS
brew install uv ffmpeg mpv
brew install --cask ollama   # optional

# Ubuntu/Debian
sudo apt install ffmpeg mpv
curl -fsSL https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh   # optional
```

### Installation

```bash
git clone https://github.com/RizhongLin/PolyglotWhisperer.git
cd PolyglotWhisperer
uv sync --all-extras

# Pull a local LLM for translation (optional)
ollama pull qwen3:8b
```

spaCy language models are downloaded automatically on first use.

<details>
<summary>Install only what you need</summary>

```bash
uv sync --extra transcribe    # Local Whisper (stable-ts, MLX)
uv sync --extra download      # URL downloading (yt-dlp)
uv sync --extra llm           # LLM translation (LiteLLM, Ollama)
uv sync --extra nlp           # spaCy NLP (POS tagging, lemmatizer)
uv sync --extra vocab         # Vocabulary analysis (wordfreq + spaCy)
uv sync --extra export        # PDF/EPUB export (WeasyPrint, ebooklib)
```

</details>

### API Keys (for cloud providers)

```bash
cp .env.example .env   # edit and add your keys
# or export directly:
export GROQ_API_KEY=gsk_...
export OPENAI_API_KEY=sk-...
```

### Usage

```bash
# Full pipeline: download → transcribe → translate → play
pgw run "https://example.com/video" --translate en --no-play

# Cloud API transcription (no local GPU needed)
pgw run "https://example.com/video" --backend api --translate en --no-play

# Batch processing
pgw run *.mp4 --translate en --no-play
pgw run urls.txt --backend api --translate en --no-play

# Transcribe only
pgw transcribe video.mp4 -l fr
pgw transcribe *.mp4 --backend api -l fr

# Translate existing subtitles
pgw translate subtitles.fr.vtt --to en

# Vocabulary analysis
pgw vocab pgw_workspace/my-video/20260217_164802/

# Playback
pgw play pgw_workspace/my-video/20260217_164802/
pgw serve pgw_workspace/my-video/20260217_164802/   # web player
```

### Configuration

Config layers (lowest to highest priority): `config/default.toml` → `~/.config/pgw/config.toml` → `./pgw.toml` → `.env` + env vars → CLI flags.

```toml
# pgw.toml
[whisper]
backend = "api"                            # "local" or "api"
api_model = "groq/whisper-large-v3-turbo"
language = "fr"

[llm]
model = "ollama_chat/qwen3:8b"
target_language = "en"
```

Environment variables use `PGW_` prefix: `PGW_WHISPER__BACKEND=api`, `PGW_LLM__MODEL=groq/llama-3.3-70b-versatile`.

### Workspace Output

```plaintext
pgw_workspace/
├── .cache/                           # Shared media cache
│   ├── audio/                        # Extracted audio (cross-workspace)
│   └── downloads/                    # yt-dlp downloads
└── my-video/
    └── 20260217_164802/
        ├── video.mp4                 # Symlinked from source
        ├── audio.wav                 # Symlinked from cache
        ├── transcription.fr.vtt      # Original subtitles
        ├── transcription.fr.txt      # Plain text
        ├── translation.en.vtt        # Translated subtitles
        ├── bilingual.fr-en.vtt       # Dual-language VTT
        ├── parallel.fr-en.pdf        # Side-by-side PDF
        ├── vocabulary.fr.json        # CEFR analysis + rare words
        ├── transcription.json        # Full Whisper result (local only)
        └── metadata.json
```

## Transcription Backends

| Backend             | Technology                                        | Pros                                                   | Limits                             |
| ------------------- | ------------------------------------------------- | ------------------------------------------------------ | ---------------------------------- |
| **Local** (default) | [stable-ts](https://github.com/jianfch/stable-ts) | Best quality, word-level timestamps, custom regrouping | Requires GPU / model downloads     |
| **Cloud API**       | [LiteLLM](https://github.com/BerriAI/litellm)     | Fast, cheap, no GPU                                    | 25 MB file limit, API key required |

```bash
# Local
pgw transcribe audio.wav -l fr                     # large-v3-turbo on MLX
pgw transcribe audio.wav -l fr --model medium       # smaller model

# Cloud API
pgw transcribe audio.wav --backend api -l fr                          # Groq (default)
pgw transcribe audio.wav --backend api --model openai/whisper-1 -l fr  # OpenAI
```

## Vocabulary Analysis

Each processed video gets a vocabulary profile: CEFR level estimation via [wordfreq](https://github.com/rspeer/wordfreq), top 30 rare words with context and translation, spaCy lemmatization to group inflected forms.

```bash
pgw vocab pgw_workspace/my-video/20260217_164802/ --top 50
```

## How It Works

```plaintext
Video/Audio/URL
  → Download (yt-dlp, cached)
  → Extract Audio (ffmpeg, cached)
  → Transcribe (local Whisper or cloud API + spaCy segmentation)
  → Translate (LLM, optional)
  → Export (VTT/TXT/bilingual VTT/PDF) + Vocabulary Analysis
  → Play (mpv dual subtitles)
```

## Tech Stack

| Component     | Technology                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Transcription | [stable-ts](https://github.com/jianfch/stable-ts) (MLX/CUDA/CPU)                                                               |
| Cloud APIs    | [LiteLLM](https://github.com/BerriAI/litellm) (Groq, OpenAI, Ollama, Claude)                                                   |
| NLP           | [spaCy](https://spacy.io/) (24 languages) + [wordfreq](https://github.com/rspeer/wordfreq)                                     |
| Export        | [WeasyPrint](https://doc.courtbouillon.org/weasyprint/stable/) (PDF) + [ebooklib](https://github.com/aerkalov/ebooklib) (EPUB) |
| Subtitles     | [pysubs2](https://github.com/tkarabela/pysubs2)                                                                                |
| Download      | [yt-dlp](https://github.com/yt-dlp/yt-dlp)                                                                                     |
| Playback      | [mpv](https://mpv.io/)                                                                                                         |
| CLI           | [Typer](https://typer.tiangolo.com/) + [Rich](https://github.com/Textualize/rich)                                              |

## Supported Languages

Whisper supports **100 languages** — run `pgw languages` for the full list. spaCy POS tagging and clitic handling covers 24 languages.

<details>
<summary>Common language codes</summary>

| Code | Language   | Code | Language | Code | Language   |
| ---- | ---------- | ---- | -------- | ---- | ---------- |
| `fr` | French     | `zh` | Chinese  | `pl` | Polish     |
| `en` | English    | `ja` | Japanese | `sv` | Swedish    |
| `de` | German     | `ko` | Korean   | `da` | Danish     |
| `es` | Spanish    | `ar` | Arabic   | `fi` | Finnish    |
| `it` | Italian    | `ru` | Russian  | `uk` | Ukrainian  |
| `pt` | Portuguese | `hi` | Hindi    | `vi` | Vietnamese |
| `nl` | Dutch      | `tr` | Turkish  |      |            |

</details>

## Roadmap

- [x] Whisper transcription (local + cloud API) with word-level timestamps
- [x] LLM translation + dual subtitle playback
- [x] spaCy subtitle segmentation + Romance clitic handling (24 languages)
- [x] Audio cache, batch processing, vocabulary analysis, parallel text export
- [x] Streaming pipeline event system
- [ ] Hosted demo (Gradio on Hugging Face Spaces)
- [ ] Speaker diarization
- [ ] Anki card generation from subtitle pairs

## License

[MIT](LICENSE)
