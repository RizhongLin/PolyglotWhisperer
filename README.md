# PolyglotWhisperer

Video and audio transcription and translation tool for language learners. Transcribe media with word-level accuracy using Whisper, clean up and translate subtitles with local or cloud LLMs, and play with dual-language subtitles.

Built for watching foreign-language media with accurate, word-for-word subtitles and their translations side by side.

## Features

- **Accurate transcription** — Word-level timestamps via stable-ts (MLX on Apple Silicon, CUDA/CPU elsewhere)
- **Smart subtitle segmentation** — Custom regrouping by punctuation, gaps, and length; spaCy POS tagging moves dangling articles/prepositions across 24 languages
- **LLM-powered cleanup** — Fix ASR errors, remove filler words, normalize punctuation
- **LLM translation** — Translate subtitles to any language using local (Ollama) or cloud LLMs
- **Dual subtitle playback** — Watch videos with original + translated subtitles simultaneously
- **Bilingual subtitles** — Single VTT file with original at bottom + translation at top, works in any player
- **Multiple output formats** — VTT (default), SRT, ASS, and plain text
- **URL support** — Download and process videos from YouTube and other sites via yt-dlp
- **Local-first** — Runs entirely offline with Ollama + Whisper, no cloud APIs required

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [ffmpeg](https://ffmpeg.org/) (audio extraction)
- [mpv](https://mpv.io/) (video playback, optional)
- [Ollama](https://ollama.com/) (local LLM, optional — for cleanup and translation)

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

# Pull a local LLM for cleanup/translation (optional)
ollama pull qwen3:8b
```

> **Note:** spaCy language models (for subtitle segmentation) are downloaded automatically on first use.

### Usage

```bash
# Full pipeline: download, transcribe, translate
pgw run "https://example.com/video" --translate en --no-play

# Play from workspace (auto-detects video + bilingual subtitles)
pgw play pgw_workspace/my-video/20260217_164802/

# Transcribe a local video or audio file
pgw transcribe ~/Videos/news.mp4 --language fr

# Translate existing subtitles
pgw translate subtitles.fr.vtt --to en

# Play with explicit subtitle file
pgw play video.mp4 --subs transcription.fr.vtt
pgw play video.mp4 --bilingual bilingual.fr-en.vtt

# Web player (opens browser, no mpv needed)
pgw serve pgw_workspace/my-video/20260217_164802/
```

### Workspace Output

Each processed file gets a workspace directory:

```plaintext
pgw_workspace/my-video/20260217_164802/
├── video.mp4                    # Symlinked source (copy as fallback)
├── audio.wav                    # Extracted audio
├── transcription.fr.vtt         # Original language subtitles
├── transcription.fr.txt         # Plain text transcript
├── translation.en.vtt           # Translated subtitles
├── translation.en.txt           # Plain text translation
├── bilingual.fr-en.vtt          # Both languages in one file (positioned)
├── transcription.json           # Full Whisper result for reprocessing
└── metadata.json                # Processing parameters and file inventory
```

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
Video/Audio/URL → Download → Extract Audio → Whisper Transcription (stable-ts)
    → Subtitle Regrouping + spaCy Function Word Fix
    → LLM Cleanup (fix ASR errors) → LLM Translation
    → Save VTT/TXT files + bilingual VTT → Play with dual subtitles in mpv
```

## Tech Stack

| Component       | Technology                                                                              |
| --------------- | --------------------------------------------------------------------------------------- |
| Transcription   | [stable-ts](https://github.com/jianfch/stable-ts) (MLX/CUDA/CPU, word-level timestamps) |
| LLM Integration | [LiteLLM](https://github.com/BerriAI/litellm) (Ollama, OpenAI, Claude, etc.)            |
| Local LLM       | [Ollama](https://ollama.com/) with Qwen 3 (default)                                     |
| Video Download  | [yt-dlp](https://github.com/yt-dlp/yt-dlp)                                              |
| NLP             | [spaCy](https://spacy.io/) (POS tagging for subtitle segmentation, 24 languages)        |
| Subtitle I/O    | [pysubs2](https://github.com/tkarabela/pysubs2)                                         |
| Video Playback  | [mpv](https://mpv.io/) (system binary, no Python binding needed)                        |
| CLI             | [Typer](https://typer.tiangolo.com/) + [Rich](https://github.com/Textualize/rich)       |

## Roadmap

- [x] Project setup
- [x] stable-ts transcription with word-level timestamps
- [x] LLM subtitle cleanup and translation
- [x] yt-dlp video download
- [x] mpv dual-subtitle playback
- [x] Full pipeline CLI (`pgw run`)
- [x] Web-based player alternative (`pgw serve`)
- [x] spaCy-based subtitle segmentation (dangling function word fix, 24 languages)
- [ ] Vocabulary extraction for language learners
- [ ] Anki card generation from subtitle pairs

## License

[MIT](LICENSE)
