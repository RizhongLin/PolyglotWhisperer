# PolyglotWhisperer

Video transcription and translation tool for language learners. Transcribe videos with word-level accuracy using Whisper, clean up and translate subtitles with local or cloud LLMs, and play with dual-language subtitles.

Built for watching foreign-language media (like Swiss French news from RTS) with accurate, word-for-word subtitles and their translations side by side.

## Features

- **Accurate transcription** — Word-level timestamps via stable-ts (MLX on Apple Silicon, CUDA/CPU elsewhere)
- **LLM-powered cleanup** — Fix ASR errors, remove filler words, normalize punctuation
- **LLM translation** — Translate subtitles to any language using local (Ollama) or cloud LLMs
- **Dual subtitle playback** — Watch videos with original + translated subtitles simultaneously
- **Bilingual subtitles** — Single VTT file with original at bottom + translation at top, works in any player
- **Multiple output formats** — VTT (default), SRT, ASS, and plain text
- **URL support** — Download and process videos from RTS, SRF, and other sites via yt-dlp
- **Local-first** — Runs entirely offline with Ollama + Whisper, no cloud APIs required

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [ffmpeg](https://ffmpeg.org/) (audio extraction)
- [mpv](https://mpv.io/) (video playback)
- [Ollama](https://ollama.com/) (local LLM, optional)

### Installation

```bash
# Clone and install
git clone https://github.com/RizhongLin/PolyglotWhisperer.git
cd PolyglotWhisperer
uv sync --all-extras

# Pull a local LLM for translation (optional)
ollama pull qwen3:8b
```

### Usage

```bash
# Full pipeline: transcribe, clean, translate, and play
pgw run "https://www.rts.ch/play/tv/..." --translate en

# Transcribe a local video
pgw transcribe ~/Videos/news.mp4 --language fr

# Translate existing subtitles
pgw translate subtitles.fr.srt --to en

# Play with dual subtitles
pgw play video.mp4 --subs subtitles.fr.srt --translation subtitles.en.srt
```

## Supported Languages

Whisper supports **100 languages** for transcription. Run `pgw languages` to see the full list.

stable-ts provides word-level timestamps natively for all languages using Whisper's built-in alignment. Use `pgw languages` to see the full list.

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

```
Video/URL → Download → Extract Audio → Whisper Transcription (stable-ts)
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
| Subtitle I/O    | [pysubs2](https://github.com/tkarabela/pysubs2)                                         |
| Video Playback  | [mpv](https://mpv.io/) via [python-mpv](https://github.com/jaseg/python-mpv)            |
| CLI             | [Typer](https://typer.tiangolo.com/) + [Rich](https://github.com/Textualize/rich)       |

## Roadmap

- [x] Project setup
- [x] stable-ts transcription with word-level timestamps
- [x] LLM subtitle cleanup and translation
- [x] yt-dlp video download (RTS/SRF support)
- [x] mpv dual-subtitle playback
- [x] Full pipeline CLI (`pgw run`)
- [ ] Web-based player alternative
- [ ] Vocabulary extraction for language learners
- [ ] Anki card generation from subtitle pairs

## License

MIT
