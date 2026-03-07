# 🎧 English Listening Practice

A **fully offline** English listening practice tool for English learners. Upload any audio or video, have it transcribed into sentence-level clips, then practice dictation with instant word-level diff feedback.

**Core loop: Listen → Write → Compare**


![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Multiple ASR engines** — Whisper (base / small / large-v3-turbo / distil-large-v3) and Qwen3-ASR (0.6B / 1.7B) with word-level forced alignment
- **Subtitle shortcut** — drop a `.srt` or `.vtt` file to skip ASR entirely
- **Sentence-level clips** — audio sliced by sentence with 200 ms padding, cached to disk
- **Dictation workspace** — type what you hear, then compare with instant word-level diff (green = correct, red = missed/wrong)
- **Loop toggle** — repeat a clip on demand without page reload
- **Resume from where you left off** — last-viewed segment persisted per audio file
- **Content-addressed cache** — re-uploading the same file loads instantly from `output/<sha256>/`
- **GPU or CPU** — select device in the sidebar; CUDA 12.1 wheels pre-configured

---

## Quick Start

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/) package manager
- `ffmpeg` system package

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Install and Run

```bash
git clone https://github.com/linttian/english-learning.git
cd english-learning

# CPU-only (default PyPI wheels)
uv sync

# NVIDIA GPU (CUDA 12.1 wheels, pre-configured in pyproject.toml)
# No extra flags needed — torch/torchaudio route to CUDA automatically
uv sync

uv run streamlit run main.py
# Opens at http://localhost:8501
```

### Optional ASR Engines

```bash
uv sync --extra parakeet    # NVIDIA Parakeet-TDT
uv sync --extra whisperx    # WhisperX (forced alignment)
```

---

## Usage

1. **Upload** an audio or video file (mp3, wav, mp4, mkv, m4a, ogg, flac, aac)
2. Optionally upload a `.srt` / `.vtt` subtitle to skip transcription
3. Click **Transcribe** — the app slices the audio into sentence clips
4. Work through each segment: listen, type what you hear, click **Compare**
5. Expand **"Reveal exact transcript"** to see the answer

---

## Project Structure

```
english-learning/
├── main.py                        # Entry point — runs Streamlit
├── pyproject.toml                 # uv config, CUDA wheel index, optional extras
├── output/                        # Runtime artifacts (gitignored)
│   └── <sha256-hash>/
│       ├── clip_000.wav … clip_NNN.wav
│       ├── segments.json          # Cached transcription
│       └── ui_state.json          # Persisted segment index
│
└── dictation/
    ├── app.py                     # Streamlit UI
    ├── models.py                  # Segment dataclass
    ├── segmentation.py            # Audio slicing + cache I/O
    ├── subtitle.py                # .srt / .vtt parser
    ├── diff.py                    # Word-level diff → colored HTML
    ├── analysis.py                # Connected-speech analysis (WIP)
    └── asr/
        ├── __init__.py            # ENGINE_REGISTRY (lazy-loading)
        ├── base.py                # ASREngine abstract base class
        ├── _text_split.py         # Fallback timestamp estimation
        ├── whisper.py             # Whisper family
        └── qwen.py                # Qwen3-ASR family
```

---

## Adding a New ASR Engine

1. Create `dictation/asr/myengine.py`, subclass `ASREngine`, implement `load(device)` and `transcribe(audio_path) -> list[Segment]`
2. Add one entry to `_ENGINE_LOADERS` in `dictation/asr/__init__.py`

That's it — the new engine appears in the sidebar dropdown automatically.

---

## Environment Notes
- Models cache to `~/.cache/huggingface/hub/`
- For CPU-only deployment, remove the `[[tool.uv.index]]` and `[tool.uv.sources]` blocks from `pyproject.toml`

---

## License

MIT
