# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Install with optional ASR engines
uv sync --extra parakeet    # NVIDIA Parakeet-TDT
uv sync --extra whisperx    # WhisperX (forced alignment)

# Run the app
uv run streamlit run main.py
# Opens at http://localhost:8501
```

There are no tests or linting configured in this project.

## Project Goal

A **fully offline, local-first** English listening practice tool. The core loop is **Listen → Write → Compare**: upload any audio/video, the app transcribes it into sentence-level clips, and the user practices dictation against each clip with instant word-level diff feedback. Everything runs locally — no cloud APIs, no accounts, no internet required after model download.

## Folder Layout

```
english-practice/
├── main.py                        # Entry point — runs Streamlit (loads .env)
├── pyproject.toml                 # uv config, CUDA wheel index, optional extras
├── .env                           # Optional runtime env (HF_ENDPOINT, etc.)
├── output/                        # Runtime artifacts (gitignored)
│   └── <sha256-hash>/
│       ├── clip_000.wav … clip_NNN.wav
│       ├── segments.json          # Cached transcription
│       ├── ui_state.json          # Persisted last-viewed segment index
│       └── ...
└── dictation/
    ├── app.py                     # Thin bootstrap + misc helpers
    ├── app_core.py                # `Application` class — high-level UI wiring
    ├── ui_templates.py            # CSS/HTML snippets (keeps HTML out of app logic)
    ├── recent_practices.py        # Recent practices rendering (tab)
    ├── analytics.py               # Practice analytics rendering (tab)
    ├── utils/                     # I/O and scanning helpers
    │   ├── __init__.py
    │   ├── io.py                  # paths, load/save, record events
    │   └── scan.py                # scan recent practice folders
    ├── models.py                  # Segment dataclass
    ├── segmentation.py            # Audio slicing + cache I/O
    ├── subtitle.py                # .srt / .vtt parser
    ├── diff.py                    # Word-level diff → colored HTML
    └── asr/
        ├── __init__.py            # ENGINE_REGISTRY (lazy-loading)
        ├── base.py                # ASREngine abstract base class
        ├── _text_split.py         # Fallback timestamp estimation
        ├── whisper.py             # Whisper family
        └── qwen.py                # Qwen3-ASR family
```

## Architecture

### Pipeline: Upload → Transcribe → Slice → Practice

```
┌─────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│  Upload  │───▶│  Transcribe   │───▶│  Slice Audio │───▶│  Dictation   │
│ (app.py) │    │ ASR or .srt   │    │(segmentation)│    │  Workspace   │
└─────────┘    └───────────────┘    └──────────────┘    └──────────────┘
     │                │                     │                    │
  SHA-256 hash    list[Segment]      clip_NNN.wav files    word-level diff
  = cache key     (timestamps+text)  + segments.json       (diff.py → HTML)
```

1. **Upload** — file content hashed (SHA-256, first 16 chars) as cache key and session ID
2. **Transcribe** — ASR engine produces `list[Segment]`, or subtitle parser if .srt/.vtt provided
3. **Slice** — pydub splits audio into `./output/<hash>/clip_NNN.wav` (16kHz mono, 200ms padding), writes `segments.json` cache
4. **Practice** — one segment at a time: audio player (with loop toggle), text input, Compare button triggers word-level diff

### Key Abstractions

**`Segment`** (`models.py`) — The single data unit flowing through every layer: index, start/end timestamps (seconds), text, clip_path, optional word-level timestamps.

**`ASREngine`** (`asr/base.py`) — Abstract base class. Subclasses implement `load(device)` and `transcribe(audio_path) -> list[Segment]`. Optional `unload()` for GPU memory release.

**`ENGINE_REGISTRY`** (`asr/__init__.py`) — Dict-like lazy registry mapping display names to engine classes. Engines are imported only when selected. To add a new engine: create a file in `asr/`, subclass `ASREngine`, add one entry to `_ENGINE_LOADERS` dict.

### ASR Engine Hierarchy

- `WhisperEngine` (large-v3-turbo) — base class for all Whisper variants, handles int8/4bit quantization on CUDA via bitsandbytes
  - `DistilWhisperEngine`, `WhisperSmallEngine`, `WhisperBaseEngine` — only override `model_id`
- `QwenEngine` (Qwen3-ASR-0.6B) — uses `qwen-asr` package with forced aligner for word timestamps
  - `QwenLargeEngine` (1.7B) — only overrides `model_id`

Word-to-sentence grouping: Whisper uses `_group_words_into_sentences()`, Qwen uses `_group_align_items_into_sentences()`. Both split on sentence-ending punctuation and force-split when exceeding max_words (15) or max_duration (8s). Both fall back to `_text_split.segments_from_plain_text()` when word timestamps are unavailable.

### Caching

Content-addressed: re-uploading the same file hits `output/<hash>/segments.json`. Cache is validated by checking all clip files still exist. UI reading position persisted to `output/<hash>/ui_state.json`.



### Environment

- `main.py` sets `HF_ENDPOINT` to `https://hf-mirror.com/` by default (Chinese HuggingFace mirror)
 - `main.py` sets `HF_ENDPOINT` to `https://hf-mirror.com/` by default (Chinese HuggingFace mirror)
 - The app also supports a project-root `.env` file. `main.py` will load simple `KEY=VALUE` lines from `.env` before applying defaults. Example:

```
HF_ENDPOINT="https://hf-mirror.com/"
```
Use `.env` to override `HF_ENDPOINT` or add other environment vars without modifying source.
- Models cache to `~/.cache/huggingface/hub/`
- Requires `ffmpeg` system package for pydub audio processing
- Python 3.12, managed with `uv`
- CUDA 12.1 wheels configured via `[[tool.uv.index]]` in pyproject.toml; remove those sections for CPU-only
- The app no longer writes `output/startup_debug.log`; it uses standard Python logging (stderr) for diagnostics to avoid creating debug files in `output/`.
 - Recent refactors split UI helpers and templates into modules under `dictation/`:
  - `dictation/app_core.py` — `Application` class / high-level UI wiring
  - `dictation/ui_templates.py` — CSS and HTML snippets (keeps HTML out of main code)
  - `dictation/recent_practices.py` — Recent practices rendering
  - `dictation/analytics.py` — Analytics rendering
  - `dictation/services/` — I/O and scanning helpers (`io.py`, `scan.py`)
  Keep this layout when making further refactors.
