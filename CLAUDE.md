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
english-learning/
├── main.py                          # Entry point — sets HF_ENDPOINT, delegates to app.main()
├── pyproject.toml                   # uv config, CUDA wheel index, optional extras
├── output/                          # Runtime artifact directory (gitignored)
│   └── <sha256-hash>/              # One directory per unique uploaded file
│       ├── clip_000.wav … clip_NNN.wav   # Per-sentence audio clips (16kHz mono)
│       ├── segments.json            # Cached transcription metadata
│       └── ui_state.json            # Persisted last-viewed segment index
│
└── english_learning/                # Main Python package
    ├── app.py                       # Streamlit UI — upload, transcription, dictation workspace
    ├── models.py                    # Core dataclass: Segment
    ├── segmentation.py              # Audio slicing (pydub/ffmpeg) + segment cache I/O
    ├── subtitle.py                  # .srt / .vtt parser → list[Segment]
    ├── diff.py                      # Word-level diff engine → colored HTML
    ├── analysis.py                  # Connected-speech analysis via local LLM (WIP, not wired into UI)
    │
    └── asr/                         # Pluggable ASR engine subsystem
        ├── __init__.py              # ENGINE_REGISTRY — lazy-loading engine lookup
        ├── base.py                  # ASREngine abstract base class
        ├── _text_split.py           # Fallback: proportional timestamp estimation from plain text
        ├── whisper.py               # Whisper family (base/small/large-v3-turbo/distil-large-v3)
        └── qwen.py                  # Qwen3-ASR family (0.6B/1.7B) with forced aligner
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
- Models cache to `~/.cache/huggingface/hub/`
- Requires `ffmpeg` system package for pydub audio processing
- Python 3.12, managed with `uv`
- CUDA 12.1 wheels configured via `[[tool.uv.index]]` in pyproject.toml; remove those sections for CPU-only
