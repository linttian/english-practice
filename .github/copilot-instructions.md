# AI Agent Guidelines

## Code Style
- Follow Python 3.12 conventions.
- Use type hints for all function signatures.
- Reference `dictation/models.py` for the `Segment` dataclass as a style example.

## Architecture
- **Core Components:**
  - `dictation/app.py`: Streamlit UI for upload, transcription, and dictation.
  - `dictation/segmentation.py`: Handles audio slicing and cache I/O.
  - `dictation/asr/`: Pluggable ASR engines (e.g., Whisper, Qwen).
  - `dictation/diff.py`: Generates word-level diffs for dictation comparison.
- **Data Flow:**
  1. Upload audio → Transcribe → Slice into clips.
  2. Clips and metadata cached under `output/<sha256-hash>/`.
  3. Dictation results saved to `ui_state.json` and `scores.json`.

## Build and Test
- Install dependencies:
  ```bash
  uv sync
  ```
- Run the app:
  ```bash
  uv run streamlit run main.py
  ```
- Optional ASR engines:
  ```bash
  uv sync --extra whisperx
  ```

## Project Conventions
- **Caching:**
  - All runtime artifacts stored under `output/<sha256-hash>/`.
  - Use `segments.json` for transcription metadata.
- **UI State:**
  - Persist last-viewed segment index in `ui_state.json`.
  - Scores saved to `scores.json`.
- **ASR Engines:**
  - Subclass `ASREngine` in `dictation/asr/`.
  - Register new engines in `ENGINE_REGISTRY`.

## Integration Points
- **External Dependencies:**
  - `torch` for GPU/CPU device handling.
  - `pydub` and `ffmpeg` for audio processing.
  - HuggingFace models cached locally.
- **Environment Variables:**
  - `HF_ENDPOINT` for HuggingFace mirror (default: `https://hf-mirror.com/`).

## Security
- Avoid hardcoding sensitive data.
- Ensure all cached files are content-addressed (SHA-256 hash).