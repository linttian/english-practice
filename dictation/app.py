"""Streamlit UI for English listening practice.

Run with:
    uv run streamlit run main.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import torch

from . import analytics, recent_practices, ui_templates
from .asr import ENGINE_REGISTRY
from .diff import build_diff_html
from .models import Segment
from .segmentation import convert_to_wav, load_cached_segments, slice_audio
from .subtitle import parse_subtitle_bytes
from .utils import io as services_io
from .utils import scan as services_scan


def _init_session_state() -> None:
    """Ensure required Streamlit session_state keys exist with sane defaults."""
    # Engine defaults
    if "engine_name" not in st.session_state:
        try:
            st.session_state.engine_name = list(ENGINE_REGISTRY.keys())[0]
        except Exception:
            st.session_state.engine_name = ""
    if "engine" not in st.session_state:
        st.session_state.engine = None

    # File / session defaults
    if "_last_file_hash" not in st.session_state:
        st.session_state._last_file_hash = None
    if "tmp_audio_path" not in st.session_state:
        st.session_state.tmp_audio_path = None
    if "original_filename" not in st.session_state:
        st.session_state.original_filename = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Navigation / UI state
    if "current_segment_idx" not in st.session_state:
        st.session_state.current_segment_idx = 0
    if "_jump_slider_widget" not in st.session_state:
        st.session_state._jump_slider_widget = 1

    # Per-audio cached data
    if "segments" not in st.session_state:
        st.session_state.segments = []
    if "dictations" not in st.session_state:
        st.session_state.dictations = {}
    if "show_loop" not in st.session_state:
        st.session_state.show_loop = {}
    if "scores" not in st.session_state:
        st.session_state.scores = {}
    if "final_score" not in st.session_state:
        st.session_state.final_score = None


def _render_sidebar() -> tuple[str, str, dict]:
    """Render sidebar and return (engine_name, device, settings)."""
    logger = logging.getLogger(__name__)
    logger.debug("render_sidebar start")
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        padding_ms = st.slider(
            "Padding (ms)",
            min_value=0,
            max_value=1000,
            value=100,
            step=50,
            help="Milliseconds of padding added to the start and end of each segment.",
        )
        min_sentence_length = st.slider(
            "Minimum Sentence Length (words)",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            help="Minimum number of words in a segment before it is split.",
        )
        max_sentence_duration = st.slider(
            "Maximum Sentence Duration (seconds)",
            min_value=1,
            max_value=60,
            value=60,
            step=1,
            help="Maximum duration of a segment in seconds before it is split.",
        )

        st.divider()

        # Recommend model based on GPU availability and memory
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )  # Convert to GB
                if gpu_memory >= 16:
                    st.info("Recommended: Use a large model for best performance.")
                elif gpu_memory >= 8:
                    st.info(
                        "Recommended: Use a medium-sized model for balanced performance."
                    )
                else:
                    st.info("Recommended: Use a small model for optimal speed.")
            except Exception:
                pass
        else:
            st.warning("No GPU detected. Using CPU-compatible models may be slower.")

        engine_name: str = st.selectbox(
            "ASR Engine",
            list(ENGINE_REGISTRY.keys()),
            help="Switch ASR engine here. The engine loads on the next Transcribe.",
        )

        device: str = st.selectbox(
            "Device",
            ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
            help="Use 'cuda' if you have an NVIDIA GPU for faster inference.",
        )

        # Engine change detection: unload old and reset
        if engine_name != st.session_state.engine_name:
            if st.session_state.engine is not None:
                try:
                    st.session_state.engine.unload()
                except Exception:
                    pass
            st.session_state.engine = None
            st.session_state.engine_name = engine_name

    settings = {
        "padding_ms": padding_ms,
        "min_sentence_length": min_sentence_length,
        "max_sentence_duration": max_sentence_duration,
    }

    return engine_name, device, settings


def _render_upload_section(engine_name: str, device: str, settings: dict) -> None:
    logger = logging.getLogger(__name__)
    logger.debug("render_upload_section start")
    st.header("1. Upload Media")

    col_audio, col_sub = st.columns(2)
    with col_audio:
        audio_file = st.file_uploader(
            "Audio / Video file",
            type=["mp3", "wav", "mp4", "mkv", "m4a", "ogg", "flac", "aac"],
            key="audio_uploader",
        )
    with col_sub:
        sub_file = st.file_uploader(
            "Subtitles (optional — skips ASR)",
            type=["srt", "vtt"],
            key="sub_uploader",
        )

    if audio_file is None:
        st.info("Upload an audio or video file to get started.")
        return

    # Compute a content hash so the same file always maps to the same cache
    file_bytes = audio_file.read()
    audio_file.seek(0)
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

    # Detect whether this is a new file
    if file_hash != st.session_state.get("_last_file_hash"):
        suffix = os.path.splitext(audio_file.name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            raw_path = tmp.name
        wav_path = convert_to_wav(raw_path)
        os.unlink(raw_path)
        st.session_state.tmp_audio_path = wav_path
        # record original filename for later metadata
        st.session_state.original_filename = audio_file.name
        st.session_state["_last_file_hash"] = file_hash
        st.session_state.session_id = file_hash
        st.session_state.current_segment_idx = _load_last_segment_idx(file_hash)

        # Try to load from cache first
        cached = load_cached_segments(file_hash)
        if cached is not None:
            st.session_state.segments = cached
            if st.session_state.current_segment_idx >= len(cached):
                st.session_state.current_segment_idx = max(0, len(cached) - 1)
            # Load UI state (dictations, show_loop, last idx) if present
            ui = _load_ui_state(file_hash)
            if ui is not None:
                st.session_state.dictations = ui.get("dictations", {})
                st.session_state.show_loop = ui.get("show_loop", {})
                st.session_state.current_segment_idx = max(
                    0, min(len(cached) - 1, ui.get("last_segment_idx", 0))
                )
            else:
                st.session_state.dictations = {}
                st.session_state.show_loop = {}
            # Load cached scores if present
            loaded = _load_scores(file_hash)
            if loaded is not None:
                by_idx, final = loaded
                st.session_state.scores = {k: v for k, v in by_idx.items()}
                if final is not None:
                    st.session_state.final_score = final
            else:
                st.session_state.scores = {seg.index: None for seg in cached}
            name = st.session_state.get("original_filename") or st.session_state.get(
                "session_id"
            )
            st.success(f"Loaded {len(cached)} segments from cache: {name}")
        else:
            st.session_state.segments = []
            st.session_state.dictations = {}
            st.session_state.show_loop = {}

    if st.button("▶ Transcribe", type="primary"):
        _run_transcription(engine_name, device, sub_file, settings)


def _run_transcription(engine_name: str, device: str, sub_file, settings: dict) -> None:
    """Run ASR (or subtitle parse) + audio slicing; store results in session state."""
    audio_path: str = st.session_state.tmp_audio_path
    session_id: str = st.session_state.session_id

    try:
        if sub_file is not None:
            with st.spinner("Parsing subtitle file…"):
                segments = parse_subtitle_bytes(sub_file.read(), sub_file.name)
        else:
            if st.session_state.engine is None:
                EngineCls = ENGINE_REGISTRY[engine_name]
                engine = EngineCls()
                with st.spinner(f"Loading {engine_name}…"):
                    engine.load(device)
                st.session_state.engine = engine

            with st.spinner("Transcribing… (this may take a few minutes)"):
                segments = st.session_state.engine.transcribe(audio_path)

        # Update output paths
        clips_dir = Path(f"./output/{session_id}/clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Slicing audio into clips…"):
            # Slice the entire list at once so returned segments include clip_path
            sliced = slice_audio(
                audio_path,
                segments,
                session_id,
                output_root="./output",
                padding_ms=settings["padding_ms"],
                min_sentence_length=settings["min_sentence_length"],
                merge_on_punctuation=True,
                max_duration=settings.get("max_sentence_duration", 60),
            )

        # Save the returned (sliced) segments to JSON in the parent directory
        _save_segments_to_json(sliced, session_id, output_root="./output")

        st.session_state.segments = sliced
        saved_idx = _load_last_segment_idx(session_id)
        st.session_state.current_segment_idx = min(saved_idx, max(0, len(sliced) - 1))
        st.session_state.dictations = {}
        st.session_state.show_loop = {}
        # Initialize scores (None = not scored yet)
        loaded = _load_scores(session_id)
        if loaded is not None:
            by_idx, final = loaded
            st.session_state.scores = {k: v for k, v in by_idx.items()}
            if final is not None:
                st.session_state.final_score = final
        else:
            st.session_state.scores = {seg.index: None for seg in sliced}
        st.success(f"Done! {len(segments)} segments found.")
        _show_transcription_success()

    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise


def _save_segments_to_json(
    segments: list[Segment], session_id: str, output_root: str = "./output"
) -> None:
    """Save the transcription segments to a JSON file in the output directory."""
    orig_fn = st.session_state.get("original_filename", None)
    return services_io.save_segments_to_json(
        segments, session_id, output_root=output_root, original_filename=orig_fn
    )


def _save_ui_state(session_id: str) -> None:
    """Save UI state (last_segment_idx, dictations, show_loop) to ui_state.json."""
    if not session_id:
        return
    payload = {
        "last_segment_idx": int(st.session_state.get("current_segment_idx", 0)),
        "dictations": {
            str(k): v for k, v in st.session_state.get("dictations", {}).items()
        },
        "show_loop": {
            str(k): bool(v) for k, v in st.session_state.get("show_loop", {}).items()
        },
    }
    return services_io.save_ui_state(session_id, payload)


def _load_ui_state(session_id: str) -> dict | None:
    """Load UI state from ui_state.json, returning the dict or None."""
    return services_io.load_ui_state(session_id)


def _scores_path(session_id: str) -> str:
    return services_io.scores_path(session_id)


def _save_scores(session_id: str, scores: dict, final: float | None = None) -> None:
    return services_io.save_scores(session_id, scores, final=final)


def _load_scores(session_id: str) -> tuple[dict, float | None] | None:
    return services_io.load_scores(session_id)


def _render_looping_audio(wav_path: str, key: str = "audio_loop") -> None:
    """Render a small looping audio player using an embedded base64 HTML audio tag.

    Using an HTML player with `loop` provides seamless looping behaviour that
    the built-in Streamlit `st.audio` playback doesn't always provide.
    """
    try:
        with open(wav_path, "rb") as fh:
            data = fh.read()
    except Exception:
        st.warning("Clip not found.")
        return

    b64 = base64.b64encode(data).decode("ascii")
    audio_html = f"""
        <audio controls autoplay loop style='width:100%'>
          <source src="data:audio/wav;base64,{b64}" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
        """
    components.html(audio_html, height=80)


def _render_embedded_audio(wav_path: str, key: str = "audio") -> None:
    """Render a standard embedded audio player for a wav file.

    Uses a base64 data URL to avoid serving static files separately.
    """
    try:
        with open(wav_path, "rb") as fh:
            data = fh.read()
    except Exception:
        st.warning("Clip not found.")
        return

    b64 = base64.b64encode(data).decode("ascii")
    audio_html = f"""
        <audio controls style='width:100%'>
          <source src="data:audio/wav;base64,{b64}" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
    """
    components.html(audio_html, height=80)


def _practice_index_path() -> str:
    return services_io.practice_index_path()


def _ui_state_path(session_id: str) -> str:
    """Return the path to ui_state.json for a session."""
    return services_io.ui_state_path(session_id)


def _load_last_segment_idx(session_id: str) -> int:
    """Load last saved segment index for a session (default 0)."""
    ui = _load_ui_state(session_id)
    if not ui:
        return 0
    try:
        return int(ui.get("last_segment_idx", 0))
    except Exception:
        return 0


def _save_last_segment_idx(session_id: str, idx: int) -> None:
    """Save only the last_segment_idx into the session's ui_state.json.

    This preserves existing dictations/show_loop if present.
    """
    if not session_id:
        return
    # Try to preserve existing UI state
    existing = _load_ui_state(session_id) or {"dictations": {}, "show_loop": {}}
    payload = {
        "last_segment_idx": int(idx or 0),
        "dictations": {str(k): v for k, v in existing.get("dictations", {}).items()},
        "show_loop": {
            str(k): bool(v) for k, v in existing.get("show_loop", {}).items()
        },
    }
    try:
        path = _ui_state_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# Use standard logging instead of a custom startup_debug file
logger = logging.getLogger(__name__)


def _get_score(scores: dict, idx: int):
    """Return the score for segment `idx`, handling int or string keys in `scores` dict."""
    if scores is None:
        return None
    # Try integer key first, then string key (Streamlit may coerce dict keys)
    val = scores.get(idx)
    if val is not None:
        return val
    return scores.get(str(idx))


def _load_practice_index() -> list:
    return services_io.load_practice_index()


def _save_practice_index(index: list) -> None:
    return services_io.save_practice_index(index)


def _record_practice_event(
    session_id: str,
    final_score: float | None,
    segments_count: int,
    label: str | None = None,
) -> None:
    """Append a practice event to the global index and per-session history."""
    return services_io.record_practice_event(
        session_id, final_score, segments_count, label
    )


def _scan_recent_practices(limit: int = 8) -> list:
    return services_scan.scan_recent_practices(limit=limit)


def _load_session_from_cache(session_id: str) -> bool:
    """Load segments, ui state and scores for a cached session into session_state."""
    # If the session output folder doesn't exist, report error.
    out_dir = Path("output") / session_id
    if not out_dir.exists():
        st.error(f"Session not found: {session_id}")
        return False
    try:
        cached = load_cached_segments(session_id)
    except Exception as exc:
        st.error(f"Failed to read session segments: {exc}")
        return False
    if cached is None:
        st.error("No cached segments found for that session.")
        return False
    st.session_state.session_id = session_id
    st.session_state.segments = cached
    # try to load meta and populate original filename for friendly display
    try:
        meta_path = Path("output") / session_id / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            fn = meta.get("source_filename")
            if fn:
                st.session_state.original_filename = fn
    except Exception:
        pass
    ui = _load_ui_state(session_id)
    if ui is not None:
        st.session_state.dictations = ui.get("dictations", {})
        st.session_state.show_loop = ui.get("show_loop", {})
        st.session_state.current_segment_idx = max(
            0, min(len(cached) - 1, ui.get("last_segment_idx", 0))
        )
    else:
        st.session_state.dictations = {}
        st.session_state.show_loop = {}
    loaded = _load_scores(session_id)
    if loaded is not None:
        by_idx, final = loaded
        st.session_state.scores = {k: v for k, v in by_idx.items()}
        if final is not None:
            st.session_state.final_score = final
    else:
        st.session_state.scores = {seg.index: None for seg in cached}
    name = st.session_state.get("original_filename") or session_id
    st.success(f"Loaded {len(cached)} segments from cache: {name}")
    # After loading into session_state, force a rerun so the UI (tabs) resets
    # to the default first tab (Workspace) and shows the Dictation Workspace.
    try:
        st.rerun()
    except Exception:
        # If rerun fails for any reason, still return True so callers know load succeeded.
        return True


# ---------------------------------------------------------------------------
# Per-segment rendering
# ---------------------------------------------------------------------------


def _render_segment(seg: Segment) -> None:
    idx = seg.index
    # Header placeholder so we can update badge after scoring without rerun
    header_ph = st.empty()
    # Show segment title with current score badge
    score_display = "—"
    scores = st.session_state.get("scores", {})
    sc = _get_score(scores, idx)
    if sc is None:
        score_display = "Not scored"
    else:
        score_display = f"{int(sc)} / 100"

    # Header shows segment number, timestamps and raw score
    header_html = (
        f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>"
        f"<div style='font-size:1.2rem;'><b>Segment {idx + 1}</b> &nbsp; "
        f"<span style='color:#888; font-size:0.9em'>{seg.start:.2f}s → {seg.end:.2f}s</span></div>"
        f"<div style='font-weight:600; color:#fff; background:#2e86c1; padding:6px 10px; border-radius:14px;'>{score_display}</div>"
        f"</div>"
    )
    header_ph.markdown(header_html, unsafe_allow_html=True)

    # clear transient flags (used previously for animations)
    if "just_scored_idx" in st.session_state:
        st.session_state.pop("just_scored_idx", None)
        st.session_state.pop("just_scored_value", None)

    # Audio player + loop toggle
    col_player, col_loop, col_skip = st.columns([5, 1, 1])

    with col_loop:
        loop_on: bool = st.toggle(
            "Loop",
            key=f"loop_{idx}",
            value=st.session_state.show_loop.get(idx, False),
        )
        st.session_state.show_loop[idx] = loop_on

    with col_skip:
        if st.button("Skip", key=f"skip_{idx}"):
            st.session_state.scores[idx] = 100  # Assign full score for skipped segments
            _save_scores(
                st.session_state.get("session_id", ""), st.session_state.scores
            )
            # mark for celebration on next render
            st.session_state.just_scored_idx = idx
            st.session_state.just_scored_value = 100
            # Advance to next segment (do not wrap past end)
            cur = st.session_state.get("current_segment_idx", 0)
            total = len(st.session_state.get("segments", []))
            if cur < total - 1:
                st.session_state.current_segment_idx = cur + 1
            try:
                _save_ui_state(st.session_state.get("session_id", ""))
            except Exception:
                pass
            st.rerun()

    with col_player:
        if seg.clip_path and os.path.exists(seg.clip_path):
            if loop_on:
                _render_looping_audio(seg.clip_path, key=f"loop_html_{idx}")
            else:
                _render_embedded_audio(seg.clip_path, key=f"audio_{idx}")
        else:
            st.warning("Clip not found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Dictation text area (transcript hidden until revealed)
    dictation: str = (
        st.text_area(
            "Your dictation:",
            value=st.session_state.dictations.get(idx, ""),
            key=f"dictation_{idx}",
            height=150,
            placeholder="Type what you hear…",
        )
        or ""
    )
    st.session_state.dictations[idx] = dictation
    # Persist dictation immediately
    try:
        _save_ui_state(st.session_state.get("session_id", ""))
    except Exception:
        pass

    # Compare button
    compare_state_key = f"_compare_shown_{idx}"
    compare_html_key = f"_compare_html_{idx}"
    # Pressing the button computes and persists the diff HTML so it survives reruns
    if (
        st.button(
            "Compare (ignore case) ↔",
            key=f"compare_{idx}",
            type="primary",
            use_container_width=True,
        )
        and dictation.strip()
    ):
        # Build diff using the original texts so displayed Reference/Your input
        # keep punctuation and case, while comparison still ignores them inside diff.py
        diff_html = build_diff_html(seg.text or "", dictation or "")
        st.session_state[compare_html_key] = diff_html
        st.session_state[compare_state_key] = True

    # If we have persisted compare HTML, display it (survives reruns and toggles)
    if st.session_state.get(compare_state_key):
        st.caption("Comparison ignores case and punctuation.")
        st.markdown(
            st.session_state.get(compare_html_key, ""),
            unsafe_allow_html=True,
        )

        # Calculate score using word-level overlap on normalized texts
        import re
        from collections import Counter

        def _norm(s: str) -> str:
            return " ".join(re.findall(r"\w+", (s or "").lower()))

        cor_norm = _norm(seg.text)
        usr_norm = _norm(dictation)

        cor_words = cor_norm.split()
        usr_words = usr_norm.split()
        if not cor_words:
            score = 100 if not usr_words else 0
        else:
            c_cor = Counter(cor_words)
            c_usr = Counter(usr_words)
            common = sum((c_cor & c_usr).values())
            score = int(round(common / len(cor_words) * 100))
        st.session_state.scores[idx] = score
        _save_scores(st.session_state.get("session_id", ""), st.session_state.scores)
        # mark for celebration on next render if perfect
        st.session_state.just_scored_idx = idx
        st.session_state.just_scored_value = score
        try:
            _save_ui_state(st.session_state.get("session_id", ""))
        except Exception:
            pass
        # Rerun to update header badge and emoji
        try:
            _save_ui_state(st.session_state.get("session_id", ""))
        except Exception:
            pass
        # Update header in-place so diff doesn't disappear
        sc = score
        score_display = f"{int(sc)} / 100"
        emoji = ""
        if sc >= 90:
            emoji = " 🥳"
        elif sc >= 75:
            emoji = " 😊"
        elif sc >= 50:
            emoji = " 😐"
        else:
            emoji = " 😢"
        header_html = (
            f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;'>"
            f"<div style='font-size:1.2rem;'><b>Segment {idx + 1}</b> &nbsp; "
            f"<span style='color:#888; font-size:0.9em'>{seg.start:.2f}s → {seg.end:.2f}s</span></div>"
            f"<div style='font-weight:600; color:#fff; background:#2e86c1; padding:6px 10px; border-radius:14px;'>{score_display}{emoji}</div>"
            f"</div>"
        )
        header_ph.markdown(header_html, unsafe_allow_html=True)

    # Show original / answer toggle (separate from Compare)
    show_key = f"show_original_{idx}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False
    toggle = st.checkbox(
        "Show original", value=st.session_state[show_key], key=show_key
    )
    if toggle:
        st.markdown(
            f"<div style='padding:12px; border:1px solid #ddd; border-radius:8px; background:#f8f9fb'><b>Original:</b><div style='margin-top:8px'>{seg.text}</div></div>",
            unsafe_allow_html=True,
        )


def _calculate_final_score(segments: list[Segment], scores: dict) -> float:
    # Weight by word count per segment
    weights = [max(1, len(s.text.split())) for s in segments]
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    weighted = 0.0
    for s, w in zip(segments, weights):
        sc = _get_score(scores, s.index)
        if sc is None:
            sc = 0
        weighted += sc * w
    return weighted / total_weight


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        st.set_page_config(
            page_title="English Listening Practice",
            page_icon="🎧",
            layout="wide",
        )
        logger.debug("main invoked")
        _init_session_state()

        # Inject CSS for tab styling via ui_templates helper
        ui_templates.inject_tab_css(font_size="1.6rem")

        engine_name, device, settings = _render_sidebar()

        tabs = st.tabs(["Workspace", "Recent Practices", "Practice Analytics"])

        # Workspace tab: upload + dictation workspace
        with tabs[0]:
            st.title("🎧 English Listening Practice")
            st.caption(
                "Upload audio or video, transcribe it, then practise Listen → Write → Compare."
            )

            _render_upload_section(engine_name, device, settings)

            segments: list[Segment] = st.session_state.get("segments", [])
            if segments:
                st.header(f"2. Dictation Workspace ({len(segments)} segments)")

                if "current_segment_idx" not in st.session_state:
                    st.session_state.current_segment_idx = 0

                # Validate index
                if st.session_state.current_segment_idx >= len(segments):
                    st.session_state.current_segment_idx = len(segments) - 1
                elif st.session_state.current_segment_idx < 0:
                    st.session_state.current_segment_idx = 0

                current_idx = st.session_state.current_segment_idx

                # Keep slider state synced with external navigation buttons.
                desired_slider_value = current_idx + 1
                if st.session_state.get("_jump_slider_widget") != desired_slider_value:
                    st.session_state["_jump_slider_widget"] = desired_slider_value

                # Persist current reading position for this audio hash.
                _save_last_segment_idx(
                    st.session_state.get("session_id", ""), current_idx
                )

                # Top navigation
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                with nav_col1:
                    if st.button(
                        "⬅ Previous",
                        use_container_width=True,
                        disabled=(current_idx <= 0),
                    ):
                        st.session_state.current_segment_idx -= 1
                        st.rerun()
                with nav_col2:
                    # We don't provide a 'value' if the key exists, let Streamlit manage the state.
                    # If we need to force an external update, we just set the un-mapped value.
                    def _on_slider_change():
                        st.session_state.current_segment_idx = (
                            st.session_state._jump_slider_widget - 1
                        )

                    st.slider(
                        "Jump to segment",
                        min_value=1,
                        max_value=len(segments),
                        key="_jump_slider_widget",
                        label_visibility="collapsed",
                        on_change=_on_slider_change,
                    )
                    st.markdown(
                        f"<p style='text-align: center; color: #666;'>Segment <b>{current_idx + 1}</b> / {len(segments)}</p>",
                        unsafe_allow_html=True,
                    )
                with nav_col3:
                    if st.button(
                        "Next ➡",
                        use_container_width=True,
                        disabled=(current_idx >= len(segments) - 1),
                    ):
                        st.session_state.current_segment_idx += 1
                        st.rerun()

                st.divider()

                # Show progress and final score button
                scores = st.session_state.get("scores", {})
                total = len(segments)
                scored_count = sum(
                    1 for s in segments if _get_score(scores, s.index) is not None
                )
                col_status, col_final = st.columns([3, 1])
                with col_status:
                    st.markdown(
                        f"**Progress:** {scored_count} / {total} segments scored"
                    )
                    progress_val = scored_count / total if total else 0
                    st.progress(progress_val)
                with col_final:
                    final_shown = st.session_state.get("final_score")
                    if scored_count == total and total > 0:
                        if st.button(
                            "Finish & Calculate Final Score", use_container_width=True
                        ):
                            final_score = _calculate_final_score(segments, scores)
                            st.session_state.final_score = final_score
                            _save_scores(
                                st.session_state.get("session_id", ""),
                                scores,
                                final=final_score,
                            )
                            try:
                                _record_practice_event(
                                    st.session_state.get("session_id", ""),
                                    final_score,
                                    len(segments),
                                )
                            except Exception:
                                pass
                            st.balloons()
                            st.success(f"Final score: {final_score:.2f} / 100")
                    elif final_shown is not None:
                        st.markdown(f"**Final:** {final_shown:.2f} / 100")

                # Render only the current segment
                _render_segment(segments[current_idx])
        # Recent Practices and Analytics tabs
        with tabs[1]:
            recent_practices.render_recent_practices_tab(
                limit=4,
                scan_callback=_scan_recent_practices,
                load_callback=_load_session_from_cache,
            )

        with tabs[2]:
            analytics.render_analytics(scan_callback=_load_practice_index)
    except Exception:
        # Log full traceback for diagnosis and re-raise so Streamlit shows error
        tb = traceback.format_exc()
        logger.exception("Unhandled exception in main:\n%s", tb)
        raise


def _show_transcription_success():
    """Show a temporary success notification after transcription."""
    st.toast("Transcription completed! Clips are saved to ./output/", icon="✅")
