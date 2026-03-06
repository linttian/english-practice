"""Streamlit UI for English listening practice.

Run with:
    uv run streamlit run main.py
"""

from __future__ import annotations

import base64
import hashlib
import html as html_lib
import json
import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from .asr import ENGINE_REGISTRY
from .diff import build_diff_html
from .models import Segment
from .segmentation import convert_to_wav, load_cached_segments, slice_audio
from .subtitle import parse_subtitle_bytes

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    st.session_state.setdefault("session_id", "")
    st.session_state.setdefault("segments", [])  # list[Segment]
    st.session_state.setdefault("engine", None)  # ASREngine instance
    st.session_state.setdefault("engine_name", None)  # str — for change detection
    st.session_state.setdefault("dictations", {})  # dict[int, str]
    st.session_state.setdefault("show_loop", {})  # dict[int, bool]
    st.session_state.setdefault("tmp_audio_path", None)
    st.session_state.setdefault("current_segment_idx", 0)


def _ui_state_path(session_id: str) -> str:
    return os.path.join("output", session_id, "ui_state.json")


def _load_last_segment_idx(session_id: str) -> int:
    if not session_id:
        return 0
    path = _ui_state_path(session_id)
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        value = int(data.get("last_segment_idx", 0))
        return max(0, value)
    except Exception:
        return 0


def _save_last_segment_idx(session_id: str, idx: int) -> None:
    if not session_id:
        return
    path = _ui_state_path(session_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"last_segment_idx": max(0, int(idx))}, fh)
    except Exception:
        # Persistence is best-effort; UI should still work if writing fails.
        pass


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[str, str]:
    """Render sidebar and return (engine_name, device)."""
    with st.sidebar:
        st.title("⚙️ Settings")
        st.divider()

        engine_name: str = st.selectbox(
            "ASR Engine",
            list(ENGINE_REGISTRY.keys()),
            help="Switch ASR engine here. The engine loads on the next Transcribe.",
        )

        device: str = st.selectbox(
            "Device",
            ["cpu", "cuda"],
            help="Use 'cuda' if you have an NVIDIA GPU for faster inference.",
        )

        st.divider()
        st.caption(
            "Clips are saved to **./output/** relative to the working directory.\n\n"
            "**System requirement:** `ffmpeg` must be installed.\n"
            "`sudo apt install ffmpeg`"
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

    return engine_name, device


# ---------------------------------------------------------------------------
# Upload and transcription
# ---------------------------------------------------------------------------


def _render_upload_section(engine_name: str, device: str) -> None:
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
        st.session_state["_last_file_hash"] = file_hash
        st.session_state.session_id = file_hash
        st.session_state.current_segment_idx = _load_last_segment_idx(file_hash)

        # Try to load from cache first
        cached = load_cached_segments(file_hash)
        if cached is not None:
            st.session_state.segments = cached
            if st.session_state.current_segment_idx >= len(cached):
                st.session_state.current_segment_idx = max(0, len(cached) - 1)
            st.session_state.dictations = {}
            st.session_state.show_loop = {}
            st.success(f"Loaded {len(cached)} segments from cache.")
            st.rerun()
        else:
            st.session_state.segments = []
            st.session_state.dictations = {}
            st.session_state.show_loop = {}

    if st.button("▶ Transcribe", type="primary"):
        _run_transcription(engine_name, device, sub_file)


def _run_transcription(engine_name: str, device: str, sub_file) -> None:
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

        with st.spinner("Slicing audio into clips…"):
            segments = slice_audio(audio_path, segments, session_id)

        st.session_state.segments = segments
        saved_idx = _load_last_segment_idx(session_id)
        st.session_state.current_segment_idx = min(saved_idx, max(0, len(segments) - 1))
        st.session_state.dictations = {}
        st.session_state.show_loop = {}
        st.success(f"Done! {len(segments)} segments found.")
        st.rerun()

    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise


# ---------------------------------------------------------------------------
# Per-segment rendering
# ---------------------------------------------------------------------------


def _render_segment(seg: Segment) -> None:
    idx = seg.index
    st.markdown(
        f"<div style='font-size: 1.2rem; margin-bottom: 10px;'>"
        f"<b>Segment {idx + 1}</b> &nbsp; "
        f"<span style='color:#888; font-size:0.9em'>{seg.start:.2f}s → {seg.end:.2f}s</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Audio player + loop toggle
    col_player, col_loop = st.columns([5, 1])

    with col_loop:
        loop_on: bool = st.toggle(
            "Loop",
            key=f"loop_{idx}",
            value=st.session_state.show_loop.get(idx, False),
        )
        st.session_state.show_loop[idx] = loop_on

    with col_player:
        if seg.clip_path and os.path.exists(seg.clip_path):
            if loop_on:
                _render_looping_audio(seg.clip_path, key=f"loop_html_{idx}")
            else:
                with open(seg.clip_path, "rb") as fh:
                    st.audio(fh.read(), format="audio/wav")
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

    # Compare button
    if (
        st.button(
            "Compare (ignore case) ↔",
            key=f"compare_{idx}",
            type="primary",
            use_container_width=True,
        )
        and dictation.strip()
    ):
        diff_html = build_diff_html(seg.text, dictation)
        st.caption("Comparison is case-insensitive.")
        st.markdown(
            f"<div style='font-size: 1.2rem; line-height: 1.6; padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-top: 10px;'>"
            f"{diff_html}</div>",
            unsafe_allow_html=True,
        )

    # Transcript reveal (hidden by default so it doesn't spoil dictation)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Reveal exact transcript", expanded=False):
        st.markdown(
            f"<div style='background:#f0f4f8; padding:20px; border-radius:8px; "
            f"font-size: 1.4rem; font-style:italic; border-left: 5px solid #2e86c1; font-weight: 500;'>"
            f"{html_lib.escape(seg.text)}</div>",
            unsafe_allow_html=True,
        )


def _render_looping_audio(clip_path: str, key: str) -> None:
    """Inject a looping audio player via an HTML component.

    ``st.audio`` does not expose the ``loop`` attribute, so we embed the
    audio as a base64 data URI in a native ``<audio>`` element.
    """
    with open(clip_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    loop_html = (
        f'<audio id="{key}" controls loop style="width:100%; margin-top:4px">'
        f'  <source src="data:audio/wav;base64,{b64}" type="audio/wav">'
        f"</audio>"
    )
    components.html(loop_html, height=60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="English Listening Practice",
        page_icon="🎧",
        layout="wide",
    )
    _init_session_state()

    engine_name, device = _render_sidebar()

    st.title("🎧 English Listening Practice")
    st.caption(
        "Upload audio or video, transcribe it, then practise Listen → Write → Compare."
    )

    _render_upload_section(engine_name, device)

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
        _save_last_segment_idx(st.session_state.get("session_id", ""), current_idx)

        # Top navigation
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button(
                "⬅ Previous", use_container_width=True, disabled=(current_idx <= 0)
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

        # Render only the current segment
        _render_segment(segments[current_idx])
