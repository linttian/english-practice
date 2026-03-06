"""Subtitle file parser: .srt and .vtt → list[Segment].

When the user uploads a subtitle file alongside the audio, the ASR step is
skipped entirely and these segments are used directly.
"""

import os
import tempfile

from .models import Segment


def parse_subtitle_file(path: str) -> list[Segment]:
    """Auto-detect SRT vs VTT by file extension and return ``list[Segment]``.

    ``segment.clip_path`` is left empty; it is filled in by
    ``segmentation.slice_audio()`` afterwards.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".srt":
        return _parse_srt(path)
    elif ext == ".vtt":
        return _parse_vtt(path)
    else:
        raise ValueError(
            f"Unsupported subtitle format: {ext!r} (expected .srt or .vtt)"
        )


def parse_subtitle_bytes(data: bytes, filename: str) -> list[Segment]:
    """Parse subtitle data from raw bytes (e.g. from ``st.file_uploader``).

    Writes to a temporary file because both ``srt`` and ``webvtt`` parsers
    work best with real file paths.
    """
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return parse_subtitle_file(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_srt(path: str) -> list[Segment]:
    try:
        import srt
    except ImportError as e:
        raise RuntimeError("srt is not installed. Run: uv sync") from e

    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    segments: list[Segment] = []
    for i, sub in enumerate(srt.parse(content)):
        segments.append(
            Segment(
                index=i,
                start=sub.start.total_seconds(),
                end=sub.end.total_seconds(),
                text=sub.content.strip(),
            )
        )
    return segments


def _parse_vtt(path: str) -> list[Segment]:
    try:
        import webvtt
    except ImportError as e:
        raise RuntimeError("webvtt-py is not installed. Run: uv sync") from e

    segments: list[Segment] = []
    for i, caption in enumerate(webvtt.read(path)):
        segments.append(
            Segment(
                index=i,
                start=_vtt_time_to_seconds(caption.start),
                end=_vtt_time_to_seconds(caption.end),
                text=caption.text.strip(),
            )
        )
    return segments


def _vtt_time_to_seconds(time_str: str) -> float:
    """Convert a WebVTT timestamp string (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = "0"
        m, s = parts
    else:
        return 0.0
    return int(h) * 3600 + int(m) * 60 + float(s)
