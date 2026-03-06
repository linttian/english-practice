"""Fallback sentence splitting for engines without word-level timestamps.

Splits a full transcript into sentence-level Segments with proportionally
estimated start/end times based on character count relative to total audio
duration.
"""

from __future__ import annotations

import re

from ..models import Segment


def segments_from_plain_text(
    full_text: str,
    audio_path: str,
) -> list[Segment]:
    """Split *full_text* into sentence-level Segments.

    Timestamps are estimated proportionally to character count relative
    to the total audio duration obtained via pydub/ffprobe.

    Returns an empty list if *full_text* is blank.
    """
    text = full_text.strip()
    if not text:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [Segment(index=0, start=0.0, end=0.0, text=text)]

    # Estimate total duration via ffprobe / pydub
    try:
        from pydub.utils import mediainfo

        info = mediainfo(audio_path)
        total_duration = float(info.get("duration", 0.0))
    except Exception:
        total_duration = 0.0

    total_chars = sum(len(s) for s in sentences) or 1
    segments: list[Segment] = []
    elapsed = 0.0
    for i, sentence in enumerate(sentences):
        proportion = len(sentence) / total_chars
        duration = total_duration * proportion
        segments.append(
            Segment(
                index=i,
                start=elapsed,
                end=elapsed + duration,
                text=sentence,
                words=[],
            )
        )
        elapsed += duration

    return segments
