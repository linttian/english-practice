"""Qwen3-ASR backend engine using the official qwen-asr package.

Suitable for Qwen3-ASR models such as Qwen3-ASR-0.6B and Qwen3-ASR-1.7B.
"""

from __future__ import annotations

import re

import torch

from ..models import Segment
from .base import ASREngine

# Matches the end of a sentence: punctuation followed by end-of-string
# or a space.  Used to decide where to cut when grouping words.
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_CLAUSE_END_RE = re.compile(r"[,;:\-]$")


def _group_align_items_into_sentences(
    items: list,
    max_words: int = 15,
    max_duration: float = 8.0,
) -> list[Segment]:
    """Group forced-aligner items into sentence-level Segments."""
    segments: list[Segment] = []
    buf_words: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0

    for item in items:
        text = item.text.strip()
        if not text:
            continue

        w_start, w_end = item.start_time, item.end_time

        if buf_start is None:
            buf_start = w_start
        buf_words.append(text)
        buf_end = w_end

        duration = buf_end - buf_start
        word_count = len(buf_words)

        is_sentence_end = bool(_SENTENCE_END_RE.search(text))
        is_clause_end = bool(_CLAUSE_END_RE.search(text))

        should_split = False
        if is_sentence_end:
            should_split = True
        elif is_clause_end and (word_count >= max_words or duration >= max_duration):
            should_split = True
        elif word_count >= max_words * 2 or duration >= max_duration * 1.5:
            # Force split if it gets way too long
            should_split = True

        if should_split:
            segments.append(
                Segment(
                    index=len(segments),
                    start=buf_start,
                    end=buf_end,
                    # For Qwen, forced aligner outputs words/tokens. We need to handle
                    # spaces carefully for English. The qwen tokenizer usually returns words
                    # sometimes with leading spaces. We join them with space, but we might want
                    # to strip and join properly.
                    text=" ".join(buf_words)
                    .replace(" ,", ",")
                    .replace(" .", ".")
                    .replace(" ?", "?")
                    .replace(" !", "!")
                    .replace(" ;", ";"),
                    words=[],
                )
            )
            buf_words = []
            buf_start = None

    if buf_words and buf_start is not None:
        segments.append(
            Segment(
                index=len(segments),
                start=buf_start,
                end=buf_end,
                text=" ".join(buf_words)
                .replace(" ,", ",")
                .replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ;", ";"),
                words=[],
            )
        )

    return segments


class QwenEngine(ASREngine):
    name: str = "qwen3-asr (0.6b)"
    model_id: str = "Qwen/Qwen3-ASR-0.6B"

    def __init__(self) -> None:
        self._model = None

    def load(self, device: str) -> None:
        from qwen_asr import Qwen3ASRModel

        # Force device mapping
        device_map = "cuda:0" if device == "cuda" else "cpu"

        # Use bfloat16 to fit smoothly into an 8GB VRAM (0.6B model takes < 2GB)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # We load a local qwen instance using transformers backend
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=8,
            max_new_tokens=4096,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device_map,
            ),
        )

    def transcribe(self, audio_path: str) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("Call load() before transcribe().")

        results = self._model.transcribe(
            audio=audio_path,
            language="English",
            return_time_stamps=True,
        )

        segments = []
        if len(results) > 0 and getattr(results[0], "time_stamps", None) is not None:
            align_result = results[0].time_stamps
            if hasattr(align_result, "items"):
                items = align_result.items
            else:
                items = align_result  # Fallback if it's directly iterable

            segments = _group_align_items_into_sentences(items)
        elif len(results) > 0 and results[0].text:
            text = results[0].text
            from ._text_split import segments_from_plain_text

            segments = segments_from_plain_text(text, audio_path)

        return segments

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
