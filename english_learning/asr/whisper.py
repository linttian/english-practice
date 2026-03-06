"""ASR engines backed by HuggingFace Transformers Whisper pipeline.

Uses ``transformers.pipeline("automatic-speech-recognition")`` with
``WhisperForConditionalGeneration`` for all Whisper-family models.

On CUDA the model is loaded with int8 quantization (bitsandbytes) to
minimise VRAM usage.  Falls back to float16 if bitsandbytes is not
installed. On CPU the model runs in float32.

Word-level timestamps (``return_timestamps="word"``) are used so that
words can be grouped into proper sentence-level segments with accurate
start/end times.
"""

from __future__ import annotations

import os
import re

# Set expandable_segments to avoid CUDA OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from ..models import Segment
from .base import ASREngine

# Matches the end of a sentence: punctuation followed by end-of-string
# or a space.  Used to decide where to cut when grouping words.
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_CLAUSE_END_RE = re.compile(r"[,;:\-]$")


def _group_words_into_sentences(
    words: list[dict],
    max_words: int = 15,
    max_duration: float = 8.0,
) -> list[Segment]:
    """Group word-level chunks into sentence-level Segments.

    Each *word* dict has ``"text"``, ``"timestamp"`` (start, end).
    A sentence boundary is placed after any word whose stripped text
    ends with ``.``, ``!``, or ``?``.
    To make segments more uniform, we also split on commas or semicolons
    if the segment has exceeded `max_words` or `max_duration`.
    """
    segments: list[Segment] = []
    buf_words: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0

    for w in words:
        text = w.get("text", "").strip()
        if not text:
            continue
        ts = w.get("timestamp", (0.0, 0.0))
        w_start = ts[0] if ts[0] is not None else buf_end
        w_end = ts[1] if ts[1] is not None else w_start

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
                    text=" ".join(buf_words),
                    words=[],
                )
            )
            buf_words = []
            buf_start = None

    # Flush remaining words as a final segment
    if buf_words and buf_start is not None:
        segments.append(
            Segment(
                index=len(segments),
                start=buf_start,
                end=buf_end,
                text=" ".join(buf_words),
                words=[],
            )
        )

    return segments


class WhisperEngine(ASREngine):
    """Whisper large-v3-turbo via HuggingFace Transformers (int8 quantized)."""

    name: str = "whisper (large-v3-turbo)"
    model_id: str = "openai/whisper-large-v3-turbo"

    def __init__(self) -> None:
        self._pipe = None

    def load(self, device: str) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        if device == "cuda":
            model = self._load_quantized_model()
        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )

        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe_device = 0 if device == "cuda" else -1

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=2,
            device=pipe_device if not getattr(model, "hf_device_map", None) else None,
        )

    def _load_quantized_model(self):
        """Load model with int8 quantization; fall back to float16."""
        from transformers import AutoModelForSpeechSeq2Seq

        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            return AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                attn_implementation="sdpa",
            )
        except (ImportError, Exception):
            # bitsandbytes not installed or failed — fall back to fp16
            return AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation="sdpa",
            )

    def transcribe(self, audio_path: str) -> list[Segment]:
        if self._pipe is None:
            raise RuntimeError("Call load() before transcribe().")

        result = self._pipe(
            audio_path,
            return_timestamps="word",
            generate_kwargs={"language": "english", "task": "transcribe"},
        )

        chunks = result.get("chunks", [])
        if chunks:
            segments = _group_words_into_sentences(chunks)
        elif result.get("text", "").strip():
            from ._text_split import segments_from_plain_text

            segments = segments_from_plain_text(result["text"], audio_path)
        else:
            segments = []

        return segments

    def unload(self) -> None:
        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DistilWhisperEngine(WhisperEngine):
    """Distil-Whisper (distil-large-v3) via HuggingFace Transformers (int8 quantized).

    ~40% smaller and ~2x faster than large-v3-turbo while retaining most
    accuracy.  Sentence-level timestamps are fully supported.
    """

    name: str = "whisper (distil-large-v3)"
    model_id: str = "distil-whisper/distil-large-v3"


class WhisperSmallEngine(WhisperEngine):
    """Whisper small via HuggingFace Transformers."""

    name: str = "whisper (small)"
    model_id: str = "openai/whisper-small"


class WhisperBaseEngine(WhisperEngine):
    """Whisper base via HuggingFace Transformers."""

    name: str = "whisper (base)"
    model_id: str = "openai/whisper-base"
