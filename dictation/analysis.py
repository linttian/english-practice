"""Connected-speech analysis using a local HuggingFace text-generation model.

Default model: ``Qwen/Qwen3.5-4B``.
The model is downloaded once to ``~/.cache/huggingface/hub`` and cached
across Streamlit re-runs via ``@st.cache_resource``.

Models that emit ``<think>…</think>`` blocks (e.g. Qwen3.5) are handled
automatically — the thinking block is stripped before JSON parsing.

Annotation types
----------------
- L  Linking       — final consonant links to the next initial vowel
                     e.g. "turn_it_on" → /tɜːr nɪ tɒn/
- E  Elision       — a sound is dropped
                     e.g. "want to" → "wanna", "last night" → /læs naɪt/
- A  Assimilation  — a sound changes to match its neighbour
                     e.g. "ten people" → /tem piːpəl/
- I  Intrusive     — an extra sound is inserted between vowel-final and
                     vowel-initial words, e.g. "I am" → /aɪ jæm/ (intrusive /j/)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid circular at type-check time

from .models import AnnotatedText, AnnotationSpan

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert phonologist specialising in English connected speech.
Analyse the English sentence below for connected speech phenomena.

Annotation types:
  L = Linking     (consonant links to next vowel-initial word)
  E = Elision     (a sound is dropped / reduced)
  A = Assimilation (a sound changes to match its neighbour)
  I = Intrusive   (an extra sound is inserted between vowels)

Return ONLY valid JSON — no markdown fences, no extra text — matching this schema exactly:
{
  "text": "<original sentence>",
  "annotations": [
    {
      "start_char": <int>,
      "end_char": <int>,
      "type": "<L|E|A|I>",
      "note": "<short phonetic explanation, e.g. want_to→wanna (Elision)>"
    }
  ]
}

If there are no connected speech phenomena, return an empty annotations array.
"""


# ---------------------------------------------------------------------------
# Pipeline loader (cached once per (model_id, device) combination)
# ---------------------------------------------------------------------------

_pipeline_cache: dict[tuple[str, str], object] = {}


def get_pipeline(model_id: str, device: str):
    """Return a cached ``transformers.pipeline`` for *model_id* on *device*.

    This function is intentionally NOT decorated with ``@st.cache_resource``
    here so that the module stays importable outside Streamlit.  The Streamlit
    app wraps this call with ``@st.cache_resource`` at the call site.
    """
    key = (model_id, device)
    if key not in _pipeline_cache:
        from transformers import pipeline

        dev = 0 if device == "cuda" else -1
        pipe = pipeline(
            "text-generation",
            model=model_id,
            device=dev,
            torch_dtype="auto",
            max_new_tokens=512,
        )
        _pipeline_cache[key] = pipe
    return _pipeline_cache[key]


# ---------------------------------------------------------------------------
# Public analysis function
# ---------------------------------------------------------------------------


def analyze_segment(text: str, segment_index: int, pipe) -> AnnotatedText:
    """Run connected-speech analysis on *text* using *pipe*.

    Returns an ``AnnotatedText`` with any detected phenomena.
    On parse failure the spans list will be empty and the raw LLM response
    is preserved in ``AnnotatedText.raw_llm_response`` for debugging.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f'Sentence: "{text}"'},
    ]

    output = pipe(messages, return_full_text=False)
    raw_response: str = output[0]["generated_text"] if output else ""

    # Strip <think>…</think> blocks emitted by reasoning-mode models (Qwen3.5)
    raw_response = _strip_thinking(raw_response)

    spans, raw = _parse_response(raw_response, text)
    return AnnotatedText(
        segment_index=segment_index,
        original_text=text,
        spans=spans,
        raw_llm_response=raw,
    )


# ---------------------------------------------------------------------------
# Thinking-block stripping (Qwen3.5 and similar reasoning models)
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` blocks from model output.

    Qwen3.5 and other reasoning-mode models emit chain-of-thought inside
    ``<think>`` tags before the actual answer.  We strip these so that only
    the final answer (the JSON payload) remains.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# JSON extraction and parsing
# ---------------------------------------------------------------------------


def _parse_response(raw: str, reference_text: str) -> tuple[list[AnnotationSpan], str]:
    """Extract JSON from *raw* and convert to ``list[AnnotationSpan]``."""
    # Strip markdown code fences if the model wrapped its answer
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Attempt direct parse first
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if data is None:
        # Could not parse — return empty annotations
        return [], raw

    spans: list[AnnotationSpan] = []
    for item in data.get("annotations", []):
        try:
            ann_type = item["type"].upper()
            if ann_type not in ("L", "E", "A", "I"):
                continue
            spans.append(
                AnnotationSpan(
                    start_char=int(item["start_char"]),
                    end_char=int(item["end_char"]),
                    annotation_type=ann_type,  # type: ignore[arg-type]
                    phonetic_note=str(item.get("note", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    # Clamp char indices to the actual text length
    text_len = len(reference_text)
    clamped: list[AnnotationSpan] = []
    for span in spans:
        s = max(0, min(span.start_char, text_len))
        e = max(s, min(span.end_char, text_len))
        if s < e:
            clamped.append(
                AnnotationSpan(
                    start_char=s,
                    end_char=e,
                    annotation_type=span.annotation_type,
                    phonetic_note=span.phonetic_note,
                )
            )

    return clamped, raw
