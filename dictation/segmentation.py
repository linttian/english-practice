"""Audio segmentation: slice an audio file into per-sentence WAV clips.

System requirement: ffmpeg must be available in PATH.
  Ubuntu/Debian/WSL:  sudo apt install ffmpeg
"""

import json
import re
import tempfile
from pathlib import Path

from .models import Segment

_CACHE_FILE = "segments.json"


def _is_sentence_final(text: str, next_text: str | None = None) -> bool:
    """Return True if the text likely ends a sentence.

    Treats '?' and '!' as sentence-final. For '.' we apply heuristics to
    avoid treating common abbreviations and initials (e.g. "Mrs.", "U.S.")
    as sentence endings.
    """
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    last_char = t[-1]
    if last_char in "!?":
        return True
    if last_char != ".":
        return False

    # Remove closing quotes/parens/brackets
    cleaned = re.sub(r"[\"'\)\]]+$", "", t)
    # Get the last token
    parts = re.split(r"\s+", cleaned)
    last_word = parts[-1] if parts else cleaned

    # If it's a sequence of single-letter initials like "U.S." or "A.B.C.",
    # we need context: treat as sentence-final if there's no following text
    # or if the following text looks like the start of a new sentence.
    if re.match(r"^(?:[A-Za-z]\.)+$", last_word):
        if not next_text:
            return True
        # if next text starts with lowercase, it's likely a continuation
        first = next_text.strip()[:1]
        if first and first.islower():
            return False
        return True

    # Honorifics/titles that almost always continue with a name (don't end sentences)
    titles = r"^(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam|Mx|Rev|Fr|Br|Sr|Sra|Sra\.|Srta|Hon|Pres|Gov|Sen|Rep|Amb|Capt|Cmdr|Lt|Col|Maj|Gen|Adm|Sgt|Cpl|Pvt)\.?$"
    if re.match(titles, last_word, re.IGNORECASE):
        return False

    # Common abbreviations (months, companies, degrees, etc.) that usually don't end sentences
    abbrev = (
        r"^(?:Sr|Jr|St|vs|v|etc|e\.g|i\.e|approx|ca|cf|fig|dept|univ|dept|est|no|sec|chap)\.?$|"
        r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?$|"
        r"^(?:Inc|Ltd|Co|Corp|LLC|PLC|GmbH|S\.A\.|SAS)\.?$|"
        r"^(?:Ph\.D|M\.D|B\.A|M\.A|B\.Sc|M\.Sc|D\.D\.S|LL\.B|J\.D|Esq)\.?$|"
        # Common country/organization abbreviations (with or without dots)
        r"^(?:US|U\.S\.|USA|U\.S\.A|UK|U\.K\.|UAE|U\.A\.E\.|PRC|P\.R\.C\.|ROC|R\.O\.C\.|EU|E\.U\.|U\.N\.|U\.S\.S\.R\.|USSR|N\.Z\.|NZ|CAN|AUS|Aus)\.?$"
    )
    if re.match(abbrev, last_word, re.IGNORECASE):
        # If there's no following text, allow sentence end
        if not next_text:
            return True
        first = next_text.strip()[:1]
        if first and first.islower():
            return False
        return True

    # Single-letter initials like "A." are likely abbreviations
    if re.match(r"^[A-Za-z]\.$", last_word):
        if not next_text:
            return True
        first = next_text.strip()[:1]
        if first and first.islower():
            return False
        return True

    return True


def _segments_to_json(segments: list[Segment]) -> list[dict]:
    return [
        {
            "index": s.index,
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "clip_path": s.clip_path,
            "words": s.words,
        }
        for s in segments
    ]


def _segments_from_json(data: list[dict]) -> list[Segment]:
    return [
        Segment(
            index=d["index"],
            start=d["start"],
            end=d["end"],
            text=d["text"],
            clip_path=d.get("clip_path", ""),
            words=d.get("words", []),
        )
        for d in data
    ]


def load_cached_segments(
    session_id: str, output_root: str = "./output"
) -> list[Segment] | None:
    """Return cached segments for *session_id*, or ``None`` if no cache exists."""
    cache_path = Path(output_root) / session_id / _CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        segments = _segments_from_json(data)
        # Verify that all clip files still exist
        if all(s.clip_path and Path(s.clip_path).exists() for s in segments):
            return segments
    except Exception:
        pass
    return None


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV for ASR compatibility.

    Uses pydub (backed by ffmpeg) so it handles all containers the app accepts:
    mp3, wav, mp4, mkv, m4a, ogg, flac, aac.

    Returns the path to a new temporary WAV file.
    """
    try:
        from pydub import AudioSegment as PydubSegment
    except ImportError as e:
        raise RuntimeError("pydub is not installed. Run: uv sync") from e

    try:
        audio = PydubSegment.from_file(input_path)
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH.\n"
                "Install it with: sudo apt install ffmpeg"
            ) from e
        raise

    audio = audio.set_frame_rate(16_000).set_channels(1)

    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_tmp.close()
    audio.export(wav_tmp.name, format="wav")
    return wav_tmp.name


def slice_audio(
    audio_path: str,
    segments: list[Segment],
    session_id: str,
    output_root: str = "./output",
    padding_ms: int = 100,
    min_sentence_length: int = 1,
    merge_on_punctuation: bool = True,
    max_duration: int = 60,  # Default max duration in seconds
) -> list[Segment]:
    """Slice *audio_path* into one WAV clip per segment with improved segmentation.

    This function ensures that segments are not overly split by merging short segments
    that are part of the same sentence. Segments shorter than `min_sentence_length` words
    are merged with the next segment if possible. Additionally, segments are merged
    based on punctuation if `merge_on_punctuation` is enabled.

    Each clip is saved to ``{output_root}/{session_id}/clip_NNN.wav`` as
    16 kHz mono WAV for maximum compatibility with Streamlit's audio player.

    A ``padding_ms``-millisecond pad is added at both ends of each clip so
    listeners have a natural lead-in / lead-out.

    ``segment.clip_path`` is set in-place on every Segment; the mutated list
    is also returned for convenience.

    Raises ``RuntimeError`` if pydub cannot find the ffmpeg binary.
    """
    try:
        from pydub import AudioSegment as PydubSegment
    except ImportError as e:
        raise RuntimeError("pydub is not installed. Run: uv sync") from e

    out_dir = Path(output_root) / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        audio = PydubSegment.from_file(audio_path)
    except FileNotFoundError as e:
        if "ffmpeg" in str(e).lower() or "avconv" in str(e).lower():
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH.\n"
                "Install it with: sudo apt install ffmpeg"
            ) from e
        raise

    audio_len_ms = len(audio)

    # Merge short segments and segments based on punctuation
    merged_segments = []
    buffer_segment = None

    for i, seg in enumerate(segments):
        if buffer_segment is None:
            buffer_segment = seg
        else:
            # Check if the current segment should be merged
            next_text = seg.text if seg and hasattr(seg, "text") else None
            if merge_on_punctuation and not _is_sentence_final(
                buffer_segment.text, next_text
            ):
                buffer_segment.end = seg.end
                buffer_segment.text += " " + seg.text
            elif len(buffer_segment.text.split()) < min_sentence_length:
                buffer_segment.end = seg.end
                buffer_segment.text += " " + seg.text
            else:
                merged_segments.append(buffer_segment)
                buffer_segment = seg

    if buffer_segment:
        merged_segments.append(buffer_segment)

    # Convert max_duration to milliseconds
    max_duration_ms = max_duration * 1000

    # Slice audio based on merged segments with max_duration constraint
    final_segments = []
    for seg in merged_segments:
        start_ms = max(0, int(seg.start * 1000) - padding_ms)
        end_ms = min(audio_len_ms, int(seg.end * 1000) + padding_ms)

        while end_ms - start_ms > max_duration_ms:
            split_point = start_ms + max_duration_ms
            clip = audio[start_ms:split_point]
            clip = clip.set_frame_rate(16_000).set_channels(1)

            out_path = clips_dir / f"clip_{len(final_segments):03d}.wav"
            clip.export(str(out_path), format="wav")

            final_segments.append(
                Segment(
                    index=len(final_segments),
                    start=start_ms / 1000,
                    end=split_point / 1000,
                    text=seg.text,
                    clip_path=str(out_path.resolve()),
                )
            )

            start_ms = split_point

        # Handle the remaining part of the segment
        clip = audio[start_ms:end_ms]
        clip = clip.set_frame_rate(16_000).set_channels(1)

        out_path = clips_dir / f"clip_{len(final_segments):03d}.wav"
        clip.export(str(out_path), format="wav")

        final_segments.append(
            Segment(
                index=len(final_segments),
                start=start_ms / 1000,
                end=end_ms / 1000,
                text=seg.text,
                clip_path=str(out_path.resolve()),
            )
        )

    # Persist segment metadata so the same file can be loaded from cache
    cache_path = out_dir / _CACHE_FILE
    cache_path.write_text(
        json.dumps(_segments_to_json(final_segments), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return final_segments
