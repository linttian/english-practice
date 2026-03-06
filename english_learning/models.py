from dataclasses import dataclass, field


@dataclass
class Segment:
    """One sentence (or sub-sentence) from the ASR or subtitle parser."""

    index: int
    start: float  # seconds from audio start
    end: float  # seconds from audio start
    text: str
    clip_path: str = ""  # absolute path to exported WAV clip; filled by segmentation.py
    words: list[dict] = field(default_factory=list)
    # Each word dict: {"word": str, "start": float, "end": float, "score": float}
