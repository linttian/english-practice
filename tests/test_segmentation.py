import os
import tempfile
import wave

from dictation import segmentation
from dictation.models import Segment


def _make_silence(path: str, duration_s: float = 3.0, rate: int = 16000):
    n_frames = int(duration_s * rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def test_merge_abbreviations():
    # Create a temporary silent wav long enough for segments
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    try:
        _make_silence(tmp_path, duration_s=4.0)

        # Segments that might be incorrectly split on periods
        segs = [
            Segment(index=0, start=0.0, end=0.5, text="Mrs."),
            Segment(index=1, start=0.5, end=1.2, text="Smith went home."),
            Segment(index=2, start=1.2, end=1.8, text="He visited the U.S."),
            Segment(index=3, start=1.8, end=2.4, text="office."),
        ]

        out = segmentation.slice_audio(
            tmp_path,
            segs,
            session_id="testsess",
            output_root="./output",
            padding_ms=0,
            min_sentence_length=1,
            merge_on_punctuation=True,
            max_duration=60,
        )

        texts = [s.text for s in out]
        print("Resulting segments:", texts)

        # Expectation: "Mrs." should merge with following text -> not create an isolated sentence
        assert any("Mrs." in t and "Smith" in t for t in texts), (
            "Mrs. should merge with Smith"
        )
        # Expect U.S. not to be treated as sentence-final (so it merges with following 'office.')
        assert any("U.S." in t and "office" in t for t in texts), (
            "U.S. should not end the sentence"
        )

        print("PASS: abbreviation merging tests")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def test_us_sentence_end():
    # Ensure a sentence that ends with 'U.S.' is treated as sentence-final
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    try:
        _make_silence(tmp_path, duration_s=3.0)

        segs = [
            Segment(index=0, start=0.0, end=1.0, text="He visited the U.S."),
            Segment(index=1, start=1.0, end=1.8, text="Next sentence starts."),
        ]

        out = segmentation.slice_audio(
            tmp_path,
            segs,
            session_id="testsess2",
            output_root="./output",
            padding_ms=0,
            min_sentence_length=1,
            merge_on_punctuation=True,
            max_duration=60,
        )

        texts = [s.text for s in out]
        print("Resulting segments (U.S. end):", texts)

        # Expectation: the U.S. segment should remain as its own sentence
        assert any(t.strip().endswith("U.S.") for t in texts), (
            "Sentence ending with U.S. should be its own segment"
        )

        print("PASS: U.S. sentence-end test")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    test_merge_abbreviations()
    print("All tests passed")
