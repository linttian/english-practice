from abc import ABC, abstractmethod

from ..models import Segment


class ASREngine(ABC):
    """Abstract base class for all ASR engines.

    All implementations must:
    - Set ``name`` as a class-level string (used in the UI sidebar).
    - Implement ``transcribe()`` to return sentence-level Segments.
    - Optionally override ``load()`` to download/initialise the model once.
    - Optionally override ``unload()`` to release GPU memory.
    """

    name: str = "base"

    def load(self, device: str) -> None:
        """Download and load the model into memory.

        Called once before the first ``transcribe()`` call.
        ``device`` is either ``"cpu"`` or ``"cuda"``.
        Default implementation is a no-op for lightweight engines.
        """

    @abstractmethod
    def transcribe(self, audio_path: str) -> list[Segment]:
        """Transcribe *audio_path* and return sentence-level segments.

        Contract:
        - Segments are sorted ascending by ``start``.
        - ``text`` is stripped of leading/trailing whitespace.
        - ``words`` list may be empty if the engine does not support
          word-level timestamps.
        """

    def unload(self) -> None:
        """Release model from memory (e.g. free GPU VRAM).

        Optional override; default is a no-op.
        """
