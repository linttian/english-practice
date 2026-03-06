"""ASR sub-package.

Import the registry to get the mapping of sidebar display names to engine classes.
Engine modules are imported lazily — only when an engine is actually selected.

Usage::

    from english_learning.asr import ENGINE_REGISTRY

    EngineCls = ENGINE_REGISTRY["whisper (large-v3-turbo)"]
    engine = EngineCls()
    engine.load("cpu")
    segments = engine.transcribe("/path/to/audio.wav")
"""

import importlib

# Maps sidebar display name → (relative module, class name).
# No engine module is imported until its entry is accessed.
_ENGINE_LOADERS: dict[str, tuple[str, str]] = {
    "whisper (small)": (".whisper", "WhisperSmallEngine"),
    "whisper (base)": (".whisper", "WhisperBaseEngine"),
    "whisper (large-v3-turbo)": (".whisper", "WhisperEngine"),
    "whisper (distil-large-v3)": (".whisper", "DistilWhisperEngine"),
    "qwen3-asr (0.6b)": (".qwen", "QwenEngine"),
    "qwen3-asr (1.7b)": (".qwen", "QwenLargeEngine"),
}


class _EngineRegistry:
    """Lazy engine registry — modules are imported only when an engine is selected.

    Supports ``list(registry.keys())``, ``registry[name]``, ``name in registry``,
    and iteration — everything Streamlit's selectbox needs.
    """

    def __init__(self) -> None:
        self._cache: dict[str, type] = {}

    # -- dict-like interface used by Streamlit and app code -----------------

    def keys(self):
        return _ENGINE_LOADERS.keys()

    def __getitem__(self, name: str) -> type:
        if name not in _ENGINE_LOADERS:
            raise KeyError(f"Unknown ASR engine: {name!r}")
        if name not in self._cache:
            module_path, class_name = _ENGINE_LOADERS[name]
            mod = importlib.import_module(module_path, package=__package__)
            self._cache[name] = getattr(mod, class_name)
        return self._cache[name]

    def __contains__(self, item: object) -> bool:
        return item in _ENGINE_LOADERS

    def __iter__(self):
        return iter(_ENGINE_LOADERS)

    def __len__(self) -> int:
        return len(_ENGINE_LOADERS)

    def __repr__(self) -> str:
        return f"_EngineRegistry({list(_ENGINE_LOADERS.keys())})"


ENGINE_REGISTRY = _EngineRegistry()

__all__ = ["ENGINE_REGISTRY"]
