"""I/O helper utilities for the dictation app.

This module centralizes file paths and read/write helpers so the main
UI file can remain focused on presentation and flow.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any


def practice_index_path() -> str:
    return os.path.join("output", "practice_index.json")


def load_practice_index() -> list:
    path = practice_index_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def save_practice_index(index: list) -> None:
    path = practice_index_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def ui_state_path(session_id: str) -> str:
    return os.path.join("output", session_id, "ui_state.json")


def scores_path(session_id: str) -> str:
    return os.path.join("output", session_id, "scores.json")


def save_segments_to_json(
    segments: list[Any],
    session_id: str,
    output_root: str = "./output",
    original_filename: str | None = None,
) -> None:
    output_dir = Path(output_root) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "segments.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [segment.__dict__ for segment in segments],
                f,
                ensure_ascii=False,
                indent=4,
            )
    except Exception:
        pass
    # Also write meta.json
    try:
        meta = {
            "saved_at": datetime.datetime.utcnow().isoformat(),
            "source_filename": original_filename,
        }
        meta_path = output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def save_ui_state(session_id: str, payload: dict) -> None:
    if not session_id:
        return
    path = ui_state_path(session_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_ui_state(session_id: str) -> dict | None:
    if not session_id:
        return None
    path = ui_state_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # normalize
        dictations = {}
        for k, v in data.get("dictations", {}).items():
            try:
                dictations[int(k)] = v
            except Exception:
                continue
        show_loop = {}
        for k, v in data.get("show_loop", {}).items():
            try:
                show_loop[int(k)] = bool(v)
            except Exception:
                continue
        last_idx = (
            int(data.get("last_segment_idx", 0))
            if data.get("last_segment_idx") is not None
            else 0
        )
        return {
            "last_segment_idx": last_idx,
            "dictations": dictations,
            "show_loop": show_loop,
        }
    except Exception:
        return None


def save_scores(session_id: str, scores: dict, final: float | None = None) -> None:
    if not session_id:
        return
    path = scores_path(session_id)
    payload = {
        "by_index": {str(k): v for k, v in (scores or {}).items()},
        "final": final,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_scores(session_id: str) -> tuple[dict, float | None] | None:
    if not session_id:
        return None
    path = scores_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "by_index" in data:
            by_idx = {int(k): v for k, v in data.get("by_index", {}).items()}
            return by_idx, data.get("final")
        if isinstance(data, dict):
            by_idx = {}
            final = data.get("final") if "final" in data else None
            for k, v in data.items():
                try:
                    ik = int(k)
                    by_idx[ik] = v
                except Exception:
                    continue
            if by_idx:
                return by_idx, final
        return None
    except Exception:
        return None


def record_practice_event(
    session_id: str,
    final_score: float | None,
    segments_count: int,
    label: str | None = None,
) -> None:
    now = datetime.datetime.utcnow().isoformat()
    event = {
        "session_id": session_id,
        "timestamp": now,
        "final_score": None if final_score is None else float(final_score),
        "segments_count": int(segments_count),
        "label": label,
    }

    # update global index
    idx = load_practice_index()
    idx.append(event)
    if len(idx) > 200:
        idx = idx[-200:]
    save_practice_index(idx)

    # per-session history
    try:
        session_dir = os.path.join("output", session_id)
        os.makedirs(session_dir, exist_ok=True)
        hist_path = os.path.join(session_dir, "practice_history.json")
        hist = []
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as fh:
                hist = json.load(fh)
        hist.append(event)
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump(hist[-100:], fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
