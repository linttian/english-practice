"""Scanning utilities for recent practices."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import List


def scan_recent_practices(limit: int = 8) -> List[dict]:
    out_dir = Path("output")
    if not out_dir.exists():
        return []
    entries = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        session_id = child.name
        seg_path = child / "segments.json"
        if not seg_path.exists():
            continue
        hist_path = child / "practice_history.json"
        last_ts = None
        final_score = None
        segments_count = None

        display_name = session_id
        meta_path = child / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("source_filename"):
                    display_name = meta.get("source_filename")
                if meta.get("saved_at"):
                    last_ts = meta.get("saved_at")
            except Exception:
                pass

        if last_ts is None and hist_path.exists():
            try:
                with open(hist_path, "r", encoding="utf-8") as fh:
                    hist = json.load(fh)
                if hist:
                    last_ts = hist[-1].get("timestamp")
                    final_score = hist[-1].get("final_score")
                    segments_count = hist[-1].get("segments_count")
            except Exception:
                pass

        if last_ts is None:
            try:
                last_ts = datetime.datetime.utcfromtimestamp(
                    seg_path.stat().st_mtime
                ).isoformat()
            except Exception:
                last_ts = datetime.datetime.utcnow().isoformat()

        entries.append(
            {
                "session_id": session_id,
                "display_name": display_name,
                "last_ts": last_ts,
                "final_score": final_score,
                "segments_count": segments_count,
            }
        )

    def _parse_ts(e):
        try:
            return datetime.datetime.fromisoformat(e["last_ts"])
        except Exception:
            return datetime.datetime.min

    entries.sort(key=_parse_ts, reverse=True)
    return entries[:limit]
