"""Render Practice Analytics tab (moved out of app.py).

This module expects a `scan_callback()` that returns the practice index
(e.g. `app._load_practice_index`) so we avoid importing app and creating
circular imports.
"""

from __future__ import annotations

import datetime
from typing import Callable

import streamlit as st

from .ui_templates import monthly_heatmap_html


def render_analytics(scan_callback: Callable[[], list]) -> None:
    idx = scan_callback()
    if not idx:
        return

    st.header("📈 Practice Analytics")
    # Controls: time range
    timescale = st.selectbox(
        "Time Range", ["7 days", "30 days", "90 days", "All"], index=1
    )
    if timescale == "7 days":
        days = 7
    elif timescale == "30 days":
        days = 30
    elif timescale == "90 days":
        days = 90
    else:
        days = None

    # Build per-day aggregates
    per_day = {}
    for ev in idx:
        try:
            d = datetime.datetime.fromisoformat(ev["timestamp"]).date()
        except Exception:
            continue
        if days is not None:
            if (datetime.date.today() - d).days > days:
                continue
        key = d.isoformat()
        entry = per_day.setdefault(key, {"count": 0, "sum_score": 0.0, "score_n": 0})
        entry["count"] += 1
        if ev.get("final_score") is not None:
            entry["sum_score"] += float(ev.get("final_score"))
            entry["score_n"] += 1

    # Sort keys
    dates = sorted(per_day.keys())
    counts = [per_day[d]["count"] for d in dates]
    avg_scores = [
        (
            per_day[d]["sum_score"] / per_day[d]["score_n"]
            if per_day[d]["score_n"] > 0
            else None
        )
        for d in dates
    ]

    if dates:
        st.subheader("Daily Practice Count")
        st.bar_chart(counts)
        st.subheader("Average Score (if available)")
        score_vals = [s for s in avg_scores if s is not None]
        if score_vals:
            st.line_chart(score_vals)

    # Monthly calendar-like heatmap for current month
    try:
        today = datetime.date.today()
        month = st.date_input("Select month (any day)", value=today)
        year = month.year
        mon = month.month
        # build day->count map for that month
        month_map = {}
        for ev in idx:
            try:
                d = datetime.datetime.fromisoformat(ev["timestamp"]).date()
            except Exception:
                continue
            if d.year == year and d.month == mon:
                month_map.setdefault(d.day, 0)
                month_map[d.day] += 1

        st.subheader("Monthly Activity Heatmap (approx)")
        st.markdown(monthly_heatmap_html(month_map, year, mon), unsafe_allow_html=True)
    except Exception:
        pass
