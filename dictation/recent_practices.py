"""Render Recent Practices tab logic (keeps HTML out of app.py).

This module provides a single function `render_recent_practices_tab` which requires
callbacks for scanning recent entries and for actually loading a session from cache.
This avoids circular imports with `app.py`.
"""

from __future__ import annotations

import datetime
from typing import Callable

import streamlit as st

from .ui_templates import recent_item_card_html


def render_recent_practices_tab(
    limit: int,
    scan_callback: Callable[[int], list],
    load_callback: Callable[[str], bool],
) -> None:
    """Render Recent Practices UI.

    - `scan_callback(limit)` should return list of entries like app's `_scan_recent_practices`.
    - `load_callback(session_id)` should load the session into `st.session_state`.
    """
    st.header("📚 Recent Practices")
    recent = scan_callback(limit)
    if not recent:
        st.caption("No cached practices yet. Completed practices will appear here.")
        return

    # Prepare single-selection behavior
    all_ids = [it["session_id"] for it in recent]

    def _select_recent(selected_id, all_ids_param):
        st.session_state["recent_selected"] = selected_id
        for sid in all_ids_param:
            key = f"recent_sel_{sid}"
            st.session_state[key] = sid == selected_id

    # Ensure keys exist
    for sid in all_ids:
        key = f"recent_sel_{sid}"
        if key not in st.session_state:
            st.session_state[key] = False

    # Render list
    for item in recent:
        session_id = item["session_id"]
        ts = item.get("last_ts")
        try:
            pretty = datetime.datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pretty = ts
        display = item.get("display_name") or pretty
        score = item.get("final_score")
        score_html = (
            f"<div style='color:#444'>🏆 Final: {int(score)}/100</div>"
            if score is not None
            else ""
        )

        cols = st.columns([0.12, 0.88])
        with cols[0]:
            st.checkbox(
                "Select",
                key=f"recent_sel_{session_id}",
                label_visibility="hidden",
                on_change=_select_recent,
                args=(session_id, all_ids),
            )
        with cols[1]:
            st.markdown(
                recent_item_card_html(display, pretty, score_html),
                unsafe_allow_html=True,
            )

    selected_id = st.session_state.get("recent_selected")
    if st.button("Load Selected", key="load_selected_recent_tab"):
        if not selected_id:
            st.info("Please select a recent practice to load.")
        else:
            load_callback(selected_id)
