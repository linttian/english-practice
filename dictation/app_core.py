"""Core Application class for the Streamlit UI.

This module contains `Application` which coordinates the high-level UI wiring.
Helpers (rendering and data functions) are injected so `app.py` stays thin
and we avoid circular imports.
"""

from __future__ import annotations

import logging
import traceback
from typing import Callable

import streamlit as st

from . import ui_templates

logger = logging.getLogger(__name__)


class Application:
    def __init__(
        self,
        init_session_state: Callable[[], None],
        render_sidebar: Callable[[], tuple[str, str, dict]],
        render_upload_section: Callable[[str, str, dict], None],
        scan_recent_practices: Callable[[int], list],
        load_session_from_cache: Callable[[str], bool],
        render_segment: Callable[[object], None],
        load_practice_index: Callable[[], list],
        analytics_module,
        recent_practices_module,
    ) -> None:
        self.init_session_state = init_session_state
        self.render_sidebar = render_sidebar
        self.render_upload_section = render_upload_section
        self.scan_recent_practices = scan_recent_practices
        self.load_session_from_cache = load_session_from_cache
        self.render_segment = render_segment
        self.load_practice_index = load_practice_index
        self.analytics_module = analytics_module
        self.recent_practices_module = recent_practices_module

    def run(self) -> None:
        try:
            st.set_page_config(
                page_title="English Listening Practice",
                page_icon="🎧",
                layout="wide",
            )
            logger.debug("Application.run invoked")

            # initialize session state
            self.init_session_state()

            # inject tab css
            ui_templates.inject_tab_css(font_size="1.6rem")

            engine_name, device, settings = self.render_sidebar()

            tabs = st.tabs(["Workspace", "Recent Practices", "Practice Analytics"])

            # Workspace tab: upload + dictation workspace
            with tabs[0]:
                st.title("🎧 English Listening Practice")
                st.caption(
                    "Upload audio or video, transcribe it, then practise Listen → Write → Compare."
                )

                self.render_upload_section(engine_name, device, settings)

                segments: list = st.session_state.get("segments", [])
                if segments:
                    st.header(f"2. Dictation Workspace ({len(segments)} segments)")

                    if "current_segment_idx" not in st.session_state:
                        st.session_state.current_segment_idx = 0

                    # Validate index
                    if st.session_state.current_segment_idx >= len(segments):
                        st.session_state.current_segment_idx = len(segments) - 1
                    elif st.session_state.current_segment_idx < 0:
                        st.session_state.current_segment_idx = 0

                    current_idx = st.session_state.current_segment_idx

                    # Keep slider state synced with external navigation buttons.
                    desired_slider_value = current_idx + 1
                    if (
                        st.session_state.get("_jump_slider_widget")
                        != desired_slider_value
                    ):
                        st.session_state["_jump_slider_widget"] = desired_slider_value

                    # Persist current reading position for this audio hash.
                    try:
                        from .app import _save_last_segment_idx

                        _save_last_segment_idx(
                            st.session_state.get("session_id", ""), current_idx
                        )
                    except Exception:
                        pass

                    # Top navigation
                    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                    with nav_col1:
                        if st.button(
                            "⬅ Previous",
                            use_container_width=True,
                            disabled=(current_idx <= 0),
                        ):
                            st.session_state.current_segment_idx -= 1
                            st.rerun()
                    with nav_col2:
                        # We don't provide a 'value' if the key exists, let Streamlit manage the state.
                        # If we need to force an external update, we just set the un-mapped value.
                        def _on_slider_change():
                            st.session_state.current_segment_idx = (
                                st.session_state._jump_slider_widget - 1
                            )

                        st.slider(
                            "Jump to segment",
                            min_value=1,
                            max_value=len(segments),
                            key="_jump_slider_widget",
                            label_visibility="collapsed",
                            on_change=_on_slider_change,
                        )
                        st.markdown(
                            f"<p style='text-align: center; color: #666;'>Segment <b>{current_idx + 1}</b> / {len(segments)}</p>",
                            unsafe_allow_html=True,
                        )
                    with nav_col3:
                        if st.button(
                            "Next ➡",
                            use_container_width=True,
                            disabled=(current_idx >= len(segments) - 1),
                        ):
                            st.session_state.current_segment_idx += 1
                            st.rerun()

                    st.divider()

                    # Show progress and final score button
                    scores = st.session_state.get("scores", {})
                    total = len(segments)
                    scored_count = sum(
                        1
                        for s in segments
                        if self._get_score(scores, s.index) is not None
                    )
                    col_status, col_final = st.columns([3, 1])
                    with col_status:
                        st.markdown(
                            f"**Progress:** {scored_count} / {total} segments scored"
                        )
                        progress_val = scored_count / total if total else 0
                        st.progress(progress_val)
                    with col_final:
                        final_shown = st.session_state.get("final_score")
                        if scored_count == total and total > 0:
                            if st.button(
                                "Finish & Calculate Final Score",
                                use_container_width=True,
                            ):
                                final_score = self._calculate_final_score(
                                    segments, scores
                                )
                                st.session_state.final_score = final_score
                                try:
                                    from .app import (
                                        _record_practice_event,
                                        _save_scores,
                                    )

                                    _save_scores(
                                        st.session_state.get("session_id", ""),
                                        scores,
                                        final=final_score,
                                    )
                                    try:
                                        _record_practice_event(
                                            st.session_state.get("session_id", ""),
                                            final_score,
                                            len(segments),
                                        )
                                    except Exception:
                                        pass
                                except Exception:
                                    pass
                                st.balloons()
                                st.success(f"Final score: {final_score:.2f} / 100")
                        elif final_shown is not None:
                            st.markdown(f"**Final:** {final_shown:.2f} / 100")

                    # Render only the current segment
                    try:
                        self.render_segment(segments[current_idx])
                    except Exception:
                        # fall back to direct call if provided as global
                        try:
                            from .app import _render_segment as fallback_render

                            fallback_render(segments[current_idx])
                        except Exception:
                            raise

            # Recent Practices and Analytics tabs
            with tabs[1]:
                # delegate to the recent_practices module for rendering
                try:
                    self.recent_practices_module.render_recent_practices_tab(
                        limit=4,
                        scan_callback=self.scan_recent_practices,
                        load_callback=self.load_session_from_cache,
                    )
                except Exception:
                    # fallback: inline logic (minimal)
                    recent = self.scan_recent_practices(limit=4)
                    if recent:
                        st.write(recent)

            with tabs[2]:
                try:
                    self.analytics_module.render_analytics(
                        scan_callback=self.load_practice_index
                    )
                except Exception:
                    # fallback: minimal indicator
                    st.info("No analytics available")

        except Exception:
            # Log full traceback for diagnosis and re-raise so Streamlit shows error
            tb = traceback.format_exc()
            logger.exception("Unhandled exception in Application.run:\n%s", tb)
            raise

    # small helpers that re-use app-level utilities when run via Application
    def _get_score(self, scores: dict, idx: int):
        # mimic earlier helper behaviour
        if scores is None:
            return None
        val = scores.get(idx)
        if val is not None:
            return val
        return scores.get(str(idx))

    def _calculate_final_score(self, segments: list, scores: dict) -> float:
        weights = [max(1, len(s.text.split())) for s in segments]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        weighted = 0.0
        for s, w in zip(segments, weights):
            sc = self._get_score(scores, s.index)
            if sc is None:
                sc = 0
            weighted += sc * w
        return weighted / total_weight
