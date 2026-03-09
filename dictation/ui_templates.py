"""HTML/CSS templates used by the Streamlit UI.

Keep HTML snippets and CSS here so `app.py` stays focused on control flow.
"""

from __future__ import annotations

import datetime
from html import escape

import streamlit as st


def inject_tab_css(font_size: str = "1.6rem") -> None:
    """Inject CSS for Streamlit tabs. Small helper so tests or callers can tune size."""
    st.markdown(
        f"""
    <style>
    /* Stronger selector and larger size to ensure Streamlit tabs show bigger text */
    [role="tablist"] > [role="tab"],
    div[role="tablist"] > button[role="tab"] {{
        font-size: {font_size} !important;
        padding: 14px 26px !important;
        border-radius: 6px 6px 0 0 !important;
    }}
    [role="tablist"] > [role="tab"][aria-selected="true"],
    div[role="tablist"] > button[role="tab"][aria-selected="true"] {{
        font-weight: 700 !important;
        color: #0b5394 !important;
        box-shadow: inset 0 -3px 0 0 #0b5394 !important;
    }}
    /* Make tab container slightly separated */
    [role="tablist"] {{ gap: 10px !important; margin-bottom: 14px !important; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def recent_item_card_html(
    display_name: str, pretty_ts: str, score_html: str = ""
) -> str:
    """Return the HTML block for a recent-practice item.

    `display_name` is already escaped by caller if needed; this helper will escape `pretty_ts`.
    """
    display = escape(display_name)
    pretty = escape(pretty_ts)
    return (
        f"<div style='display:flex;align-items:center;gap:10px;padding:8px;border-radius:6px;border:1px solid #eee;margin-bottom:8px'>"
        f"<div style='flex:1'>"
        f"<div style='font-weight:600'>🔖 {display}</div>"
        f"<div style='color:#666;font-size:0.9em'>📅 {pretty}</div>"
        f"{score_html}"
        f"</div>"
        f"</div>"
    )


def monthly_heatmap_html(month_map: dict, year: int, mon: int) -> str:
    """Build a simple monthly heatmap HTML for analytics tab.

    `month_map` maps day->count. Caller computes days_in_month.
    """
    days_in_month = (
        datetime.date(year + int(mon / 12), (mon % 12) + 1, 1)
        - datetime.timedelta(days=1)
    ).day

    max_c = max(month_map.values()) if month_map else 0
    html = ["<div style='display:flex; flex-wrap:wrap; gap:6px; max-width:700px;'>"]
    for day in range(1, days_in_month + 1):
        c = month_map.get(day, 0)
        intensity = 0
        if max_c > 0:
            intensity = int(255 - (c / max_c) * 200)
        color = f"rgb({intensity},{255},{intensity})"
        html.append(
            f"<div style='width:28px; height:28px; background:{color}; display:flex; align-items:center; justify-content:center; border-radius:4px;' title='{escape(str(c))} times'>{day}</div>"
        )
    html.append("</div>")
    return "".join(html)
