"""Word-level diff rendering: compare user dictation to the reference transcript.

Returns an HTML string suitable for ``st.markdown(html, unsafe_allow_html=True)``.

Color coding
------------
- Correct / equal  →  no highlight
- Missing word     →  red background + strikethrough  (word in ref, absent in user)
- Wrong word       →  red strikethrough for ref word + green for user word
- Extra word       →  yellow background  (word in user, absent in ref)
"""

import difflib
import html as html_lib
import re

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_diff_html(reference: str, user_input: str) -> str:
    """Return an HTML string highlighting word-level differences.

    Comparison is case-insensitive and strips punctuation from token edges so
    that "hello," and "hello" are treated as the same word.
    """
    ref_tokens = _tokenize(reference)
    usr_tokens = _tokenize(user_input)

    # Normalised keys for comparison
    ref_keys = [_normalise(t) for t in ref_tokens]
    usr_keys = [_normalise(t) for t in usr_tokens]

    matcher = difflib.SequenceMatcher(None, ref_keys, usr_keys, autojunk=False)
    opcodes = matcher.get_opcodes()

    ref_parts: list[str] = []
    usr_parts: list[str] = []

    for tag, i1, i2, j1, j2 in opcodes:
        ref_chunk = ref_tokens[i1:i2]
        usr_chunk = usr_tokens[j1:j2]

        if tag == "equal":
            for w in ref_chunk:
                ref_parts.append(_span(w, color=None))
            for w in usr_chunk:
                usr_parts.append(_span(w, color=None))

        elif tag == "replace":
            for w in ref_chunk:
                ref_parts.append(_span(w, color="#ffcccc"))
            for w in usr_chunk:
                usr_parts.append(
                    _span(
                        w, color="#ffcccc", extra_style="text-decoration:line-through"
                    )
                )
            # If user typed fewer words than reference in this replaced chunk,
            # show explicit missing markers on the user line.
            if len(ref_chunk) > len(usr_chunk):
                for w in ref_chunk[len(usr_chunk) :]:
                    usr_parts.append(
                        _span(
                            f"[missing:{w}]",
                            color="#ffcccc",
                            extra_style="opacity:0.85",
                        )
                    )

        elif tag == "delete":
            # Present in reference, missing from user input.
            # Show missing markers on user line only.
            for w in ref_chunk:
                ref_parts.append(_span(w, color="#ffcccc"))
                usr_parts.append(
                    _span(f"[missing:{w}]", color="#ffcccc", extra_style="opacity:0.85")
                )

        elif tag == "insert":
            # Extra words the user typed that aren't in the reference
            for w in usr_chunk:
                usr_parts.append(_span(w, color="#ffffcc"))

    ref_html = " ".join(ref_parts)
    usr_html = " ".join(usr_parts)

    return (
        "<div style='font-size:1.05em; line-height:1.8'>"
        f"<p><b>Reference:</b> {ref_html}</p>"
        f"<p><b>Your input:</b> {usr_html}</p>"
        "<p style='font-size:0.85em; color:#888'>"
        "<span style='background:#ffcccc; text-decoration:line-through'>wrong</span>&nbsp;"
        "<span style='background:#ffcccc'>[missing:word]</span>&nbsp;"
        "<span style='background:#ffffcc'>extra word</span>"
        "</p>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens, preserving punctuation."""
    return text.split()


def _normalise(token: str) -> str:
    """Lowercase and strip leading/trailing punctuation for comparison."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token.lower())


def _span(word: str, *, color: str | None, extra_style: str = "") -> str:
    escaped = html_lib.escape(word)
    if color is None and not extra_style:
        return escaped
    styles = []
    if color:
        styles.append(f"background:{color}")
    if extra_style:
        styles.append(extra_style)
    style_str = "; ".join(styles)
    return f'<span style="{style_str}">{escaped}</span>'
