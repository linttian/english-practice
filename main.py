"""Entry point for the English Listening Practice app.

Run with:
    uv run streamlit run main.py
"""

import os


def _load_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ if not set.

    This avoids adding an extra runtime dependency while still allowing
    environment configuration from the project root.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        # Best-effort: if reading .env fails, fall back to defaults below
        pass


# Load .env from project root (optional)
_load_env()

from dictation.app import main

if __name__ == "__main__":
    main()
