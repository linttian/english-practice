"""Entry point for the English Listening Practice app.

Run with:
    uv run streamlit run main.py
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
from dictation.app import main

if __name__ == "__main__":
    main()
