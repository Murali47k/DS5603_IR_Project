"""
USER INTERFACE MODULE
Interactive Streamlit app for SciDocs search.

Default engine: citation-boosted hybrid retrieval (BM25 + FAISS + citation_aware_boost).
Launch from the repository root:

    .venv/bin/streamlit run src/ui/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

venv_dir = ROOT / ".venv"
for sp in venv_dir.glob("lib/python*/site-packages"):
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))

from src.ui.search_service import SearchEngine, build_search_engine
from src.ui.settings import load_ui_settings

__all__ = ["SearchEngine", "build_search_engine", "load_ui_settings"]
