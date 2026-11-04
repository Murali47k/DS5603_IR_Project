"""
USER INTERFACE MODULE
Interactive Streamlit app for SciDocs search.

Default engine: citation-boosted hybrid retrieval (BM25 + FAISS + citation_aware_boost).
Launch from the repository root:

    .venv/bin/streamlit run src/ui/app.py
"""

from src.ui.search_service import SearchEngine, build_search_engine
from src.ui.settings import load_ui_settings

__all__ = ["SearchEngine", "build_search_engine", "load_ui_settings"]
