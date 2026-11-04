"""UI defaults loaded from config.yaml, with safe fallbacks from config.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    DENSE_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    PROJECT_ROOT,
    VECTOR_INDEX_DIR,
)
from src.vectordb.constants import INDEX_HNSW

DEFAULT_CITATION_BOOST = 0.4
YAML_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class UISettings:
    dense_model: str
    dense_fallback: str
    k: int
    candidate_k: int
    bm25_weight: float
    dense_weight: float
    normalize_scores: bool
    citation_boost_weight: float
    index_type: str
    data_dir: str
    vector_index_dir: str
    snippet_chars: int = 320
    similar_k: int = 5


def _yaml_data(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def load_ui_settings(path: Path = YAML_PATH) -> UISettings:
    data = _yaml_data(path)
    no_normalize = bool(data.get("no_normalize", False))
    return UISettings(
        dense_model=str(data.get("dense_model") or DENSE_MODEL_NAME),
        dense_fallback=str(data.get("dense_fallback") or "sentence-transformers/all-MiniLM-L6-v2"),
        k=int(data.get("k") or DEFAULT_TOP_K),
        candidate_k=int(data.get("candidate_k") or 50),
        bm25_weight=float(data.get("bm25_weight", HYBRID_BM25_WEIGHT)),
        dense_weight=float(data.get("dense_weight", HYBRID_DENSE_WEIGHT)),
        normalize_scores=not no_normalize,
        citation_boost_weight=float(
            data.get("citation_boost_weight", DEFAULT_CITATION_BOOST)
        ),
        index_type=str(data.get("index_type") or INDEX_HNSW),
        data_dir=str(DATA_DIR),
        vector_index_dir=str(VECTOR_INDEX_DIR),
    )
