"""
VectorDBRetriever: a retriever backed by a FAISS index.

It has the same retrieve() output as the other retrievers:
list of (doc_id, score), best first, ties broken by doc ID.
So fusion, reranking and evaluation work with it unchanged.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from config import DEFAULT_TOP_K, DENSE_MODEL_NAME, VECTOR_INDEX_DIR
from src.base_models.base import BaseRetriever
from src.base_models.utils import Corpus, is_blank, top_k_results
from src.vectordb.constants import EMBEDDINGS_SUBDIR, FAISS_MISSING_ID, INDEX_HNSW
from src.vectordb.embeddings import encode_queries, get_or_create_embeddings
from src.vectordb.index_builder import build_index, set_search_params
from src.vectordb.store import load_index, save_index


def drop_missing_hits(
    scores: Sequence[float], positions: Sequence[int], doc_ids: Sequence[str]
) -> Tuple[List[str], List[float]]:
    """Filter out missing FAISS result slots (-1) and map indices to document IDs."""
    ids, kept_scores = [], []
    for score, position in zip(scores, positions):
        if position == FAISS_MISSING_ID:
            continue
        ids.append(doc_ids[int(position)])
        kept_scores.append(float(score))
    return ids, kept_scores


def hits_from_search_row(
    scores: Sequence[float], positions: Sequence[int], doc_ids: Sequence[str], k: int
) -> List[Tuple[str, float]]:
    """Convert raw FAISS search output row to top-k (doc_id, score) tuples."""
    ids, kept_scores = drop_missing_hits(scores, positions, doc_ids)
    if not ids:
        return []
    return top_k_results(ids, kept_scores, k)


class VectorDBRetriever(BaseRetriever):
    """Retriever backed by a FAISS vector index."""

    def __init__(
        self,
        index: Optional[faiss.Index],
        doc_ids: Sequence[str],
        encoder: Any,
        index_type: str = INDEX_HNSW,
        model_name: str = DENSE_MODEL_NAME,
        search_params: Optional[Dict[str, Any]] = None,
    ):
        self.index = index
        self.doc_ids = list(doc_ids)
        self.encoder = encoder
        self.index_type = index_type
        self.model_name = model_name
        self.search_params = dict(search_params or {})

    # ---------- ways to create one ----------

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        encoder: Any,
        index_type: str = INDEX_HNSW,
        model_name: str = DENSE_MODEL_NAME,
        build_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> "VectorDBRetriever":
        """Build an index from corpus texts and return an initialized retriever."""
        cache_dir = Path(cache_dir) if cache_dir else Path(VECTOR_INDEX_DIR) / EMBEDDINGS_SUBDIR
        doc_ids, embeddings = get_or_create_embeddings(
            corpus, encoder, model_name, cache_dir, use_cache=use_cache, show_progress=show_progress
        )
        if not doc_ids:
            return cls(None, [], encoder, index_type, model_name, search_params)
        index = build_index(index_type, embeddings, **(build_params or {}))
        set_search_params(index, index_type, **(search_params or {}))
        return cls(index, doc_ids, encoder, index_type, model_name, search_params)

    @classmethod
    def from_saved(cls, index_dir: Path, encoder: Any) -> "VectorDBRetriever":
        """Load an existing index, doc IDs, and metadata from disk."""
        index, doc_ids, meta = load_index(index_dir)
        return cls(index, doc_ids, encoder, meta["index_type"], meta.get("model_name", ""), meta.get("search_params"))

    # ---------- using it ----------

    def save(self, index_dir: Path) -> Path:
        """Persist index, document IDs, and metadata to disk."""
        if self.index is None:
            raise ValueError("Nothing to save: the index is empty")
        return save_index(self.index, self.doc_ids, self.index_type, index_dir, self.search_params, self.model_name)

    def set_search_params(self, **search_params: Any) -> None:
        """Dynamically update search parameters like nprobe or ef_search."""
        self.search_params.update(search_params)
        if self.index is not None:
            set_search_params(self.index, self.index_type, **search_params)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a query ordered by similarity score."""
        if k <= 0:
            raise ValueError("k must be positive")
        if is_blank(query) or self.index is None or not self.doc_ids:
            return []
        query_vec = encode_queries(self.encoder, [query])
        scores, positions = self.index.search(query_vec, min(k, len(self.doc_ids)))
        return hits_from_search_row(scores[0], positions[0], self.doc_ids, k)
