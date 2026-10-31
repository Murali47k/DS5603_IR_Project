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
    """
    Description: Turn FAISS positions into doc IDs and remove the -1 "no answer" slots.
    Inputs: scores, positions (one query's row from index.search), doc_ids.
        Reads global FAISS_MISSING_ID.
    Outputs: (ids, scores) with only real answers.
    Dependencies: none.
    Utilities: called by VectorDBRetriever.retrieve(), benchmark.
    """
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
    """
    Description: One query's FAISS result row -> final (doc_id, score) list with the
        project's tie rule (higher score first, then smaller doc ID).
    Inputs: scores, positions, doc_ids, k.
    Outputs: list of (doc_id, score), at most k long.
    Dependencies: drop_missing_hits(), top_k_results().
    Utilities: called by VectorDBRetriever.retrieve(), benchmark.
    """
    ids, kept_scores = drop_missing_hits(scores, positions, doc_ids)
    if not ids:
        return []
    return top_k_results(ids, kept_scores, k)


class VectorDBRetriever(BaseRetriever):
    """
    Description: Retriever that searches a FAISS index of document vectors.
    Inputs: index, doc_ids, encoder (object with encode(list) -> array),
        index_type, model_name, search_params.
    Outputs: retrieve() returns [(doc_id, score), ...].
    Dependencies: encode_queries(), hits_from_search_row(), store functions.
    Utilities: used by the UI, fusion, reranking and evaluation code.
    """

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
    ) -> "VectorDBRetriever":
        """
        Description: Encode the corpus (using the disk cache) and build an index.
        Inputs: corpus, encoder, index_type, model_name, build_params, search_params,
            cache_dir (default VECTOR_INDEX_DIR/embeddings), use_cache.
        Outputs: a ready VectorDBRetriever.
        Dependencies: get_or_create_embeddings(), build_index(), set_search_params().
        Utilities: called by users of the package and tests.
        """
        cache_dir = Path(cache_dir) if cache_dir else Path(VECTOR_INDEX_DIR) / EMBEDDINGS_SUBDIR
        doc_ids, embeddings = get_or_create_embeddings(corpus, encoder, model_name, cache_dir, use_cache=use_cache)
        if not doc_ids:
            return cls(None, [], encoder, index_type, model_name, search_params)
        index = build_index(index_type, embeddings, **(build_params or {}))
        set_search_params(index, index_type, **(search_params or {}))
        return cls(index, doc_ids, encoder, index_type, model_name, search_params)

    @classmethod
    def from_saved(cls, index_dir: Path, encoder: Any) -> "VectorDBRetriever":
        """
        Description: Load an index that was saved earlier with save().
        Inputs: index_dir, encoder (must be the same model used to build it).
        Outputs: a ready VectorDBRetriever.
        Dependencies: load_index().
        Utilities: called by users of the package and tests.
        """
        index, doc_ids, meta = load_index(index_dir)
        return cls(index, doc_ids, encoder, meta["index_type"], meta.get("model_name", ""), meta.get("search_params"))

    # ---------- using it ----------

    def save(self, index_dir: Path) -> Path:
        """
        Description: Save the index and ID map to a folder.
        Inputs: index_dir.
        Outputs: the folder path.
        Dependencies: save_index().
        Utilities: called by users of the package.
        """
        if self.index is None:
            raise ValueError("Nothing to save: the index is empty")
        return save_index(self.index, self.doc_ids, self.index_type, index_dir, self.search_params, self.model_name)

    def set_search_params(self, **search_params: Any) -> None:
        """
        Description: Change nprobe / ef_search without rebuilding.
        Inputs: nprobe and/or ef_search.
        Outputs: nothing; updates the index and self.search_params.
        Dependencies: index_builder.set_search_params().
        Utilities: called by users and the benchmark.
        """
        self.search_params.update(search_params)
        if self.index is not None:
            set_search_params(self.index, self.index_type, **search_params)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """
        Description: Top-k (doc_id, score) pairs for a query, best first.
        Inputs: query text, k (must be > 0).
        Outputs: list of (doc_id, float score).
        Dependencies: encode_queries(), index.search(), hits_from_search_row().
        Utilities: BaseRetriever.retrieve_ids(), evaluation, fusion.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if is_blank(query) or self.index is None or not self.doc_ids:
            return []
        query_vec = encode_queries(self.encoder, [query])
        scores, positions = self.index.search(query_vec, min(k, len(self.doc_ids)))
        return hits_from_search_row(scores[0], positions[0], self.doc_ids, k)
