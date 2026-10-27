"""
Optional in-memory dense retriever (exact cosine similarity).

This is a lightweight baseline: encode documents once, then score queries
with cosine similarity. Approximate nearest-neighbour indexes (FAISS, etc.)
belong to ``src/vectordb`` and are not implemented here.

A pretrained SentenceTransformer is loaded only when no encoder is supplied.
Unit tests should inject a local encoder so they never download a model.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from config import DEFAULT_TOP_K, DENSE_MODEL_NAME

from src.base_models.base import BaseRetriever
from src.base_models.utils import (
    Corpus,
    corpus_texts,
    is_blank,
    top_k_results,
)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


class DenseRetriever(BaseRetriever):
    """
    Optional dense baseline using pretrained embeddings.

    Args:
        corpus: SciDocs-style mapping of document ID to document.
        model_name: Hugging Face / sentence-transformers checkpoint.
            Used only when ``encoder`` is not provided.
        encoder: Optional object with ``encode(texts) -> ndarray``. Inject this
            in tests to avoid network access and model downloads.
    """

    def __init__(
        self,
        corpus: Corpus,
        model_name: str = DENSE_MODEL_NAME,
        encoder: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.doc_ids, texts = corpus_texts(corpus)
        self.encoder = encoder
        self.doc_embeddings: Optional[np.ndarray] = None

        if encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)

        if self.doc_ids:
            embeddings = self.encoder.encode(texts)
            self.doc_embeddings = _l2_normalize(embeddings)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if is_blank(query) or not self.doc_ids or self.doc_embeddings is None:
            return []

        query_vec = _l2_normalize(self.encoder.encode([query]))
        scores = (self.doc_embeddings @ query_vec.T).reshape(-1)
        return top_k_results(self.doc_ids, scores, k)
