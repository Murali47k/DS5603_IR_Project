"""BM25 lexical retriever built with rank-bm25."""

import re
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from config import BM25_B, BM25_K1, DEFAULT_TOP_K

from src.base_models.base import BaseRetriever
from src.base_models.utils import (
    Corpus,
    as_float_scores,
    corpus_texts,
    is_blank,
    top_k_results,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer shared by documents and queries."""
    return _TOKEN_RE.findall(str(text).lower())


class BM25Retriever(BaseRetriever):
    """
    BM25Okapi retriever with a one-time document index.

    Parameters ``k1`` and ``b`` default to the project values in ``config.py``.
    Documents are tokenised once during initialisation.
    """

    def __init__(
        self,
        corpus: Corpus,
        k1: float = BM25_K1,
        b: float = BM25_B,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.doc_ids, texts = corpus_texts(corpus)
        self._tokenized_corpus: List[List[str]] = [tokenize(text) for text in texts]
        self._bm25: Optional[BM25Okapi] = None

        if self.doc_ids:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus,
                k1=k1,
                b=b,
                epsilon=epsilon,
            )

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if is_blank(query) or not self.doc_ids or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scores = as_float_scores(raw_scores)
        return top_k_results(self.doc_ids, scores, k)
