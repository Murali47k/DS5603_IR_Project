"""TF-IDF lexical retriever built with scikit-learn."""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import DEFAULT_TOP_K

from src.base_models.base import BaseRetriever
from src.base_models.utils import (
    Corpus,
    as_float_scores,
    corpus_texts,
    is_blank,
    top_k_results,
)


class TfidfRetriever(BaseRetriever):
    """
    Sparse TF-IDF retriever with a one-time document index.

    The document-term matrix is fitted in ``__init__`` and reused for every
    query. Query vectors are transformed with the same vocabulary; the corpus
    is never re-vectorised at query time.
    """

    def __init__(
        self,
        corpus: Corpus,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.doc_ids, texts = corpus_texts(corpus)
        self.vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            norm="l2",
            min_df=1,
        )
        self.doc_matrix = None

        if not self.doc_ids:
            return

        try:
            self.doc_matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (all documents blank). Keep IDs, no matrix.
            self.doc_matrix = None

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        if is_blank(query) or not self.doc_ids:
            return []

        if self.doc_matrix is None or self.doc_matrix.shape[1] == 0:
            scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        else:
            query_vec = self.vectorizer.transform([query])
            scores = as_float_scores(query_vec @ self.doc_matrix.T)

        return top_k_results(self.doc_ids, scores, k)
