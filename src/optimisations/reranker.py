"""
Cross-Encoder Re-ranking Module
Re-ranks candidate scientific documents by query-document relevance.
"""

from typing import Any, Dict, List, Tuple

from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME, RERANK_TOP_K


class CrossEncoderReranker:
    """Re-rank retrieved documents using a Cross-Encoder."""

    def __init__( 
        self, model_name: str = RERANKER_MODEL_NAME,
        top_k: int = RERANK_TOP_K,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.model = CrossEncoder(model_name)
        self.top_k = top_k

    def rerank(
        self, query: str,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> List[Tuple[str, float]]:
        """
        Args:
            query: User's search query.
            candidates: List of (doc_id, document) pairs.
                Each document should contain 'title' and/or 'text'.

        Returns:
            List of (doc_id, relevance_score) sorted by descending score.
        """
        if not query.strip() or not candidates:
            return []

        candidates = candidates[:self.top_k]

        pairs = []
        for doc_id, doc in candidates:
            title = doc.get("title", "")
            text = doc.get("text", "")
            document_text = f"{title}. {text}".strip()
            pairs.append((query, document_text))

        scores = self.model.predict(pairs)

        ranked_results = [(doc_id, float(score)) for (doc_id, _), score in zip(candidates, scores)]

        return sorted(ranked_results, key=lambda item: item[1], reverse=True,)
