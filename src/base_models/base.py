"""Common interface for baseline retrieval models."""

from abc import ABC, abstractmethod
from typing import List, Tuple

from config import DEFAULT_TOP_K

from src.base_models.utils import ranked_ids


class BaseRetriever(ABC):
    """
    Baseline retriever interface.

    Corpus format (same as `src.data_loader.load_scidocs`):
        doc_id -> {"title": str, "text": str, ...}
        or doc_id -> raw document string

    Retrieval results:
        list of (document_id, score), sorted by descending score.
        Ties are broken by ascending document ID.
        These tuples are compatible with weighted_score_fusion and
        citation_aware_boost. Pass just the IDs to reciprocal_rank_fusion.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Tuple[str, float]]:
        """Return the top-k (document_id, score) pairs for a query."""

    def retrieve_ids(self, query: str, k: int = DEFAULT_TOP_K) -> List[str]:
        """Return only ranked document IDs (for Reciprocal Rank Fusion)."""
        return ranked_ids(self.retrieve(query, k=k))
