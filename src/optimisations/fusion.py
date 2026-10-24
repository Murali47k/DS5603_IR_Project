"""
Hybrid Search Fusion Module
Combines ranked results from multiple retrieval systems.
"""

from typing import Dict, List, Tuple
from config import RRF_K, HYBRID_BM25_WEIGHT, HYBRID_DENSE_WEIGHT


def reciprocal_rank_fusion(
    ranked_results: List[List[str]], k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Combine ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_results: List of ranked document-ID lists.
        k: RRF smoothing constant.

    Returns:
        List of (document_id, fused_score), sorted by descending score.
    """
    scores: Dict[str, float] = {}

    for result_list in ranked_results:
        for rank, doc_id in enumerate(result_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def weighted_score_fusion(
    bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]],
    bm25_weight: float = HYBRID_BM25_WEIGHT, dense_weight: float = HYBRID_DENSE_WEIGHT,
) -> List[Tuple[str, float]]:
    """
    Combine BM25 and dense retrieval scores using weighted addition.

    Args:
        bm25_results: List of (document_id, score) pairs.
        dense_results: List of (document_id, score) pairs.
        bm25_weight: Weight assigned to BM25 scores.
        dense_weight: Weight assigned to dense scores.

    Returns:
        List of (document_id, fused_score), sorted by descending score.
    """
    scores: Dict[str, float] = {}

    for doc_id, score in bm25_results:
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight * score

    for doc_id, score in dense_results:
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * score

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)