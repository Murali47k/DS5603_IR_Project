"""
Citation-Aware Boosting Module
Adjusts retrieval scores using document citation counts.
"""

import math
from typing import Dict, List, Tuple


def citation_aware_boost(
    ranked_results: List[Tuple[str, float]],
    citation_counts: Dict[str, int],
    boost_weight: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    Boost document scores based on citation counts.

    Args:
        ranked_results: List of (document_id, retrieval_score) pairs.
        citation_counts: Mapping of document IDs to citation counts.
        boost_weight: Strength of the citation-based boost (>= 0).

    Returns:
        List of (document_id, adjusted_score) pairs,
        sorted by descending adjusted score.
    """

    if boost_weight < 0:
        raise ValueError("boost_weight must be non-negative")

    if not ranked_results:
        return []

    max_citations = max(
        (max(0, count) for count in citation_counts.values()),
        default=0,
    )

    boosted_results = []

    for doc_id, score in ranked_results:
        citations = max(0, citation_counts.get(doc_id, 0))

        if max_citations > 0:
            normalized_citations = (math.log1p(citations) / math.log1p(max_citations))
        else:
            normalized_citations = 0.0

        adjusted_score = score * (1.0 + boost_weight * normalized_citations)

        boosted_results.append((doc_id, adjusted_score))

    return sorted(boosted_results, key=lambda item: item[1], reverse=True)
