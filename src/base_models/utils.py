"""Shared helpers for baseline retrievers."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

Corpus = Mapping[str, Union[str, Mapping[str, Any]]]


def document_text(doc: Union[str, Mapping[str, Any]]) -> str:
    """Flatten a SciDocs-style document (or raw string) into searchable text."""
    if isinstance(doc, str):
        return doc
    title = str(doc.get("title") or "")
    text = str(doc.get("text") or "")
    return f"{title} {text}".strip()


def corpus_texts(corpus: Corpus) -> Tuple[List[str], List[str]]:
    """
    Return (doc_ids, texts) in a stable order.

    Document IDs are sorted so index construction is deterministic even if
    the input mapping does not preserve insertion order.
    """
    doc_ids = sorted(corpus.keys(), key=lambda doc_id: str(doc_id))
    texts = [document_text(corpus[doc_id]) for doc_id in doc_ids]
    return doc_ids, texts


def is_blank(query: str) -> bool:
    return not str(query).strip()


def top_k_results(
    doc_ids: Sequence[str],
    scores: Sequence[float],
    k: int,
) -> List[Tuple[str, float]]:
    """
    Rank documents by descending score, breaking ties by document ID.

    Returns at most k (doc_id, score) pairs. Original IDs are preserved.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not doc_ids:
        return []

    n = len(doc_ids)
    k = min(k, n)
    order = sorted(
        range(n),
        key=lambda i: (-float(scores[i]), doc_ids[i]),
    )
    return [(doc_ids[i], float(scores[i])) for i in order[:k]]


def ranked_ids(results: List[Tuple[str, float]]) -> List[str]:
    """Extract the ranked ID list expected by reciprocal_rank_fusion."""
    return [doc_id for doc_id, _ in results]


def as_float_scores(scores: Any) -> np.ndarray:
    if hasattr(scores, "toarray"):
        scores = scores.toarray()
    return np.asarray(scores, dtype=np.float64).reshape(-1)
