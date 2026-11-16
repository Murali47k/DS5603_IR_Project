"""
Query-level overlap and ranking analysis for comparison mode.

Set operations and correlations use document IDs. Missing ranks are excluded;
they are never filled with zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

COMPARE_METHOD_ORDER: Tuple[str, ...] = (
    "bm25",
    "tfidf",
    "dense",
    "citation_hybrid",
    "cross_encoder",
    "rrf",
    "weighted",
)


@dataclass
class CorrelationResult:
    coefficient: Optional[float]
    n_common: int
    n_only_a: int
    n_only_b: int
    reason: str = ""


@dataclass
class OverlapSummary:
    k: int
    method_ids: Dict[str, List[str]]
    jaccard: np.ndarray
    shared_by_all: List[str]
    unique_ids: Dict[str, List[str]]
    unique_counts: Dict[str, int]
    total_unique_docs: int = 0
    sharing_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class RankMover:
    doc_id: str
    ranks: Dict[str, int]
    spread: int


@dataclass
class ComparisonInsights:
    highest_overlap: Optional[Tuple[str, str, float]] = None
    lowest_overlap: Optional[Tuple[str, str, float]] = None
    most_unique_method: Optional[Tuple[str, int]] = None
    largest_rank_movers: List[RankMover] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def ordered_unique_ids(doc_ids: Sequence[str], k: int) -> List[str]:
    """Preserve first-seen rank order and drop duplicate IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    seen = set()
    ordered: List[str] = []
    for doc_id in doc_ids:
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)
        if len(ordered) >= k:
            break
    return ordered


def ids_from_results(results: Sequence, k: int) -> List[str]:
    """Extract unique document IDs from RankedPaper-like objects."""
    raw = []
    for hit in results:
        paper = getattr(hit, "paper", None)
        doc_id = getattr(paper, "doc_id", None) if paper is not None else None
        if doc_id is None and isinstance(hit, (tuple, list)) and hit:
            doc_id = hit[0]
        if doc_id is not None:
            raw.append(str(doc_id))
    return ordered_unique_ids(raw, k)


def rank_map_from_results(results: Sequence, k: int) -> Dict[str, int]:
    """Map document ID -> 1-based rank, first occurrence only."""
    ids = ids_from_results(results, k)
    return {doc_id: rank for rank, doc_id in enumerate(ids, start=1)}


def jaccard(ids_a: Iterable[str], ids_b: Iterable[str]) -> float:
    set_a = set(ids_a)
    set_b = set(ids_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def method_id_lists(responses: Mapping[str, object], k: int, order: Sequence[str] = COMPARE_METHOD_ORDER) -> Dict[str, List[str]]:
    lists: Dict[str, List[str]] = {}
    for key in order:
        response = responses.get(key)
        results = getattr(response, "results", None) if response is not None else None
        lists[key] = ids_from_results(results or [], k)
    return lists


def overlap_matrix(method_ids: Mapping[str, Sequence[str]], order: Sequence[str]) -> np.ndarray:
    n = len(order)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, key_i in enumerate(order):
        for j, key_j in enumerate(order):
            matrix[i, j] = jaccard(method_ids.get(key_i, []), method_ids.get(key_j, []))
    return matrix


def shared_by_all(method_ids: Mapping[str, Sequence[str]], order: Sequence[str]) -> List[str]:
    sets = [set(method_ids.get(key, [])) for key in order]
    if not sets or any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def unique_to_each(method_ids: Mapping[str, Sequence[str]], order: Sequence[str]) -> Dict[str, List[str]]:
    sets = {key: set(method_ids.get(key, [])) for key in order}
    unique: Dict[str, List[str]] = {}
    for key in order:
        others = set()
        for other in order:
            if other != key:
                others |= sets[other]
        unique[key] = sorted(sets[key] - others)
    return unique


def intersection_of(method_ids: Mapping[str, Sequence[str]], keys: Sequence[str]) -> List[str]:
    if not keys:
        return []
    sets = [set(method_ids.get(key, [])) for key in keys]
    if any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def summarize_overlap(
    responses: Mapping[str, object],
    k: int,
    order: Sequence[str] = COMPARE_METHOD_ORDER,
) -> OverlapSummary:
    method_ids = method_id_lists(responses, k, order)
    unique_ids = unique_to_each(method_ids, order)
    all_ids = set()
    for ids in method_ids.values():
        all_ids.update(ids)
    doc_freq = {
        doc_id: sum(1 for ids in method_ids.values() if doc_id in ids)
        for doc_id in all_ids
    }
    n_order = len(order)
    distribution = {
        "1 method only": sum(1 for f in doc_freq.values() if f == 1),
        "2–3 methods": sum(1 for f in doc_freq.values() if 2 <= f <= 3),
        "4–6 methods": sum(1 for f in doc_freq.values() if 4 <= f < n_order),
        "All 7 methods": sum(1 for f in doc_freq.values() if f == n_order),
    }
    return OverlapSummary(
        k=k,
        method_ids=method_ids,
        jaccard=overlap_matrix(method_ids, order),
        shared_by_all=shared_by_all(method_ids, order),
        unique_ids=unique_ids,
        unique_counts={key: len(ids) for key, ids in unique_ids.items()},
        total_unique_docs=len(all_ids),
        sharing_distribution=distribution,
    )


def pairwise_rank_correlation(
    ranks_a: Mapping[str, int],
    ranks_b: Mapping[str, int],
    kind: str = "spearman",
) -> CorrelationResult:
    """Correlate ranks on the intersection of retrieved IDs only.

    Documents missing from either list are counted in ``n_only_a`` / ``n_only_b``
    and are not assigned a sentinel rank of 0.
    """
    set_a = set(ranks_a)
    set_b = set(ranks_b)
    common = sorted(set_a & set_b)
    only_a = len(set_a - set_b)
    only_b = len(set_b - set_a)
    if len(common) < 2:
        return CorrelationResult(
            coefficient=None,
            n_common=len(common),
            n_only_a=only_a,
            n_only_b=only_b,
            reason="Need at least two shared documents to compute rank correlation.",
        )

    x = [int(ranks_a[doc_id]) for doc_id in common]
    y = [int(ranks_b[doc_id]) for doc_id in common]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return CorrelationResult(
            coefficient=None,
            n_common=len(common),
            n_only_a=only_a,
            n_only_b=only_b,
            reason="Rank correlation is undefined when one method assigns the same rank to every shared document.",
        )

    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError as exc:  # pragma: no cover
        return CorrelationResult(
            None, len(common), only_a, only_b, f"scipy is unavailable: {exc}"
        )

    if kind == "kendall":
        coef, _p = kendalltau(x, y)
    elif kind == "spearman":
        coef, _p = spearmanr(x, y)
    else:
        raise ValueError("kind must be 'spearman' or 'kendall'")

    if coef is None or (isinstance(coef, float) and math.isnan(float(coef))):
        return CorrelationResult(
            None,
            len(common),
            only_a,
            only_b,
            "Correlation was undefined for these ranks.",
        )
    return CorrelationResult(float(coef), len(common), only_a, only_b)


def correlation_matrix(
    rank_maps: Mapping[str, Mapping[str, int]],
    order: Sequence[str],
    kind: str = "spearman",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coefficients, n_common) with NaN where correlation is unavailable."""
    n = len(order)
    coefs = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int32)
    for i, key_i in enumerate(order):
        for j, key_j in enumerate(order):
            result = pairwise_rank_correlation(rank_maps.get(key_i, {}), rank_maps.get(key_j, {}), kind=kind)
            counts[i, j] = result.n_common
            if result.coefficient is not None:
                coefs[i, j] = result.coefficient
            elif key_i == key_j and result.n_common >= 1:
                coefs[i, j] = 1.0
    return coefs, counts


def rank_movers(
    rank_maps: Mapping[str, Mapping[str, int]],
    order: Sequence[str],
    min_methods: int = 2,
    limit: int = 12,
) -> List[RankMover]:
    present: Dict[str, Dict[str, int]] = {}
    for method in order:
        for doc_id, rank in rank_maps.get(method, {}).items():
            present.setdefault(doc_id, {})[method] = int(rank)
    movers = []
    for doc_id, ranks in present.items():
        if len(ranks) < min_methods:
            continue
        values = list(ranks.values())
        movers.append(RankMover(doc_id=doc_id, ranks=ranks, spread=max(values) - min(values)))
    movers.sort(key=lambda item: (-item.spread, item.doc_id))
    return movers[:limit]


def rank_mover_segments(
    mover: RankMover,
    order: Sequence[str] = COMPARE_METHOD_ORDER,
) -> List[Tuple[List[int], List[int]]]:
    """Break a paper's ranks into contiguous method index segments.

    If a paper appears in method 0 (BM25) and method 2 (Dense) but not method 1 (TF-IDF),
    a straight line must not connect (0, r0) and (2, r2) because that falsely implies
    it had an intermediate rank at method 1.
    This returns continuous slices [([0], [r0]), ([2], [r2])] so only consecutively
    ranked methods are joined by lines.
    """
    segments: List[Tuple[List[int], List[int]]] = []
    current_xs: List[int] = []
    current_ys: List[int] = []

    for i, method in enumerate(order):
        if method in mover.ranks:
            rank = mover.ranks[method]
            if current_xs and current_xs[-1] != i - 1:
                segments.append((current_xs, current_ys))
                current_xs = [i]
                current_ys = [rank]
            else:
                current_xs.append(i)
                current_ys.append(rank)
        else:
            if current_xs:
                segments.append((current_xs, current_ys))
                current_xs = []
                current_ys = []

    if current_xs:
        segments.append((current_xs, current_ys))

    return segments


def off_diagonal_pairs(matrix: np.ndarray, order: Sequence[str]) -> List[Tuple[str, str, float]]:
    pairs = []
    for i, key_i in enumerate(order):
        for j, key_j in enumerate(order):
            if j <= i:
                continue
            pairs.append((key_i, key_j, float(matrix[i, j])))
    return pairs


def build_insights(
    summary: OverlapSummary,
    rank_maps: Mapping[str, Mapping[str, int]],
    order: Sequence[str] = COMPARE_METHOD_ORDER,
    labels: Optional[Mapping[str, str]] = None,
) -> ComparisonInsights:
    labels = labels or {key: key for key in order}
    notes: List[str] = []
    pairs = off_diagonal_pairs(summary.jaccard, order)
    highest = max(pairs, key=lambda item: item[2]) if pairs else None
    lowest = min(pairs, key=lambda item: item[2]) if pairs else None
    unique_counts = [(key, summary.unique_counts.get(key, 0)) for key in order]
    most_unique = max(unique_counts, key=lambda item: item[1]) if unique_counts else None
    movers = rank_movers(rank_maps, order)

    if highest:
        notes.append(
            f"Highest overlap: {labels.get(highest[0], highest[0])} and "
            f"{labels.get(highest[1], highest[1])} share {highest[2] * 100:.0f}% of papers (Jaccard)."
        )
    if lowest:
        notes.append(
            f"Lowest overlap: {labels.get(lowest[0], lowest[0])} and "
            f"{labels.get(lowest[1], lowest[1])} share {lowest[2] * 100:.0f}% of papers (Jaccard)."
        )
    notes.append(
        f"{len(summary.shared_by_all)} paper(s) appear in every method's top-{summary.k}."
    )
    if most_unique and most_unique[1] > 0:
        notes.append(
            f"{labels.get(most_unique[0], most_unique[0])} has the most unique papers "
            f"({most_unique[1]} not retrieved by any other method)."
        )
    only_one = []
    for method, ids in summary.unique_ids.items():
        for doc_id in ids:
            only_one.append((method, doc_id))
    if only_one:
        preview = ", ".join(f"{doc_id} ({labels.get(method, method)})" for method, doc_id in only_one[:5])
        extra = "" if len(only_one) <= 5 else f" and {len(only_one) - 5} more"
        notes.append(f"Papers retrieved by only one method include {preview}{extra}.")
    if movers:
        top = movers[0]
        notes.append(
            f"Largest rank change among papers retrieved by multiple methods: "
            f"{top.doc_id} (spread {top.spread} rank positions)."
        )

    return ComparisonInsights(
        highest_overlap=highest,
        lowest_overlap=lowest,
        most_unique_method=most_unique,
        largest_rank_movers=movers,
        notes=notes,
    )


METHOD_SPECS = {
    "bm25": {
        "name": "BM25",
        "category": "Lexical Baseline",
        "formula": r"\text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}",
        "description": "Probabilistic lexical baseline using BM25Okapi over the SciDocs corpus. Uses term frequency saturation and document length normalization to score exact word matches.",
        "parameters": "k₁ = 1.5, b = 0.75, ε = 0.25",
    },
    "tfidf": {
        "name": "TF-IDF",
        "category": "Lexical Baseline",
        "formula": r"\text{score}(D, Q) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}",
        "description": "Vector-space lexical baseline using scikit-learn TfidfVectorizer. Computes L2-normalized cosine similarity between query and document term vectors.",
        "parameters": "L2 normalized cosine, fitted once across corpus",
    },
    "dense": {
        "name": "Dense Retrieval (FAISS)",
        "category": "Dense Bi-Encoder",
        "formula": r"\text{score}(D, Q) = \langle \mathbf{e}(Q), \mathbf{e}(D) \rangle",
        "description": "Dense semantic retriever with sentence-transformers/all-MiniLM-L6-v2 embeddings indexed in a FAISS HNSW graph. Performs Approximate Nearest Neighbour (ANN) search capturing conceptual similarity without requiring exact keyword matches.",
        "parameters": "all-MiniLM-L6-v2 (384-dim), FAISS HNSW graph index",
    },
    "citation_hybrid": {
        "name": "Citation-Boosted Retrieval",
        "category": "Hybrid Optimization",
        "formula": r"S_{\text{boosted}} = S_{\text{weighted}} \cdot \left(1 + w_{\text{boost}} \cdot \frac{\log(1 + c)}{\log(1 + c_{\max})}\right)",
        "description": "Our highest-performing method (nDCG@10 = 0.22877). Re-scores normalized BM25 + Dense candidates using a leak-free logarithmic citation boost factor based on document citation counts.",
        "parameters": "w_boost = 0.4, w_bm25 = 0.4, w_dense = 0.6, candidate_k = 50",
    },
    "cross_encoder": {
        "name": "Cross-Encoder Neural Reranking",
        "category": "Neural Reranker",
        "formula": r"S_{\text{CE}}(Q, D) = \text{CrossEncoder}([Q; D])",
        "description": "Re-ranks top-50 candidates from weighted hybrid search using ms-marco-MiniLM-L-6-v2. Evaluates full cross-attention between query and document text on CPU.",
        "parameters": "cross-encoder/ms-marco-MiniLM-L-6-v2, candidate_k = 50, CPU inference",
    },
    "rrf": {
        "name": "Reciprocal Rank Fusion (RRF)",
        "category": "Rank Fusion",
        "formula": r"S_{\text{RRF}}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k_{\text{rrf}} + r_m(d)}",
        "description": "Rank-based fusion combining ranked candidate lists from BM25 and Dense without relying on raw score magnitudes. Robust to score distribution differences.",
        "parameters": "k_rrf = 60, candidate_k = 50",
    },
    "weighted": {
        "name": "Weighted Score Fusion",
        "category": "Hybrid Fusion",
        "formula": r"S_{\text{weighted}} = w_{\text{bm25}} \cdot \tilde{S}_{\text{bm25}} + w_{\text{dense}} \cdot \tilde{S}_{\text{dense}}",
        "description": "Fuses BM25 and Dense scores with linear weights after per-query Min-Max normalization to [0, 1]. Combines exact keyword matching with dense conceptual similarity.",
        "parameters": "w_bm25 = 0.4, w_dense = 0.6, candidate_k = 50, Min-Max normalized",
    },
}
