"""Hand-checkable overlap and ranking tests. No SciDocs download."""

from __future__ import annotations

import pytest

from src.ui.comparison import (
    COMPARE_METHOD_ORDER,
    build_insights,
    correlation_matrix,
    ids_from_results,
    intersection_of,
    jaccard,
    ordered_unique_ids,
    overlap_matrix,
    pairwise_rank_correlation,
    rank_map_from_results,
    shared_by_all,
    summarize_overlap,
    unique_to_each,
)
from src.ui.metadata import PaperRecord
from src.ui.search_service import RankedPaper, SearchResponse


def _paper(doc_id: str, title: str = "") -> PaperRecord:
    return PaperRecord(doc_id=doc_id, title=title or doc_id, abstract="")


def _hits(ids: list[str]) -> list[RankedPaper]:
    return [
        RankedPaper(rank=i, paper=_paper(doc_id, title="Shared Title"), final_score=1.0 / i)
        for i, doc_id in enumerate(ids, start=1)
    ]


def test_ordered_unique_ids_drops_duplicates_and_preserves_first_rank():
    assert ordered_unique_ids(["a", "b", "a", "c", "b"], k=10) == ["a", "b", "c"]
    assert ordered_unique_ids(["a", "b", "c"], k=2) == ["a", "b"]


def test_jaccard_is_intersection_over_union():
    # |{1,2,3} ∩ {2,3,4}| / |{1,2,3,4}| = 2/4
    assert jaccard(["d1", "d2", "d3"], ["d2", "d3", "d4"]) == 0.5
    assert jaccard([], []) == 1.0
    assert jaccard(["a"], []) == 0.0


def test_overlap_uses_document_ids_not_titles():
    left = _hits(["paper-a", "paper-b"])
    right = _hits(["paper-c"])
    # Identical titles, different IDs → no overlap.
    assert ids_from_results(left, k=10) == ["paper-a", "paper-b"]
    assert jaccard(ids_from_results(left, 10), ids_from_results(right, 10)) == 0.0


def test_seven_by_seven_overlap_matrix_and_uniques():
    method_ids = {
        "bm25": ["a", "b", "c"],
        "tfidf": ["a", "b", "c"],
        "dense": ["a", "x"],
        "citation_hybrid": ["a", "b"],
        "cross_encoder": ["a", "z"],
        "rrf": ["a", "b", "y"],
        "weighted": ["a", "b"],
    }
    matrix = overlap_matrix(method_ids, COMPARE_METHOD_ORDER)
    assert matrix.shape == (7, 7)
    assert matrix[0, 1] == 1.0  # BM25 vs TF-IDF, identical sets
    assert abs(matrix[0, 2] - (1 / 4)) < 1e-9  # {a,b,c} vs {a,x}
    assert shared_by_all(method_ids, COMPARE_METHOD_ORDER) == ["a"]
    unique = unique_to_each(method_ids, COMPARE_METHOD_ORDER)
    assert unique["dense"] == ["x"]
    assert unique["cross_encoder"] == ["z"]
    assert unique["bm25"] == []
    assert intersection_of(method_ids, ["bm25", "weighted"]) == ["a", "b"]


def test_spearman_excludes_missing_documents_instead_of_zero_fill():
    ranks_a = {"d1": 1, "d2": 2, "d3": 3}
    ranks_b = {"d1": 1, "d2": 2}  # d3 missing from B
    dropped = pairwise_rank_correlation(ranks_a, ranks_b, kind="spearman")
    assert dropped.n_common == 2
    assert dropped.n_only_a == 1
    assert dropped.n_only_b == 0
    assert dropped.coefficient == pytest.approx(1.0)

    # Filling the missing rank with 0 would change the correlation.
    filled_a = [1, 2, 3]
    filled_b = [1, 2, 0]
    from scipy.stats import spearmanr

    filled_rho, _ = spearmanr(filled_a, filled_b)
    assert filled_rho != dropped.coefficient


def test_correlation_unavailable_with_fewer_than_two_shared_ids():
    result = pairwise_rank_correlation({"a": 1, "b": 2}, {"c": 1}, kind="kendall")
    assert result.coefficient is None
    assert result.n_common == 0
    assert "shared documents" in result.reason


def test_summarize_overlap_and_insights_from_search_responses():
    compared = {
        key: SearchResponse(query="q", method=key, results=_hits(["shared", key]))
        for key in COMPARE_METHOD_ORDER
    }
    compared["bm25"] = SearchResponse(query="q", method="bm25", results=_hits(["shared", "only-bm25"]))
    summary = summarize_overlap(compared, k=10)
    assert "shared" in summary.shared_by_all
    assert "only-bm25" in summary.unique_ids["bm25"]
    rank_maps = {key: rank_map_from_results(compared[key].results, 10) for key in COMPARE_METHOD_ORDER}
    insights = build_insights(summary, rank_maps, COMPARE_METHOD_ORDER, {k: k for k in COMPARE_METHOD_ORDER})
    assert insights.notes
    assert insights.highest_overlap is not None
    coefs, counts = correlation_matrix(rank_maps, COMPARE_METHOD_ORDER, kind="spearman")
    assert coefs.shape == (7, 7)
    assert counts[0, 0] >= 1


def test_rank_mover_segments_breaks_on_unranked_intermediate_methods():
    from src.ui.comparison import RankMover, rank_mover_segments

    # COMPARE_METHOD_ORDER is:
    # 0: bm25, 1: tfidf, 2: dense, 3: citation_hybrid, 4: cross_encoder, 5: rrf, 6: weighted
    # Paper present in bm25 (0) and dense (2), missing in tfidf (1)
    mover = RankMover(doc_id="doc-gap", ranks={"bm25": 1, "dense": 3}, spread=2)
    segments = rank_mover_segments(mover, COMPARE_METHOD_ORDER)
    # Must NOT connect index 0 to 2 across index 1
    assert len(segments) == 2
    assert segments[0] == ([0], [1])
    assert segments[1] == ([2], [3])

    # Paper present in dense (2), citation_hybrid (3), cross_encoder (4) consecutively
    mover_contig = RankMover(doc_id="doc-contig", ranks={"dense": 5, "citation_hybrid": 2, "cross_encoder": 1}, spread=4)
    segments_contig = rank_mover_segments(mover_contig, COMPARE_METHOD_ORDER)
    assert len(segments_contig) == 1
    assert segments_contig[0] == ([2, 3, 4], [5, 2, 1])


def test_sharing_distribution_and_total_unique_docs():
    from src.ui.comparison import summarize_overlap

    compared = {
        "bm25": SearchResponse(query="q", method="bm25", results=_hits(["shared_all", "shared_pair", "only_bm25"])),
        "tfidf": SearchResponse(query="q", method="tfidf", results=_hits(["shared_all", "shared_pair", "only_tfidf"])),
        "dense": SearchResponse(query="q", method="dense", results=_hits(["shared_all", "only_dense"])),
        "citation_hybrid": SearchResponse(query="q", method="citation_hybrid", results=_hits(["shared_all"])),
        "cross_encoder": SearchResponse(query="q", method="cross_encoder", results=_hits(["shared_all"])),
        "rrf": SearchResponse(query="q", method="rrf", results=_hits(["shared_all"])),
        "weighted": SearchResponse(query="q", method="weighted", results=_hits(["shared_all"])),
    }
    summary = summarize_overlap(compared, k=10)
    assert summary.total_unique_docs == 5
    assert summary.shared_by_all == ["shared_all"]
    assert summary.sharing_distribution["All 7 methods"] == 1
    assert summary.sharing_distribution["2–3 methods"] == 1  # "shared_pair" appears in bm25 & tfidf (2 methods)
    assert summary.sharing_distribution["1 method only"] == 3  # only_bm25, only_tfidf, only_dense


def test_method_specs_complete_and_valid():
    from src.ui.comparison import METHOD_SPECS

    assert len(METHOD_SPECS) == 7
    for method in COMPARE_METHOD_ORDER:
        assert method in METHOD_SPECS
        spec = METHOD_SPECS[method]
        assert spec["name"]
        assert spec["category"]
        assert spec["description"]
        assert spec["formula"]
        assert spec["parameters"]
