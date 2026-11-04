"""Synthetic tests for the SciDocs search UI pipeline.

Does not download SciDocs, FAISS indexes, or pretrained models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.base_models.base import BaseRetriever
from src.ui.metadata import (
    PaperRecord,
    load_paper_catalog,
    parse_jsonl_record,
    year_bounds,
)
from src.ui.search_service import SearchEngine, build_search_engine
from src.ui.settings import UISettings


class FakeRetriever(BaseRetriever):
    def __init__(self, hits: List[Tuple[str, float]], doc_ids: List[str] | None = None):
        self.hits = list(hits)
        self.doc_ids = list(doc_ids or [doc_id for doc_id, _ in hits])
        self.index = None
        self.encoder = None

    def retrieve(self, query: str, k: int = 10):
        if not str(query).strip():
            return []
        return list(self.hits[:k])


class FakeIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int):
        scores = self.vectors @ query.reshape(-1)
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.reshape(1, -1)

    def reconstruct(self, position: int):
        return self.vectors[int(position)]


def _settings(**overrides) -> UISettings:
    values = dict(
        dense_model="fake",
        dense_fallback="fake",
        k=10,
        candidate_k=10,
        bm25_weight=0.4,
        dense_weight=0.6,
        normalize_scores=True,
        citation_boost_weight=0.4,
        index_type="hnsw",
        data_dir=".",
        vector_index_dir=".",
    )
    values.update(overrides)
    return UISettings(**values)


def _catalog() -> dict[str, PaperRecord]:
    return {
        "old-low": PaperRecord(
            "old-low", "Old Low Cite", "abstract one",
            authors=["Ada"], year=2001, citation_count=1, cited_by=["x"],
        ),
        "new-high": PaperRecord(
            "new-high", "New High Cite", "abstract two",
            authors=["Grace"], year=2020, citation_count=100,
            cited_by=["a"] * 100, references=["old-low"],
        ),
        "mid": PaperRecord(
            "mid", "Mid Paper", "abstract three",
            authors=["Alan"], year=2010, citation_count=10, cited_by=["z"] * 10,
        ),
    }


def _engine(dense: FakeRetriever | None = None) -> SearchEngine:
    catalog = _catalog()
    corpus = {
        doc_id: {"title": rec.title, "text": rec.abstract}
        for doc_id, rec in catalog.items()
    }
    bm25 = FakeRetriever([("old-low", 0.90), ("new-high", 0.85), ("mid", 0.10)])
    if dense is None:
        dense = FakeRetriever([("old-low", 0.90), ("new-high", 0.85), ("mid", 0.10)])
    return SearchEngine(corpus, catalog, bm25, dense, _settings())


def test_parse_jsonl_record_reads_scidocs_metadata():
    record = parse_jsonl_record(
        {
            "_id": "paper-1",
            "title": "Graph Nets",
            "text": "A paper about graphs.",
            "metadata": {
                "authors": ["1", "2"],
                "year": 2018,
                "cited_by": ["a", "b", "c"],
                "references": ["r1"],
            },
        }
    )
    assert record is not None
    assert record.doc_id == "paper-1"
    assert record.year == 2018
    assert record.citation_count == 3
    assert record.references == ["r1"]


def test_load_paper_catalog_uses_jsonl_cited_by_not_qrels(tmp_path: Path):
    corpus = {"doc-a": {"title": "Cats", "text": "yarn"}}
    scidocs = tmp_path / "scidocs"
    scidocs.mkdir()
    (scidocs / "corpus.jsonl").write_text(
        json.dumps(
            {
                "_id": "doc-a",
                "title": "Cats",
                "text": "yarn",
                "metadata": {"year": 2011, "authors": ["Ada"], "cited_by": ["c1", "c2"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = load_paper_catalog(corpus, data_dir=tmp_path)
    assert catalog["doc-a"].year == 2011
    assert catalog["doc-a"].authors == ["Ada"]
    assert catalog["doc-a"].citation_count == 2


def test_year_bounds_avoids_equal_slider_endpoints():
    catalog = {"a": PaperRecord("a", "A", "", year=2015)}
    low, high = year_bounds(catalog)
    assert high > low


def test_search_returns_hybrid_results_with_score_breakdown():
    engine = _engine()
    response = engine.search("graphs", k=3, boost_weight=0.4)
    assert response.method == "citation_hybrid"
    assert [hit.paper.doc_id for hit in response.results]
    top = response.results[0]
    assert top.bm25_score is not None
    assert top.dense_score is not None
    assert top.fused_score is not None
    assert top.citation_multiplier is not None
    assert top.paper.authors


def test_citation_boost_can_promote_highly_cited_paper():
    engine = _engine()
    unboosted = engine.search("graphs", k=3, boost_weight=0.0)
    boosted = engine.search("graphs", k=3, boost_weight=1.0)
    assert unboosted.results[0].paper.doc_id == "old-low"
    assert boosted.results[0].paper.doc_id == "new-high"


def test_year_and_citation_filters():
    engine = _engine()
    filtered = engine.search(
        "graphs", k=10, boost_weight=0.0, year_min=2015, year_max=2025, min_citations=50
    )
    ids = [hit.paper.doc_id for hit in filtered.results]
    assert ids == ["new-high"]


def test_empty_query_returns_no_results():
    engine = _engine()
    response = engine.search("   ")
    assert response.results == []


def test_compare_keeps_hybrid_when_dense_is_missing():
    engine = _engine()
    engine.dense = None
    engine.dense_error = "no faiss"
    compared = engine.compare("graphs", k=2, boost_weight=0.4)
    assert compared["dense"].results == []
    assert compared["dense"].error
    assert compared["citation_hybrid"].results
    assert compared["bm25"].results


def test_find_similar_excludes_query_document():
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dense = FakeRetriever([])
    dense.doc_ids = ["old-low", "new-high", "mid"]
    dense.index = FakeIndex(vectors)
    engine = _engine(dense=dense)
    neighbours = engine.find_similar("old-low", k=2)
    ids = [hit.paper.doc_id for hit in neighbours]
    assert "old-low" not in ids
    assert ids[0] == "new-high"


def test_build_search_engine_without_dataset():
    catalog = _catalog()
    corpus = {doc_id: {"title": rec.title, "text": rec.abstract} for doc_id, rec in catalog.items()}
    engine = build_search_engine(
        settings=_settings(),
        corpus=corpus,
        catalog=catalog,
        bm25=FakeRetriever([("mid", 1.0)]),
        dense=FakeRetriever([("mid", 1.0)]),
        load_dataset=False,
    )
    results = engine.search("query", k=1).results
    assert results[0].paper.doc_id == "mid"


if __name__ == "__main__":
    import tempfile
    from pathlib import Path as _Path

    test_parse_jsonl_record_reads_scidocs_metadata()
    with tempfile.TemporaryDirectory() as tmp:
        test_load_paper_catalog_uses_jsonl_cited_by_not_qrels(_Path(tmp))
    test_year_bounds_avoids_equal_slider_endpoints()
    test_search_returns_hybrid_results_with_score_breakdown()
    test_citation_boost_can_promote_highly_cited_paper()
    test_year_and_citation_filters()
    test_empty_query_returns_no_results()
    test_compare_keeps_hybrid_when_dense_is_missing()
    test_find_similar_excludes_query_document()
    test_build_search_engine_without_dataset()
    print("UI tests passed!")
