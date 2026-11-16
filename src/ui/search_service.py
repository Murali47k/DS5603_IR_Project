"""
Search pipeline for the SciDocs UI.

Default ranking is citation-boosted hybrid retrieval:
  BM25 + FAISS dense -> weighted_score_fusion -> citation_aware_boost

Retrievers are built once and reused. Qrels are never used for ranking.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.base_models import BM25Retriever, TfidfRetriever
from src.base_models.base import BaseRetriever
from src.base_models.utils import corpus_texts, document_text, ranked_ids
from src.optimisations.reranker import CrossEncoderReranker
from src.data_loader import load_scidocs
from src.evaluation.baselines import load_dense_encoder
from src.optimisations.citation_boost import citation_aware_boost
from src.optimisations.fusion import (
    _min_max_normalize,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from src.ui.metadata import (
    PaperRecord,
    citation_counts_from_catalog,
    load_paper_catalog,
    lookup_display_papers,
    lookup_papers,
)
from src.ui.settings import UISettings, load_ui_settings
from src.vectordb.build_db import get_index_dir
from src.vectordb.embeddings import (
    build_cache_key,
    encode_queries,
    get_cache_paths,
    get_or_create_embeddings,
    to_float32,
)
from src.vectordb.retriever import VectorDBRetriever, hits_from_search_row
from src.vectordb.store import get_index_paths

Hit = Tuple[str, float]


@dataclass
class RankedPaper:
    rank: int
    paper: PaperRecord
    final_score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    fused_score: Optional[float] = None
    citation_multiplier: Optional[float] = None
    method: str = "citation_hybrid"
    tfidf_score: Optional[float] = None
    norm_bm25_score: Optional[float] = None
    norm_dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    rrf_bm25_term: Optional[float] = None
    rrf_dense_term: Optional[float] = None
    ce_score: Optional[float] = None
    candidate_rank: Optional[int] = None


@dataclass
class RetrievalPool:
    query: str
    bm25_hits: List[Hit]
    dense_hits: List[Hit]


@dataclass
class SearchResponse:
    query: str
    method: str
    results: List[RankedPaper]
    pool: Optional[RetrievalPool] = None
    error: str = ""
    latency_ms: Optional[float] = None


def _score_map(hits: Sequence[Hit]) -> Dict[str, float]:
    return {doc_id: float(score) for doc_id, score in hits}


def _multiplier(fused: float, boosted: float) -> float:
    if fused == 0.0:
        return 1.0 if boosted == 0.0 else 0.0
    return float(boosted) / float(fused)


def _in_year_range(year: Optional[int], year_min: Optional[int], year_max: Optional[int]) -> bool:
    if year_min is None and year_max is None:
        return True
    if year is None:
        return True
    if year_min is not None and year < year_min:
        return False
    if year_max is not None and year > year_max:
        return False
    return True


class SearchEngine:
    """Reusable hybrid search engine. Safe to use without Streamlit."""

    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, Any]],
        catalog: Mapping[str, PaperRecord],
        bm25: BaseRetriever,
        dense: Optional[BaseRetriever],
        settings: UISettings,
        dense_error: str = "",
        tfidf: Optional[BaseRetriever] = None,
        reranker: Optional[Any] = None,
    ):
        self.corpus = corpus
        self.catalog = catalog
        self.bm25 = bm25
        self.dense = dense
        self.tfidf = tfidf
        self.settings = settings
        self.dense_error = dense_error
        self._reranker = reranker
        self._reranker_error = ""
        self._tfidf_error = ""
        self.citation_counts = citation_counts_from_catalog(catalog, corpus.keys())
        self._id_to_pos = {}
        if dense is not None and getattr(dense, "doc_ids", None):
            self._id_to_pos = {doc_id: i for i, doc_id in enumerate(dense.doc_ids)}

    def paper(self, doc_id: str) -> PaperRecord:
        record = self.catalog.get(doc_id)
        if record is not None:
            return record
        doc = self.corpus.get(doc_id) or {}
        return PaperRecord(
            doc_id=doc_id,
            title=str(doc.get("title") or "Untitled paper"),
            abstract=str(doc.get("text") or ""),
            citation_count=self.citation_counts.get(doc_id, 0),
        )

    def retrieve_pool(
        self, query: str, candidate_k: Optional[int] = None
    ) -> Tuple[RetrievalPool, float, float]:
        pool_k = max(int(candidate_k or self.settings.candidate_k), 1)
        bm25_hits: List[Hit] = []
        dense_hits: List[Hit] = []
        bm25_ms = 0.0
        dense_ms = 0.0
        if query.strip():
            started = time.perf_counter()
            bm25_hits = self.bm25.retrieve(query, k=pool_k)
            bm25_ms = (time.perf_counter() - started) * 1000.0
            if self.dense is not None:
                started = time.perf_counter()
                dense_hits = self.dense.retrieve(query, k=pool_k)
                dense_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalPool(query=query, bm25_hits=bm25_hits, dense_hits=dense_hits), bm25_ms, dense_ms

    def rank_hybrid(
        self,
        pool: RetrievalPool,
        boost_weight: float,
        k: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
        method: str = "citation_hybrid",
    ) -> List[RankedPaper]:
        if boost_weight < 0:
            raise ValueError("boost_weight must be non-negative")
        fused = weighted_score_fusion(
            pool.bm25_hits,
            pool.dense_hits,
            bm25_weight=self.settings.bm25_weight,
            dense_weight=self.settings.dense_weight,
            normalize=self.settings.normalize_scores,
        )
        boosted = citation_aware_boost(
            fused,
            self.citation_counts,
            boost_weight=boost_weight,
        )
        return self._decorate(
            boosted,
            pool,
            fused_map=_score_map(fused),
            boosted_map=_score_map(boosted),
            k=k or self.settings.k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            method=method,
        )

    def rank_weighted(
        self,
        pool: RetrievalPool,
        k: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
    ) -> List[RankedPaper]:
        fused = weighted_score_fusion(
            pool.bm25_hits,
            pool.dense_hits,
            bm25_weight=self.settings.bm25_weight,
            dense_weight=self.settings.dense_weight,
            normalize=self.settings.normalize_scores,
        )
        norm_bm25 = (
            _min_max_normalize(pool.bm25_hits)
            if self.settings.normalize_scores
            else _score_map(pool.bm25_hits)
        )
        norm_dense = (
            _min_max_normalize(pool.dense_hits)
            if self.settings.normalize_scores
            else _score_map(pool.dense_hits)
        )
        extra_meta = {
            doc_id: {
                "norm_bm25_score": norm_bm25.get(doc_id),
                "norm_dense_score": norm_dense.get(doc_id),
                "fused_score": s,
            }
            for doc_id, s in fused
        }
        fused_map = _score_map(fused)
        return self._decorate(
            fused,
            pool,
            fused_map=fused_map,
            boosted_map=fused_map,
            k=k or self.settings.k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            method="weighted",
            apply_multiplier=False,
            extra_meta=extra_meta,
        )

    def rank_rrf(
        self,
        pool: RetrievalPool,
        k: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
    ) -> List[RankedPaper]:
        bm25_order = ranked_ids(pool.bm25_hits)
        dense_order = ranked_ids(pool.dense_hits)
        fused = reciprocal_rank_fusion(
            [bm25_order, dense_order],
            k=self.settings.rrf_k,
        )
        bm25_ranks = {doc_id: r for r, doc_id in enumerate(bm25_order, 1)}
        dense_ranks = {doc_id: r for r, doc_id in enumerate(dense_order, 1)}
        rrf_k = self.settings.rrf_k
        extra_meta = {}
        for doc_id, s in fused:
            r_b = bm25_ranks.get(doc_id)
            r_d = dense_ranks.get(doc_id)
            extra_meta[doc_id] = {
                "bm25_rank": r_b,
                "dense_rank": r_d,
                "rrf_bm25_term": (1.0 / (rrf_k + r_b)) if r_b else 0.0,
                "rrf_dense_term": (1.0 / (rrf_k + r_d)) if r_d else 0.0,
                "fused_score": s,
            }
        fused_map = _score_map(fused)
        return self._decorate(
            fused,
            pool,
            fused_map=fused_map,
            boosted_map=fused_map,
            k=k or self.settings.k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            method="rrf",
            apply_multiplier=False,
            extra_meta=extra_meta,
        )

    def rank_cross_encoder(
        self,
        pool: RetrievalPool,
        query: str,
        k: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
    ) -> Tuple[List[RankedPaper], str]:
        reranker = self._get_reranker()
        if reranker is None:
            return [], self._reranker_error or "Cross-encoder reranker is unavailable."
        top_k = k or self.settings.k
        pool_k = max(top_k, int(self.settings.candidate_k))
        fused = weighted_score_fusion(
            pool.bm25_hits,
            pool.dense_hits,
            bm25_weight=self.settings.bm25_weight,
            dense_weight=self.settings.dense_weight,
            normalize=self.settings.normalize_scores,
        )[:pool_k]
        candidates = [(doc_id, self.corpus.get(doc_id, {})) for doc_id, _ in fused]
        try:
            reranked = reranker.rerank(query, candidates)
        except Exception as exc:  # noqa: BLE001
            return [], f"{type(exc).__name__}: {exc}"
        fused_map = _score_map(fused)
        reranked_map = _score_map(reranked)
        cand_ranks = {doc_id: r for r, (doc_id, _) in enumerate(fused, 1)}
        extra_meta = {
            doc_id: {
                "candidate_rank": cand_ranks.get(doc_id),
                "ce_score": s,
                "fused_score": fused_map.get(doc_id),
            }
            for doc_id, s in reranked
        }
        return self._decorate(
            reranked,
            pool,
            fused_map=fused_map,
            boosted_map=reranked_map,
            k=top_k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            method="cross_encoder",
            apply_multiplier=False,
            extra_meta=extra_meta,
        ), ""

    def rank_single(
        self,
        hits: Sequence[Hit],
        method: str,
        pool: Optional[RetrievalPool] = None,
        k: Optional[int] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
    ) -> List[RankedPaper]:
        fused_map = _score_map(hits)
        extra_meta = {}
        if method == "tfidf":
            extra_meta = {doc_id: {"tfidf_score": s} for doc_id, s in hits}
        elif method == "bm25":
            extra_meta = {doc_id: {"bm25_score": s} for doc_id, s in hits}
        elif method == "dense":
            extra_meta = {doc_id: {"dense_score": s} for doc_id, s in hits}
        return self._decorate(
            list(hits),
            pool,
            fused_map=fused_map,
            boosted_map=fused_map,
            k=k or self.settings.k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
            method=method,
            apply_multiplier=False,
            extra_meta=extra_meta,
        )

    def search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        boost_weight: Optional[float] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
        candidate_k: Optional[int] = None,
    ) -> SearchResponse:
        if not query or not str(query).strip():
            return SearchResponse(query=query or "", method="citation_hybrid", results=[])
        warning = ""
        if self.dense is None:
            warning = self.dense_error or "Dense FAISS retriever is unavailable; ranking BM25 with citation boosting only."
        pool, _bm25_ms, _dense_ms = self.retrieve_pool(query, candidate_k=candidate_k)
        results = self.rank_hybrid(
            pool,
            boost_weight=self.settings.citation_boost_weight if boost_weight is None else boost_weight,
            k=k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
        )
        return SearchResponse(
            query=query,
            method="citation_hybrid",
            results=results,
            pool=pool,
            error=warning,
        )

    def compare(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        boost_weight: Optional[float] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        min_citations: int = 0,
        candidate_k: Optional[int] = None,
    ) -> Dict[str, SearchResponse]:
        top_k = k or self.settings.k
        weight = self.settings.citation_boost_weight if boost_weight is None else boost_weight
        empty_keys = (
            "bm25",
            "tfidf",
            "dense",
            "citation_hybrid",
            "cross_encoder",
            "rrf",
            "weighted",
        )
        if not query or not str(query).strip():
            empty = SearchResponse(query=query or "", method="empty", results=[])
            return {key: empty for key in empty_keys}

        pool_k = max(top_k, int(candidate_k or self.settings.candidate_k))
        pool, bm25_ms, dense_ms = self.retrieve_pool(query, candidate_k=pool_k)
        dense_warning = ""
        if self.dense is None:
            dense_warning = self.dense_error or "Dense FAISS retriever is unavailable; fusion uses BM25 candidates only."

        rank_kwargs = dict(
            k=top_k,
            year_min=year_min,
            year_max=year_max,
            min_citations=min_citations,
        )

        started = time.perf_counter()
        tfidf_hits, tfidf_error = self._retrieve_tfidf(query, pool_k)
        tfidf_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        bm25_results = self.rank_single(pool.bm25_hits, "bm25", pool, **rank_kwargs)
        bm25_rank_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        dense_results = self.rank_single(pool.dense_hits, "dense", pool, **rank_kwargs)
        dense_rank_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        tfidf_results = self.rank_single(tfidf_hits, "tfidf", pool, **rank_kwargs)
        tfidf_rank_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        rrf_results = self.rank_rrf(pool, **rank_kwargs)
        rrf_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        weighted_results = self.rank_weighted(pool, **rank_kwargs)
        weighted_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        citation_results = self.rank_hybrid(pool, weight, **rank_kwargs)
        citation_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        ce_results, ce_error = self.rank_cross_encoder(pool, query, **rank_kwargs)
        ce_ms = (time.perf_counter() - started) * 1000.0
        ce_messages = [msg for msg in (dense_warning, ce_error) if msg]

        return {
            "bm25": SearchResponse(
                query=query,
                method="bm25",
                results=bm25_results,
                pool=pool,
                latency_ms=bm25_ms + bm25_rank_ms,
            ),
            "tfidf": SearchResponse(
                query=query,
                method="tfidf",
                results=tfidf_results,
                pool=pool,
                error=tfidf_error,
                latency_ms=tfidf_ms + tfidf_rank_ms,
            ),
            "dense": SearchResponse(
                query=query,
                method="dense",
                results=dense_results,
                pool=pool,
                error=dense_warning,
                latency_ms=dense_ms + dense_rank_ms,
            ),
            "citation_hybrid": SearchResponse(
                query=query,
                method="citation_hybrid",
                results=citation_results,
                pool=pool,
                error=dense_warning,
                latency_ms=bm25_ms + dense_ms + citation_ms,
            ),
            "cross_encoder": SearchResponse(
                query=query,
                method="cross_encoder",
                results=ce_results,
                pool=pool,
                error=" ".join(ce_messages),
                latency_ms=bm25_ms + dense_ms + ce_ms,
            ),
            "rrf": SearchResponse(
                query=query,
                method="rrf",
                results=rrf_results,
                pool=pool,
                error=dense_warning,
                latency_ms=bm25_ms + dense_ms + rrf_ms,
            ),
            "weighted": SearchResponse(
                query=query,
                method="weighted",
                results=weighted_results,
                pool=pool,
                error=dense_warning,
                latency_ms=bm25_ms + dense_ms + weighted_ms,
            ),
        }

    def cited_by(self, doc_id: str) -> List[PaperRecord]:
        return lookup_papers(self.catalog, self.paper(doc_id).cited_by)

    def references(self, doc_id: str) -> List[PaperRecord]:
        return lookup_papers(self.catalog, self.paper(doc_id).references)

    def visible_cited_by(self, doc_id: str) -> List[PaperRecord]:
        return lookup_display_papers(self.catalog, self.paper(doc_id).cited_by)

    def visible_references(self, doc_id: str) -> List[PaperRecord]:
        return lookup_display_papers(self.catalog, self.paper(doc_id).references)

    def citation_list_counts(self, doc_id: str) -> Dict[str, int]:
        paper = self.paper(doc_id)
        cited = paper.cited_by
        refs = paper.references
        return {
            "cited_recorded": len(cited),
            "cited_visible": len(lookup_display_papers(self.catalog, cited)),
            "refs_recorded": len(refs),
            "refs_visible": len(lookup_display_papers(self.catalog, refs)),
        }

    def find_similar(self, doc_id: str, k: int = 5) -> List[RankedPaper]:
        if self.dense is None or getattr(self.dense, "index", None) is None:
            return []
        vector = self._document_vector(doc_id)
        if vector is None:
            return []
        search_k = min(max(k + 1, 2), len(self.dense.doc_ids))
        scores, positions = self.dense.index.search(vector, search_k)
        hits = hits_from_search_row(scores[0], positions[0], self.dense.doc_ids, search_k)
        hits = [(hid, score) for hid, score in hits if hid != doc_id][:k]
        ranked = []
        for rank, (hid, score) in enumerate(hits, start=1):
            ranked.append(
                RankedPaper(
                    rank=rank,
                    paper=self.paper(hid),
                    final_score=float(score),
                    dense_score=float(score),
                    method="similar",
                )
            )
        return ranked

    def _get_tfidf(self) -> Optional[BaseRetriever]:
        if self.tfidf is not None:
            return self.tfidf
        if self._tfidf_error:
            return None
        try:
            self.tfidf = TfidfRetriever(self.corpus)
            return self.tfidf
        except Exception as exc:  # noqa: BLE001
            self._tfidf_error = f"{type(exc).__name__}: {exc}"
            return None

    def _retrieve_tfidf(self, query: str, k: int) -> Tuple[List[Hit], str]:
        retriever = self._get_tfidf()
        if retriever is None:
            return [], self._tfidf_error or "TF-IDF retriever is unavailable."
        try:
            return retriever.retrieve(query, k=k), ""
        except Exception as exc:  # noqa: BLE001
            return [], f"{type(exc).__name__}: {exc}"

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker
        if self._reranker_error:
            return None
        try:
            self._reranker = CrossEncoderReranker(
                model_name=self.settings.reranker_model,
                top_k=max(int(self.settings.candidate_k), 1),
                device="cpu",
            )
            return self._reranker
        except Exception as exc:  # noqa: BLE001
            self._reranker_error = f"{type(exc).__name__}: {exc}"
            return None

    def _document_vector(self, doc_id: str) -> Optional[np.ndarray]:
        pos = self._id_to_pos.get(doc_id)
        index = getattr(self.dense, "index", None)
        if pos is not None and index is not None:
            try:
                reconstructed = index.reconstruct(int(pos))
                return to_float32(np.asarray(reconstructed).reshape(1, -1))
            except Exception:
                pass
        doc = self.corpus.get(doc_id)
        if not doc or getattr(self.dense, "encoder", None) is None:
            return None
        text = document_text(doc)
        if not text.strip():
            return None
        return encode_queries(self.dense.encoder, [text])

    def _decorate(
        self,
        ranked_hits: Sequence[Hit],
        pool: Optional[RetrievalPool],
        fused_map: Mapping[str, float],
        boosted_map: Mapping[str, float],
        k: int,
        year_min: Optional[int],
        year_max: Optional[int],
        min_citations: int,
        method: str,
        apply_multiplier: bool = True,
        extra_meta: Optional[Mapping[str, Dict[str, Any]]] = None,
    ) -> List[RankedPaper]:
        bm25_map = _score_map(pool.bm25_hits) if pool else {}
        dense_map = _score_map(pool.dense_hits) if pool else {}
        results: List[RankedPaper] = []
        for doc_id, score in ranked_hits:
            paper = self.paper(doc_id)
            if paper.citation_count < min_citations:
                continue
            if not _in_year_range(paper.year, year_min, year_max):
                continue
            fused = fused_map.get(doc_id)
            boosted = boosted_map.get(doc_id, float(score))
            meta = (extra_meta or {}).get(doc_id, {})
            bm25_score = meta.get("bm25_score", bm25_map.get(doc_id))
            dense_score = meta.get("dense_score", dense_map.get(doc_id))
            if method == "bm25":
                bm25_score = float(score)
            elif method == "dense":
                dense_score = float(score)

            results.append(
                RankedPaper(
                    rank=len(results) + 1,
                    paper=paper,
                    final_score=float(score),
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                    fused_score=meta.get("fused_score", fused),
                    citation_multiplier=(
                        _multiplier(fused, boosted) if apply_multiplier and fused is not None else None
                    ),
                    method=method,
                    tfidf_score=meta.get("tfidf_score", float(score) if method == "tfidf" else None),
                    norm_bm25_score=meta.get("norm_bm25_score"),
                    norm_dense_score=meta.get("norm_dense_score"),
                    bm25_rank=meta.get("bm25_rank"),
                    dense_rank=meta.get("dense_rank"),
                    rrf_bm25_term=meta.get("rrf_bm25_term"),
                    rrf_dense_term=meta.get("rrf_dense_term"),
                    ce_score=meta.get("ce_score", float(score) if method == "cross_encoder" else None),
                    candidate_rank=meta.get("candidate_rank"),
                )
            )
            if len(results) >= k:
                break
        return results


def _saved_index_dir(settings: UISettings) -> Path:
    return get_index_dir(settings.index_type, settings.dense_model, Path(settings.vector_index_dir))


def _index_is_saved(index_dir: Path) -> bool:
    index_path, ids_path, meta_path = get_index_paths(index_dir)
    return index_path.is_file() and ids_path.is_file() and meta_path.is_file()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cached_embedding_paths(corpus: Mapping[str, Any], model_name: str, cache_dir: Path) -> Tuple[Path, Path]:
    doc_ids, texts = corpus_texts(corpus)
    return get_cache_paths(Path(cache_dir), build_cache_key(model_name, doc_ids, texts))


def _ensure_faiss_index_saved(
    corpus: Mapping[str, Any],
    settings: UISettings,
    model_name: str,
    saved: Path,
) -> None:
    """Persist a FAISS index without building HNSW in the UI process.

    Cached embeddings may already exist. Index construction is done in a
    child Python process that imports FAISS only, matching the working
    standalone HNSW build. The UI process then loads the saved index.
    """
    if _index_is_saved(saved):
        return

    cache_dir = Path(settings.vector_index_dir) / "embeddings"
    vectors_path, ids_path = _cached_embedding_paths(corpus, model_name, cache_dir)
    if not (vectors_path.is_file() and ids_path.is_file()):
        encoder = load_dense_encoder(model_name, local_files_only=False, device="cpu")
        get_or_create_embeddings(
            corpus,
            encoder,
            model_name,
            cache_dir,
            use_cache=True,
            show_progress=True,
        )
        vectors_path, ids_path = _cached_embedding_paths(corpus, model_name, cache_dir)

    if not (vectors_path.is_file() and ids_path.is_file()):
        raise FileNotFoundError(f"Cached embeddings were not written to {cache_dir}")

    root = _project_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.vectordb.index_builder import build_and_save_from_npy; "
            "build_and_save_from_npy("
            f"{str(vectors_path)!r}, {str(ids_path)!r}, {str(saved)!r}, "
            f"{settings.index_type!r}, {model_name!r})",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not _index_is_saved(saved):
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"FAISS index build failed (exit {completed.returncode}). {detail}"
        )


def _build_dense_retriever(corpus: Mapping[str, Any], settings: UISettings):
    errors: List[str] = []
    candidates = []
    for name in (settings.dense_model, settings.dense_fallback):
        if name and name not in candidates:
            candidates.append(name)

    for name in candidates:
        try:
            saved = get_index_dir(settings.index_type, name, Path(settings.vector_index_dir))
            _ensure_faiss_index_saved(corpus, settings, name, saved)
            encoder = load_dense_encoder(
                name,
                local_files_only=False,
                device="cpu",
            )
            retriever = VectorDBRetriever.from_saved(saved, encoder)
            retriever.model_name = name
            return retriever, ""
        except Exception as exc:  # noqa: BLE001 — UI must still start on BM25-only
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return None, " ; ".join(errors) if errors else "Dense retriever could not be created."


def build_search_engine(
    settings: Optional[UISettings] = None,
    corpus: Optional[Mapping[str, Mapping[str, Any]]] = None,
    catalog: Optional[Mapping[str, PaperRecord]] = None,
    bm25: Optional[BaseRetriever] = None,
    dense: Optional[BaseRetriever] = None,
    tfidf: Optional[BaseRetriever] = None,
    reranker: Optional[Any] = None,
    load_dataset: bool = True,
) -> SearchEngine:
    """
    Build the production search engine.

    Tests should pass corpus/catalog/retrievers and set load_dataset=False.
    """
    settings = settings or load_ui_settings()
    if corpus is None:
        if not load_dataset:
            raise ValueError("corpus is required when load_dataset is False")
        corpus, _queries, _qrels = load_scidocs(data_dir=settings.data_dir)
    if catalog is None:
        catalog = load_paper_catalog(corpus, data_dir=Path(settings.data_dir))
    if bm25 is None:
        bm25 = BM25Retriever(corpus)
    dense_error = ""
    if dense is None and load_dataset:
        dense, dense_error = _build_dense_retriever(corpus, settings)
    return SearchEngine(
        corpus=corpus,
        catalog=catalog,
        bm25=bm25,
        dense=dense,
        settings=settings,
        dense_error=dense_error,
        tfidf=tfidf,
        reranker=reranker,
    )
