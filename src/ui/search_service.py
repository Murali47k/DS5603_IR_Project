"""
Search pipeline for the SciDocs UI.

Default ranking is citation-boosted hybrid retrieval:
  BM25 + FAISS dense -> weighted_score_fusion -> citation_aware_boost

Retrievers are built once and reused. Qrels are never used for ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.base_models import BM25Retriever
from src.base_models.base import BaseRetriever
from src.base_models.utils import document_text
from src.data_loader import load_scidocs
from src.evaluation.baselines import load_dense_encoder
from src.optimisations.citation_boost import citation_aware_boost
from src.optimisations.fusion import weighted_score_fusion
from src.ui.metadata import (
    PaperRecord,
    citation_counts_from_catalog,
    load_paper_catalog,
    lookup_papers,
)
from src.ui.settings import UISettings, load_ui_settings
from src.vectordb.build_db import get_index_dir
from src.vectordb.embeddings import encode_queries, to_float32
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
    ):
        self.corpus = corpus
        self.catalog = catalog
        self.bm25 = bm25
        self.dense = dense
        self.settings = settings
        self.dense_error = dense_error
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

    def retrieve_pool(self, query: str, candidate_k: Optional[int] = None) -> RetrievalPool:
        pool_k = max(int(candidate_k or self.settings.candidate_k), 1)
        bm25_hits = self.bm25.retrieve(query, k=pool_k) if query.strip() else []
        dense_hits: List[Hit] = []
        if self.dense is not None and query.strip():
            dense_hits = self.dense.retrieve(query, k=pool_k)
        return RetrievalPool(query=query, bm25_hits=bm25_hits, dense_hits=dense_hits)

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
        pool = self.retrieve_pool(query, candidate_k=candidate_k)
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
        if not query or not str(query).strip():
            empty = SearchResponse(query=query or "", method="empty", results=[])
            return {"bm25": empty, "dense": empty, "citation_hybrid": empty}

        pool = self.retrieve_pool(query, candidate_k=candidate_k)
        bm25_resp = SearchResponse(
            query=query,
            method="bm25",
            results=self.rank_single(
                pool.bm25_hits, "bm25", pool, k=top_k,
                year_min=year_min, year_max=year_max, min_citations=min_citations,
            ),
            pool=pool,
        )
        dense_warning = ""
        if self.dense is None:
            dense_warning = self.dense_error or "Dense FAISS retriever is unavailable."
            dense_resp = SearchResponse(
                query=query, method="dense", results=[],
                error=dense_warning, pool=pool,
            )
        else:
            dense_resp = SearchResponse(
                query=query,
                method="dense",
                results=self.rank_single(
                    pool.dense_hits, "dense", pool, k=top_k,
                    year_min=year_min, year_max=year_max, min_citations=min_citations,
                ),
                pool=pool,
            )
        hybrid_resp = SearchResponse(
            query=query,
            method="citation_hybrid",
            results=self.rank_hybrid(
                pool, weight, k=top_k,
                year_min=year_min, year_max=year_max, min_citations=min_citations,
            ),
            pool=pool,
            error=dense_warning,
        )
        return {"bm25": bm25_resp, "dense": dense_resp, "citation_hybrid": hybrid_resp}

    def cited_by(self, doc_id: str) -> List[PaperRecord]:
        return lookup_papers(self.catalog, self.paper(doc_id).cited_by)

    def references(self, doc_id: str) -> List[PaperRecord]:
        return lookup_papers(self.catalog, self.paper(doc_id).references)

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
            results.append(
                RankedPaper(
                    rank=len(results) + 1,
                    paper=paper,
                    final_score=float(score),
                    bm25_score=bm25_map.get(doc_id),
                    dense_score=dense_map.get(doc_id),
                    fused_score=fused,
                    citation_multiplier=(
                        _multiplier(fused, boosted) if apply_multiplier and fused is not None else None
                    ),
                    method=method,
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


def _build_dense_retriever(corpus: Mapping[str, Any], settings: UISettings):
    errors: List[str] = []
    candidates = []
    for name in (settings.dense_model, settings.dense_fallback):
        if name and name not in candidates:
            candidates.append(name)

    for name in candidates:
        try:
            encoder = load_dense_encoder(name, local_files_only=False)
            saved = get_index_dir(settings.index_type, name, Path(settings.vector_index_dir))
            if _index_is_saved(saved):
                retriever = VectorDBRetriever.from_saved(saved, encoder)
                retriever.model_name = name
                return retriever, ""
            retriever = VectorDBRetriever.from_corpus(
                corpus,
                encoder,
                index_type=settings.index_type,
                model_name=name,
                cache_dir=Path(settings.vector_index_dir) / "embeddings",
                use_cache=True,
            )
            try:
                retriever.save(saved)
            except Exception:
                pass
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
    )
