"""Optimisation evaluation on BEIR SciDocs (test split).

Three approaches are measured and compared against each other and the published
baseline numbers:

  1. RRF Hybrid      – BM25 + FAISS-Dense ranked lists fused with Reciprocal Rank Fusion.
  2. Weighted Hybrid – BM25 + FAISS-Dense scores combined with configurable weights.
  3. Citation Hybrid – Weighted hybrid scores further boosted by citation counts.

All three share the same BM25 retriever and the same VectorDBRetriever (backed by
whichever FAISS index type you choose — HNSW by default), so results are directly
comparable.  The output rows use the same column schema as ``baseline_eval`` so the
two CSV files can be concatenated for a single comparison table.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    DENSE_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    OUTPUTS_DIR,
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
    RRF_K,
    VECTOR_INDEX_DIR,
)
from src.base_models import BM25Retriever
from src.base_models.base import BaseRetriever
from src.data_loader import get_document_citations
from src.evaluation.baselines import (
    FALLBACK_DENSE_MODEL,
    load_dense_encoder,
    load_official_scidocs,
)
from src.evaluation.metrics import (
    evaluate_beir_results,
    metrics_at_k,
    ranked_hits_to_beir_results,
)
from src.optimisations.citation_boost import citation_aware_boost
from src.optimisations.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.optimisations.reranker import CrossEncoderReranker
from src.vectordb import VectorDBRetriever
from src.vectordb.constants import EMBEDDINGS_SUBDIR, INDEX_HNSW


# ---- latency helpers (same contract as baselines.py) ----

def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def _latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    """Mean and percentile latencies in ms, same schema as baselines.py."""
    if not latencies:
        return {
            "n_timed_queries": 0,
            "query_latency_mean_ms": 0.0,
            "query_latency_p50_ms": 0.0,
            "query_latency_p95_ms": 0.0,
            "query_latency_total_seconds": 0.0,
        }
    ms = [x * 1000.0 for x in latencies]
    return {
        "n_timed_queries": len(latencies),
        "query_latency_mean_ms": round(statistics.fmean(ms), 4),
        "query_latency_p50_ms": round(_percentile(ms, 50), 4),
        "query_latency_p95_ms": round(_percentile(ms, 95), 4),
        "query_latency_total_seconds": round(sum(latencies), 4),
    }


# ---- VectorDB builder (wraps the teammate's VectorDBRetriever) ----

def _build_vector_retriever(
    corpus: Mapping[str, Any],
    model_name: str = DENSE_MODEL_NAME,
    fallback_model: str = FALLBACK_DENSE_MODEL,
    index_type: str = INDEX_HNSW,
    allow_download: bool = True,
    use_cache: bool = True,
) -> tuple[Optional[VectorDBRetriever], Dict[str, Any]]:
    """Load an encoder then build a FAISS VectorDBRetriever over the corpus.

    Tries the primary model locally first, then the fallback locally, then
    (optionally) downloads — same fallback chain as try_build_dense_retriever in
    baselines.py so the two suites behave consistently.
    """
    cache_dir = Path(VECTOR_INDEX_DIR) / EMBEDDINGS_SUBDIR

    candidates = [(model_name, True), (fallback_model, True)]
    if allow_download:
        candidates += [(model_name, False), (fallback_model, False)]

    attempts: List[str] = []
    seen: set = set()
    for name, local_only in candidates:
        key = (name, local_only)
        if not name or key in seen:
            continue
        seen.add(key)
        mode = "local" if local_only else "download-allowed"
        try:
            load_start = time.perf_counter()
            encoder = load_dense_encoder(name, local_files_only=local_only)
            model_load_seconds = time.perf_counter() - load_start

            index_start = time.perf_counter()
            retriever = VectorDBRetriever.from_corpus(
                corpus, encoder,
                index_type=index_type,
                model_name=name,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            index_seconds = time.perf_counter() - index_start

            return retriever, {
                "status": "ok",
                "dense_model": name,
                "index_type": index_type,
                "model_load_seconds": round(model_load_seconds, 4),
                "index_seconds": round(index_seconds, 4),
                "attempts": attempts + [f"{name} ({mode}): ok"],
            }
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{name} ({mode}): {type(exc).__name__}: {exc}")

    return None, {
        "status": "skipped",
        "dense_model": model_name,
        "index_type": index_type,
        "error": " ; ".join(attempts) if attempts else "no dense model configured",
        "attempts": attempts,
        "model_load_seconds": None,
        "index_seconds": None,
    }


# ---- per-query fusion helpers ----

def _retrieve_rrf(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    rrf_k: int = RRF_K,
) -> tuple[Dict[str, Dict[str, float]], List[float]]:
    """Retrieve with RRF fusion over candidate pool and return (beir_results, per-query latencies)."""
    per_query_hits: Dict[str, Any] = {}
    latencies: List[float] = []

    items = list(queries.items())
    try:
        from tqdm import tqdm
        items = tqdm(items, desc="Approach 1 (RRF queries)", leave=True)
    except ImportError:
        pass

    pool_k = max(k, candidate_k)
    for query_id, query_text in items:
        start = time.perf_counter()
        bm25_ids = bm25.retrieve_ids(query_text, k=pool_k)
        dense_ids = dense.retrieve_ids(query_text, k=pool_k)
        hits = reciprocal_rank_fusion([bm25_ids, dense_ids], k=rrf_k)[:k]
        latencies.append(time.perf_counter() - start)
        per_query_hits[query_id] = hits

    return ranked_hits_to_beir_results(per_query_hits), latencies


def _retrieve_weighted(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    normalize: bool = True,
) -> tuple[Dict[str, Dict[str, float]], List[float]]:
    """Retrieve with weighted score fusion over candidate pool and return (beir_results, latencies)."""
    per_query_hits: Dict[str, Any] = {}
    latencies: List[float] = []

    items = list(queries.items())
    try:
        from tqdm import tqdm
        items = tqdm(items, desc="Approach 2 (Weighted queries)", leave=True)
    except ImportError:
        pass

    pool_k = max(k, candidate_k)
    for query_id, query_text in items:
        start = time.perf_counter()
        bm25_hits = bm25.retrieve(query_text, k=pool_k)
        dense_hits = dense.retrieve(query_text, k=pool_k)
        hits = weighted_score_fusion(
            bm25_hits, dense_hits,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
            normalize=normalize,
        )[:k]
        latencies.append(time.perf_counter() - start)
        per_query_hits[query_id] = hits

    return ranked_hits_to_beir_results(per_query_hits), latencies


def _retrieve_citation_boosted(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    citation_counts: Mapping[str, int],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    boost_weight: float = 0.1,
    normalize: bool = True,
) -> tuple[Dict[str, Dict[str, float]], List[float]]:
    """Weighted fusion over candidate pool then citation-count boosting."""
    per_query_hits: Dict[str, Any] = {}
    latencies: List[float] = []

    items = list(queries.items())
    try:
        from tqdm import tqdm
        items = tqdm(items, desc="Approach 3 (Citation queries)", leave=True)
    except ImportError:
        pass

    pool_k = max(k, candidate_k)
    for query_id, query_text in items:
        start = time.perf_counter()
        bm25_hits = bm25.retrieve(query_text, k=pool_k)
        dense_hits = dense.retrieve(query_text, k=pool_k)
        fused = weighted_score_fusion(
            bm25_hits, dense_hits,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
            normalize=normalize,
        )
        hits = citation_aware_boost(fused, citation_counts, boost_weight=boost_weight)[:k]
        latencies.append(time.perf_counter() - start)
        per_query_hits[query_id] = hits

    return ranked_hits_to_beir_results(per_query_hits), latencies


def _retrieve_reranked(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    corpus: Mapping[str, Any],
    reranker: CrossEncoderReranker,
    queries: Mapping[str, str],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    normalize: bool = True,
) -> tuple[Dict[str, Dict[str, float]], List[float]]:
    """Weighted hybrid candidate generation followed by Cross-Encoder neural re-ranking."""
    per_query_hits: Dict[str, Any] = {}
    latencies: List[float] = []

    items = list(queries.items())
    try:
        from tqdm import tqdm
        items = tqdm(items, desc="Approach 4 (Cross-Encoder queries)", leave=True)
    except ImportError:
        pass

    pool_k = max(k, candidate_k)
    for query_id, query_text in items:
        start = time.perf_counter()
        bm25_hits = bm25.retrieve(query_text, k=pool_k)
        dense_hits = dense.retrieve(query_text, k=pool_k)
        fused = weighted_score_fusion(
            bm25_hits, dense_hits,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
            normalize=normalize,
        )[:pool_k]
        candidate_docs = [(doc_id, corpus.get(doc_id, {})) for doc_id, _ in fused]
        reranked = reranker.rerank(query_text, candidate_docs)[:k]
        latencies.append(time.perf_counter() - start)
        per_query_hits[query_id] = reranked

    return ranked_hits_to_beir_results(per_query_hits), latencies


# ---- public evaluation functions ----

def evaluate_rrf_hybrid(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    rrf_k: int = RRF_K,
) -> tuple[Dict[str, float], List[float]]:
    """Evaluate BM25 + FAISS-Dense fused with Reciprocal Rank Fusion."""
    results, latencies = _retrieve_rrf(bm25, dense, queries, k=k, candidate_k=candidate_k, rrf_k=rrf_k)
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(k,)), k=k)
    return metrics, latencies


def evaluate_weighted_hybrid(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    normalize: bool = True,
) -> tuple[Dict[str, float], List[float]]:
    """Evaluate BM25 + FAISS-Dense combined with normalized score weights."""
    results, latencies = _retrieve_weighted(
        bm25, dense, queries, k=k, candidate_k=candidate_k,
        bm25_weight=bm25_weight, dense_weight=dense_weight,
        normalize=normalize,
    )
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(k,)), k=k)
    return metrics, latencies


def evaluate_citation_hybrid(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    citation_counts: Mapping[str, int],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    boost_weight: float = 0.1,
    normalize: bool = True,
) -> tuple[Dict[str, float], List[float]]:
    """Evaluate weighted hybrid re-ranked by citation-count signal."""
    results, latencies = _retrieve_citation_boosted(
        bm25, dense, queries, citation_counts, k=k, candidate_k=candidate_k,
        bm25_weight=bm25_weight, dense_weight=dense_weight,
        boost_weight=boost_weight,
        normalize=normalize,
    )
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(k,)), k=k)
    return metrics, latencies


def evaluate_reranked_hybrid(
    bm25: BaseRetriever,
    dense: BaseRetriever,
    corpus: Mapping[str, Any],
    reranker: CrossEncoderReranker,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    normalize: bool = True,
) -> tuple[Dict[str, float], List[float]]:
    """Evaluate Weighted Hybrid candidates re-ranked by Cross-Encoder."""
    results, latencies = _retrieve_reranked(
        bm25, dense, corpus, reranker, queries,
        k=k, candidate_k=candidate_k,
        bm25_weight=bm25_weight, dense_weight=dense_weight,
        normalize=normalize,
    )
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(k,)), k=k)
    return metrics, latencies


# ---- suite runner ----

def run_optimisation_suite(
    corpus: Mapping[str, Any],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
    candidate_k: int = 50,
    dense_model: str = DENSE_MODEL_NAME,
    dense_fallback: str = FALLBACK_DENSE_MODEL,
    index_type: str = INDEX_HNSW,
    allow_download: bool = True,
    use_cache: bool = True,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    normalize_scores: bool = True,
    rrf_k: int = RRF_K,
    citation_boost_weight: float = 0.1,
    citation_file: Optional[Path | str] = None,
    allow_qrels_citations: bool = False,
    include_reranker: bool = False,
    reranker_model: str = RERANKER_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """Build shared BM25 + VectorDBRetriever and evaluate optimisation approaches.

    The VectorDB uses whichever FAISS index type your teammate implemented (HNSW by
    default). Embeddings are cached on disk so the second run is fast.

    Returns a list of result rows with the same column schema as ``run_baseline_suite``
    so the two outputs can be compared side by side.
    """
    rows: List[Dict[str, Any]] = []

    # Build BM25 once — shared across all approaches.
    print(f"\n[1/5] Building BM25 index over {len(corpus)} documents...")
    bm25_start = time.perf_counter()
    bm25 = BM25Retriever(corpus)
    bm25_index_seconds = round(time.perf_counter() - bm25_start, 4)
    print(f"BM25 index ready in {bm25_index_seconds:.2f}s.")

    # Build the VectorDB retriever (FAISS-backed) once — shared across approaches.
    print(f"\n[2/5] Setting up VectorDB retriever ({index_type}) with model '{dense_model}'...")
    dense, dense_meta = _build_vector_retriever(
        corpus,
        model_name=dense_model,
        fallback_model=dense_fallback,
        index_type=index_type,
        allow_download=allow_download,
        use_cache=use_cache,
    )

    if dense is None:
        print(f"Warning: VectorDBRetriever unavailable: {dense_meta.get('error', '')}")
        for name in ("rrf_hybrid", "weighted_hybrid", "citation_hybrid"):
            rows.append({
                "model": name,
                "status": "skipped",
                "error": f"VectorDBRetriever unavailable: {dense_meta.get('error', '')}",
            })
        return rows

    # Citation counts: clean extraction (metadata or external file; qrels fallback requires explicit opt-in)
    citation_counts = get_document_citations(
        corpus,
        qrels=qrels if allow_qrels_citations else None,
        citation_file=citation_file,
        allow_qrels_fallback=allow_qrels_citations,
    )

    # Shared metadata attached to all rows.
    shared_meta = {
        "dense_model": dense_meta.get("dense_model"),
        "index_type": dense_meta.get("index_type"),
        "bm25_index_seconds": bm25_index_seconds,
        "model_load_seconds": dense_meta.get("model_load_seconds"),
        "index_seconds": dense_meta.get("index_seconds"),
        "candidate_k": candidate_k,
    }

    # 1. RRF Hybrid
    print(f"\n[3/5] Evaluating Approach 1: Reciprocal Rank Fusion (k={k}, candidate_k={candidate_k}, rrf_k={rrf_k})...")
    metrics, latencies = evaluate_rrf_hybrid(
        bm25, dense, queries, qrels, k=k, candidate_k=candidate_k, rrf_k=rrf_k
    )
    rows.append({
        "model": "rrf_hybrid",
        "status": "ok",
        **shared_meta,
        "rrf_k": rrf_k,
        **metrics,
        **_latency_stats(latencies),
        "error": "",
    })

    # 2. Weighted Score Hybrid
    print(
        f"\n[4/5] Evaluating Approach 2: Weighted Score Fusion "
        f"(BM25={bm25_weight}, Dense={dense_weight}, normalize={normalize_scores})..."
    )
    metrics, latencies = evaluate_weighted_hybrid(
        bm25, dense, queries, qrels, k=k, candidate_k=candidate_k,
        bm25_weight=bm25_weight, dense_weight=dense_weight,
        normalize=normalize_scores,
    )
    rows.append({
        "model": "weighted_hybrid",
        "status": "ok",
        **shared_meta,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "normalized": normalize_scores,
        **metrics,
        **_latency_stats(latencies),
        "error": "",
    })

    # 3. Citation-Boosted Hybrid
    print(f"\n[5/5] Evaluating Approach 3: Citation-Boosted Hybrid (boost_weight={citation_boost_weight})...")
    metrics, latencies = evaluate_citation_hybrid(
        bm25, dense, queries, qrels, citation_counts, k=k, candidate_k=candidate_k,
        bm25_weight=bm25_weight, dense_weight=dense_weight,
        boost_weight=citation_boost_weight,
        normalize=normalize_scores,
    )
    rows.append({
        "model": "citation_hybrid",
        "status": "ok",
        **shared_meta,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "citation_boost_weight": citation_boost_weight,
        "normalized": normalize_scores,
        **metrics,
        **_latency_stats(latencies),
        "error": "",
    })

    # 4. Cross-Encoder Reranker (Optional Approach 4)
    if include_reranker:
        print(f"\n[Optional] Evaluating Approach 4: Cross-Encoder Reranking ('{reranker_model}')...")
        try:
            reranker = CrossEncoderReranker(model_name=reranker_model, top_k=candidate_k)
            metrics, latencies = evaluate_reranked_hybrid(
                bm25, dense, corpus, reranker, queries, qrels,
                k=k, candidate_k=candidate_k,
                bm25_weight=bm25_weight, dense_weight=dense_weight,
                normalize=normalize_scores,
            )
            rows.append({
                "model": "cross_encoder_hybrid",
                "status": "ok",
                **shared_meta,
                "reranker_model": reranker_model,
                "bm25_weight": bm25_weight,
                "dense_weight": dense_weight,
                **metrics,
                **_latency_stats(latencies),
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Cross-Encoder reranking failed: {exc}")
            rows.append({
                "model": "cross_encoder_hybrid",
                "status": "failed",
                "error": str(exc),
            })

    return rows


def save_optimisation_report(
    rows: List[Dict[str, Any]],
    dataset_summary: Mapping[str, Any],
    output_dir: Optional[Path] = None,
    k: int = DEFAULT_TOP_K,
    stem: str = "optimisation_eval",
) -> Dict[str, Path]:
    """Write results to JSON and CSV, same contract as ``save_report`` in baselines."""
    output_dir = Path(output_dir) if output_dir else Path(OUTPUTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_summary.get("dataset", "scidocs"),
        "split": dataset_summary.get("split", "test"),
        "k": k,
        "protocol": (
            "BEIR SciDocs test split; EvaluateRetrieval.evaluate "
            "(pytrec_eval NDCG, MAP, Recall, P); ignore_identical_ids=True. "
            "Dense retrieval via VectorDBRetriever (FAISS). "
            "Ranking does not use qrels."
        ),
        "dataset_summary": dict(dataset_summary),
        "models": rows,
    }
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}
