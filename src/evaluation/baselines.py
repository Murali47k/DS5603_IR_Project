"""Reproducible baseline evaluation on BEIR SciDocs (test split)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from config import DATA_DIR, DEFAULT_TOP_K, DENSE_MODEL_NAME, OUTPUTS_DIR
from src.base_models import BM25Retriever, DenseRetriever, TfidfRetriever
from src.base_models.base import BaseRetriever
from src.data_loader import load_scidocs
from src.evaluation.metrics import (
    evaluate_beir_results,
    metrics_at_k,
    ranked_hits_to_beir_results,
)

FALLBACK_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LATENCY_DEFINITION = (
    "Indexing time is wall-clock time of retriever construction "
    "(TF-IDF fit, BM25 tokenisation, or dense model load + corpus encoding) "
    "measured with time.perf_counter. Query latency is the per-query wall-clock "
    "time of retriever.retrieve(query_text, k), also via time.perf_counter, "
    "averaged over evaluation queries. Metric computation is excluded. "
    "Qrels are never passed to the retriever."
)


def describe_dataset(
    corpus: Mapping[str, Any],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Any]:
    """Sanity-check BEIR SciDocs dictionaries before evaluation."""
    if not corpus or not queries or not qrels:
        raise ValueError("corpus, queries, and qrels must be non-empty")

    doc_id = next(iter(corpus))
    query_id = next(iter(queries))
    doc = corpus[doc_id]
    if not isinstance(doc, dict) or ("title" not in doc and "text" not in doc):
        raise ValueError("corpus values must be dicts with title and/or text")
    if not isinstance(queries[query_id], str):
        raise ValueError("queries must map query_id -> str")
    rels = qrels[query_id] if query_id in qrels else next(iter(qrels.values()))
    if not isinstance(rels, dict) or not all(isinstance(v, int) for v in rels.values()):
        raise ValueError("qrels must map query_id -> {doc_id: int relevance}")
    if set(qrels) != set(queries):
        raise ValueError("query IDs and qrel IDs must match for the evaluation split")

    n_qrel_pairs = sum(len(r) for r in qrels.values())
    n_positive_pairs = sum(
        1 for rels in qrels.values() for score in rels.values() if int(score) > 0
    )
    n_queries_in_corpus = sum(1 for qid in queries if qid in corpus)
    return {
        "n_documents": len(corpus),
        "n_queries": len(queries),
        "n_qrel_queries": len(qrels),
        "n_qrel_pairs": n_qrel_pairs,
        "n_positive_pairs": n_positive_pairs,
        "n_queries_in_corpus": n_queries_in_corpus,
        "sample_doc_id": str(doc_id),
        "sample_query_id": str(query_id),
        "sample_doc_keys": sorted(str(k) for k in doc.keys()),
        "query_id_in_corpus": query_id in corpus,
        "mean_relevant_per_query": round(n_positive_pairs / len(qrels), 3),
    }


def load_official_scidocs(
    data_dir: Optional[str] = None,
    split: str = "test",
) -> tuple[Dict[str, Any], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Load the BEIR SciDocs split via the project data loader (no extra download if present)."""
    corpus, queries, qrels = load_scidocs(
        data_dir=data_dir or str(DATA_DIR),
        split=split,
    )
    summary = describe_dataset(corpus, queries, qrels)
    summary["split"] = split
    summary["dataset"] = "scidocs"
    return corpus, queries, qrels, summary


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def retrieve_queries(
    retriever: BaseRetriever,
    queries: Mapping[str, str],
    k: int,
) -> tuple[Dict[str, Dict[str, float]], List[float]]:
    """Run retrieval for every query. Does not use qrels."""
    per_query_hits = {}
    latencies: List[float] = []
    items = list(queries.items())
    try:
        from tqdm import tqdm

        items = tqdm(items, desc="queries", leave=False)
    except ImportError:
        pass
    for query_id, query_text in items:
        start = time.perf_counter()
        hits = retriever.retrieve(query_text, k=k)
        latencies.append(time.perf_counter() - start)
        per_query_hits[query_id] = hits
    return ranked_hits_to_beir_results(per_query_hits), latencies


def evaluate_retriever(
    retriever: BaseRetriever,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
) -> tuple[Dict[str, float], List[float]]:
    results, latencies = retrieve_queries(retriever, queries, k=k)
    metrics = metrics_at_k(
        evaluate_beir_results(qrels, results, k_values=(k,)),
        k=k,
    )
    return metrics, latencies


def latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
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
        "query_latency_p50_ms": round(percentile(ms, 50), 4),
        "query_latency_p95_ms": round(percentile(ms, 95), 4),
        "query_latency_total_seconds": round(sum(latencies), 4),
    }


def load_dense_encoder(model_name: str, local_files_only: bool = False):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


def try_build_dense_retriever(
    corpus: Mapping[str, Any],
    model_name: str = DENSE_MODEL_NAME,
    fallback_model: str = FALLBACK_DENSE_MODEL,
    allow_download: bool = True,
) -> tuple[Optional[DenseRetriever], Dict[str, Any]]:
    """
    Build DenseRetriever without failing the lexical baselines.

    Tries the requested model locally, then the fallback locally, then
    (optionally) downloads the fallback.
    """
    attempts: List[str] = []
    candidates = [
        (model_name, True),
        (fallback_model, True),
    ]
    if allow_download:
        candidates.extend(
            [
                (model_name, False),
                (fallback_model, False),
            ]
        )

    seen = set()
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
            retriever = DenseRetriever(corpus, model_name=name, encoder=encoder)
            index_seconds = time.perf_counter() - index_start
            return retriever, {
                "status": "ok",
                "dense_model": name,
                "model_load_seconds": round(model_load_seconds, 4),
                "index_seconds": round(index_seconds, 4),
                "attempts": attempts + [f"{name} ({mode}): ok"],
            }
        except Exception as exc:  # noqa: BLE001 - must not abort lexical eval
            attempts.append(f"{name} ({mode}): {type(exc).__name__}: {exc}")

    return None, {
        "status": "skipped",
        "dense_model": model_name,
        "error": " ; ".join(attempts) if attempts else "no dense model configured",
        "attempts": attempts,
        "model_load_seconds": None,
        "index_seconds": None,
    }


def run_lexical_model(
    name: str,
    factory: Callable[[], BaseRetriever],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int,
) -> Dict[str, Any]:
    start = time.perf_counter()
    retriever = factory()
    index_seconds = time.perf_counter() - start
    metrics, latencies = evaluate_retriever(retriever, queries, qrels, k=k)
    row = {
        "model": name,
        "status": "ok",
        "index_seconds": round(index_seconds, 4),
        "model_load_seconds": 0.0,
        **metrics,
        **latency_stats(latencies),
        "error": "",
    }
    return row


def run_baseline_suite(
    corpus: Mapping[str, Any],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K,
    dense_model: str = DENSE_MODEL_NAME,
    dense_fallback: str = FALLBACK_DENSE_MODEL,
    allow_download: bool = True,
    run_dense: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        run_lexical_model(
            "tfidf",
            lambda: TfidfRetriever(corpus),
            queries,
            qrels,
            k=k,
        )
    )
    rows.append(
        run_lexical_model(
            "bm25",
            lambda: BM25Retriever(corpus),
            queries,
            qrels,
            k=k,
        )
    )
    if not run_dense:
        rows.append(
            {
                "model": "dense",
                "status": "skipped",
                "error": "dense evaluation disabled",
                "dense_model": dense_model,
            }
        )
        return rows

    retriever, meta = try_build_dense_retriever(
        corpus,
        model_name=dense_model,
        fallback_model=dense_fallback,
        allow_download=allow_download,
    )
    if retriever is None:
        rows.append({"model": "dense", **meta})
        return rows

    metrics, latencies = evaluate_retriever(retriever, queries, qrels, k=k)
    rows.append(
        {
            "model": "dense",
            "status": "ok",
            "dense_model": meta.get("dense_model"),
            "index_seconds": meta.get("index_seconds"),
            "model_load_seconds": meta.get("model_load_seconds"),
            **metrics,
            **latency_stats(latencies),
            "error": "",
        }
    )
    return rows


def save_report(
    rows: List[Dict[str, Any]],
    dataset_summary: Mapping[str, Any],
    output_dir: Optional[Path] = None,
    k: int = DEFAULT_TOP_K,
    stem: str = "baseline_eval",
) -> Dict[str, Path]:
    output_dir = Path(output_dir) if output_dir else Path(OUTPUTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_summary.get("dataset", "scidocs"),
        "split": dataset_summary.get("split", "test"),
        "k": k,
        "protocol": (
            "BEIR SciDocs test split; EvaluateRetrieval.evaluate "
            "(pytrec_eval NDCG, MAP, Recall, P); ignore_identical_ids=True. "
            "Ranking does not use qrels."
        ),
        "latency_definition": LATENCY_DEFINITION,
        "dataset_summary": dict(dataset_summary),
        "models": rows,
    }
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}
