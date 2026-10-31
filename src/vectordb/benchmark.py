"""
Benchmark the index types: speed vs accuracy.

For each index we record: build time, search time per query, size in bytes,
nDCG/MAP/Recall/P @k, and "recall vs exact" (how many of the exact top-k
the approximate index also found).

Query encoding is done ONCE, outside the timer. Otherwise the slow encoder
would hide the real difference between the indexes.
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, DEFAULT_TOP_K, DENSE_MODEL_NAME, OUTPUTS_DIR, VECTOR_INDEX_DIR
from src.vectordb.constants import (
    BENCHMARK_PLAN,
    BENCHMARK_STEM,
    EMBEDDINGS_SUBDIR,
    INDEX_FLAT,
    INDEX_IVFPQ,
)
from src.vectordb.embeddings import encode_queries, get_or_create_embeddings
from src.vectordb.index_builder import build_index, set_search_params
from src.vectordb.retriever import hits_from_search_row
from src.vectordb.store import measure_index_bytes


def compute_overlap_recall(
    exact_ids: Sequence[Sequence[str]], approx_ids: Sequence[Sequence[str]], k: int
) -> float:
    """
    Description: Average share of the exact top-k that the approximate index also returned.
    Inputs: exact_ids and approx_ids (one list of doc IDs per query), k.
    Outputs: float between 0 and 1.
    Dependencies: none.
    Utilities: called by run_one_setting().
    """
    if not exact_ids:
        return 0.0
    shares = []
    for exact, approx in zip(exact_ids, approx_ids):
        wanted = set(exact[:k])
        shares.append(len(wanted & set(approx[:k])) / len(wanted) if wanted else 1.0)
    return float(statistics.fmean(shares))


def search_one_by_one(
    index: Any, query_vectors: np.ndarray, doc_ids: Sequence[str], k: int
) -> Tuple[List[List[Tuple[str, float]]], List[float]]:
    """
    Description: Search each query separately and time only index.search().
    Inputs: index, query_vectors (n, dim), doc_ids, k.
    Outputs: (hits per query, seconds per query).
    Dependencies: hits_from_search_row().
    Utilities: called by run_one_setting().
    """
    all_hits, latencies = [], []
    k_search = min(k, len(doc_ids))
    for row in query_vectors:
        start = time.perf_counter()
        scores, positions = index.search(row.reshape(1, -1), k_search)
        latencies.append(time.perf_counter() - start)
        all_hits.append(hits_from_search_row(scores[0], positions[0], doc_ids, k))
    return all_hits, latencies


def score_hits(
    query_ids: Sequence[str], all_hits: Sequence[Sequence[Tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]], k: int,
) -> Dict[str, float]:
    """
    Description: nDCG/MAP/Recall/P @k using the same BEIR code as the baselines.
    Inputs: query_ids, all_hits (same order), qrels, k.
    Outputs: dict like {"NDCG@10": ...}.
    Dependencies: src.evaluation.metrics.
    Utilities: called by run_one_setting().
    """
    from src.evaluation.metrics import evaluate_beir_results, metrics_at_k, ranked_hits_to_beir_results

    results = ranked_hits_to_beir_results(dict(zip(query_ids, all_hits)))
    return metrics_at_k(evaluate_beir_results(qrels, results, k_values=(k,)), k=k)


def latency_summary_ms(latencies: Sequence[float]) -> Dict[str, float]:
    """
    Description: Mean and 95th percentile of latencies, in milliseconds.
    Inputs: latencies in seconds.
    Outputs: dict with mean and p95.
    Dependencies: numpy.
    Utilities: called by run_one_setting().
    """
    if not latencies:
        return {"latency_mean_ms": 0.0, "latency_p95_ms": 0.0}
    ms = np.asarray(latencies) * 1000.0
    return {
        "latency_mean_ms": round(float(ms.mean()), 4),
        "latency_p95_ms": round(float(np.percentile(ms, 95)), 4),
    }


def run_one_setting(
    name: str, index: Any, index_type: str, search_params: Dict[str, Any], build_seconds: float,
    query_ids: Sequence[str], query_vectors: np.ndarray, doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]], exact_ids: Sequence[Sequence[str]], k: int,
) -> Dict[str, Any]:
    """
    Description: Apply search settings, run all queries, and build one result row.
    Inputs: see signature. Search settings are applied to the index.
    Outputs: dict row for the report.
    Dependencies: set_search_params(), search_one_by_one(), score_hits(),
        compute_overlap_recall(), latency_summary_ms(), measure_index_bytes().
    Utilities: called by run_benchmark().
    """
    set_search_params(index, index_type, **search_params)
    all_hits, latencies = search_one_by_one(index, query_vectors, doc_ids, k)
    approx_ids = [[doc_id for doc_id, _ in hits] for hits in all_hits]
    return {
        "name": name,
        "index_type": index_type,
        "search_params": json.dumps(search_params),
        "build_seconds": round(build_seconds, 4),
        "index_mb": round(measure_index_bytes(index) / 1e6, 3),
        f"recall_vs_exact@{k}": round(compute_overlap_recall(exact_ids, approx_ids, k), 5),
        **latency_summary_ms(latencies),
        **score_hits(query_ids, all_hits, qrels, k),
    }


def make_setting_name(index_type: str, search_params: Mapping[str, Any]) -> str:
    """
    Description: Readable row name, e.g. "ivf nprobe=4".
    Inputs: index_type, search_params.
    Outputs: string.
    Dependencies: none.
    Utilities: called by run_benchmark().
    """
    suffix = " ".join(f"{key}={value}" for key, value in search_params.items())
    return f"{index_type} {suffix}".strip()


def compute_exact_top_ids(
    embeddings: np.ndarray, query_vectors: np.ndarray, doc_ids: Sequence[str], k: int
) -> List[List[str]]:
    """
    Description: The true top-k for every query, found with the exact Flat index.
    Inputs: embeddings, query_vectors, doc_ids, k.
    Outputs: list of ID lists.
    Dependencies: build_index(), search_one_by_one().
    Utilities: called by run_benchmark().
    """
    flat = build_index(INDEX_FLAT, embeddings)
    hits, _ = search_one_by_one(flat, query_vectors, doc_ids, k)
    return [[doc_id for doc_id, _ in row] for row in hits]


def run_benchmark(
    doc_ids: Sequence[str], embeddings: np.ndarray, query_ids: Sequence[str],
    query_vectors: np.ndarray, qrels: Mapping[str, Mapping[str, int]],
    k: int = DEFAULT_TOP_K, plan: Sequence = BENCHMARK_PLAN, skip_pq: bool = False,
) -> List[Dict[str, Any]]:
    """
    Description: Build each index in the plan once, then try each search setting.
    Inputs: vectors, queries, qrels, k, plan (default BENCHMARK_PLAN), skip_pq.
    Outputs: list of result rows.
    Dependencies: compute_exact_top_ids(), build_index(), run_one_setting(), make_setting_name().
    Utilities: called by main().
    """
    exact_ids = compute_exact_top_ids(embeddings, query_vectors, doc_ids, k)
    rows = []
    for index_type, build_params, search_settings in plan:
        if skip_pq and index_type == INDEX_IVFPQ:
            continue
        start = time.perf_counter()
        index = build_index(index_type, embeddings, **build_params)
        build_seconds = time.perf_counter() - start
        for search_params in search_settings:
            rows.append(run_one_setting(
                make_setting_name(index_type, search_params), index, index_type, search_params,
                build_seconds, query_ids, query_vectors, doc_ids, qrels, exact_ids, k,
            ))
    return rows


def save_benchmark_report(rows: List[Dict[str, Any]], output_dir: Path, stem: str = BENCHMARK_STEM) -> Dict[str, Path]:
    """
    Description: Write the rows to JSON and CSV.
    Inputs: rows, output_dir, stem.
    Outputs: {"json": path, "csv": path}. Creates two files.
    Dependencies: json, pandas.
    Utilities: called by main().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / f"{stem}.json", output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}


def parse_args(argv=None) -> argparse.Namespace:
    """
    Description: Read command line options.
    Inputs: argv (optional list). Reads globals for defaults.
    Outputs: Namespace.
    Dependencies: argparse.
    Utilities: called by main().
    """
    parser = argparse.ArgumentParser(description="Benchmark FAISS index types on SciDocs.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model", default=DENSE_MODEL_NAME)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-queries", type=int, default=0, help="0 = use all queries (use a small number for a quick test)")
    parser.add_argument("--skip-pq", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Description: Load SciDocs, encode (cached), run the benchmark, save and print the report.
    Inputs: argv.
    Outputs: exit code 0. Writes outputs/vectordb_benchmark.json/.csv and the embedding cache.
    Dependencies: load_official_scidocs(), load_dense_encoder(), get_or_create_embeddings(),
        encode_queries(), run_benchmark(), save_benchmark_report().
    Utilities: command line entry point.
    """
    from src.evaluation.baselines import load_dense_encoder, load_official_scidocs

    args = parse_args(argv)
    corpus, queries, qrels, _ = load_official_scidocs(data_dir=args.data_dir)
    encoder = load_dense_encoder(args.model)
    cache_dir = Path(VECTOR_INDEX_DIR) / EMBEDDINGS_SUBDIR
    doc_ids, embeddings = get_or_create_embeddings(corpus, encoder, args.model, cache_dir)

    query_ids = list(queries)[: args.max_queries or None]
    query_vectors = encode_queries(encoder, [queries[qid] for qid in query_ids])
    rows = run_benchmark(doc_ids, embeddings, query_ids, query_vectors, qrels, k=args.k, skip_pq=args.skip_pq)

    paths = save_benchmark_report(rows, Path(args.output_dir))
    print(pd.DataFrame(rows).drop(columns=["search_params"]).to_string(index=False))
    print(f"Wrote {paths['json']} and {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
