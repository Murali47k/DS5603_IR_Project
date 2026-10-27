"""
VALIDATION & BENCHMARKING
Scope: System evaluation (NDCG@k, MAP), computational cost profiling, and comparison reports.

Baseline SciDocs evaluation lives in ``baselines.py`` / ``run_baselines.py``.
Hybrid fusion and citation boosting are evaluated separately by those modules' owners.
"""

from src.evaluation.baselines import (
    LATENCY_DEFINITION,
    describe_dataset,
    evaluate_retriever,
    load_official_scidocs,
    run_baseline_suite,
    save_report,
)
from src.evaluation.metrics import (
    evaluate_beir_results,
    metrics_at_k,
    ranked_hits_to_beir_results,
)

__all__ = [
    "LATENCY_DEFINITION",
    "describe_dataset",
    "evaluate_beir_results",
    "evaluate_retriever",
    "load_official_scidocs",
    "metrics_at_k",
    "ranked_hits_to_beir_results",
    "run_baseline_suite",
    "save_report",
]
