"""
OPTIMISATIONS & HYBRID MODELS PACKAGE
Scope: Hybrid search (RRF / Weighted Score Fusion), Cross-Encoder Re-ranking, and Citation-Aware Boosting.
Evaluation of all three approaches lives in eval_optimisations.py / run_optimisations.py.
"""

from src.optimisations.eval_optimisations import (
    evaluate_citation_hybrid,
    evaluate_rrf_hybrid,
    evaluate_weighted_hybrid,
    run_optimisation_suite,
    save_optimisation_report,
)

__all__ = [
    "evaluate_rrf_hybrid",
    "evaluate_weighted_hybrid",
    "evaluate_citation_hybrid",
    "run_optimisation_suite",
    "save_optimisation_report",
]
