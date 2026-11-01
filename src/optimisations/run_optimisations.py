"""CLI: evaluate hybrid optimisation approaches on BEIR SciDocs.

Usage examples:

    # Full run with defaults (HNSW index, dense model cached or downloaded)
    python -m src.optimisations.run_optimisations

    # Skip downloading the dense model
    python -m src.optimisations.run_optimisations --no-download-dense

    # Choose a different FAISS index type
    python -m src.optimisations.run_optimisations --index-type ivf

    # Custom weights and output location
    python -m src.optimisations.run_optimisations \\
        --bm25-weight 0.4 --dense-weight 0.6 \\
        --citation-boost-weight 0.15 \\
        --output-dir outputs/

Results are printed as JSON and saved as both JSON and CSV so they can be
compared directly against outputs/baseline_eval.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    DENSE_MODEL_NAME,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    OUTPUTS_DIR,
    RRF_K,
)
from src.evaluation.baselines import FALLBACK_DENSE_MODEL, load_official_scidocs
from src.vectordb.constants import ALL_INDEX_TYPES, INDEX_HNSW
from src.optimisations.eval_optimisations import (
    run_optimisation_suite,
    save_optimisation_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Pre-parse --config to load YAML defaults if available
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file.")
    conf_args, _ = conf_parser.parse_known_args(argv)

    yaml_config = {}
    config_path = Path(conf_args.config)
    if config_path.is_file():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    # Replace hyphen with underscore for argparse compatibility
                    yaml_config = {k.replace("-", "_"): v for k, v in loaded.items()}
        except Exception as exc:
            print(f"Warning: Could not read config file {config_path}: {exc}")

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hybrid optimisation approaches on BEIR SciDocs "
            "(RRF Hybrid, Weighted Hybrid, Citation-Boosted Hybrid)."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file (default: config.yaml).")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--split", default="test", help="BEIR split (SciDocs is test-only).")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-model", default=DENSE_MODEL_NAME)
    parser.add_argument("--dense-fallback", default=FALLBACK_DENSE_MODEL)
    parser.add_argument(
        "--no-download-dense",
        action="store_true",
        help="Do not download a dense checkpoint; skip if not cached.",
    )
    parser.add_argument("--candidate-k", type=int, default=50, help="Initial candidate pool retrieved per method before fusion (default: 50).")
    parser.add_argument("--bm25-weight", type=float, default=HYBRID_BM25_WEIGHT)
    parser.add_argument("--dense-weight", type=float, default=HYBRID_DENSE_WEIGHT)
    parser.add_argument("--no-normalize", action="store_true", help="Disable Min-Max score normalization in weighted fusion.")
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--citation-boost-weight", type=float, default=0.1)
    parser.add_argument("--citation-file", default=None, help="Path to JSON file with doc_id -> citation count mapping.")
    parser.add_argument(
        "--allow-qrels-citations",
        action="store_true",
        help="Fallback to qrels in-degree for citation counts (WARNING: introduces test leakage on test split).",
    )
    parser.add_argument(
        "--include-reranker",
        action="store_true",
        help="Evaluate Approach 4: Cross-Encoder reranking over hybrid candidates.",
    )
    parser.add_argument("--reranker-model", default=None, help="Hugging Face checkpoint for Cross-Encoder reranker.")
    parser.add_argument(
        "--index-type", default=INDEX_HNSW, choices=ALL_INDEX_TYPES,
        help="FAISS index type used by VectorDBRetriever (default: hnsw).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Re-encode the corpus even if embeddings are already cached on disk.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--stem", default="optimisation_eval")

    if yaml_config:
        parser.set_defaults(**yaml_config)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    corpus, queries, qrels, summary = load_official_scidocs(
        data_dir=args.data_dir,
        split=args.split,
    )
    print("SciDocs split summary:")
    print(json.dumps(summary, indent=2))

    rows = run_optimisation_suite(
        corpus,
        queries,
        qrels,
        k=args.k,
        candidate_k=args.candidate_k,
        dense_model=args.dense_model,
        dense_fallback=args.dense_fallback,
        index_type=args.index_type,
        allow_download=not args.no_download_dense,
        use_cache=not args.no_cache,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        normalize_scores=not args.no_normalize,
        rrf_k=args.rrf_k,
        citation_boost_weight=args.citation_boost_weight,
        citation_file=args.citation_file,
        allow_qrels_citations=args.allow_qrels_citations,
        include_reranker=args.include_reranker,
        reranker_model=args.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    paths = save_optimisation_report(
        rows,
        summary,
        output_dir=args.output_dir,
        k=args.k,
        stem=args.stem,
    )

    print(json.dumps(rows, indent=2))
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
