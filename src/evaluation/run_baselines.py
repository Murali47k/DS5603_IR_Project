"""CLI: evaluate TF-IDF, BM25, and optional dense retrieval on BEIR SciDocs."""

from __future__ import annotations

import argparse
import json
import sys

from config import DATA_DIR, DEFAULT_TOP_K, DENSE_MODEL_NAME, OUTPUTS_DIR
from src.evaluation.baselines import (
    FALLBACK_DENSE_MODEL,
    LATENCY_DEFINITION,
    load_official_scidocs,
    run_baseline_suite,
    save_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate lexical and optional dense baselines on the official "
            "BEIR SciDocs test split (nDCG@k, MAP@k, Recall@k, P@k)."
        )
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--split", default="test", help="BEIR split (SciDocs is test-only).")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-model", default=DENSE_MODEL_NAME)
    parser.add_argument("--dense-fallback", default=FALLBACK_DENSE_MODEL)
    parser.add_argument(
        "--no-download-dense",
        action="store_true",
        help="Do not download a dense checkpoint; skip dense if it is not cached.",
    )
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--stem", default="baseline_eval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus, queries, qrels, summary = load_official_scidocs(
        data_dir=args.data_dir,
        split=args.split,
    )
    print("SciDocs split summary:")
    print(json.dumps(summary, indent=2))
    print(LATENCY_DEFINITION)

    rows = run_baseline_suite(
        corpus,
        queries,
        qrels,
        k=args.k,
        dense_model=args.dense_model,
        dense_fallback=args.dense_fallback,
        allow_download=not args.no_download_dense,
        run_dense=not args.skip_dense,
    )
    paths = save_report(
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
