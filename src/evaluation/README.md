# Baseline evaluation (BEIR SciDocs)

Evaluates the TF-IDF, BM25, and optional dense retrievers on the **official
BEIR SciDocs test split**.

## Protocol

- Dataset: BEIR SciDocs, `split="test"` via `src.data_loader.load_scidocs`.
- Queries: `queries.jsonl` (query ID → query text). Qrels: `qrels/test.tsv`.
- Ranking never uses qrels.
- Metrics: BEIR `EvaluateRetrieval.evaluate` (pytrec_eval) with
  `ignore_identical_ids=True`.
- Cut-off: k=10 → nDCG@10, MAP@10, Recall@10, P@10.
- Indexing/encoding time is measured separately from query latency. Query
  latency is mean `time.perf_counter()` around `retrieve(query, k)` only.

If `data/scidocs/corpus.jsonl` already exists, it is reused and not downloaded
again.

## Run SciDocs

From the repository root, using the project virtualenv:

```bash
.venv/bin/python -m src.evaluation.run_baselines \
  --dense-model sentence-transformers/all-MiniLM-L6-v2
```

Dense checkpoints are configurable:

```bash
.venv/bin/python -m src.evaluation.run_baselines \
  --dense-model sentence-transformers/all-MiniLM-L6-v2
```

`--skip-dense` or `--no-download-dense` leaves lexical baselines running if a
model cannot be loaded. Errors are stored in the report.

Results are written to `outputs/baseline_eval.json` and
`outputs/baseline_eval.csv` (gitignored).

## Tests (no SciDocs download)

```bash
.venv/bin/python -m pytest tests/evaluation tests/base_models tests/optimisations/test_fusion.py tests/optimisations/test_citation_boost.py -q
```
