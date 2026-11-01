# DS5603 Information Retrieval Project: Scientific Document Retrieval System

## Repository Structure

```
.
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── base_models/
│   │   └── README.md          # TF-IDF, BM25, optional dense baselines
│   ├── evaluation/
│   ├── optimisations/
│   ├── ui/
│   ├── vectordb/
│   └── data_loader.py
├── tests/                     # pytest-style unit tests
├── .gitignore
├── config.py
├── README.md
└── requirements.txt
```

## Base retrieval models

See [`src/base_models/README.md`](src/base_models/README.md) for corpus format, retrieval output, and examples.

Quick start from the repository root:

```python
from src.data_loader import load_scidocs
from src.base_models import TfidfRetriever, BM25Retriever

corpus, queries, qrels = load_scidocs()
retriever = BM25Retriever(corpus)
hits = retriever.retrieve(queries[next(iter(queries))], k=10)
# hits: list of (doc_id, score)
```
## To run all test

```bash
pytest -v
```

## Baseline SciDocs evaluation

See [`src/evaluation/README.md`](src/evaluation/README.md). From the repo root:

```bash
venv/bin/python -m src.evaluation.run_baselines
```

Uses the local `data/scidocs` test split when present. Writes `outputs/baseline_eval.json` and `.csv`.

## Hybrid & Optimisation Evaluation

See [`src/optimisations/README.md`](src/optimisations/README.md). To run the full hybrid evaluation:

```bash
venv/bin/python -m src.optimisations.run_optimisations
```

Configuration defaults are maintained in [`config.yaml`](config.yaml).

## Benchmark Results (BEIR SciDocs Test Split, $k=10$)

| Model | Architecture | nDCG@10 | MAP@10 | Recall@10 | P@10 | Mean Latency |
|:---|:---|:---|:---|:---|:---|:---|
| **TF-IDF** | Lexical Baseline (scikit-learn) | 0.13636 | 0.07691 | 0.14347 | 0.07060 | 4.8 ms |
| **BM25** | Lexical Baseline (rank-bm25) | 0.14950 | 0.08688 | 0.15428 | 0.07620 | 1.8 ms |
| **Dense (MiniLM)** | Exhaustive Dot-Product (all-MiniLM-L6-v2) | 0.21641 | 0.12938 | 0.23087 | 0.11400 | ~120 ms |
| **RRF Hybrid** | Reciprocal Rank Fusion ($K_{\text{cand}}=50$) | 0.19782 | 0.11774 | 0.20922 | 0.10330 | 134.0 ms |
| **Weighted Hybrid** | Normalized Score Fusion (0.4 BM25 + 0.6 Dense) | **0.21340** | **0.12852** | **0.22782** | **0.11250** | 137.8 ms |
| **Citation Hybrid** | Leak-Free Citation-Boosted Hybrid | **0.21340** | **0.12852** | **0.22782** | **0.11250** | 143.1 ms |