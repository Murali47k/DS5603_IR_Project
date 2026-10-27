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

```bash
python -m pytest tests/base_models tests/evaluation tests/optimisations/test_fusion.py tests/optimisations/test_citation_boost.py -q
```

## Baseline SciDocs evaluation

See [`src/evaluation/README.md`](src/evaluation/README.md). From the repo root:

```bash
.venv/bin/python -m src.evaluation.run_baselines
```

Uses the local `data/scidocs` test split when present. Writes `outputs/baseline_eval.json` and `.csv` (gitignored).