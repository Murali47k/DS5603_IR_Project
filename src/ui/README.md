# SciDocs search UI

Streamlit app for citation-boosted hybrid search over BEIR SciDocs.

Default ranking matches the strongest reported optimisation result:

`BM25 + FAISS dense → weighted_score_fusion → citation_aware_boost (w = 0.4)`

## Launch

From the repository root:

```bash
.venv/bin/streamlit run src/ui/app.py
```

The engine loads SciDocs once, reuses the BM25 index, and loads a saved FAISS
index from `data/vector_index/` when present. Qrels are never used for ranking.

## Features

- Paper cards: title, authors, year, abstract snippet, citation-count badge
- Score breakdown: BM25, dense/FAISS, fused hybrid, citation multiplier
- Citation graph: cited-by and references for the selected paper
- Find similar papers from the current paper vector (top-5 neighbours)
- Sidebar: citation-boost weight, year range, minimum citations, result count
- Comparison mode: BM25 vs dense vs citation hybrid

## Tests

```bash
.venv/bin/python -m pytest tests/ui -q
```

Tests use a tiny synthetic corpus and fake retrievers. They do not download
SciDocs or model weights.
