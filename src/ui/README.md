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
- Comparison dashboard: seven approaches (BM25, TF-IDF, Dense FAISS, Citation-Boosted Hybrid, Cross-Encoder, Reciprocal Rank Fusion, Weighted Score Fusion)
- Section 1 (Search & Controls): Single query search, Top-10 / Top-50 cutoffs, method selector
- Section 2 (Method Results): Method-aware score breakdowns, academic explanation, formulas, latency, citation graph, similar papers
- Section 3 (Benchmark Comparison): Offline SciDocs test-split metrics (nDCG@10, MAP@10, Recall@10, P@10, latency) vs live latency with multi-metric comparison charts
- Section 4 (Retrieval Overlap): 7×7 Jaccard similarity matrix (% shared), degree-of-sharing distribution (UpSet-style sharing breakdown), unique papers, and multi-method intersection
- Section 5 (Ranking Analysis): Slope/bump rank comparison chart with broken-line fix across unretrieved methods, paper highlighting selector, Spearman ρ and Kendall τ rank correlation heatmaps with common-count matrix, and automated query insights
- Query-level caching in session state avoids re-encoding queries, re-invoking models, or rebuilding FAISS indexes when switching approaches or cutoffs
- Missing component scores stay as dashes (—), indicating absence from the candidate pool rather than a score of zero

## Tests

```bash
.venv/bin/python -m pytest tests/ui -q
```

Tests use a tiny synthetic corpus and fake retrievers. They do not download
SciDocs or model weights.
