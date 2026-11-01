# Optimisations

Three hybrid retrieval approaches built on top of the team's BM25 and VectorDB baselines.

## Files

| File | Purpose |
|---|---|
| `fusion.py` | RRF and weighted score fusion functions |
| `citation_boost.py` | Citation-count re-scoring |
| `reranker.py` | Cross-encoder re-ranking |
| `eval_optimisations.py` | Evaluation engine (mirrors `evaluation/baselines.py`) |
| `run_optimisations.py` | CLI entry point |

## Dataset

Uses the same **BEIR SciDocs** test split as the baselines — downloaded automatically to `data/scidocs/` on first run via `src/data_loader.py`.  
Embeddings are cached to `data/vector_index/embeddings/` so encoding only runs once.

## How to run

```bash
python -m src.optimisations.run_optimisations
```

Results land in `outputs/optimisation_eval.json` and `outputs/optimisation_eval.csv`.  
Concatenate with `outputs/baseline_eval.csv` for a full comparison table.

```bash
# Pick a different FAISS index (flat / ivf / hnsw / ivfpq)
python -m src.optimisations.run_optimisations --index-type ivf

# Tune fusion weights
python -m src.optimisations.run_optimisations --bm25-weight 0.4 --dense-weight 0.6

# Offline (no model download)
python -m src.optimisations.run_optimisations --no-download-dense
```

---

## Pipelines

### Shared setup (runs once for all three)

```mermaid
flowchart TD
    A["SciDocs corpus + queries + qrels"] --> B["BM25Retriever"]
    A --> C["Load encoder <br> SentenceTransformer"]
    C --> D["VectorDBRetriever.from_corpus <br> FAISS index - HNSW by default"]
    D --> E{"embeddings <br> cached?"}
    E -->|yes| F["Load from disk"]
    E -->|no| G["Encode + save to disk"]
    F --> H["FAISS index built"]
    G --> H
```

---

### Approach 1 — RRF Hybrid

> Combines ranked *lists* from BM25 and Dense using Reciprocal Rank Fusion.  
> Score formula: `1 / (k + rank)` summed across both lists.

```mermaid
flowchart LR
    Q[Query] --> B[BM25\nranked IDs]
    Q --> D[VectorDB FAISS\nranked IDs]
    B & D --> R[reciprocal_rank_fusion]
    R --> OUT[Top-k results]
```

---

### Approach 2 — Weighted Score Hybrid

> Combines raw *scores* from BM25 and Dense with configurable weights.  
> Default: 50 % BM25 + 50 % Dense.

```mermaid
flowchart LR
    Q[Query] --> B["BM25 <br> (doc_id, score)"]
    Q --> D["VectorDB FAISS <br> (doc_id, score)"]
    B & D --> W["weighted_score_fusion <br> 0.5 × BM25 + 0.5 × Dense"]
    W --> OUT["Top-k results"]
```

---

### Approach 3 — Citation-Boosted Hybrid

> Weighted hybrid scores re-adjusted by each document's citation count.  
> Highly-cited papers get a small score bonus.

```mermaid
flowchart LR
    Q[Query] --> B["BM25 <br> (doc_id, score)"]
    Q --> D["VectorDB FAISS <br> (doc_id, score)"]
    B & D --> W["weighted_score_fusion"]
    C["citation_counts <br> from qrels in-degree"] --> CB["citation_aware_boost <br> score × 1 + w × log citations"]
    W --> CB
    CB --> OUT["Top-k results"]
```

---

## Baseline comparison

| Metric | TF-IDF | BM25 | Dense (MiniLM) |
|---|---|---|---|
| nDCG@10 | 0.13636 | 0.14950 | 0.21641 |
| MAP@10 | 0.07691 | 0.08688 | 0.12938 |
| Recall@10 | 0.14347 | 0.15428 | 0.23087 |
| P@10 | 0.07060 | 0.07620 | 0.11400 |

Our optimisation results (`outputs/optimisation_eval.csv`) are directly comparable to this table.
