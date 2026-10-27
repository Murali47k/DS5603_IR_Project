# Base retrieval models

Lexical (and optional dense) baselines for SciDocs. Hybrid fusion, citation
boosting, reranking, evaluation, and vector databases are implemented in other
packages.

## Models

| Class | Method | Indexing |
| --- | --- | --- |
| `TfidfRetriever` | scikit-learn TF-IDF + cosine similarity | Document-term matrix built once in `__init__` |
| `BM25Retriever` | `rank_bm25.BM25Okapi` | Documents tokenised once in `__init__` |
| `DenseRetriever` (optional) | Pretrained sentence embeddings + cosine similarity | Document vectors encoded once in `__init__` |

`DenseRetriever` is an in-memory exact-search baseline. Do not treat it as a
replacement for FAISS / Chroma work in `src/vectordb`.

## Corpus format

Same dictionaries as `src.data_loader.load_scidocs`:

```python
corpus: dict[str, dict]   # doc_id -> {"title": str, "text": str, ...}
queries: dict[str, str]   # query_id -> query text
```

Raw strings are also accepted as document values. Original document IDs are
preserved in the results; they are never replaced by positional indices.

## Retrieval format

```python
from src.data_loader import load_scidocs
from src.base_models import TfidfRetriever, BM25Retriever

corpus, queries, qrels = load_scidocs()  # qrels are evaluation-only; not used here

tfidf = TfidfRetriever(corpus)
bm25 = BM25Retriever(corpus)  # k1 / b default to config.BM25_K1 / BM25_B

query = queries[next(iter(queries))]
hits = bm25.retrieve(query, k=10)
# -> [("doc_id", score), ...] sorted by descending score, ties by doc_id

ids = bm25.retrieve_ids(query, k=10)  # for reciprocal_rank_fusion
```

- Empty or whitespace-only queries return `[]`.
- An empty corpus returns `[]`.
- If `k` is larger than the corpus, every document is returned (except for
  empty queries).
- `k` must be a positive integer.

These tuples match `weighted_score_fusion` and `citation_aware_boost`. Ranked
ID lists match `reciprocal_rank_fusion`.

## Optional dense baseline

```python
from src.base_models import DenseRetriever

# Downloads a pretrained checkpoint on first use (not used in unit tests).
dense = DenseRetriever(corpus)  # model_name defaults to config.DENSE_MODEL_NAME
hits = dense.retrieve(query, k=10)
```

Pass `encoder=` (any object with `encode(texts) -> ndarray`) to avoid loading a
pretrained model.

## Tests

From the repository root (so `config` and `src` import cleanly):

```bash
python -m pytest tests/base_models tests/optimisations/test_fusion.py tests/optimisations/test_citation_boost.py -q
```

Or run a file directly:

```bash
PYTHONPATH=. python tests/base_models/test_retrievers.py
```

Tests use a tiny synthetic corpus. They do not download SciDocs or any
pretrained model.
