"""
BASE MODELS PACKAGE
Scope: Baseline retrieval systems (TF-IDF, BM25, optional dense embeddings) on SciDocs.

Dense retrieval here is exact in-memory cosine search. Vector-database / ANN
indexing remains the responsibility of ``src/vectordb``.
"""

from src.base_models.base import BaseRetriever
from src.base_models.bm25 import BM25Retriever, tokenize
from src.base_models.dense import DenseRetriever
from src.base_models.tfidf import TfidfRetriever
from src.base_models.utils import ranked_ids

__all__ = [
    "BaseRetriever",
    "TfidfRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "ranked_ids",
    "tokenize",
]
