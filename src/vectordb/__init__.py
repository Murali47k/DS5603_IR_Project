"""
VECTOR DATABASE PACKAGE
Scope: Vector indexing, storage, low-latency approximate nearest neighbor (ANN) search, and metadata management.

Files:
    constants.py     settings and index names
    embeddings.py    encode the corpus and cache vectors on disk
    index_builder.py build Flat / IVF / HNSW / IVF-PQ FAISS indexes
    store.py         save / load an index with its doc-ID map
    retriever.py     VectorDBRetriever (same output as the other retrievers)
    benchmark.py     speed vs accuracy comparison of the index types
"""

from src.vectordb.index_builder import build_index, set_search_params
from src.vectordb.retriever import VectorDBRetriever
from src.vectordb.store import load_index, save_index

__all__ = [
    "VectorDBRetriever",
    "build_index",
    "set_search_params",
    "save_index",
    "load_index",
]
