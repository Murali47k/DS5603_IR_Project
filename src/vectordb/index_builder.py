"""
Build FAISS indexes (Flat, IVF, HNSW, IVF-PQ) from normalised vectors.

All indexes use INNER PRODUCT. With normalised vectors this equals cosine
similarity, and a higher score means a better match.
"""

from typing import Any, Callable, Dict, Optional

import faiss
import numpy as np

from src.vectordb.constants import (
    ALL_INDEX_TYPES,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_FLAT,
    INDEX_HNSW,
    INDEX_IVF,
    INDEX_IVFPQ,
    IVF_NLIST,
    IVF_NPROBE,
    PQ_M,
    PQ_NBITS,
)
from src.vectordb.embeddings import to_float32


def get_embedding_dim(embeddings: np.ndarray) -> int:
    """Return the embedding dimension from a 2D embeddings array."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    return int(embeddings.shape[1])


def _safe_nlist(n_vectors: int, nlist: int) -> int:
    """Bound cluster count to at most the number of vectors, minimum 1."""
    return max(1, min(nlist, n_vectors))


def _safe_pq_m(dim: int, pq_m: int) -> int:
    """Find the largest divisor of dim that is <= pq_m."""
    for candidate in range(min(pq_m, dim), 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _make_quantizer(dim: int) -> faiss.IndexFlatIP:
    """Create a flat inner-product quantizer for coarse cluster centroids."""
    return faiss.IndexFlatIP(dim)


def build_flat(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index for exact search."""
    data = to_float32(embeddings)
    index = faiss.IndexFlatIP(get_embedding_dim(data))
    index.add(data)
    return index


def build_ivf(embeddings: np.ndarray, nlist: int = IVF_NLIST, nprobe: int = IVF_NPROBE) -> faiss.Index:
    """Build an IVF flat index partitioning vectors into clusters."""
    data = to_float32(embeddings)
    dim = get_embedding_dim(data)
    nlist = _safe_nlist(data.shape[0], nlist)
    index = faiss.IndexIVFFlat(_make_quantizer(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(data)
    index.add(data)
    set_search_params(index, INDEX_IVF, nprobe=nprobe)
    return index


def build_hnsw(
    embeddings: np.ndarray,
    m: int = HNSW_M,
    ef_construction: int = HNSW_EF_CONSTRUCTION,
    ef_search: int = HNSW_EF_SEARCH,
) -> faiss.Index:
    """Build an HNSW graph index for fast approximate nearest neighbor search."""
    data = to_float32(embeddings)
    index = faiss.IndexHNSWFlat(get_embedding_dim(data), m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    set_search_params(index, INDEX_HNSW, ef_search=ef_search)
    return index


def build_ivfpq(
    embeddings: np.ndarray,
    nlist: int = IVF_NLIST,
    nprobe: int = IVF_NPROBE,
    pq_m: int = PQ_M,
    pq_nbits: int = PQ_NBITS,
) -> faiss.Index:
    """Build an IVF-PQ index with product quantization for reduced memory usage."""
    data = to_float32(embeddings)
    dim = get_embedding_dim(data)
    nlist = _safe_nlist(data.shape[0], nlist)
    index = faiss.IndexIVFPQ(
        _make_quantizer(dim), dim, nlist, _safe_pq_m(dim, pq_m), pq_nbits, faiss.METRIC_INNER_PRODUCT
    )
    index.train(data)
    index.add(data)
    set_search_params(index, INDEX_IVFPQ, nprobe=nprobe)
    return index


# Which function builds which index type.
INDEX_BUILDERS: Dict[str, Callable[..., faiss.Index]] = {
    INDEX_FLAT: build_flat,
    INDEX_IVF: build_ivf,
    INDEX_HNSW: build_hnsw,
    INDEX_IVFPQ: build_ivfpq,
}


def build_index(index_type: str, embeddings: np.ndarray, **build_params: Any) -> faiss.Index:
    """Construct a FAISS index of the specified type."""
    if index_type not in INDEX_BUILDERS:
        raise ValueError(f"Unknown index_type '{index_type}'. Choose from {ALL_INDEX_TYPES}")
    return INDEX_BUILDERS[index_type](embeddings, **build_params)


def set_search_params(
    index: faiss.Index,
    index_type: str,
    nprobe: Optional[int] = None,
    ef_search: Optional[int] = None,
) -> None:
    """Update runtime search parameters (nprobe for IVF, efSearch for HNSW)."""
    if index_type in (INDEX_IVF, INDEX_IVFPQ) and nprobe is not None:
        index.nprobe = int(nprobe)
    if index_type == INDEX_HNSW and ef_search is not None:
        index.hnsw.efSearch = int(ef_search)
