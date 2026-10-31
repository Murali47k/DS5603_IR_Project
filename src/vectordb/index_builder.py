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
    """
    Description: Read the vector size from the data (do not trust config.EMBEDDING_DIM,
        which may not match the model: SPECTER=768, MiniLM=384).
    Inputs: embeddings (2D array).
    Outputs: int dimension.
    Dependencies: none.
    Utilities: called by every build_* function.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty 2D array")
    return int(embeddings.shape[1])


def _safe_nlist(n_vectors: int, nlist: int) -> int:
    """
    Description: Never ask for more clusters than there are vectors.
    Inputs: n_vectors, nlist.
    Outputs: a valid nlist (at least 1).
    Dependencies: none.
    Utilities: called by build_ivf(), build_ivfpq().
    """
    return max(1, min(nlist, n_vectors))


def _safe_pq_m(dim: int, pq_m: int) -> int:
    """
    Description: PQ needs dim to divide evenly by m. Pick the biggest m <= pq_m that does.
    Inputs: dim, pq_m.
    Outputs: valid m (at least 1).
    Dependencies: none.
    Utilities: called by build_ivfpq().
    """
    for candidate in range(min(pq_m, dim), 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _make_quantizer(dim: int) -> faiss.IndexFlatIP:
    """
    Description: The small exact index IVF uses to find the nearest cluster centres.
    Inputs: dim.
    Outputs: IndexFlatIP.
    Dependencies: faiss.
    Utilities: called by build_ivf(), build_ivfpq().
    """
    return faiss.IndexFlatIP(dim)


def build_flat(embeddings: np.ndarray) -> faiss.Index:
    """
    Description: Exact search index. Slow on big data, but always correct.
    Inputs: embeddings.
    Outputs: filled IndexFlatIP.
    Dependencies: get_embedding_dim(), to_float32().
    Utilities: called by build_index().
    """
    data = to_float32(embeddings)
    index = faiss.IndexFlatIP(get_embedding_dim(data))
    index.add(data)
    return index


def build_ivf(embeddings: np.ndarray, nlist: int = IVF_NLIST, nprobe: int = IVF_NPROBE) -> faiss.Index:
    """
    Description: Cluster the vectors into nlist groups. A query only looks into
        nprobe groups, so it is faster but may miss some answers.
    Inputs: embeddings, nlist, nprobe.
    Outputs: trained and filled IndexIVFFlat.
    Dependencies: _safe_nlist(), _make_quantizer(), set_search_params().
    Utilities: called by build_index().
    """
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
    """
    Description: Build a graph where close vectors are linked. Search walks the graph.
    Inputs: embeddings, m (links per node), ef_construction, ef_search.
    Outputs: filled IndexHNSWFlat.
    Dependencies: get_embedding_dim(), set_search_params().
    Utilities: called by build_index().
    """
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
    """
    Description: IVF plus compressed vectors. Uses much less memory; scores are approximate.
    Inputs: embeddings, nlist, nprobe, pq_m, pq_nbits.
    Outputs: trained and filled IndexIVFPQ.
    Dependencies: _safe_nlist(), _safe_pq_m(), _make_quantizer(), set_search_params().
    Utilities: called by build_index().
    """
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
    """
    Description: Build any supported index by name.
    Inputs: index_type (one of ALL_INDEX_TYPES), embeddings, build_params (passed on).
        Reads global INDEX_BUILDERS.
    Outputs: a filled FAISS index.
    Dependencies: INDEX_BUILDERS functions.
    Utilities: called by VectorDBRetriever.from_corpus(), benchmark.
    """
    if index_type not in INDEX_BUILDERS:
        raise ValueError(f"Unknown index_type '{index_type}'. Choose from {ALL_INDEX_TYPES}")
    return INDEX_BUILDERS[index_type](embeddings, **build_params)


def set_search_params(
    index: faiss.Index,
    index_type: str,
    nprobe: Optional[int] = None,
    ef_search: Optional[int] = None,
) -> None:
    """
    Description: Change how hard the index searches. Bigger = slower but more accurate.
        Settings that do not apply to this index type are ignored.
    Inputs: index, index_type, nprobe (IVF/IVFPQ), ef_search (HNSW).
    Outputs: nothing. Changes the index object in place.
    Dependencies: faiss.
    Utilities: called by build_ivf(), build_hnsw(), build_ivfpq(), store.load_index(), benchmark.
    """
    if index_type in (INDEX_IVF, INDEX_IVFPQ) and nprobe is not None:
        index.nprobe = int(nprobe)
    if index_type == INDEX_HNSW and ef_search is not None:
        index.hnsw.efSearch = int(ef_search)
