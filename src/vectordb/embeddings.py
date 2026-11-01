"""
Turn documents into vectors, and remember them on disk.

Encoding the whole corpus is the slowest step, so we do it once and save it.
Cached vectors are L2-normalised float32, so inner product == cosine similarity.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from src.base_models.utils import Corpus, corpus_texts
from src.vectordb.constants import (
    CACHE_KEY_LENGTH,
    EMBEDDING_DTYPE,
    EMBEDDINGS_FILE_SUFFIX,
    ENCODE_BATCH_SIZE,
    IDS_FILE_SUFFIX,
)


def to_float32(matrix: Any) -> np.ndarray:
    """Convert input to a 2D C-contiguous float32 array for FAISS."""
    array = np.asarray(matrix, dtype=EMBEDDING_DTYPE)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array)


def normalize_rows(matrix: Any) -> np.ndarray:
    """L2-normalize each row to unit length (zero rows remain zero)."""
    array = to_float32(matrix)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return to_float32(array / norms)


def _split_into_batches(items: Sequence[str], batch_size: int) -> List[Sequence[str]]:
    """Split items into batches of up to batch_size."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def encode_texts(encoder: Any, texts: Sequence[str], batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """Encode texts in batches and return normalized vector embeddings."""
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=EMBEDDING_DTYPE)
    parts = [np.asarray(encoder.encode(list(batch))) for batch in _split_into_batches(texts, batch_size)]
    return normalize_rows(np.vstack(parts))


def encode_queries(encoder: Any, queries: Sequence[str]) -> np.ndarray:
    """Encode query strings into normalized vector embeddings."""
    return encode_texts(encoder, list(queries))


def build_cache_key(model_name: str, doc_ids: Sequence[str], texts: Sequence[str]) -> str:
    """Compute a short hash of the model name and corpus content for cache invalidation."""
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    for doc_id, text in zip(doc_ids, texts):
        hasher.update(b"\x00" + str(doc_id).encode("utf-8"))
        hasher.update(b"\x01" + text.encode("utf-8"))
    return hasher.hexdigest()[:CACHE_KEY_LENGTH]


def get_cache_paths(cache_dir: Path, cache_key: str) -> Tuple[Path, Path]:
    """Return the vector (.npy) and ID (.json) file paths for a cache key."""
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"{cache_key}{EMBEDDINGS_FILE_SUFFIX}",
        cache_dir / f"{cache_key}{IDS_FILE_SUFFIX}",
    )


def save_embeddings(cache_dir: Path, cache_key: str, doc_ids: Sequence[str], embeddings: np.ndarray) -> None:
    """Write embeddings and corresponding doc IDs to disk cache."""
    vectors_path, ids_path = get_cache_paths(cache_dir, cache_key)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, embeddings)
    ids_path.write_text(json.dumps(list(doc_ids)), encoding="utf-8")


def load_embeddings(cache_dir: Path, cache_key: str):
    """Load cached vectors and doc IDs, returning None if missing or corrupted."""
    vectors_path, ids_path = get_cache_paths(cache_dir, cache_key)
    if not (vectors_path.is_file() and ids_path.is_file()):
        return None
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    embeddings = to_float32(np.load(vectors_path))
    if len(doc_ids) != embeddings.shape[0]:
        return None
    return doc_ids, embeddings


def get_or_create_embeddings(
    corpus: Corpus,
    encoder: Any,
    model_name: str,
    cache_dir: Path,
    batch_size: int = ENCODE_BATCH_SIZE,
    use_cache: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """Retrieve embeddings from disk cache if available, or compute and cache them."""
    doc_ids, texts = corpus_texts(corpus)
    if not doc_ids:
        return [], np.zeros((0, 0), dtype=EMBEDDING_DTYPE)

    cache_key = build_cache_key(model_name, doc_ids, texts)
    if use_cache:
        cached = load_embeddings(cache_dir, cache_key)
        if cached is not None:
            return cached

    embeddings = encode_texts(encoder, texts, batch_size)
    if use_cache:
        save_embeddings(cache_dir, cache_key, doc_ids, embeddings)
    return doc_ids, embeddings
