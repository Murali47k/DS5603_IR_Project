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
    """
    Description: Make a 2D, C-ordered float32 array (FAISS needs exactly this).
    Inputs: matrix (anything array-like). Reads global EMBEDDING_DTYPE.
    Outputs: 2D float32 numpy array. No globals changed.
    Dependencies: numpy.
    Utilities: called by normalize_rows(), index_builder functions, retriever.
    """
    array = np.asarray(matrix, dtype=EMBEDDING_DTYPE)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.ascontiguousarray(array)


def normalize_rows(matrix: Any) -> np.ndarray:
    """
    Description: Scale every row to length 1 (all-zero rows are left as zeros).
    Inputs: matrix (anything array-like).
    Outputs: normalised float32 array. No globals changed.
    Dependencies: to_float32().
    Utilities: called by encode_texts(), get_or_create_embeddings().
    """
    array = to_float32(matrix)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return to_float32(array / norms)


def _split_into_batches(items: Sequence[str], batch_size: int) -> List[Sequence[str]]:
    """
    Description: Cut a list into chunks of at most batch_size items.
    Inputs: items, batch_size (must be > 0).
    Outputs: list of chunks. No globals changed.
    Dependencies: none.
    Utilities: called by encode_texts().
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def encode_texts(encoder: Any, texts: Sequence[str], batch_size: int = ENCODE_BATCH_SIZE) -> np.ndarray:
    """
    Description: Encode texts batch by batch and return normalised vectors.
    Inputs: encoder (object with encode(list) -> array), texts, batch_size.
    Outputs: array of shape (len(texts), dim). No globals changed.
    Dependencies: _split_into_batches(), normalize_rows().
    Utilities: called by get_or_create_embeddings(), encode_queries().
    """
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=EMBEDDING_DTYPE)
    parts = [np.asarray(encoder.encode(list(batch))) for batch in _split_into_batches(texts, batch_size)]
    return normalize_rows(np.vstack(parts))


def encode_queries(encoder: Any, queries: Sequence[str]) -> np.ndarray:
    """
    Description: Encode query texts into normalised vectors.
    Inputs: encoder, queries.
    Outputs: array of shape (len(queries), dim).
    Dependencies: encode_texts().
    Utilities: called by VectorDBRetriever.retrieve(), benchmark.
    """
    return encode_texts(encoder, list(queries))


def build_cache_key(model_name: str, doc_ids: Sequence[str], texts: Sequence[str]) -> str:
    """
    Description: Make a short hash from model name + ids + texts. If any of them
        change, the key changes, so an old cache is never reused by mistake.
    Inputs: model_name, doc_ids, texts. Reads global CACHE_KEY_LENGTH.
    Outputs: hex string. No globals changed.
    Dependencies: hashlib.
    Utilities: called by get_or_create_embeddings().
    """
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    for doc_id, text in zip(doc_ids, texts):
        hasher.update(b"\x00" + str(doc_id).encode("utf-8"))
        hasher.update(b"\x01" + text.encode("utf-8"))
    return hasher.hexdigest()[:CACHE_KEY_LENGTH]


def get_cache_paths(cache_dir: Path, cache_key: str) -> Tuple[Path, Path]:
    """
    Description: Work out the two file paths used for one cache entry.
    Inputs: cache_dir, cache_key. Reads globals EMBEDDINGS_FILE_SUFFIX, IDS_FILE_SUFFIX.
    Outputs: (vectors_path, ids_path).
    Dependencies: pathlib.
    Utilities: called by save_embeddings(), load_embeddings(), get_or_create_embeddings().
    """
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"{cache_key}{EMBEDDINGS_FILE_SUFFIX}",
        cache_dir / f"{cache_key}{IDS_FILE_SUFFIX}",
    )


def save_embeddings(cache_dir: Path, cache_key: str, doc_ids: Sequence[str], embeddings: np.ndarray) -> None:
    """
    Description: Write vectors (.npy) and their doc IDs (.json) to disk.
    Inputs: cache_dir, cache_key, doc_ids, embeddings.
    Outputs: nothing. Creates two files on disk.
    Dependencies: get_cache_paths().
    Utilities: called by get_or_create_embeddings().
    """
    vectors_path, ids_path = get_cache_paths(cache_dir, cache_key)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, embeddings)
    ids_path.write_text(json.dumps(list(doc_ids)), encoding="utf-8")


def load_embeddings(cache_dir: Path, cache_key: str):
    """
    Description: Read a cache entry back. Returns None if it is missing or broken.
    Inputs: cache_dir, cache_key.
    Outputs: (doc_ids, embeddings) or None.
    Dependencies: get_cache_paths(), to_float32().
    Utilities: called by get_or_create_embeddings().
    """
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
    """
    Description: Return (doc_ids, vectors) for the corpus. Loads from disk if the
        same corpus + model was encoded before; otherwise encodes and saves.
    Inputs: corpus, encoder, model_name, cache_dir, batch_size, use_cache.
    Outputs: (sorted doc_ids, normalised float32 vectors). Writes cache files.
    Dependencies: corpus_texts(), build_cache_key(), load_embeddings(),
        encode_texts(), save_embeddings().
    Utilities: called by VectorDBRetriever.from_corpus(), benchmark.
    """
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
