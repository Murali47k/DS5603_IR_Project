"""
Save and load an index together with its ID map.

FAISS only knows positions 0, 1, 2, ... so we keep a list that turns each
position back into the real document ID. Both must be saved together.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss

from src.vectordb.constants import IDS_FILE_NAME, INDEX_FILE_NAME, META_FILE_NAME
from src.vectordb.index_builder import set_search_params


def get_index_paths(index_dir: Path) -> Tuple[Path, Path, Path]:
    """Return paths to the index, IDs, and metadata files in the directory."""
    index_dir = Path(index_dir)
    return index_dir / INDEX_FILE_NAME, index_dir / IDS_FILE_NAME, index_dir / META_FILE_NAME


def check_ids_match_index(index: faiss.Index, doc_ids: Sequence[str]) -> None:
    """Validate that the count of doc IDs matches the number of vectors in the index."""
    if index.ntotal != len(doc_ids):
        raise ValueError(f"Index has {index.ntotal} vectors but {len(doc_ids)} doc IDs")


def measure_index_bytes(index: faiss.Index) -> int:
    """Return the serialized byte size of the FAISS index."""
    return int(faiss.serialize_index(index).nbytes)


def save_index(
    index: faiss.Index,
    doc_ids: Sequence[str],
    index_type: str,
    index_dir: Path,
    search_params: Optional[Dict[str, Any]] = None,
    model_name: str = "",
) -> Path:
    """Serialize FAISS index, document ID mapping, and metadata to disk."""
    check_ids_match_index(index, doc_ids)
    index_path, ids_path, meta_path = get_index_paths(index_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(list(doc_ids)), encoding="utf-8")
    meta = {
        "index_type": index_type,
        "model_name": model_name,
        "n_vectors": int(index.ntotal),
        "dim": int(index.d),
        "search_params": search_params or {},
        "index_bytes": measure_index_bytes(index),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Path(index_dir)


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[str], Dict[str, Any]]:
    """Load serialized FAISS index, document ID list, and metadata from disk."""
    index_path, ids_path, meta_path = get_index_paths(index_dir)
    if not (index_path.is_file() and ids_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError(f"No saved index found in {index_dir}")
    index = faiss.read_index(str(index_path))
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_ids_match_index(index, doc_ids)
    set_search_params(index, meta["index_type"], **meta.get("search_params", {}))
    return index, doc_ids, meta
