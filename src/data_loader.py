"""
=============================================================================
DATA LOADER MODULE 
=============================================================================
Purpose:
- Download the BEIR 'scidocs' dataset if not present.
- Load corpus, queries, and qrels into standard dictionaries.
- Parse paper metadata (title, text/abstract, citations, metadata).
=============================================================================
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from beir import util
from beir.datasets.data_loader import GenericDataLoader

SCIDOCS_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip"
)

# <project_root>/data  (this file lives in <project_root>/src/)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_scidocs_dataset(data_dir: Optional[str] = None) -> str:
    """
    Download and unzip SciDocs into `data_dir/scidocs` if it isn't there yet.
    Returns the path to the extracted dataset folder.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "scidocs"
    if (dataset_path / "corpus.jsonl").exists():
        return str(dataset_path)

    # Downloads data/scidocs.zip, extracts to data/scidocs/
    return util.download_and_unzip(SCIDOCS_URL, str(data_dir))


def load_scidocs(
    data_dir: Optional[str] = None, split: str = "test"
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load SciDocs (downloading it first if needed).

    Returns:
       - corpus: doc_id -> {'title': str, 'text': str, ...}
       - queries: query_id -> query_text
       - qrels: query_id -> {doc_id: relevance_score}
    """
    dataset_path = download_scidocs_dataset(data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split=split)
    return corpus, queries, qrels


def get_document_citations(
    corpus: Dict[str, Dict[str, Any]],
    qrels: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, int]:
    """
    Return doc_id -> citation_count.

    BEIR's SciDocs has NO citation-count field, so:
    1. If a doc has a 'citations' / 'citation_count' value in its metadata, use it.
    2. Otherwise, if `qrels` is given, use in-degree: the number of queries
       (papers) that cite the doc (relevance > 0). This is a proxy computed
       only over the loaded split, not the true global citation count.
    3. Docs with neither get 0.
    """
    counts: Dict[str, int] = {doc_id: 0 for doc_id in corpus}

    # Fallback: in-degree from the qrels graph
    if qrels:
        for doc_scores in qrels.values():
            for doc_id, score in doc_scores.items():
                if score > 0 and doc_id in counts:
                    counts[doc_id] += 1

    # Preferred: explicit metadata if the corpus provides it
    for doc_id, doc in corpus.items():
        meta = doc.get("metadata") or {}
        for key in ("citation_count", "citations"):
            value = meta.get(key, doc.get(key))
            if isinstance(value, int):
                counts[doc_id] = value
                break
            if isinstance(value, (list, tuple)):
                counts[doc_id] = len(value)
                break

    return counts


if __name__ == "__main__":
    corpus, queries, qrels = load_scidocs()
    citations = get_document_citations(corpus, qrels)
    print(f"Docs: {len(corpus)} | Queries: {len(queries)} | Qrels: {len(qrels)}")
    first_id = next(iter(corpus))
    print(first_id, corpus[first_id]["title"], "| citations:", citations[first_id])