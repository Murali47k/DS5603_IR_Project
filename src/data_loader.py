"""
=============================================================================
DATA LOADER MODULE (Shared Utility)
Maintained by: Shared / Person 1
=============================================================================
Purpose:
- Download the BEIR 'scidocs' dataset if not present.
- Load corpus, queries, and qrels into standard dictionaries.
- Parse paper metadata (title, text/abstract, citations, metadata).
=============================================================================
"""

from typing import Dict, Tuple, Any

def download_scidocs_dataset(data_dir: str = "./data") -> str:
    """
    TODO:
    1. Check if dataset already exists at data_dir / 'scidocs'.
    2. If not, use BEIR's `util.download_and_unzip(url, data_dir)`.
    3. Return the directory path of the extracted dataset.
    """
    # NOTE: DO NOT IMPLEMENT CODE YET. Implement when team starts sprint.
    pass


def load_scidocs(data_dir: str = "./data", split: str = "test") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    TODO:
    1. Use `GenericDataLoader(data_folder=dataset_path).load(split=split)` from BEIR.
    2. Returns:
       - corpus: dict mapping doc_id -> {'title': str, 'text': str, 'metadata': dict}
       - queries: dict mapping query_id -> query_text
       - qrels: dict mapping query_id -> {doc_id: relevance_score}
    """
    # NOTE: DO NOT IMPLEMENT CODE YET.
    pass


def get_document_citations(corpus: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """
    TODO:
    1. Parse citation information / counts from corpus metadata if available.
    2. If SciDocs metadata does not have direct citation counts, compute graph in-degree
       or mock/extract from metadata dictionary.
    3. Return a mapping: doc_id -> citation_count.
    """
    # NOTE: DO NOT IMPLEMENT CODE YET.
    pass
