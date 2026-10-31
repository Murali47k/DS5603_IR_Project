"""
Build and save the vector database for SciDocs (command line tool).

Example:
    python -m src.vectordb.build_db --model sentence-transformers/all-MiniLM-L6-v2 --index-type hnsw

What it does:
    1. Loads SciDocs.
    2. Encodes every paper into a vector (cached on disk, so the next run is fast).
    3. Builds the FAISS index.
    4. Saves the index + doc-ID map to data/vector_index/<model>_<index type>/.
    5. Runs a small demo search so you can see it works.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from config import DATA_DIR, DENSE_MODEL_NAME, VECTOR_INDEX_DIR
from src.vectordb.constants import ALL_INDEX_TYPES, EMBEDDINGS_SUBDIR, INDEX_HNSW
from src.vectordb.retriever import VectorDBRetriever

DEMO_QUERY_COUNT = 3     # how many example queries to show after building
DEMO_TOP_K = 5           # how many results to show per example query
TITLE_MAX_CHARS = 80     # long titles are cut to this length when printing


def make_model_slug(model_name: str) -> str:
    """
    Description: Turn a model name into a safe folder name ("a/b" -> "a_b").
    Inputs: model_name.
    Outputs: string.
    Dependencies: none.
    Utilities: called by get_index_dir().
    """
    return model_name.replace("/", "_").replace("\\", "_")


def get_index_dir(index_type: str, model_name: str, base_dir: Path = VECTOR_INDEX_DIR) -> Path:
    """
    Description: Folder where this model + index type is saved.
        Different models make different vectors, so each gets its own folder.
    Inputs: index_type, model_name, base_dir (default config.VECTOR_INDEX_DIR).
    Outputs: Path.
    Dependencies: make_model_slug().
    Utilities: called by main(); also use it when loading later.
    """
    return Path(base_dir) / f"{make_model_slug(model_name)}_{index_type}"


def format_hit_line(rank: int, doc_id: str, score: float, corpus: Mapping[str, Any]) -> str:
    """
    Description: One printable line for a search result.
    Inputs: rank, doc_id, score, corpus (to look up the title). Reads TITLE_MAX_CHARS.
    Outputs: string.
    Dependencies: none.
    Utilities: called by print_demo_results().
    """
    title = str(corpus.get(doc_id, {}).get("title", ""))[:TITLE_MAX_CHARS]
    return f"  {rank}. {score:.4f}  {doc_id}  {title}"


def print_demo_results(
    retriever: VectorDBRetriever, corpus: Mapping[str, Any], queries: Mapping[str, str]
) -> None:
    """
    Description: Search a few real queries and print the results.
    Inputs: retriever, corpus, queries. Reads DEMO_QUERY_COUNT, DEMO_TOP_K.
    Outputs: prints to the screen.
    Dependencies: format_hit_line(), retriever.retrieve().
    Utilities: called by main().
    """
    for query_id in list(queries)[:DEMO_QUERY_COUNT]:
        print(f"\nQuery {query_id}: {queries[query_id][:TITLE_MAX_CHARS]}")
        for rank, (doc_id, score) in enumerate(retriever.retrieve(queries[query_id], k=DEMO_TOP_K), 1):
            print(format_hit_line(rank, doc_id, score, corpus))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Description: Read command line options.
    Inputs: argv (optional). Reads global defaults.
    Outputs: Namespace.
    Dependencies: argparse.
    Utilities: called by main().
    """
    parser = argparse.ArgumentParser(description="Build and save the SciDocs vector database.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model", default=DENSE_MODEL_NAME)
    parser.add_argument("--index-type", default=INDEX_HNSW, choices=ALL_INDEX_TYPES)
    parser.add_argument("--output-dir", default=None, help="default: data/vector_index/<model>_<index type>")
    parser.add_argument("--no-cache", action="store_true", help="re-encode even if vectors are cached")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Description: Load data, build the index, save it, and run the demo search.
    Inputs: argv.
    Outputs: exit code 0. Writes the embedding cache and the saved index folder.
    Dependencies: load_official_scidocs(), load_dense_encoder(),
        VectorDBRetriever.from_corpus(), retriever.save(), print_demo_results().
    Utilities: command line entry point.
    """
    from src.evaluation.baselines import load_dense_encoder, load_official_scidocs

    args = parse_args(argv)
    corpus, queries, _qrels, _summary = load_official_scidocs(data_dir=args.data_dir)
    encoder = load_dense_encoder(args.model)

    start = time.perf_counter()
    retriever = VectorDBRetriever.from_corpus(
        corpus, encoder, index_type=args.index_type, model_name=args.model,
        cache_dir=Path(VECTOR_INDEX_DIR) / EMBEDDINGS_SUBDIR, use_cache=not args.no_cache,
    )
    print(f"Built '{args.index_type}' index over {len(retriever.doc_ids)} docs in {time.perf_counter() - start:.1f}s")

    out_dir = Path(args.output_dir) if args.output_dir else get_index_dir(args.index_type, args.model)
    retriever.save(out_dir)
    print(f"Saved to {out_dir}")

    print_demo_results(retriever, corpus, queries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
