"""BEIR/pytrec_eval metric helpers for baseline retrieval."""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from beir.retrieval.evaluation import EvaluateRetrieval

RankedHits = List[Tuple[str, float]]
BeirResults = Dict[str, Dict[str, float]]
Qrels = Mapping[str, Mapping[str, int]]


def ranked_hits_to_beir_results(
    per_query_hits: Mapping[str, Iterable[Tuple[str, float]]],
) -> BeirResults:
    """Convert retriever output to the BEIR results dict: query_id -> {doc_id: score}."""
    results: BeirResults = {}
    for query_id, hits in per_query_hits.items():
        results[str(query_id)] = {str(doc_id): float(score) for doc_id, score in hits}
    return results


def evaluate_beir_results(
    qrels: Qrels,
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int] = (10,),
    ignore_identical_ids: bool = True,
) -> Dict[str, float]:
    """
    Official BEIR evaluation: nDCG@k, MAP@k, Recall@k, P@k via pytrec_eval.

    ``results`` is copied because ``EvaluateRetrieval.evaluate`` may drop
    documents whose ID equals the query ID.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")
    for k in k_values:
        if int(k) <= 0:
            raise ValueError("k_values must be positive")

    qrels_dict = {
        str(qid): {str(doc_id): int(rel) for doc_id, rel in rels.items()}
        for qid, rels in qrels.items()
    }
    results_copy: BeirResults = {
        str(qid): {str(doc_id): float(score) for doc_id, score in hits.items()}
        for qid, hits in results.items()
    }

    ndcg, mapped, recall, precision = EvaluateRetrieval.evaluate(
        qrels_dict,
        results_copy,
        k_values=list(k_values),
        ignore_identical_ids=ignore_identical_ids,
    )
    merged = {}
    merged.update(ndcg)
    merged.update(mapped)
    merged.update(recall)
    merged.update(precision)
    return merged


def metrics_at_k(metrics: Mapping[str, float], k: int) -> Dict[str, float]:
    """Select the four official BEIR cut-off metrics at k."""
    return {
        f"NDCG@{k}": float(metrics[f"NDCG@{k}"]),
        f"MAP@{k}": float(metrics[f"MAP@{k}"]),
        f"Recall@{k}": float(metrics[f"Recall@{k}"]),
        f"P@{k}": float(metrics[f"P@{k}"]),
    }
