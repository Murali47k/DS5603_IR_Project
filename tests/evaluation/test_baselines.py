"""Synthetic tests for BEIR-style metric calculation and result handling.

Does not download SciDocs or pretrained models.
"""

from pathlib import Path

from src.base_models import TfidfRetriever
from src.evaluation.baselines import (
    describe_dataset,
    evaluate_retriever,
    run_baseline_suite,
    save_report,
    try_build_dense_retriever,
)
from src.evaluation.metrics import (
    evaluate_beir_results,
    metrics_at_k,
    ranked_hits_to_beir_results,
)


def test_ranked_hits_preserve_query_and_doc_ids():
    hits = {
        "query-paper": [("doc-alpha", 1.5), ("doc-beta", 0.25)],
    }
    results = ranked_hits_to_beir_results(hits)
    assert set(results) == {"query-paper"}
    assert results["query-paper"]["doc-alpha"] == 1.5
    assert results["query-paper"]["doc-beta"] == 0.25


def test_perfect_ranking_metrics_at_10():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    results = {"q1": {"d1": 2.0, "d2": 1.0, "d3": 0.1}}
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(10,)), k=10)
    assert metrics["NDCG@10"] == 1.0
    assert metrics["MAP@10"] == 1.0
    assert metrics["Recall@10"] == 1.0
    assert metrics["P@10"] == 0.2


def test_empty_ranking_has_zero_metrics():
    qrels = {"q1": {"d1": 1}}
    results = {"q1": {}}
    metrics = metrics_at_k(evaluate_beir_results(qrels, results, k_values=(10,)), k=10)
    assert metrics["NDCG@10"] == 0.0
    assert metrics["MAP@10"] == 0.0
    assert metrics["Recall@10"] == 0.0
    assert metrics["P@10"] == 0.0


def test_identical_query_doc_id_is_ignored():
    qrels = {"doc-self": {"doc-rel": 1}}
    results = {"doc-self": {"doc-self": 9.0, "doc-rel": 1.0}}
    metrics = metrics_at_k(
        evaluate_beir_results(qrels, results, k_values=(10,), ignore_identical_ids=True),
        k=10,
    )
    assert metrics["Recall@10"] == 1.0
    assert metrics["NDCG@10"] == 1.0


def test_describe_dataset_counts_positive_qrels():
    corpus = {"d1": {"title": "Cats", "text": "yarn"}, "d2": {"title": "Dogs", "text": "bark"}}
    queries = {"q1": "cats"}
    qrels = {"q1": {"d1": 1, "d2": 0}}
    summary = describe_dataset(corpus, queries, qrels)
    assert summary["n_qrel_pairs"] == 2
    assert summary["n_positive_pairs"] == 1
    assert summary["mean_relevant_per_query"] == 1.0


def test_describe_dataset_rejects_mismatched_qrels():
    corpus = {"d1": {"title": "Cats", "text": "yarn"}}
    queries = {"q1": "cats"}
    qrels = {"q2": {"d1": 1}}
    try:
        describe_dataset(corpus, queries, qrels)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_retriever_does_not_need_full_scidocs():
    corpus = {
        "doc-cats": {"title": "Domestic Cats", "text": "Cats sit on mats and chase yarn."},
        "doc-dogs": {"title": "Domestic Dogs", "text": "Dogs chase balls and bark loudly."},
        "doc-quantum": {"title": "Quantum Circuits", "text": "Qubits and entanglement."},
    }
    queries = {"q-cats": "cats on mats"}
    qrels = {"q-cats": {"doc-cats": 1}}
    retriever = TfidfRetriever(corpus)
    metrics, latencies = evaluate_retriever(retriever, queries, qrels, k=10)
    assert len(latencies) == 1
    assert latencies[0] >= 0.0
    assert metrics["Recall@10"] == 1.0
    assert metrics["NDCG@10"] > 0.0


def test_save_report_writes_json_and_csv(tmp_path: Path):
    rows = [
        {
            "model": "tfidf",
            "status": "ok",
            "NDCG@10": 0.5,
            "MAP@10": 0.4,
            "Recall@10": 0.3,
            "P@10": 0.2,
            "index_seconds": 0.01,
            "query_latency_mean_ms": 1.2,
        }
    ]
    summary = {"dataset": "synthetic", "split": "test", "n_queries": 1}
    paths = save_report(rows, summary, output_dir=tmp_path, k=10, stem="unit")
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    text = paths["json"].read_text(encoding="utf-8")
    assert "NDCG@10" in text
    assert "synthetic" in text
    csv_text = paths["csv"].read_text(encoding="utf-8")
    assert "tfidf" in csv_text


def test_dense_load_failure_skips_without_raising():
    corpus = {"d1": {"title": "Cats", "text": "yarn"}}
    retriever, meta = try_build_dense_retriever(
        corpus,
        model_name="this-model-does-not-exist-locally-xyz",
        fallback_model="also-missing-xyz",
        allow_download=False,
    )
    assert retriever is None
    assert meta["status"] == "skipped"
    assert meta["error"]


def test_run_baseline_suite_lexical_only_on_synthetic_corpus():
    corpus = {
        "doc-cats": {"title": "Domestic Cats", "text": "Cats sit on mats."},
        "doc-dogs": {"title": "Domestic Dogs", "text": "Dogs chase balls."},
    }
    queries = {"q1": "cats mats"}
    qrels = {"q1": {"doc-cats": 1}}
    rows = run_baseline_suite(
        corpus,
        queries,
        qrels,
        k=10,
        run_dense=False,
    )
    by_name = {row["model"]: row for row in rows}
    assert by_name["tfidf"]["status"] == "ok"
    assert by_name["bm25"]["status"] == "ok"
    assert by_name["dense"]["status"] == "skipped"
    assert by_name["tfidf"]["Recall@10"] == 1.0
    assert by_name["bm25"]["Recall@10"] == 1.0
    assert "index_seconds" in by_name["tfidf"]
    assert "query_latency_mean_ms" in by_name["tfidf"]


if __name__ == "__main__":
    from pathlib import Path as _Path

    test_ranked_hits_preserve_query_and_doc_ids()
    test_perfect_ranking_metrics_at_10()
    test_empty_ranking_has_zero_metrics()
    test_identical_query_doc_id_is_ignored()
    test_describe_dataset_counts_positive_qrels()
    test_describe_dataset_rejects_mismatched_qrels()
    test_evaluate_retriever_does_not_need_full_scidocs()
    test_save_report_writes_json_and_csv(_Path("outputs"))
    test_dense_load_failure_skips_without_raising()
    test_run_baseline_suite_lexical_only_on_synthetic_corpus()
    print("Evaluation tests passed!")
