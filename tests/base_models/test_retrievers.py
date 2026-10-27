"""Unit tests for TF-IDF, BM25, and optional dense retrievers.

Uses a tiny synthetic corpus only. No SciDocs download and no pretrained models.
"""

from typing import List, Tuple

import numpy as np

from src.base_models import BM25Retriever, DenseRetriever, TfidfRetriever, ranked_ids
from src.optimisations.fusion import reciprocal_rank_fusion, weighted_score_fusion


TINY_CORPUS = {
    "paper-cats": {"title": "Domestic Cats", "text": "Cats sit on mats and chase yarn."},
    "paper-dogs": {"title": "Domestic Dogs", "text": "Dogs chase balls and bark loudly."},
    "paper-quantum": {"title": "Quantum Circuits", "text": "Qubits and quantum entanglement."},
}

IDENTICAL_CORPUS = {
    "zebra-id": {"title": "Same", "text": "identical overlapping tokens"},
    "alpha-id": {"title": "Same", "text": "identical overlapping tokens"},
}


class BagOfCharsEncoder:
    """Deterministic local encoder so dense tests never hit the network."""

    def encode(self, texts):
        dim = 26
        rows = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float64)
            for ch in str(text).lower():
                if "a" <= ch <= "z":
                    vec[ord(ch) - 97] += 1.0
            rows.append(vec)
        return np.vstack(rows)


def _assert_result_shape(results: List[Tuple[str, float]], expected_ids=None):
    assert isinstance(results, list)
    for doc_id, score in results:
        assert isinstance(doc_id, str)
        assert isinstance(score, float)
        if expected_ids is not None:
            assert doc_id in expected_ids


def test_tfidf_index_and_retrieval():
    retriever = TfidfRetriever(TINY_CORPUS)
    assert retriever.doc_matrix is not None
    assert retriever.doc_matrix.shape[0] == len(TINY_CORPUS)

    results = retriever.retrieve("cats on mats", k=2)
    _assert_result_shape(results, TINY_CORPUS)
    assert results[0][0] == "paper-cats"
    assert results[0][1] >= results[1][1]


def test_bm25_index_and_retrieval():
    retriever = BM25Retriever(TINY_CORPUS, k1=1.5, b=0.75)
    assert retriever._bm25 is not None

    results = retriever.retrieve("quantum qubits", k=2)
    _assert_result_shape(results, TINY_CORPUS)
    assert results[0][0] == "paper-quantum"
    assert results[0][1] >= results[1][1]


def test_dense_with_local_encoder():
    retriever = DenseRetriever(TINY_CORPUS, encoder=BagOfCharsEncoder())
    results = retriever.retrieve("cats mats yarn", k=2)
    _assert_result_shape(results, TINY_CORPUS)
    assert results[0][0] == "paper-cats"


def test_document_ids_preserved():
    for retriever in (
        TfidfRetriever(TINY_CORPUS),
        BM25Retriever(TINY_CORPUS),
        DenseRetriever(TINY_CORPUS, encoder=BagOfCharsEncoder()),
    ):
        ids = {doc_id for doc_id, _ in retriever.retrieve("dogs bark", k=3)}
        assert ids.issubset(set(TINY_CORPUS))
        assert "paper-dogs" in ids


def test_k_one_and_k_larger_than_corpus():
    retriever = TfidfRetriever(TINY_CORPUS)
    one = retriever.retrieve("dogs", k=1)
    assert len(one) == 1
    assert one[0][0] == "paper-dogs"

    all_hits = retriever.retrieve("dogs", k=100)
    assert len(all_hits) == len(TINY_CORPUS)
    scores = [score for _, score in all_hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_and_empty_corpus():
    retriever = BM25Retriever(TINY_CORPUS)
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []

    empty_tfidf = TfidfRetriever({})
    empty_bm25 = BM25Retriever({})
    empty_dense = DenseRetriever({}, encoder=BagOfCharsEncoder())
    assert empty_tfidf.retrieve("cats", k=5) == []
    assert empty_bm25.retrieve("cats", k=5) == []
    assert empty_dense.retrieve("cats", k=5) == []


def test_repeated_queries_reuse_index():
    retriever = TfidfRetriever(TINY_CORPUS)
    matrix_id = id(retriever.doc_matrix)
    first = retriever.retrieve("cats", k=3)
    second = retriever.retrieve("cats", k=3)
    assert first == second
    assert id(retriever.doc_matrix) == matrix_id

    bm25 = BM25Retriever(TINY_CORPUS)
    index_id = id(bm25._bm25)
    assert bm25.retrieve("dogs", k=3) == bm25.retrieve("dogs", k=3)
    assert id(bm25._bm25) == index_id


def test_stable_tie_breaking():
    retriever = TfidfRetriever(IDENTICAL_CORPUS)
    results = retriever.retrieve("identical overlapping tokens", k=2)
    assert [doc_id for doc_id, _ in results] == ["alpha-id", "zebra-id"]
    assert results[0][1] == results[1][1]


def test_invalid_k_raises():
    retriever = TfidfRetriever(TINY_CORPUS)
    try:
        retriever.retrieve("cats", k=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_results_compatible_with_fusion():
    tfidf = TfidfRetriever(TINY_CORPUS)
    bm25 = BM25Retriever(TINY_CORPUS)
    query = "cats and dogs"

    tfidf_hits = tfidf.retrieve(query, k=3)
    bm25_hits = bm25.retrieve(query, k=3)

    fused_scores = weighted_score_fusion(bm25_hits, tfidf_hits)
    fused_rrf = reciprocal_rank_fusion(
        [ranked_ids(bm25_hits), ranked_ids(tfidf_hits)]
    )
    assert fused_scores
    assert fused_rrf
    assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in fused_scores)


def test_blank_documents_do_not_crash():
    corpus = {
        "empty": {"title": "", "text": ""},
        "full": {"title": "Neural ranking", "text": "Learning to rank documents."},
        "other": {"title": "Domestic cats", "text": "Cats sit on mats."},
    }
    tfidf = TfidfRetriever(corpus)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, encoder=BagOfCharsEncoder())
    assert tfidf.retrieve("ranking", k=3)[0][0] == "full"
    assert bm25.retrieve("ranking", k=3)[0][0] == "full"
    assert dense.retrieve("ranking", k=3)[0][0] == "full"


if __name__ == "__main__":
    test_tfidf_index_and_retrieval()
    test_bm25_index_and_retrieval()
    test_dense_with_local_encoder()
    test_document_ids_preserved()
    test_k_one_and_k_larger_than_corpus()
    test_empty_query_and_empty_corpus()
    test_repeated_queries_reuse_index()
    test_stable_tie_breaking()
    test_invalid_k_raises()
    test_results_compatible_with_fusion()
    test_blank_documents_do_not_crash()
    print("Base model tests passed!")
