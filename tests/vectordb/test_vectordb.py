"""Tests for src/vectordb. Uses a fake encoder and fake corpus: no downloads."""

import random

import numpy as np
import pytest

from src.base_models import DenseRetriever
from src.optimisations.fusion import weighted_score_fusion
from src.vectordb import VectorDBRetriever
from src.vectordb.benchmark import compute_overlap_recall
from src.vectordb.constants import (
    ALL_INDEX_TYPES, INDEX_FLAT, INDEX_HNSW, INDEX_IVF, INDEX_IVFPQ,
)
from src.vectordb.embeddings import build_cache_key, get_or_create_embeddings, normalize_rows
from src.vectordb.index_builder import build_index
from src.vectordb.retriever import drop_missing_hits

# ---- small global settings for tests ----
FAKE_CORPUS_SIZE = 300
WORDS = ["cat", "dog", "quantum", "graph", "neural", "ranking", "vector", "search",
         "paper", "citation", "index", "query", "tensor", "language", "model", "mats"]
SMALL_BUILD_PARAMS = {
    INDEX_FLAT: {},
    INDEX_IVF: {"nlist": 4, "nprobe": 4},
    INDEX_HNSW: {"m": 8, "ef_construction": 40, "ef_search": 40},
    INDEX_IVFPQ: {"nlist": 4, "nprobe": 4, "pq_m": 13},
}


class BagOfCharsEncoder:
    """Fake encoder: counts letters. Also counts how many texts it was asked to encode."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        rows = []
        for text in texts:
            self.calls += 1
            vec = np.zeros(26)
            for ch in str(text).lower():
                if "a" <= ch <= "z":
                    vec[ord(ch) - 97] += 1.0
            rows.append(vec)
        return np.vstack(rows)


def make_corpus(size=FAKE_CORPUS_SIZE, seed=0):
    rng = random.Random(seed)
    return {
        f"doc-{i:04d}": {"title": " ".join(rng.choices(WORDS, k=3)), "text": " ".join(rng.choices(WORDS, k=12))}
        for i in range(size)
    }


def make_retriever(index_type, tmp_path, corpus=None):
    return VectorDBRetriever.from_corpus(
        corpus or make_corpus(), BagOfCharsEncoder(), index_type=index_type, model_name="fake",
        build_params=SMALL_BUILD_PARAMS[index_type], cache_dir=tmp_path / "cache",
    )


def test_normalize_rows_gives_unit_length_float32():
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out.dtype == np.float32
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-6
    assert np.all(out[1] == 0)


def test_embedding_cache_is_reused(tmp_path):
    corpus = make_corpus(20)
    encoder = BagOfCharsEncoder()
    ids1, emb1 = get_or_create_embeddings(corpus, encoder, "fake", tmp_path)
    calls_after_first = encoder.calls
    ids2, emb2 = get_or_create_embeddings(corpus, encoder, "fake", tmp_path)
    assert encoder.calls == calls_after_first  # second time: nothing re-encoded
    assert ids1 == ids2 and np.array_equal(emb1, emb2)


def test_cache_key_changes_when_text_or_model_changes():
    base = build_cache_key("m", ["a"], ["hello"])
    assert base != build_cache_key("m", ["a"], ["hello!"])
    assert base != build_cache_key("other", ["a"], ["hello"])


def test_flat_matches_exact_dense_retriever(tmp_path):
    corpus = make_corpus(50)
    exact = DenseRetriever(corpus, encoder=BagOfCharsEncoder())
    flat = make_retriever(INDEX_FLAT, tmp_path, corpus)
    expected = dict(exact.retrieve("quantum graph search", k=10))
    got = flat.retrieve("quantum graph search", k=10)
    assert len(got) == 10
    for doc_id, score in got:
        if doc_id in expected:
            assert abs(score - expected[doc_id]) < 1e-5


@pytest.mark.parametrize("index_type", ALL_INDEX_TYPES)
def test_every_index_type_returns_valid_sorted_hits(index_type, tmp_path):
    retriever = make_retriever(index_type, tmp_path)
    hits = retriever.retrieve("neural ranking model", k=5)
    assert 0 < len(hits) <= 5
    for doc_id, score in hits:
        assert isinstance(doc_id, str) and isinstance(score, float)
        assert doc_id.startswith("doc-")
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("index_type", ALL_INDEX_TYPES)
def test_save_and_load_gives_same_results(index_type, tmp_path):
    retriever = make_retriever(index_type, tmp_path)
    retriever.save(tmp_path / "saved")
    loaded = VectorDBRetriever.from_saved(tmp_path / "saved", BagOfCharsEncoder())
    assert loaded.index_type == index_type
    assert loaded.retrieve("dog cat", k=5) == retriever.retrieve("dog cat", k=5)


def test_missing_faiss_slots_are_skipped(tmp_path):
    # nprobe=1 with a huge k: FAISS cannot fill every slot and returns -1.
    retriever = make_retriever(INDEX_IVF, tmp_path)
    retriever.set_search_params(nprobe=1)
    hits = retriever.retrieve("cat", k=FAKE_CORPUS_SIZE)
    assert len(hits) <= FAKE_CORPUS_SIZE
    assert all(isinstance(doc_id, str) for doc_id, _ in hits)


def test_drop_missing_hits_helper():
    ids, scores = drop_missing_hits([0.9, 0.5, -1e9], [1, 0, -1], ["a", "b"])
    assert ids == ["b", "a"] and scores == [0.9, 0.5]


def test_ties_are_broken_by_doc_id(tmp_path):
    corpus = {
        "zebra": {"title": "Same", "text": "identical words"},
        "alpha": {"title": "Same", "text": "identical words"},
    }
    retriever = VectorDBRetriever.from_corpus(
        corpus, BagOfCharsEncoder(), index_type=INDEX_FLAT, model_name="fake", cache_dir=tmp_path
    )
    assert [d for d, _ in retriever.retrieve("identical words", k=2)] == ["alpha", "zebra"]


def test_edge_cases(tmp_path):
    retriever = make_retriever(INDEX_FLAT, tmp_path)
    assert retriever.retrieve("", k=5) == []
    assert retriever.retrieve("   ", k=5) == []
    with pytest.raises(ValueError):
        retriever.retrieve("cat", k=0)
    empty = VectorDBRetriever.from_corpus({}, BagOfCharsEncoder(), cache_dir=tmp_path)
    assert empty.retrieve("cat", k=5) == []


def test_unknown_index_type_raises():
    with pytest.raises(ValueError):
        build_index("nope", np.ones((3, 4), dtype=np.float32))


def test_results_work_with_fusion_and_retrieve_ids(tmp_path):
    retriever = make_retriever(INDEX_HNSW, tmp_path)
    hits = retriever.retrieve("cat dog", k=5)
    assert retriever.retrieve_ids("cat dog", k=5) == [d for d, _ in hits]
    assert weighted_score_fusion(hits, hits)


def test_overlap_recall_helper():
    exact = [["a", "b"], ["c", "d"]]
    approx = [["a", "x"], ["c", "d"]]
    assert compute_overlap_recall(exact, approx, k=2) == 0.75
    assert compute_overlap_recall([], [], k=2) == 0.0
