from src.optimisations.fusion import (
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


def test_reciprocal_rank_fusion():
    ranked_results = [
        ["doc1", "doc2", "doc3"],
        ["doc2", "doc1", "doc4"],
    ]

    fused = reciprocal_rank_fusion(ranked_results)

    assert len(fused) == 4
    assert fused[0][0] == "doc1" or fused[0][0] == "doc2"

    scores = dict(fused)
    assert scores["doc1"] > 0
    assert scores["doc2"] > 0


def test_weighted_score_fusion():
    bm25_results = [
        ("doc1", 0.8),
        ("doc2", 0.6),
    ]

    dense_results = [
        ("doc2", 0.9),
        ("doc3", 0.7),
    ]

    fused = weighted_score_fusion(
        bm25_results,
        dense_results,
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    scores = dict(fused)

    assert len(fused) == 3
    assert abs(scores["doc1"] - 0.4) < 1e-9
    assert abs(scores["doc2"] - 0.75) < 1e-9
    assert abs(scores["doc3"] - 0.35) < 1e-9
    assert fused[0][0] == "doc2"


if __name__ == "__main__":
    test_reciprocal_rank_fusion()
    test_weighted_score_fusion()
    print("Fusion tests passed!")
