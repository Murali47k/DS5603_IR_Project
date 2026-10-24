from src.optimisations.citation_boost import citation_aware_boost

def test_citation_aware_boost():
    results = [
        ("doc1", 0.8),
        ("doc2", 0.7),
        ("doc3", 0.6),
    ]

    citations = {
        "doc1": 10,
        "doc2": 100,
        "doc3": 0,
    }

    boosted = citation_aware_boost(
        results,
        citations,
        boost_weight=0.1,
    )

    scores = dict(boosted)
    
    assert len(boosted) == 3
    assert scores["doc1"] > 0.8
    assert scores["doc2"] > 0.7
    assert scores["doc3"] == 0.6


if __name__ == "__main__":
    test_citation_aware_boost()
    print("Citation boost test passed!")
