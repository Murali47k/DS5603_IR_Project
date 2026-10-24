from src.optimisations.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(top_k=3)

candidates = [
    ("doc1", {"title": "Graph Neural Networks", "text": "Graph learning methods."}),
    ("doc2", {"title": "Information Retrieval", "text": "Search ranking methods."}),
    ("doc3", {"title": "Quantum Computing", "text": "Quantum circuit simulation."}),
]

results = reranker.rerank(
    "Methods for ranking scientific documents",
    candidates,
)

print(results)