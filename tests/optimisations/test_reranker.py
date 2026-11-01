import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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