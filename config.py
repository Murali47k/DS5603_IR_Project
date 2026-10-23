"""
GLOBAL CONFIGURATION FILE
Modify paths and hyperparameter defaults here.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"

# Create directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Configuration (BEIR SciDocs)
DATASET_NAME = "scidocs"
# SciDocs dataset URL from BEIR
BEIR_DATASET_URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"

# Model Checkpoints
# Dense Model for embeddings
# Suggested scientific dense model: "allenai/specter" or "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MODEL_NAME = "allenai/specter"

# Cross-Encoder for reranking
# Suggested reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2" or "BAAI/bge-reranker-base"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Hyperparameters
DEFAULT_TOP_K = 10
RERANK_TOP_K = 50
BM25_K1 = 1.5
BM25_B = 0.75

# Hybrid Search Weights 
HYBRID_BM25_WEIGHT = 0.5
HYBRID_DENSE_WEIGHT = 0.5
RRF_K = 60 # Parameter for Reciprocal Rank Fusion

# Vector DB Settings 
VECTOR_DB_TYPE = "faiss"  # Options: "faiss", "chroma"
EMBEDDING_DIM = 768       # SPECTER produces 768-dim embeddings (MiniLM is 384)
DISTANCE_METRIC = "cosine" # Options: "cosine", "inner_product", "l2"
