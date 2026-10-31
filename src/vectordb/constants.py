"""
Settings for the vector database package.

Everything you may want to tune lives here, so no other file has "magic numbers".
"""

# ---- Index type names (used everywhere instead of typing strings by hand) ----
INDEX_FLAT = "flat"      # exact search, the "truth" baseline
INDEX_IVF = "ivf"        # clusters; only a few clusters are searched
INDEX_HNSW = "hnsw"      # graph index; usually fast and accurate
INDEX_IVFPQ = "ivfpq"    # IVF + compressed vectors (small memory)
ALL_INDEX_TYPES = (INDEX_FLAT, INDEX_IVF, INDEX_HNSW, INDEX_IVFPQ)

# ---- IVF settings ----
IVF_NLIST = 128          # how many clusters to build
IVF_NPROBE = 8           # how many clusters to look into per query

# ---- HNSW settings ----
HNSW_M = 32                  # links per node (more = better but bigger)
HNSW_EF_CONSTRUCTION = 200   # effort while building
HNSW_EF_SEARCH = 64          # effort while searching

# ---- Product Quantization (PQ) settings ----
PQ_M = 48                # number of pieces each vector is cut into
PQ_NBITS = 8             # bits per piece (8 means 256 codes per piece)

# ---- Embedding settings ----
ENCODE_BATCH_SIZE = 64
EMBEDDING_DTYPE = "float32"      # FAISS only understands float32
EMBEDDINGS_SUBDIR = "embeddings"  # inside VECTOR_INDEX_DIR
CACHE_KEY_LENGTH = 16            # characters of the hash used in file names

# ---- File names used when saving ----
INDEX_FILE_NAME = "index.faiss"
IDS_FILE_NAME = "doc_ids.json"
META_FILE_NAME = "meta.json"
EMBEDDINGS_FILE_SUFFIX = ".npy"
IDS_FILE_SUFFIX = ".ids.json"

# FAISS gives this number when it has fewer than k answers
FAISS_MISSING_ID = -1

# ---- Benchmark plan ----
# Each entry: (index type, settings used to BUILD, list of settings used to SEARCH).
# We build once, then try each search setting. This shows speed vs accuracy.
BENCHMARK_PLAN = [
    (INDEX_FLAT, {}, [{}]),
    (
        INDEX_IVF,
        {"nlist": IVF_NLIST},
        [{"nprobe": 1}, {"nprobe": 4}, {"nprobe": 16}, {"nprobe": 64}],
    ),
    (
        INDEX_HNSW,
        {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
        [{"ef_search": 16}, {"ef_search": 64}, {"ef_search": 256}],
    ),
    (
        INDEX_IVFPQ,
        {"nlist": IVF_NLIST, "pq_m": PQ_M, "pq_nbits": PQ_NBITS},
        [{"nprobe": 8}, {"nprobe": 32}],
    ),
]

BENCHMARK_STEM = "vectordb_benchmark"
