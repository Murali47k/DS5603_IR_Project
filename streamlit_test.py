import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.ui.search_service import build_search_engine
from src.data_loader import load_scidocs
from src.ui.settings import load_ui_settings
from src.vectordb.retriever import VectorDBRetriever

st.title("SciDocs Diagnostic Test")
st.write("Streamlit started successfully")

@st.cache_resource
def load_engine():
    settings = load_ui_settings()
    corpus, _, _ = load_scidocs(data_dir=settings.data_dir)

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    dense = VectorDBRetriever.from_saved(
        Path("/tmp/scidocs_test_index"), encoder
    )

    return build_search_engine(
        settings=settings,
        corpus=corpus,
        load_dataset=False,
        dense=dense
    )

st.write("Loading search engine...")
engine = load_engine()
st.success("Hybrid engine loaded successfully!")

query = st.text_input("Test query", "graph neural networks")

if st.button("Search"):
    response = engine.search(query, k=5)
    for paper in response.results:
        st.write(paper.paper.title)
