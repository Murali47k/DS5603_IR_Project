"""
SciDocs academic search UI.

Launch from the repository root:

    .venv/bin/streamlit run src/ui/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ui.metadata import year_bounds
from src.ui.search_service import RankedPaper, SearchEngine, SearchResponse, build_search_engine

METHOD_LABELS = {
    "citation_hybrid": "Citation-boosted hybrid",
    "bm25": "BM25 (lexical)",
    "dense": "Dense (FAISS)",
    "similar": "Nearest neighbours",
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1280px; padding-top: 1.4rem; }
        h1, h2, h3 { font-family: "Source Serif 4", "Georgia", serif; letter-spacing: -0.02em; }
        .scidocs-hero { padding: 0.2rem 0 0.8rem 0; }
        .scidocs-kicker {
            color: #3d5a80; font-weight: 600; font-size: 0.82rem;
            text-transform: uppercase; letter-spacing: 0.12em;
        }
        .paper-card {
            border: 1px solid #d9e2ec; border-radius: 14px; padding: 1rem 1.1rem 0.85rem;
            background: #ffffff; margin-bottom: 0.85rem;
            box-shadow: 0 1px 2px rgba(16, 42, 67, 0.04);
        }
        .paper-rank { color: #3d5a80; font-weight: 700; font-size: 0.85rem; }
        .paper-title { font-size: 1.12rem; font-weight: 700; color: #102a43; line-height: 1.35; margin: 0.15rem 0 0.35rem; }
        .paper-meta { color: #486581; font-size: 0.9rem; margin-bottom: 0.45rem; }
        .paper-snippet { color: #334e68; font-size: 0.95rem; line-height: 1.5; }
        .badge {
            display: inline-block; padding: 0.16rem 0.55rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 700; margin-right: 0.35rem;
        }
        .badge-cite { background: #e0f2fe; color: #075985; }
        .badge-score { background: #ecfdf3; color: #166534; }
        .badge-method { background: #f1f5f9; color: #334155; }
        .score-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.5rem; margin-top: 0.6rem; }
        .score-cell { background: #f8fafc; border-radius: 10px; padding: 0.5rem 0.6rem; }
        .score-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #627d98; }
        .score-value { font-size: 0.95rem; font-weight: 700; color: #102a43; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading SciDocs corpus, BM25 index, and FAISS retriever…")
def get_engine() -> SearchEngine:
    return build_search_engine()


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def _fmt_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown authors"
    return ", ".join(authors[:8]) + ("…" if len(authors) > 8 else "")


def _fmt_year(year: int | None) -> str:
    return str(year) if year else "Year unknown"


def render_score_breakdown(hit: RankedPaper) -> None:
    st.markdown(
        f"""
        <div class="score-grid">
          <div class="score-cell"><div class="score-label">BM25</div><div class="score-value">{_fmt_score(hit.bm25_score)}</div></div>
          <div class="score-cell"><div class="score-label">Dense / FAISS</div><div class="score-value">{_fmt_score(hit.dense_score)}</div></div>
          <div class="score-cell"><div class="score-label">Fused hybrid</div><div class="score-value">{_fmt_score(hit.fused_score)}</div></div>
          <div class="score-cell"><div class="score-label">Cite multiplier</div><div class="score-value">{_fmt_score(hit.citation_multiplier)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph_lists(engine: SearchEngine, doc_id: str) -> None:
    cited = engine.cited_by(doc_id)
    refs = engine.references(doc_id)
    left, right = st.columns(2)
    with left:
        st.caption(f"Cited by ({len(cited)})")
        if not cited:
            st.write("No citing papers are recorded for this document.")
        else:
            for paper in cited[:25]:
                year = f" ({paper.year})" if paper.year else ""
                st.markdown(f"- **{paper.title}**{year}  \n`{paper.doc_id}`")
    with right:
        st.caption(f"References ({len(refs)})")
        if not refs:
            st.write("No references are recorded for this document.")
        else:
            for paper in refs[:25]:
                year = f" ({paper.year})" if paper.year else ""
                st.markdown(f"- **{paper.title}**{year}  \n`{paper.doc_id}`")


def render_card(engine: SearchEngine, hit: RankedPaper, key_prefix: str) -> None:
    paper = hit.paper
    title = html.escape(paper.title)
    meta = html.escape(f"{_fmt_authors(paper.authors)} · {_fmt_year(paper.year)}")
    snippet = html.escape(paper.snippet())
    method = html.escape(METHOD_LABELS.get(hit.method, hit.method))
    doc_short = html.escape(paper.doc_id[:12] + ("…" if len(paper.doc_id) > 12 else ""))
    st.markdown(
        f"""
        <div class="paper-card">
          <div class="paper-rank">#{hit.rank} · {method}</div>
          <div class="paper-title">{title}</div>
          <div class="paper-meta">{meta}</div>
          <div>
            <span class="badge badge-cite">{paper.citation_count} citations</span>
            <span class="badge badge-score">score {hit.final_score:.4f}</span>
            <span class="badge badge-method">{doc_short}</span>
          </div>
          <p class="paper-snippet">{snippet}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Score breakdown, citations, and similar papers", expanded=False):
        render_score_breakdown(hit)
        st.markdown("**Citation graph**")
        render_graph_lists(engine, paper.doc_id)
        if st.button("Find similar papers", key=f"{key_prefix}-similar-{paper.doc_id}"):
            with st.spinner("Searching FAISS for nearest papers…"):
                similar = engine.find_similar(paper.doc_id, k=engine.settings.similar_k)
            if not similar:
                st.info("No similar papers were found. The FAISS index may be unavailable for this document.")
            else:
                for neighbour in similar:
                    nyear = f" ({neighbour.paper.year})" if neighbour.paper.year else ""
                    st.markdown(
                        f"{neighbour.rank}. **{neighbour.paper.title}**{nyear}  \n"
                        f"cosine/IP score `{neighbour.final_score:.4f}` · `{neighbour.paper.doc_id}`"
                    )


def render_result_column(engine: SearchEngine, response: SearchResponse, key_prefix: str) -> None:
    st.subheader(METHOD_LABELS.get(response.method, response.method))
    if response.error:
        st.warning(response.error)
    if not response.results:
        st.info("No papers matched this query and the current filters.")
        return
    for hit in response.results:
        render_card(engine, hit, key_prefix=f"{key_prefix}-{hit.paper.doc_id}")


def sidebar_controls(engine: SearchEngine) -> dict:
    settings = engine.settings
    min_year, max_year = year_bounds(engine.catalog)
    st.sidebar.header("Ranking & filters")
    boost = st.sidebar.slider(
        "Citation boost weight (w)",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.citation_boost_weight),
        step=0.05,
        help="Re-ranks the current candidate pool with citation_aware_boost. Default w = 0.4.",
    )
    years = st.sidebar.slider(
        "Publication year",
        min_value=int(min_year),
        max_value=int(max_year),
        value=(int(min_year), int(max_year)),
    )
    min_cites = st.sidebar.number_input(
        "Minimum citation count",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
    )
    top_k = st.sidebar.slider("Results per method", min_value=1, max_value=25, value=int(settings.k))
    compare = st.sidebar.toggle("Comparison mode", value=False)
    st.sidebar.caption(
        f"Hybrid weights: BM25 {settings.bm25_weight:.2f} + dense {settings.dense_weight:.2f}. "
        f"FAISS index: {settings.index_type}. Model: {settings.dense_model}."
    )
    return {
        "boost_weight": float(boost),
        "year_min": int(years[0]),
        "year_max": int(years[1]),
        "min_citations": int(min_cites),
        "k": int(top_k),
        "compare": bool(compare),
    }


def main() -> None:
    st.set_page_config(page_title="SciDocs Search", page_icon="📚", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="scidocs-hero">
          <div class="scidocs-kicker">DS5603 · SciDocs</div>
          <h1>Scientific document search</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "Default ranking is **citation-boosted hybrid retrieval**: BM25 lexical search fused with "
        "FAISS dense search, then `citation_aware_boost` (w = 0.4). Indexes are built once and reused."
    )

    try:
        engine = get_engine()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the search engine: {type(exc).__name__}: {exc}")
        st.stop()

    if engine.dense_error and engine.dense is None:
        st.error(f"FAISS dense retrieval failed to load: {engine.dense_error}")

    controls = sidebar_controls(engine)
    with st.form("search_form"):
        query = st.text_input(
            "Search query",
            placeholder="e.g. graph neural networks for citation recommendation",
            label_visibility="collapsed",
        )
        search_clicked = st.form_submit_button("Search", type="primary")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a search query to retrieve papers.")
            return
        st.session_state["last_query"] = query.strip()

    active_query = st.session_state.get("last_query", "")
    if not active_query:
        return

    params = dict(
        k=controls["k"],
        boost_weight=controls["boost_weight"],
        year_min=controls["year_min"],
        year_max=controls["year_max"],
        min_citations=controls["min_citations"],
    )
    with st.spinner("Retrieving and ranking papers…"):
        try:
            if controls["compare"]:
                compared = engine.compare(active_query, **params)
                response = compared["citation_hybrid"]
            else:
                compared = None
                response = engine.search(active_query, **params)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {type(exc).__name__}: {exc}")
            return

        if compared is not None:
            cols = st.columns(3)
            order = ("bm25", "dense", "citation_hybrid")
            for column, key in zip(cols, order):
                with column:
                    render_result_column(engine, compared[key], key_prefix=key)
        else:
            if response.error:
                st.warning(response.error)
            if not response.results:
                st.info("No papers matched this query and the current filters.")
            else:
                st.caption(
                    f"{len(response.results)} results · {METHOD_LABELS[response.method]} · "
                    f"w = {controls['boost_weight']:.2f}"
                )
                for hit in response.results:
                    render_card(engine, hit, key_prefix="hybrid")


if __name__ == "__main__":
    main()
