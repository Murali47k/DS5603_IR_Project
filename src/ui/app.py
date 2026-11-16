"""
SciDocs academic search UI.

Launch from the repository root:

    .venv/bin/streamlit run src/ui/app.py
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure the application consistently uses the project's .venv
venv_dir = ROOT / ".venv"
venv_python = venv_dir / "bin" / "python"
venv_streamlit = venv_dir / "bin" / "streamlit"

if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    if venv_streamlit.exists() and os.environ.get("_STREAMLIT_VENV_REEXEC") != "1":
        os.environ["_STREAMLIT_VENV_REEXEC"] = "1"
        os.environ.setdefault("HOME", str(ROOT))
        try:
            os.execv(str(venv_streamlit), [str(venv_streamlit), "run", str(Path(__file__).resolve()), *sys.argv[1:]])
        except Exception:
            pass
    for sp in venv_dir.glob("lib/python*/site-packages"):
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

os.environ.setdefault("HOME", str(ROOT))

# Prevent permission issues with matplotlib font cache
mpl_cache_dir = ROOT / "data" / ".matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.ui.comparison import (
    COMPARE_METHOD_ORDER,
    METHOD_SPECS,
    build_insights,
    correlation_matrix,
    intersection_of,
    rank_map_from_results,
    rank_mover_segments,
    rank_movers,
    summarize_overlap,
)
from src.ui.compare_views import render_head_to_head, render_rank_matrix
from src.ui.metadata import year_bounds
from src.ui.search_service import RankedPaper, SearchEngine, SearchResponse, build_search_engine

METHOD_LABELS = {key: spec["name"] for key, spec in METHOD_SPECS.items()}
METHOD_LABELS["similar"] = "Nearest neighbours"

METHOD_BLURBS = {key: spec["description"] for key, spec in METHOD_SPECS.items()}

# Offline SciDocs test-split numbers from the project README. Not live-query scores.
OFFLINE_BENCHMARKS = [
    {"key": "bm25", "ndcg10": 0.14950, "map10": 0.08688, "recall10": 0.15428, "p10": 0.07620, "latency_ms": 1.8},
    {"key": "tfidf", "ndcg10": 0.13636, "map10": 0.07691, "recall10": 0.14347, "p10": 0.07060, "latency_ms": 4.8},
    {"key": "dense", "ndcg10": 0.21641, "map10": 0.12938, "recall10": 0.23087, "p10": 0.11400, "latency_ms": 120.0},
    {"key": "citation_hybrid", "ndcg10": 0.22877, "map10": 0.13964, "recall10": 0.24117, "p10": 0.11910, "latency_ms": 87.9},
    {"key": "cross_encoder", "ndcg10": 0.17317, "map10": 0.10054, "recall10": 0.18385, "p10": 0.09070, "latency_ms": 3493.3},
    {"key": "rrf", "ndcg10": 0.19782, "map10": 0.11774, "recall10": 0.20922, "p10": 0.10330, "latency_ms": 88.4},
    {"key": "weighted", "ndcg10": 0.21340, "map10": 0.12852, "recall10": 0.22782, "p10": 0.11250, "latency_ms": 90.0},
]

BENCHMARK_BY_KEY = {row["key"]: row for row in OFFLINE_BENCHMARKS}

MISSING_SCORE_HELP = (
    "A dash (—) means that retriever did not return this paper in its candidate pool. "
    "It is not a score of zero. Fusion still treats a missing component as 0 internally."
)
DENSE_BENCHMARK_NOTE = (
    "Offline Dense nDCG/MAP figures are the exhaustive MiniLM baseline. "
    "Live dense results in this dashboard use the same encoder through the FAISS index loaded by the UI."
)
COMPARE_CACHE_MAX_K = 50


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
        .score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.5rem; margin-top: 0.6rem; }
        .score-cell { background: #f8fafc; border-radius: 10px; padding: 0.5rem 0.6rem; }
        .score-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #627d98; }
        .score-value { font-size: 0.95rem; font-weight: 700; color: #102a43; }
        .muted-note { color: #627d98; font-size: 0.82rem; line-height: 1.45; margin: 0.35rem 0 0.7rem; }
        .compare-method { font-size: 0.92rem; color: #486581; margin-bottom: 0.6rem; }
        .insight-box {
            background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 12px;
            padding: 0.85rem 1rem; margin: 0.4rem 0 0.8rem;
        }
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
    method = hit.method
    cells: list[tuple[str, str]] = []

    if method == "bm25":
        cells.append(("Retrieval Model", "BM25 (rank-bm25)"))
        cells.append(("Raw BM25 Score", _fmt_score(hit.final_score)))
        cells.append(("Dense FAISS Score", _fmt_score(hit.dense_score)))
        cells.append(("Paper Citations", str(hit.paper.citation_count)))

    elif method == "tfidf":
        cells.append(("Retrieval Model", "TF-IDF (scikit-learn)"))
        cells.append(("TF-IDF Cosine", _fmt_score(hit.final_score)))
        cells.append(("BM25 Score", _fmt_score(hit.bm25_score)))
        cells.append(("Paper Citations", str(hit.paper.citation_count)))

    elif method == "dense":
        cells.append(("Retrieval Model", "FAISS HNSW (MiniLM)"))
        cells.append(("Dense Cosine / IP", _fmt_score(hit.final_score)))
        cells.append(("BM25 Score", _fmt_score(hit.bm25_score)))
        cells.append(("Paper Citations", str(hit.paper.citation_count)))

    elif method == "weighted":
        cells.append(("Norm BM25 (40%)", _fmt_score(hit.norm_bm25_score)))
        cells.append(("Norm Dense (60%)", _fmt_score(hit.norm_dense_score)))
        cells.append(("Fused Hybrid Score", _fmt_score(hit.final_score)))
        cells.append(("Raw BM25 / Dense", f"{_fmt_score(hit.bm25_score)} / {_fmt_score(hit.dense_score)}"))

    elif method == "citation_hybrid":
        cells.append(("Pre-Boost Hybrid", _fmt_score(hit.fused_score)))
        cells.append(("Cite Multiplier", _fmt_score(hit.citation_multiplier)))
        cells.append(("Boosted Final Score", _fmt_score(hit.final_score)))
        cells.append(("Citations", f"{hit.paper.citation_count} cites"))

    elif method == "rrf":
        r_b = f"#{hit.bm25_rank}" if hit.bm25_rank else "—"
        r_d = f"#{hit.dense_rank}" if hit.dense_rank else "—"
        cells.append(("BM25 Cand. Rank", r_b))
        cells.append(("Dense Cand. Rank", r_d))
        term_b = f"{hit.rrf_bm25_term:.5f}" if hit.rrf_bm25_term else "0.0"
        term_d = f"{hit.rrf_dense_term:.5f}" if hit.rrf_dense_term else "0.0"
        cells.append(("1/(60+r) Sum", f"{term_b} + {term_d}"))
        cells.append(("Fused RRF Score", _fmt_score(hit.final_score)))

    elif method == "cross_encoder":
        cand_r = f"#{hit.candidate_rank}" if hit.candidate_rank else "—"
        cells.append(("Hybrid Cand. Rank", cand_r))
        cells.append(("Hybrid Cand. Score", _fmt_score(hit.fused_score)))
        cells.append(("Cross-Encoder Score", _fmt_score(hit.final_score)))
        cells.append(("Neural Architecture", "ms-marco-MiniLM (CPU)"))

    else:
        cells.append(("Rank score", _fmt_score(hit.final_score)))
        cells.append(("BM25", _fmt_score(hit.bm25_score)))
        cells.append(("Dense", _fmt_score(hit.dense_score)))
        cells.append(("Cite Multiplier", _fmt_score(hit.citation_multiplier)))

    grid_items = "".join(
        f'<div class="score-cell"><div class="score-label">{html.escape(label)}</div><div class="score-value">{html.escape(val)}</div></div>'
        for label, val in cells
    )
    st.markdown(f'<div class="score-grid">{grid_items}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="muted-note">{html.escape(MISSING_SCORE_HELP)}</p>', unsafe_allow_html=True)


def _graph_caption(label: str, visible: int, recorded: int) -> str:
    hidden = max(recorded - visible, 0)
    if hidden:
        return f"{label} ({visible} in SciDocs · {hidden} unresolved IDs hidden)"
    return f"{label} ({visible})"


def render_graph_lists(engine: SearchEngine, doc_id: str) -> None:
    counts = engine.citation_list_counts(doc_id)
    cited = engine.visible_cited_by(doc_id)
    refs = engine.visible_references(doc_id)
    left, right = st.columns(2)
    with left:
        st.caption(_graph_caption("Cited by", counts["cited_visible"], counts["cited_recorded"]))
        if not cited:
            if counts["cited_recorded"]:
                st.write("Recorded citing papers are outside the loaded SciDocs corpus, so they are hidden.")
            else:
                st.write("No citing papers are recorded for this document.")
        else:
            for paper in cited[:25]:
                year = f" ({paper.year})" if paper.year else ""
                st.markdown(f"- **{paper.title}**{year}")
    with right:
        st.caption(_graph_caption("References", counts["refs_visible"], counts["refs_recorded"]))
        if not refs:
            if counts["refs_recorded"]:
                st.write("Recorded references are outside the loaded SciDocs corpus, so they are hidden.")
            else:
                st.write("No references are recorded for this document.")
        else:
            for paper in refs[:25]:
                year = f" ({paper.year})" if paper.year else ""
                st.markdown(f"- **{paper.title}**{year}")


def render_card(engine: SearchEngine, hit: RankedPaper, key_prefix: str) -> None:
    paper = hit.paper
    title = html.escape(paper.title)
    meta = html.escape(f"{_fmt_authors(paper.authors)} · {_fmt_year(paper.year)}")
    snippet = html.escape(paper.snippet())
    method = html.escape(METHOD_LABELS.get(hit.method, hit.method))
    st.markdown(
        f"""
        <div class="paper-card">
          <div class="paper-rank">#{hit.rank} · {method}</div>
          <div class="paper-title">{title}</div>
          <div class="paper-meta">{meta}</div>
          <div>
            <span class="badge badge-cite">{paper.citation_count} citations</span>
            <span class="badge badge-score">score {hit.final_score:.4f}</span>
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


def render_method_results(engine: SearchEngine, response: SearchResponse, k: int, key_prefix: str) -> None:
    spec = METHOD_SPECS.get(response.method, {})
    label = spec.get("name", METHOD_LABELS.get(response.method, response.method))
    category = spec.get("category", "")
    formula = spec.get("formula", "")
    description = spec.get("description", "")
    params = spec.get("parameters", "")

    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;">'
        f'<h3 style="margin: 0;">{html.escape(label)}</h3>'
        f'<span class="badge badge-method">{html.escape(category)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(description)
    if formula:
        st.latex(formula)
    if params:
        st.caption(f"**Configuration / Hyperparameters**: {params}")

    if response.error:
        st.warning(response.error)
    hits = response.results[:k]
    if not hits:
        st.info("No papers matched this query and the current filters.")
        return
    if response.latency_ms is not None:
        st.caption(f"⏱️ **Live query latency**: `{response.latency_ms:.1f} ms` · Showing top {len(hits)} ranked papers")
    for hit in hits:
        render_card(engine, hit, key_prefix=f"{key_prefix}-{hit.paper.doc_id}")


def _offline_metric_cell(value) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.5f}"


def render_benchmark_section(compared: dict[str, SearchResponse]) -> None:
    st.markdown("### Benchmark comparison (offline metrics vs. this query)")
    st.markdown("**Offline SciDocs held-out test split evaluation ($k = 10$) vs. Live query retrieval time**")
    st.caption(
        "Offline metrics are computed over the official BEIR SciDocs evaluation suite (1,000 queries) with "
        "ground-truth relevance judgements (`qrels`). Live retrieval latency reflects the single query entered above. "
        f"{DENSE_BENCHMARK_NOTE}"
    )
    rows = []
    for key in COMPARE_METHOD_ORDER:
        bench = BENCHMARK_BY_KEY.get(key)
        live = compared.get(key)
        rows.append(
            {
                "Approach": METHOD_LABELS.get(key, key),
                "Category": METHOD_SPECS.get(key, {}).get("category", ""),
                "nDCG@10": _offline_metric_cell(bench["ndcg10"] if bench else None),
                "MAP@10": _offline_metric_cell(bench["map10"] if bench else None),
                "Recall@10": _offline_metric_cell(bench["recall10"] if bench else None),
                "P@10": _offline_metric_cell(bench["p10"] if bench else None),
                "Mean Latency (offline)": (
                    f"{bench['latency_ms']:.1f} ms" if bench else "Unavailable"
                ),
                "This Query (live)": (
                    f"{live.latency_ms:.1f} ms" if live and live.latency_ms is not None else "Unavailable"
                ),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

    metric_choice = st.radio(
        "Select benchmark metric to compare across approaches:",
        options=["nDCG@10", "MAP@10", "Recall@10", "Precision@10", "Mean Latency (ms)"],
        horizontal=True,
        key="benchmark_metric_choice",
    )
    metric_map = {
        "nDCG@10": "ndcg10",
        "MAP@10": "map10",
        "Recall@10": "recall10",
        "Precision@10": "p10",
        "Mean Latency (ms)": "latency_ms",
    }
    col_key = metric_map[metric_choice]
    chart_df = pd.DataFrame(
        [
            {
                "Approach": METHOD_LABELS[row["key"]],
                metric_choice: row[col_key],
            }
            for row in OFFLINE_BENCHMARKS
        ]
    )
    st.caption(f"Offline {metric_choice} comparison across all 7 evaluated approaches on SciDocs ($k=10$).")
    st.bar_chart(chart_df.set_index("Approach"))


def _heatmap_figure(matrix: np.ndarray, labels: list[str], title: str, fmt: str, vmin, vmax, cmap: str):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        square=True,
        linewidths=0.4,
        mask=mask if mask.any() else None,
        cbar_kws={"shrink": 0.75},
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_overlap_section(engine: SearchEngine, compared: dict[str, SearchResponse], k: int) -> None:
    st.markdown("### Retrieval overlap analysis")
    st.caption(
        f"Pairwise Jaccard similarity across all seven approaches at cutoff $k = {k}$: "
        r"$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$. "
        "Calculated strictly using unique document IDs (titles and duplicates are never conflated)."
    )
    summary = summarize_overlap(compared, k=k, order=COMPARE_METHOD_ORDER)
    labels = [METHOD_LABELS[key] for key in COMPARE_METHOD_ORDER]
    fig = _heatmap_figure(
        summary.jaccard,
        labels,
        f"Jaccard Overlap Matrix (% of Shared IDs at Top-{k})",
        fmt=".0%",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Papers Retrieved", summary.total_unique_docs)
        st.caption(f"Union size across all 7 methods at top-{k}")
    with col2:
        st.metric("Complete Consensus Papers", len(summary.shared_by_all))
        st.caption("Papers retrieved by ALL 7 approaches")
    with col3:
        unique_total = sum(summary.unique_counts.values())
        st.metric("Total Single-Method Papers", unique_total)
        st.caption("Papers retrieved by only 1 approach")

    left, right = st.columns(2)
    with left:
        st.markdown("**Degree-of-Sharing Distribution**")
        st.caption("How many methods retrieved the same papers (sharing breakdown):")
        dist_df = pd.DataFrame(
            {
                "Category": list(summary.sharing_distribution.keys()),
                "Number of Papers": list(summary.sharing_distribution.values()),
            }
        )
        st.bar_chart(dist_df.set_index("Category"))

    with right:
        st.markdown("**Unique Papers Per Method**")
        st.caption("Number of papers retrieved exclusively by each method:")
        unique_df = pd.DataFrame(
            {
                "Approach": [METHOD_LABELS[key] for key in COMPARE_METHOD_ORDER],
                "Unique papers": [summary.unique_counts[key] for key in COMPARE_METHOD_ORDER],
            }
        )
        st.bar_chart(unique_df.set_index("Approach"))

    with st.expander("Inspect unique and consensus paper titles", expanded=False):
        if summary.shared_by_all:
            st.markdown(f"**Papers retrieved by all seven methods ({len(summary.shared_by_all)}):**")
            for doc_id in summary.shared_by_all:
                paper = engine.paper(doc_id)
                st.markdown(f"- `{doc_id}`: **{paper.title}** ({paper.citation_count} cites)")
        else:
            st.info("No single paper was retrieved by all seven methods at this cutoff.")

        st.markdown("**Exclusively retrieved papers by method:**")
        any_unique = False
        for key in COMPARE_METHOD_ORDER:
            u_ids = summary.unique_ids.get(key, [])
            if u_ids:
                any_unique = True
                st.markdown(f"- **{METHOD_LABELS[key]}** ({len(u_ids)} unique):")
                for uid in u_ids[:5]:
                    p = engine.paper(uid)
                    st.markdown(f"  - `{uid}`: {p.title} ({p.citation_count} cites)")
                if len(u_ids) > 5:
                    st.caption(f"  ... and {len(u_ids) - 5} more")
        if not any_unique:
            st.write("No method retrieved unique papers at this cutoff.")

    st.markdown("**Interactive Multi-Method Intersection**")
    st.caption("Select two or more approaches to inspect their shared documents:")
    selected = st.multiselect(
        "Intersect selected methods",
        options=list(COMPARE_METHOD_ORDER),
        default=list(COMPARE_METHOD_ORDER),
        format_func=lambda key: METHOD_LABELS[key],
        key="overlap_methods_multiselect",
    )
    shared = intersection_of(summary.method_ids, selected)
    st.write(
        f"**{len(shared)}** paper(s) appear in every selected method at top-{k}."
    )
    if shared:
        for doc_id in shared[:15]:
            paper = engine.paper(doc_id)
            st.markdown(f"- `{doc_id}` · **{paper.title}** ({paper.citation_count} citations)")


def render_ranking_section(engine: SearchEngine, compared: dict[str, SearchResponse], k: int) -> None:
    st.markdown("### Ranking analysis")
    st.caption(
        "Rank agreement measures whether approaches rank their shared documents in the same relative order. "
        "Missing documents are strictly excluded from pairwise rank correlation; they are never treated as rank zero."
    )
    rank_maps = {
        key: rank_map_from_results(compared[key].results, k) for key in COMPARE_METHOD_ORDER
    }
    labels = [METHOD_LABELS[key] for key in COMPARE_METHOD_ORDER]
    spearman, n_common = correlation_matrix(rank_maps, COMPARE_METHOD_ORDER, kind="spearman")
    kendall, _n_kendall = correlation_matrix(rank_maps, COMPARE_METHOD_ORDER, kind="kendall")

    corr_tab, kendall_tab = st.tabs(["Spearman rank correlation (ρ)", "Kendall's tau (τ)"])
    with corr_tab:
        fig = _heatmap_figure(
            spearman,
            labels,
            f"Spearman ρ on Shared Documents (top-{k})",
            fmt=".2f",
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            r"""
            **Spearman's $\rho$ Formula**:
            $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
            where $d_i = \text{rank}_A(d_i) - \text{rank}_B(d_i)$ is the rank difference on shared document $d_i$, and $n$ is the number of shared documents ($n \ge 2$).
            Grey/masked cells indicate fewer than two shared documents, where rank correlation is mathematically undefined.
            """
        )
    with kendall_tab:
        fig = _heatmap_figure(
            kendall,
            labels,
            f"Kendall's τ on Shared Documents (top-{k})",
            fmt=".2f",
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            r"""
            **Kendall's $\tau$ Formula**:
            $$\tau = \frac{C - D}{\frac{1}{2} n (n - 1)}$$
            where $C$ is the number of concordant pairs and $D$ is the number of discordant pairs.
            $\tau = 1.0$ indicates identical relative ordering, while $\tau = -1.0$ indicates completely inverted ordering.
            """
        )

    n_df = pd.DataFrame(n_common, index=labels, columns=labels)
    st.caption("Number of shared document IDs ($N_{\text{common}}$) available for each correlation pair:")
    st.dataframe(n_df, use_container_width=True)

    movers = rank_movers(rank_maps, COMPARE_METHOD_ORDER, min_methods=2, limit=20)
    if movers:
        st.markdown("#### Rank comparison across methods (Slope / Bump chart)")
        st.caption(
            "Visualises how the rank of the same document changes between approaches (rank #1 is best). "
            "Continuous lines join only consecutive retrieved methods to avoid false rank interpolation."
        )

        mover_options = [mover.doc_id for mover in movers]
        format_mover_title = lambda did: f"{engine.paper(did).title[:55]}… [spread: {next(m.spread for m in movers if m.doc_id == did)}]"

        selected_mover_ids = st.multiselect(
            "Select papers to highlight in plot (defaults to top 6 rank movers)",
            options=mover_options,
            default=mover_options[:6],
            format_func=format_mover_title,
            key="rank_movers_selected",
        )

        active_movers = [m for m in movers if m.doc_id in selected_mover_ids]
        if active_movers:
            fig, ax = plt.subplots(figsize=(10.0, 5.5))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(active_movers), 1)))

            for idx, mover in enumerate(active_movers):
                color = colors[idx % 10]
                label_text = f"#{idx+1}: {engine.paper(mover.doc_id).title[:35]}…"
                segments = rank_mover_segments(mover, COMPARE_METHOD_ORDER)
                plotted_label = False
                for seg_xs, seg_ys in segments:
                    ax.plot(
                        seg_xs,
                        seg_ys,
                        marker="o",
                        linewidth=2.0,
                        markersize=6,
                        color=color,
                        label=label_text if not plotted_label else None,
                    )
                    plotted_label = True

            ax.set_xticks(list(range(len(COMPARE_METHOD_ORDER))))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.invert_yaxis()
            ax.set_ylabel("Rank Position (1 = Best)", fontsize=10)
            ax.set_ylim(k + 1, 0.5)
            ax.set_title(f"Rank Movement Across Approaches (Top-{k})", fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("**Rank Movement Table (Papers in 2+ methods)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Paper Title": engine.paper(item.doc_id).title,
                        "ID": item.doc_id,
                        "Rank Spread": item.spread,
                        **{METHOD_LABELS[m]: f"#{item.ranks[m]}" if m in item.ranks else "—" for m in COMPARE_METHOD_ORDER},
                    }
                    for item in movers
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No paper appears in two or more methods at this cutoff, so a rank comparison plot is not available.")

    insights = build_insights(summarize_overlap(compared, k=k), rank_maps, COMPARE_METHOD_ORDER, METHOD_LABELS)
    st.markdown("#### Automated Query Insights & Observations")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    for note in insights.notes:
        st.markdown(f"- {note}")
    st.markdown("</div>", unsafe_allow_html=True)


def comparison_cache_key(query: str, params: dict) -> tuple:
    return (
        query,
        float(params["boost_weight"]),
        int(params["year_min"]),
        int(params["year_max"]),
        int(params["min_citations"]),
        int(COMPARE_CACHE_MAX_K),
    )


def get_compared_results(engine: SearchEngine, query: str, params: dict) -> dict[str, SearchResponse]:
    key = comparison_cache_key(query, params)
    cached = st.session_state.get("_compare_cache")
    if cached and cached.get("key") == key:
        return cached["value"]
    compared = engine.compare(
        query,
        k=COMPARE_CACHE_MAX_K,
        boost_weight=params["boost_weight"],
        year_min=params["year_min"],
        year_max=params["year_max"],
        min_citations=params["min_citations"],
        candidate_k=max(COMPARE_CACHE_MAX_K, engine.settings.candidate_k),
    )
    st.session_state["_compare_cache"] = {"key": key, "value": compared}
    return compared


COMPARE_VIEWS = (
    "Overview (all 7)",
    "Head-to-head",
    "Single method",
    "Benchmarks & deep analysis",
)


def render_comparison_dashboard(engine: SearchEngine, compared: dict[str, SearchResponse]) -> None:
    st.markdown("## Comparison Dashboard")
    st.caption(
        "Seven retrieval methods on the same query: BM25, TF-IDF, Dense, Citation-boosted, "
        "Cross-encoder, RRF and Weighted. Results are cached, so switching view, method or cutoff "
        "does not re-run any retriever."
    )
    top_row = st.columns((3, 1))
    with top_row[0]:
        view = st.radio("View", COMPARE_VIEWS, horizontal=True, key="compare_view", label_visibility="collapsed")
    with top_row[1]:
        top_k = st.radio("Top-k cutoff", options=[10, 50], horizontal=True, key="compare_top_k")
    k = int(top_k)
    st.markdown("---")

    if view == COMPARE_VIEWS[0]:
        render_rank_matrix(st, engine, compared, k=k)
    elif view == COMPARE_VIEWS[1]:
        render_head_to_head(st, engine, compared, k=k)
    elif view == COMPARE_VIEWS[2]:
        selected = st.selectbox(
            "Retrieval method",
            options=list(COMPARE_METHOD_ORDER),
            format_func=lambda key: f"{METHOD_LABELS[key]} ({METHOD_SPECS.get(key, {}).get('category', '')})",
            key="compare_method_select",
        )
        render_method_results(engine, compared[selected], k=k, key_prefix=f"cmp-{selected}")
    else:
        render_benchmark_section(compared)
        st.markdown("---")
        render_overlap_section(engine, compared, k=k)
        st.markdown("---")
        render_ranking_section(engine, compared, k=k)


def sidebar_controls(engine: SearchEngine) -> dict:
    settings = engine.settings
    min_year, max_year = year_bounds(engine.catalog)
    st.sidebar.header("Ranking & filters")
    compare = st.sidebar.toggle(
        "Comparison mode",
        value=False,
        help="Compare seven retrieval approaches on the same query. Default search stays citation-boosted hybrid.",
    )
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
    if compare:
        top_k = int(settings.k)
        st.sidebar.caption("Comparison mode uses the Top-10 / Top-50 control in the main panel.")
    else:
        top_k = st.sidebar.slider("Results per method", min_value=1, max_value=25, value=int(settings.k))
    st.sidebar.caption(
        f"Hybrid weights: BM25 {settings.bm25_weight:.2f} + dense {settings.dense_weight:.2f}. "
        f"RRF k = {settings.rrf_k}. FAISS index: {settings.index_type}. Model: {settings.dense_model}."
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
    try:
        if controls["compare"]:
            with st.spinner(
                "Ranking the query with seven approaches. "
                "The first comparison may build TF-IDF and load the cross-encoder on CPU…"
            ):
                compared = get_compared_results(engine, active_query, params)
            render_comparison_dashboard(engine, compared)
        else:
            with st.spinner("Retrieving and ranking papers…"):
                response = engine.search(active_query, **params)
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
    except Exception as exc:  # noqa: BLE001
        st.error(f"Search failed: {type(exc).__name__}: {exc}")
        return


if __name__ == "__main__":
    main()