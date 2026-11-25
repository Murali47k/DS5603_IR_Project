"""Readable views for comparison mode: rank matrix and head-to-head.

Pure helpers (``build_rank_table``, ``movement``, ``head_to_head_rows``) have no
Streamlit dependency so they can be unit-tested. The ``render_*`` functions are
thin wrappers around them.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.ui.comparison import (
    COMPARE_METHOD_ORDER,
    jaccard,
    rank_map_from_results,
    summarize_overlap,
)

SHORT_LABELS: Dict[str, str] = {
    "bm25": "BM25",
    "tfidf": "TF-IDF",
    "dense": "Dense",
    "citation_hybrid": "Citation-boosted",
    "cross_encoder": "Cross-encoder",
    "rrf": "RRF",
    "weighted": "Weighted",
}

# Fairness note: some methods are not independent retrievers.
METHOD_NOTES: Dict[str, str] = {
    "citation_hybrid": "Weighted fusion of BM25 + Dense, then citation boost.",
    "cross_encoder": "Reranks the top-50 Weighted (BM25 + Dense) candidates, so it can only reorder papers that Weighted already found.",
    "rrf": "Rank fusion of BM25 + Dense.",
    "weighted": "Min-max normalised BM25 + Dense, 0.4 / 0.6.",
}

COMPARE_CSS = """
<style>
.h2h-head { font-weight: 700; font-size: 1.0rem; margin: 0 0 0.4rem; }
.h2h-row {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.5rem 0.65rem; margin-bottom: 0.35rem;
    border: 1px solid #d9e2ec; border-radius: 10px; background: #ffffff;
}
.h2h-rank { min-width: 2rem; font-weight: 700; color: #3d5a80; }
.h2h-body { flex: 1; min-width: 0; }
.h2h-title { color: #102a43; font-weight: 600; font-size: 0.92rem; line-height: 1.3; }
.h2h-meta { color: #627d98; font-size: 0.76rem; margin-top: 0.1rem; }
.mv {
    display: inline-block; min-width: 3.1rem; text-align: center;
    padding: 0.12rem 0.45rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; white-space: nowrap;
}
.mv-up { background: #dcfce7; color: #166534; }
.mv-down { background: #fee2e2; color: #991b1b; }
.mv-same { background: #f1f5f9; color: #475569; }
.mv-new { background: #fef3c7; color: #92400e; }
.mv-gone { background: #e2e8f0; color: #475569; }
</style>
"""


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def rank_maps_for(compared: Mapping[str, object], k: int, order: Sequence[str] = COMPARE_METHOD_ORDER) -> Dict[str, Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    for key in order:
        response = compared.get(key)
        results = getattr(response, "results", None) or []
        maps[key] = rank_map_from_results(results, k)
    return maps


def build_rank_table(
    rank_maps: Mapping[str, Mapping[str, int]],
    order: Sequence[str] = COMPARE_METHOD_ORDER,
) -> pd.DataFrame:
    """Rows = union of every method's top-k. Columns = methods (rank or NaN).

    Sorted by how many methods returned the paper, then by mean rank, so the
    papers everyone agrees on sit at the top.
    """
    doc_ids = sorted({doc_id for key in order for doc_id in rank_maps.get(key, {})})
    data = {
        key: [rank_maps.get(key, {}).get(doc_id, np.nan) for doc_id in doc_ids]
        for key in order
    }
    df = pd.DataFrame(data, index=pd.Index(doc_ids, name="doc_id"), dtype="float64")
    if df.empty:
        df["found_by"] = pd.Series(dtype="int64")
        df["mean_rank"] = pd.Series(dtype="float64")
        return df
    df["found_by"] = df[list(order)].notna().sum(axis=1).astype(int)
    df["mean_rank"] = df[list(order)].mean(axis=1, skipna=True)
    return df.sort_values(["found_by", "mean_rank"], ascending=[False, True], kind="mergesort")


@dataclass
class Movement:
    kind: str      # up | down | same | new
    delta: int     # positive = moved up (better rank) in B vs A
    text: str
    css: str


def movement(rank_a: Optional[int], rank_b: int) -> Movement:
    """Movement of a paper in list B relative to list A."""
    if rank_a is None:
        return Movement("new", 0, "NEW", "mv-new")
    delta = int(rank_a) - int(rank_b)
    if delta > 0:
        return Movement("up", delta, f"▲ {delta}", "mv-up")
    if delta < 0:
        return Movement("down", delta, f"▼ {-delta}", "mv-down")
    return Movement("same", 0, "＝", "mv-same")


def head_to_head_rows(
    ranks_a: Mapping[str, int], ranks_b: Mapping[str, int]
) -> Tuple[List[Tuple[str, int, Movement]], List[Tuple[str, int, Optional[int]]]]:
    """Return (rows for B with movement vs A, rows for A with rank in B or None)."""
    b_rows = [
        (doc_id, rank, movement(ranks_a.get(doc_id), rank))
        for doc_id, rank in sorted(ranks_b.items(), key=lambda kv: kv[1])
    ]
    a_rows = [
        (doc_id, rank, ranks_b.get(doc_id))
        for doc_id, rank in sorted(ranks_a.items(), key=lambda kv: kv[1])
    ]
    return b_rows, a_rows


def _rank_color(rank: float, k: int) -> str:
    """Solid colour (light base) so dark text stays readable in light and dark themes."""
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return "background-color: #e5e7eb; color: #4b5563; font-weight: 700"
    t = min(max((float(rank) - 1.0) / max(k - 1, 1), 0.0), 1.0)
    # rank 1 -> strong green, rank k -> pale green
    r = int(26 + (226 - 26) * t ** 0.7)
    g = int(152 + (240 - 152) * t ** 0.7)
    b = int(80 + (230 - 80) * t ** 0.7)
    return f"background-color: rgb({r},{g},{b}); color: #0b2e1a; font-weight: 600"


def _style_map(styler, func, subset):
    fn = getattr(styler, "map", None) or styler.applymap
    return fn(func, subset=subset)


# --------------------------------------------------------------------------
# Streamlit renderers
# --------------------------------------------------------------------------

def _paper_line(engine, doc_id: str) -> Tuple[str, str]:
    paper = engine.paper(doc_id)
    year = f" · {paper.year}" if paper.year else ""
    return paper.title, f"{paper.citation_count} citations{year} · {doc_id}"


def render_rank_matrix(st, engine, compared: Mapping[str, object], k: int) -> None:
    order = list(COMPARE_METHOD_ORDER)
    rank_maps = rank_maps_for(compared, k, order)
    table = build_rank_table(rank_maps, order)

    st.markdown("#### All 7 methods at a glance")
    st.caption(
        f"Each row is a paper that appears in at least one method's top-{k}. "
        "The number is that method's rank for the paper (1 = best). "
        f"A grey cell with a dash means the method did not return it in its top-{k}. "
        "Green cells are ranked high and pale cells are ranked low. "
        "Papers at the top are the ones most methods agree on. Click a column header to sort."
    )

    empty = [SHORT_LABELS[m] for m in order if not rank_maps[m]]
    if empty:
        details = "; ".join(
            f"{SHORT_LABELS[m]}: {getattr(compared.get(m), 'error', '') or 'no results'}" for m in order if not rank_maps[m]
        )
        st.warning(f"No results for: {', '.join(empty)}. {details}")

    if table.empty:
        st.info("No papers were returned by any method for this query and filters.")
        return

    c1, c2 = st.columns(2)
    with c1:
        min_found = st.slider(
            "Show papers returned by at least N methods",
            min_value=1, max_value=len(order), value=1, key="matrix_min_found",
            help="Set to 7 to see only papers every method agrees on.",
        )
    with c2:
        row_cap = st.slider(
            "Rows to show", min_value=10, max_value=100, value=30, step=10, key="matrix_rows",
        )

    view = table[table["found_by"] >= min_found]
    total_rows = len(view)
    view = view.head(int(row_cap))

    display = pd.DataFrame(
        {
            "Paper": [engine.paper(d).title for d in view.index],
            "Cites": [engine.paper(d).citation_count for d in view.index],
            **{SHORT_LABELS[m]: view[m].values for m in order},
            "Found by": view["found_by"].values,
        }
    )
    method_cols = [SHORT_LABELS[m] for m in order]
    styler = display.style.format({c: "{:.0f}" for c in method_cols}, na_rep="-")
    styler = _style_map(styler, lambda v: _rank_color(v, k), subset=method_cols)
    column_config = {
        "Paper": st.column_config.TextColumn("Paper", width="medium"),
        "Cites": st.column_config.NumberColumn("Cites", width="small"),
        "Found by": st.column_config.NumberColumn("Found by", width="small"),
        **{c: st.column_config.NumberColumn(c, width="small") for c in method_cols},
    }
    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=True,
        column_config=column_config,
        height=min(80 + 36 * len(display), 760),
    )
    st.caption(f"Showing {len(display)} of {total_rows} papers · union of the 7 top-{k} lists has {len(table)} papers.")

    # --- compact overlap summary -------------------------------------------------
    st.markdown("#### How similar are the lists? (top-k overlap)")
    summary = summarize_overlap(compared, k=k, order=order)
    labels = [SHORT_LABELS[m] for m in order]
    jac = pd.DataFrame(summary.jaccard, index=labels, columns=labels)

    def _blue(v: float) -> str:
        t = min(max(float(v), 0.0), 1.0)
        r = int(241 - (241 - 59) * t)
        g = int(245 - (245 - 130) * t)
        b = int(249 - (249 - 246) * t)
        txt = "#ffffff" if t > 0.6 else "#0f172a"
        return f"background-color: rgb({r},{g},{b}); color: {txt}"

    jstyler = _style_map(jac.style.format("{:.0%}"), _blue, subset=labels)
    st.dataframe(jstyler, use_container_width=True)
    st.caption(
        "Jaccard overlap = shared papers ÷ papers in either list. 100% = identical sets "
        "(order ignored). Low values mean the methods retrieve different papers."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Papers in the union", summary.total_unique_docs)
    m2.metric("Returned by all 7", len(summary.shared_by_all))
    m3.metric("Returned by only 1 method", sum(summary.unique_counts.values()))

    notes = [f"**{SHORT_LABELS[m]}**: {METHOD_NOTES[m]}" for m in order if m in METHOD_NOTES]
    with st.expander("How the hybrid methods are built (read before comparing)", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")
        st.markdown(
            "- **BM25, TF-IDF, Dense** are independent retrievers. The other four reuse BM25 + Dense "
            "candidates, so they naturally overlap with each other."
        )


def _render_h2h_list(st, engine, rows, header: str, kind: str) -> None:
    st.markdown(f'<div class="h2h-head">{html.escape(header)}</div>', unsafe_allow_html=True)
    parts: List[str] = []
    for doc_id, rank, extra in rows:
        title, meta = _paper_line(engine, doc_id)
        if kind == "b":
            badge = f'<span class="mv {extra.css}">{html.escape(extra.text)}</span>'
        else:
            if extra is None:
                badge = '<span class="mv mv-gone">dropped</span>'
            else:
                badge = f'<span class="mv mv-same">→ #{extra}</span>'
        parts.append(
            '<div class="h2h-row">'
            f'<div class="h2h-rank">#{rank}</div>'
            f'<div class="h2h-body"><div class="h2h-title">{html.escape(title)}</div>'
            f'<div class="h2h-meta">{html.escape(meta)}</div></div>'
            f"{badge}</div>"
        )
    st.markdown("".join(parts) or "<p>No results.</p>", unsafe_allow_html=True)


def render_head_to_head(st, engine, compared: Mapping[str, object], k: int) -> None:
    order = list(COMPARE_METHOD_ORDER)
    st.markdown(COMPARE_CSS, unsafe_allow_html=True)
    st.markdown("#### Head-to-head")
    st.caption(
        "Pick a **baseline (A)** and a **method (B)**. The right-hand list shows how each paper in B "
        "moved compared with A: ▲ moved up, ▼ moved down, ＝ same rank, NEW = not in A's top-"
        f"{k}. The left-hand list shows where A's papers went in B."
    )
    c1, c2 = st.columns(2)
    with c1:
        key_a = st.selectbox(
            "Method A (baseline)", order, index=order.index("bm25"),
            format_func=lambda m: SHORT_LABELS[m], key="h2h_a",
        )
    with c2:
        key_b = st.selectbox(
            "Method B (compare against A)", order, index=order.index("citation_hybrid"),
            format_func=lambda m: SHORT_LABELS[m], key="h2h_b",
        )
    for key in dict.fromkeys((key_a, key_b)):
        if key in METHOD_NOTES:
            st.caption(f"ℹ️ **{SHORT_LABELS[key]}**: {METHOD_NOTES[key]}")
    if key_a == key_b:
        st.info("Choose two different methods to see how rankings change.")
        return

    ranks_a = rank_map_from_results(getattr(compared.get(key_a), "results", None) or [], k)
    ranks_b = rank_map_from_results(getattr(compared.get(key_b), "results", None) or [], k)
    if not ranks_a or not ranks_b:
        empty = SHORT_LABELS[key_a] if not ranks_a else SHORT_LABELS[key_b]
        err = getattr(compared.get(key_a if not ranks_a else key_b), "error", "")
        st.warning(f"{empty} returned no results. {err}")
        return

    shared = set(ranks_a) & set(ranks_b)
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Shared in top-{k}", f"{len(shared)} / {k}")
    m2.metric("Jaccard", f"{jaccard(ranks_a, ranks_b):.0%}")
    m3.metric("NEW in B", len(set(ranks_b) - set(ranks_a)))

    b_rows, a_rows = head_to_head_rows(ranks_a, ranks_b)
    left, right = st.columns(2)
    with left:
        _render_h2h_list(st, engine, a_rows, f"A · {SHORT_LABELS[key_a]}", "a")
    with right:
        _render_h2h_list(st, engine, b_rows, f"B · {SHORT_LABELS[key_b]}", "b")