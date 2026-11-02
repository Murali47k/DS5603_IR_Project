"""
USER INTERFACE MODULE
Scope: Interactive UI for retrieval tasks and scientific document visualization.

TODOs for UI Development:
1. Core Search Engine:
   - Use Citation-Boosted Hybrid Retrieval (BM25 + FAISS Dense + Citation Boost, w=0.4)
     as the default search engine, as it achieved the top benchmark score (0.22877 nDCG@10).
   - Module reference: `src.optimisations.citation_boost.citation_aware_boost`

2. Result Cards & Metadata Visualization:
   - Display Paper Title, Authors, Year, Abstract snippet, and a prominent Citation Count badge.
   - Show relevance score breakdown (Lexical + Semantic + Citation Boost multiplier).

3. Citation & Graph Exploration:
   - "Cited By" / "References" modal: Inspect which papers cite or are cited by the selected document.
   - "Find Similar Papers": One-click button taking the current paper embedding to find top-5 nearest neighbors.

4. Interactive Controls & Filtering:
   - Dynamic Citation Boost slider (w in [0.0, 1.0]) to re-rank results in real-time.
   - Filters for publication year range and minimum citation threshold.

5. (Bonus) Model Comparison View:
   - Optional toggle/split-view comparing pure BM25 vs. Dense vs. Citation Hybrid.
"""
