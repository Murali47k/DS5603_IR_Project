"""Paper metadata and citation-graph helpers for the SciDocs UI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.data_loader import DEFAULT_DATA_DIR


@dataclass
class PaperRecord:
    doc_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    citation_count: int = 0
    cited_by: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    def snippet(self, max_chars: int = 320) -> str:
        text = " ".join((self.abstract or "").split())
        if not text:
            return "No abstract available."
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "…"


def _as_id_list(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _as_authors(value: Any) -> List[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, (list, tuple)):
        authors = []
        for item in value:
            if item is None:
                continue
            authors.append(str(item).strip())
        return [name for name in authors if name]
    return []


def _as_year(value: Any) -> Optional[int]:
    if isinstance(value, int) and 1000 <= value <= 2100:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        year = int(value.strip())
        if 1000 <= year <= 2100:
            return year
    return None


def parse_jsonl_record(entry: Mapping[str, Any]) -> Optional[PaperRecord]:
    doc_id = str(entry.get("_id") or entry.get("doc_id") or "").strip()
    if not doc_id:
        return None
    meta = entry.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    title = str(entry.get("title") or meta.get("title") or "").strip()
    abstract = str(entry.get("text") or entry.get("abstract") or "").strip()
    authors = _as_authors(meta.get("authors") or entry.get("authors"))
    year = _as_year(meta.get("year") or entry.get("year"))
    cited_by = _as_id_list(meta.get("cited_by") or entry.get("cited_by"))
    references = _as_id_list(meta.get("references") or entry.get("references"))
    return PaperRecord(
        doc_id=doc_id,
        title=title or "Untitled paper",
        abstract=abstract,
        authors=authors,
        year=year,
        citation_count=len(cited_by),
        cited_by=cited_by,
        references=references,
    )


def load_jsonl_records(path: Path) -> Dict[str, PaperRecord]:
    records: Dict[str, PaperRecord] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = parse_jsonl_record(entry)
            if record is not None:
                records[record.doc_id] = record
    return records


def load_paper_catalog(
    corpus: Mapping[str, Mapping[str, Any]],
    data_dir: Optional[Path] = None,
) -> Dict[str, PaperRecord]:
    """
    Build UI paper records from the BEIR corpus plus SciDocs JSONL metadata.

    BEIR's GenericDataLoader keeps title/text and drops authors, year, and
    citation lists. Those fields are read from corpus.jsonl / queries.jsonl.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    catalog: Dict[str, PaperRecord] = {}
    for doc_id, doc in corpus.items():
        catalog[str(doc_id)] = PaperRecord(
            doc_id=str(doc_id),
            title=str((doc or {}).get("title") or "").strip() or "Untitled paper",
            abstract=str((doc or {}).get("text") or "").strip(),
        )

    jsonl_records = {}
    jsonl_records.update(load_jsonl_records(data_dir / "scidocs" / "queries.jsonl"))
    jsonl_records.update(load_jsonl_records(data_dir / "scidocs" / "corpus.jsonl"))
    for doc_id, extra in jsonl_records.items():
        current = catalog.get(doc_id)
        if current is None:
            catalog[doc_id] = extra
            continue
        current.authors = extra.authors
        current.year = extra.year
        current.cited_by = extra.cited_by
        current.references = extra.references
        if extra.title and current.title == "Untitled paper":
            current.title = extra.title
        if extra.abstract and not current.abstract:
            current.abstract = extra.abstract

    # Citation counts come from SciDocs `cited_by` lists, never from qrels.
    for rec in catalog.values():
        rec.citation_count = len(rec.cited_by)
    return catalog


def citation_counts_from_catalog(
    catalog: Mapping[str, PaperRecord],
    doc_ids: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    if doc_ids is None:
        return {doc_id: rec.citation_count for doc_id, rec in catalog.items()}
    return {doc_id: catalog.get(doc_id, PaperRecord(doc_id, "", "")).citation_count for doc_id in doc_ids}


def lookup_papers(
    catalog: Mapping[str, PaperRecord],
    doc_ids: Sequence[str],
) -> List[PaperRecord]:
    results = []
    for doc_id in doc_ids:
        record = catalog.get(doc_id)
        if record is None:
            results.append(
                PaperRecord(
                    doc_id=doc_id,
                    title="Paper not present in the loaded SciDocs corpus",
                    abstract="",
                )
            )
        else:
            results.append(record)
    return results


def year_bounds(catalog: Mapping[str, PaperRecord]) -> tuple[int, int]:
    years = [rec.year for rec in catalog.values() if rec.year is not None]
    if not years:
        return 1900, 2026
    low, high = min(years), max(years)
    if low == high:
        return low, high + 1
    return low, high
