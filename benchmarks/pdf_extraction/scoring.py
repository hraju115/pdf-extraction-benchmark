"""Score extractor outputs against ground truth and collect run statistics into CSV files.

    python -m benchmarks.pdf_extraction.scoring
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmarks.pdf_extraction.corpus.make_scans import read_page_truth
from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry
from benchmarks.pdf_extraction.extractors.base import RunInfo, read_run
from benchmarks.pdf_extraction.metrics.facts import fact_recall
from benchmarks.pdf_extraction.metrics.tables import extract_tables, score_tables
from benchmarks.pdf_extraction.metrics.text import mean_page_cer, mean_page_similarity
from benchmarks.pdf_extraction.paths import OUT_DIR, RESULTS_DIR, TRUTH_DIR

SCORE_FIELDS = ["extractor", "doc_id", "tier", "bucket", "source", "status", "sampled", "metric", "value"]
RUN_FIELDS = ["extractor", "doc_id", "tier", "bucket", "source", "machine", "status", "sampled", "pages", "seconds",
              "seconds_per_page", "peak_rss_mb", "version", "error"]


def iter_runs(out_root: Path):
    """Yield (extractor, doc_id, out_dir, RunInfo) for every finished run under ``out_root``."""
    for extractor_dir in sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith(".")) if out_root.exists() else []:
        for doc_dir in sorted(p for p in extractor_dir.iterdir() if p.is_dir()):
            run = read_run(doc_dir)
            if run is not None:
                yield extractor_dir.name, doc_dir.name, doc_dir, run


def _pred_pages(doc_dir: Path) -> dict[int, str]:
    path = doc_dir / "pages.json"
    if not path.exists():
        return {}
    return {p["page"]: p["text"] for p in json.loads(path.read_text(encoding="utf-8"))}


def load_table_truth(path: Path) -> list[tuple[int | None, str]]:
    """Table truth as (page, html) pairs. Legacy files hold plain HTML strings (page unknown -> None)."""
    items = json.loads(path.read_text(encoding="utf-8"))
    truth: list[tuple[int | None, str]] = []
    for item in items:
        if isinstance(item, str):
            truth.append((None, item))
        else:
            truth.append((item.get("page"), item["html"]))
    return truth


def _in_range(page: int | None, run: RunInfo) -> bool:
    """A sampled run is judged only on truth items whose page it was asked to read (unknown pages always count)."""
    if page is None or not run.page_range:
        return True
    first, last = run.page_range
    return first <= page <= last


def _processed_truth(truth: dict[int, str], run: RunInfo) -> dict[int, str]:
    """Only pages the run was asked to process count: a sampled run is not penalized for the rest."""
    if run.page_range:
        first, last = run.page_range
        return {page: text for page, text in truth.items() if first <= page <= last}
    return truth


def score_extraction(out_root: Path, truth_dir: Path, entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    """One row per (extractor, document, metric). A failed run scores 0 on every metric it has truth for."""
    by_id = {e.id: e for e in entries}
    rows: list[dict[str, Any]] = []

    for extractor, doc_id, doc_dir, run in iter_runs(out_root):
        entry = by_id.get(doc_id)
        ok = run.status == "ok"
        markdown = (doc_dir / "result.md").read_text(encoding="utf-8") if ok and (doc_dir / "result.md").exists() else ""
        base = {
            "extractor": extractor, "doc_id": doc_id, "tier": entry.tier if entry else "", "bucket": entry.bucket if entry else "",
            "source": entry.source if entry else "",
            "status": run.status, "sampled": run.sampled,
        }

        def add(metric: str, value: float) -> None:
            rows.append({**base, "metric": metric, "value": round(value, 4)})

        pages_truth = truth_dir / f"{doc_id}.pages.json"
        if pages_truth.exists():
            truth = _processed_truth(read_page_truth(pages_truth), run)
            pred = _pred_pages(doc_dir) if ok else {}
            similarity = mean_page_similarity(pred, truth)
            if similarity is not None:  # None: no page of this document has truth text
                add("text_similarity", similarity)
                add("cer", (mean_page_cer(pred, truth) or 0.0) if ok else 1.0)

        tables_truth = truth_dir / f"{doc_id}.tables.json"
        if tables_truth.exists():
            tables = [html for page, html in load_table_truth(tables_truth) if _in_range(page, run)]
            if tables:
                scores = score_tables(extract_tables(markdown), tables)
                add("teds", scores["teds"])
                add("table_detection_recall", scores["detection_recall"])

        figures_truth = truth_dir / f"{doc_id}.figures.json"
        if figures_truth.exists():
            figures = [f for f in json.loads(figures_truth.read_text(encoding="utf-8")) if _in_range(f.get("page"), run)]
            if figures:
                add("figure_fact_recall", sum(fact_recall(markdown, f["facts"]) for f in figures) / len(figures))
    return rows


def machine_label(hardware: dict[str, Any] | None) -> str:
    """'cpu', or 'gpu:<model>' from the run's recorded hardware, so runs from different machines stay distinguishable."""
    gpu = (hardware or {}).get("gpu")
    return f"gpu:{str(gpu).split(',')[0].strip()}" if gpu else "cpu"


def collect_runs(out_root: Path, entries: list[ManifestEntry]) -> list[dict[str, Any]]:
    by_id = {e.id: e for e in entries}
    rows = []
    for extractor, doc_id, _dir, run in iter_runs(out_root):
        entry = by_id.get(doc_id)
        rows.append({
            "extractor": extractor, "doc_id": doc_id, "tier": entry.tier if entry else "", "bucket": entry.bucket if entry else "",
            "source": entry.source if entry else "", "machine": machine_label(run.hardware),
            "status": run.status, "sampled": run.sampled, "pages": run.pages_processed, "seconds": run.seconds,
            "seconds_per_page": round(run.seconds / run.pages_processed, 3) if run.pages_processed else "",
            "peak_rss_mb": run.peak_rss_mb, "version": run.version, "error": run.error[:200].replace("\n", " "),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    from benchmarks.pdf_extraction.corpus.build import all_entries

    entries = all_entries()
    write_csv(RESULTS_DIR / "extraction_scores.csv", score_extraction(OUT_DIR, TRUTH_DIR, entries), SCORE_FIELDS)
    write_csv(RESULTS_DIR / "runs.csv", collect_runs(OUT_DIR, entries), RUN_FIELDS)
    print(f"wrote {RESULTS_DIR}/extraction_scores.csv and runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
