"""Build the comparison report from the CSVs written by the scoring stages.

    python -m benchmarks.pdf_extraction.report.build_report
Writes report/out/report.md, plots/*.png and failure_gallery.html.
"""

from __future__ import annotations

import base64
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

from benchmarks.pdf_extraction.corpus.make_scans import read_page_truth  # noqa: E402
from benchmarks.pdf_extraction.extractors.base import hardware_info  # noqa: E402
from benchmarks.pdf_extraction.metrics.text import edit_similarity  # noqa: E402
from benchmarks.pdf_extraction.config import BenchConfig, load_config  # noqa: E402
from benchmarks.pdf_extraction.paths import BENCH_DIR, CONFIG_PATH, OUT_DIR, PDF_DIR, RESULTS_DIR, TRUTH_DIR  # noqa: E402
from benchmarks.pdf_extraction.scoring import iter_runs, machine_label, read_csv  # noqa: E402

REPORT_DIR = BENCH_DIR / "report" / "out"
TIERS = ("T1", "T2", "T3", "T4")
BUCKETS = ("S", "M", "L", "XL")
SOURCES = ("real", "scan", "generated")

#: Primary extraction metric per tier and whether higher is better.
TIER_METRIC = {"T1": ("text_similarity", True), "T2": ("teds", True), "T3": ("cer", False), "T4": ("figure_fact_recall", True)}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pivot(rows: list[dict[str, Any]], row_key: str, col_key: str, value_key: str,
          where: Callable[[dict[str, Any]], bool] = lambda r: True) -> dict[str, dict[str, float | None]]:
    """Mean of ``value_key`` grouped by (row_key, col_key); empty numeric cells are skipped."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not where(row) or row.get(value_key) in ("", None):
            continue
        cells[(row[row_key], row[col_key])].append(float(row[value_key]))
    out: dict[str, dict[str, float | None]] = defaultdict(dict)
    for (r, c), values in cells.items():
        out[r][c] = mean(values)
    return out


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def pivot_table(data: dict[str, dict[str, float | None]], columns: tuple[str, ...], digits: int = 2) -> str:
    return md_table(["extractor", *columns], [[name, *[fmt(cells.get(c), digits) for c in columns]] for name, cells in sorted(data.items())])


def label_sampled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sampled runs get their own label so they are never averaged with full runs (spec 3.2)."""
    return [{**r, "extractor": f"{r['extractor']} (sampled)"} if str(r.get("sampled")) == "True" else r for r in rows]


# ── speed ────────────────────────────────────────────────────────────────────


def fit_speed(run_rows: list[dict[str, Any]], group_key: str | None = None) -> dict[Any, tuple[float, float, int]]:
    """Least-squares seconds = fixed + per_page * pages over full ok runs; (fixed, per_page, n_docs) per extractor.

    With ``group_key`` the keys are (extractor, row[group_key]). With fewer than 3 documents the fit is unstable,
    so fixed is 0 and per_page is the mean seconds/page.
    """
    points: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for r in run_rows:
        if r["status"] == "ok" and str(r.get("sampled")) != "True" and float(r["pages"] or 0) > 0:
            key = (r["extractor"], r[group_key]) if group_key else r["extractor"]
            points[key].append((float(r["pages"]), float(r["seconds"])))
    fit: dict[Any, tuple[float, float, int]] = {}
    for key, pts in points.items():
        if len(pts) < 3 or len({p for p, _ in pts}) < 2:  # a line through one page count is undefined
            fit[key] = (0.0, sum(s / p for p, s in pts) / len(pts), len(pts))
            continue
        x = np.array([p for p, _ in pts])
        y = np.array([s for _, s in pts])
        slope, intercept = np.polyfit(x, y, 1)
        fit[key] = (max(0.0, float(intercept)), max(0.0, float(slope)), len(pts))
    return fit


def per_page_table(fit: dict[tuple[str, str], tuple[float, float, int]], columns: tuple[str, ...]) -> str:
    """Pivot a grouped ``fit_speed`` result into extractor x group cells of fitted seconds per page."""
    data: dict[str, dict[str, float | None]] = defaultdict(dict)
    for (name, group), (_fixed, per_page, _n) in fit.items():
        data[name][group] = per_page
    return pivot_table(data, columns, 3)


# ── recommendation ───────────────────────────────────────────────────────────


def recommend(retrieval_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]], chunk_mode: str) -> dict[str, Any]:
    """Best extractor per tier, best single extractor, and the hit@5 a per-tier router would have achieved."""
    rows = [r for r in retrieval_rows if r["chunk_mode"] == chunk_mode and not r["extractor"].endswith("(sampled)")]
    seconds: dict[str, list[float]] = defaultdict(list)
    for run in run_rows:
        if run["seconds_per_page"] not in ("", None):
            seconds[run["extractor"]].append(float(run["seconds_per_page"]))
    speed = {name: mean(values) or 0.0 for name, values in seconds.items()}

    by_tier = pivot(rows, "extractor", "tier", "hit5")
    best_by_tier: dict[str, str] = {}
    for tier in TIERS:
        scored = [(-(cells[tier]), speed.get(name, 0.0), name) for name, cells in by_tier.items() if cells.get(tier) is not None]
        if scored:
            best_by_tier[tier] = sorted(scored)[0][2]

    overall = {name: mean([float(r["hit5"]) for r in rows if r["extractor"] == name]) for name in {r["extractor"] for r in rows}}
    best_single = max(overall.items(), key=lambda kv: (kv[1] or 0.0, kv[0]), default=(None, None))
    routed = [float(r["hit5"]) for r in rows if best_by_tier.get(r["tier"]) == r["extractor"]]
    router = mean(routed)
    gain = (router - best_single[1]) if router is not None and best_single[1] is not None else None
    return {"best_by_tier": best_by_tier, "best_single": best_single, "router_estimate": router, "router_gain": gain}


# ── failure gallery ──────────────────────────────────────────────────────────


def worst_pages(out_root: Path, truth_dir: Path, per_extractor: int = 3) -> list[dict[str, Any]]:
    """The lowest-similarity pages for each extractor, across documents that have page-level text truth."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for extractor, doc_id, doc_dir, run in iter_runs(out_root):
        truth_path = truth_dir / f"{doc_id}.pages.json"
        if run.status != "ok" or not truth_path.exists() or not (doc_dir / "pages.json").exists():
            continue
        truth = read_page_truth(truth_path)
        pred = {p["page"]: p["text"] for p in json.loads((doc_dir / "pages.json").read_text(encoding="utf-8"))}
        for page, truth_text in truth.items():
            if page in pred:
                found[extractor].append({"extractor": extractor, "doc_id": doc_id, "page": page,
                                         "similarity": edit_similarity(pred[page], truth_text), "pred": pred[page], "truth": truth_text})
    worst: list[dict[str, Any]] = []
    for items in found.values():
        worst += sorted(items, key=lambda i: i["similarity"])[:per_extractor]
    return worst


def render_gallery(items: list[dict[str, Any]], pdf_dir: Path) -> str:
    cards = []
    for item in items:
        image = ""
        pdf_path = pdf_dir / f"{item['doc_id']}.pdf"
        if pdf_path.exists():
            with pymupdf.open(pdf_path) as pdf:
                png = pdf[item["page"] - 1].get_pixmap(dpi=70).tobytes("png")
            image = f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}">'
        cards.append(
            f"<section><h3>{html.escape(item['extractor'])} — {html.escape(item['doc_id'])} p.{item['page']} "
            f"(similarity {item['similarity']:.2f})</h3><div class='row'>{image}"
            f"<div><h4>Extracted</h4><pre>{html.escape(item['pred'][:1200])}</pre></div>"
            f"<div><h4>Truth</h4><pre>{html.escape(item['truth'][:1200])}</pre></div></div></section>"
        )
    style = ("body{font-family:sans-serif;margin:16px}.row{display:flex;gap:12px;align-items:flex-start}"
             "pre{white-space:pre-wrap;max-width:420px;font-size:11px;background:#f4f4f4;padding:6px}img{max-width:300px;border:1px solid #ccc}")
    return f"<!doctype html><meta charset='utf-8'><title>Failure gallery</title><style>{style}</style><h1>Worst pages per extractor</h1>{''.join(cards)}"


# ── plots ────────────────────────────────────────────────────────────────────


def plot_scaling(run_rows: list[dict[str, Any]], field_name: str, ylabel: str, path: Path) -> None:
    """One log-log subplot per tier, one line per extractor."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    by_tier: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in run_rows:
        if row["status"] == "ok" and row[field_name] not in ("", None) and float(row["pages"] or 0) > 0 and float(row[field_name]) > 0:
            by_tier[row["tier"]][row["extractor"]].append((float(row["pages"]), float(row[field_name])))  # log axes need > 0
    for ax, tier in zip(axes.flat, TIERS):
        for name, points in sorted(by_tier[tier].items()):
            points.sort()
            ax.plot(*zip(*points), marker="o", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(tier)
        ax.set_xlabel("pages processed")
        ax.set_ylabel(ylabel)
        if by_tier[tier]:
            ax.legend(fontsize=6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────────────


def failure_matrix(run_rows: list[dict[str, Any]]) -> str:
    """Documents that at least one extractor failed on, one column per extractor: the status of each failed run.

    Quality tables score a failed run as 0, so this is where a reader tells "failed" from "extracted nothing".
    """
    failed = [r for r in run_rows if r["status"] != "ok"]
    if not failed:
        return "No failed runs."
    extractors = sorted({r["extractor"] for r in run_rows})
    pages = {r["doc_id"]: int(r.get("pages") or 0) for r in run_rows}
    for r in run_rows:  # a failed run reports 0 pages; take the count from any successful run of the document
        if r["status"] == "ok" and int(r.get("pages") or 0) > pages.get(r["doc_id"], 0):
            pages[r["doc_id"]] = int(r["pages"])
    status = {(r["extractor"], r["doc_id"]): r["status"] for r in run_rows}
    docs = sorted({r["doc_id"] for r in failed}, key=lambda d: (-pages.get(d, 0), d))
    body = [[f"{d} ({pages.get(d, 0)} pp)"] + [status.get((e, d), "") if status.get((e, d)) != "ok" else "" for e in extractors]
            for d in docs]
    return md_table(["document"] + extractors, body)


def capability_matrix(cfg: BenchConfig | None = None) -> str:
    """One row per config.yaml entry (variants included) with its adapter's capabilities.

    Registry adapters without a config entry are left out. A variant with ``disable_ocr`` has OCR turned off.
    """
    from benchmarks.pdf_extraction.extractors import load_extractor

    cfg = cfg or load_config(CONFIG_PATH)
    keys = ["page_numbers", "bboxes", "tables", "figures", "ocr"]
    body = []
    for name, entry in cfg.extractors.items():
        adapter = cfg.adapter_for(name)
        try:
            extractor = load_extractor(adapter)
        except ImportError:
            continue
        caps = dict(extractor.capabilities)
        if entry.get("disable_ocr"):
            caps["ocr"] = False
        notes = [f"{adapter} variant" if adapter != name else "",
                 "disabled" if not entry.get("enabled", False) else ""]
        label = name + "".join(f" ({n})" for n in notes if n)
        body.append([label] + ["yes" if caps.get(k) else "no" for k in keys])
    return md_table(["extractor", *keys], body)


def build_report(results_dir: Path = RESULTS_DIR, out_dir: Path = REPORT_DIR, out_root: Path = OUT_DIR,
                 truth_dir: Path = TRUTH_DIR, pdf_dir: Path = PDF_DIR) -> Path:
    runs_raw = read_csv(results_dir / "runs.csv")
    # Sampled runs carry their own label from here on, so no table or recommendation mixes them with full runs.
    scores = label_sampled(read_csv(results_dir / "extraction_scores.csv"))
    runs = label_sampled(runs_raw)
    # Retrieval rows carry no sampled column; they inherit it from the run of the same (extractor, document).
    sampled_runs = {(r["extractor"], r["doc_id"]) for r in runs_raw if r["sampled"] == "True"}

    def from_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return label_sampled([{**r, "sampled": str((r["extractor"], r["doc_id"]) in sampled_runs)} for r in rows])

    retrieval = from_runs(read_csv(results_dir / "retrieval_scores.csv"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = sorted({r["extractor"] for r in runs_raw if r["sampled"] == "True"})
    # Runs recorded before total_ram_gb existed lack it; then describe the machine building the report instead.
    machines: dict[str, dict[str, Any]] = {}
    for *_, run in iter_runs(out_root):
        if run.hardware.get("total_ram_gb"):
            machines.setdefault(machine_label(run.hardware), run.hardware)
    hardware = next(iter(machines.values()), None)
    where = "recorded with the runs"
    if hardware is None:
        hardware, where = hardware_info(), "of the machine building this report; the runs did not record RAM"
    if len(machines) > 1:  # outputs from several machines were merged: keep their speed and memory apart
        hardware_line = (f"Runs come from {len(machines)} machines: "
                         + "; ".join(f"{m} ({h.get('cpus', '?')} CPUs, {h.get('total_ram_gb', '?')} GB RAM"
                                     + (f", GPU {h['gpu']}" if h.get("gpu") else "") + ")" for m, h in machines.items())
                         + ". Speed, memory and failure rows are labelled `extractor @machine`; quality rows are not "
                         "machine-specific.")
        runs_by_extractor = runs  # the recommendation joins retrieval rows on the plain extractor name
        runs = [{**r, "extractor": f"{r['extractor']} @{r.get('machine') or 'cpu'}"} for r in runs]
    else:
        runs_by_extractor = runs
        hardware_line = (f"Hardware ({where}): {hardware.get('cpus', '?')} CPUs, {hardware.get('total_ram_gb', '?')} GB RAM, "
                         f"no GPU unless stated (`{json.dumps(hardware)}`).")
    sections = ["# PDF extraction benchmark report", hardware_line,
                f"Sampled runs (first pages only, not comparable to full runs): {', '.join(sampled) or 'none'}.", "",
                "## Capability matrix", capability_matrix(), "", "## Extraction quality by tier"]
    for tier, (metric, higher) in TIER_METRIC.items():
        data = pivot(scores, "extractor", "bucket", "value", where=lambda r, t=tier, m=metric: r["tier"] == t and r["metric"] == m)
        if not data:
            continue
        if tier == "T1":  # real born-digital documents: the page truth is PyMuPDF's own text layer
            sections += ["### T1: agreement with PyMuPDF's text layer (reference, not truth — pymupdf is the reference itself)",
                         f"{metric} by page-count bucket (higher is closer to PyMuPDF). Retrieval hit@5 below is the T1 headline.",
                         pivot_table(data, BUCKETS), ""]
        else:
            sections += [f"### {tier}: {metric} ({'higher' if higher else 'lower'} is better), by page-count bucket", pivot_table(data, BUCKETS), ""]
    tables_recall = pivot(scores, "extractor", "tier", "value", where=lambda r: r["metric"] == "table_detection_recall")
    if tables_recall:
        sections += ["### Table detection recall by tier", pivot_table(tables_recall, TIERS), ""]

    for mode in sorted({r["chunk_mode"] for r in retrieval}):
        by_tier = pivot(retrieval, "extractor", "tier", "hit5", where=lambda r, m=mode: r["chunk_mode"] == m)
        by_bucket = pivot(retrieval, "extractor", "bucket", "hit5", where=lambda r, m=mode: r["chunk_mode"] == m)
        sections += [f"## Retrieval hit@5 — chunking: {mode}", "### By tier", pivot_table(by_tier, TIERS),
                     "### By page-count bucket", pivot_table(by_bucket, BUCKETS), ""]

    if runs:
        overall = fit_speed(runs)
        speed = pivot(runs, "extractor", "bucket", "seconds_per_page", where=lambda r: r["status"] == "ok")
        memory = pivot(runs, "extractor", "bucket", "peak_rss_mb", where=lambda r: r["status"] == "ok")
        failures: dict[str, int] = defaultdict(int)
        for r in runs:
            if r["status"] != "ok":
                failures[r["extractor"]] += 1
        ok = lambda r: r["status"] == "ok"  # noqa: E731
        sections += ["## Speed",
                     "Per-document seconds include interpreter start and model loading; the fit separates that fixed cost. "
                     "The fit is least squares of seconds on pages over full (not sampled) successful runs; with fewer than "
                     "3 documents it falls back to mean seconds per page and no fixed cost.",
                     md_table(["extractor", "fixed s", "s/page", "docs"],
                              [[n, f"{f:.1f}", f"{pp:.3f}", str(d)] for n, (f, pp, d) in sorted(overall.items())]), "",
                     "### Seconds per page by page-count bucket (raw, includes fixed cost)", pivot_table(speed, BUCKETS, 3), "",
                     "## Speed by kind of PDF",
                     "Fitted seconds per page (fixed cost removed where at least 3 documents allow a fit).",
                     "### Seconds per page by tier", per_page_table(fit_speed(runs, "tier"), TIERS), "",
                     "### Seconds per page by source", per_page_table(fit_speed(runs, "source"), SOURCES), "",
                     "## Peak memory (MB)", pivot_table(memory, BUCKETS, 0), "",
                     "## Peak memory by kind",
                     "### Peak memory (MB) by tier", pivot_table(pivot(runs, "extractor", "tier", "peak_rss_mb", where=ok), TIERS, 0), "",
                     "### Peak memory (MB) by source", pivot_table(pivot(runs, "extractor", "source", "peak_rss_mb", where=ok), SOURCES, 0), "",
                     "## Failures", md_table(["extractor", "failed runs"],
                                          [[n, str(failures.get(n, 0))] for n in sorted({r['extractor'] for r in runs})]), "",
                     "### Failed runs by document",
                     "A failed run scores 0 on every quality metric above (oom = exceeded the memory cap, "
                     "timeout = exceeded the time cap, error = the tool raised). Empty cells are successful runs.",
                     failure_matrix(runs), ""]
        plot_scaling(runs, "seconds", "seconds per document", out_dir / "plots" / "time_vs_pages.png")
        plot_scaling(runs, "peak_rss_mb", "peak memory (MB)", out_dir / "plots" / "memory_vs_pages.png")
        sections += ["![time vs pages](plots/time_vs_pages.png)", "![memory vs pages](plots/memory_vs_pages.png)", ""]

    if retrieval:
        mode = "structure_preserving" if any(r["chunk_mode"] == "structure_preserving" for r in retrieval) else retrieval[0]["chunk_mode"]
        rec = recommend(retrieval, runs_by_extractor, mode)
        best_name, best_score = rec["best_single"]
        sections += [f"## Recommendation (from hit@5, chunking: {mode})",
                     "Best extractor per tier: " + (", ".join(f"**{t}** → {n}" for t, n in rec["best_by_tier"].items()) or "n/a"),
                     f"Best single extractor overall: **{best_name}** (hit@5 {fmt(best_score)}).",
                     f"A per-tier router using those picks would score hit@5 {fmt(rec['router_estimate'])} "
                     f"({'+' if (rec['router_gain'] or 0) >= 0 else ''}{fmt(rec['router_gain'])} vs the best single extractor). "
                     "This is an estimate from tier-level picks, not a measured per-page router.", ""]

    gallery = render_gallery(worst_pages(out_root, truth_dir), pdf_dir)
    (out_dir / "failure_gallery.html").write_text(gallery, encoding="utf-8")
    sections.append("Failure gallery: [failure_gallery.html](failure_gallery.html)")
    report = out_dir / "report.md"
    report.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return report


def main() -> int:
    print(f"wrote {build_report()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
