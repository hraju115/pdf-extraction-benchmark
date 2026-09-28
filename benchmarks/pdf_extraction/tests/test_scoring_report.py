"""Tests for extraction scoring, question verification and report building."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from benchmarks.pdf_extraction.corpus.generated import build_generated, write_truth
from benchmarks.pdf_extraction.corpus.make_scans import rasterize_pdf, write_page_truth
from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry
from benchmarks.pdf_extraction.corpus.questions import Question, inherit_scan_questions
from benchmarks.pdf_extraction.corpus.verify import verify_questions
from benchmarks.pdf_extraction.extractors.base import ExtractionResult, PageResult, RunInfo, write_outputs, write_run
from benchmarks.pdf_extraction.report.build_report import (
    build_report,
    md_table,
    pivot,
    recommend,
    render_gallery,
    worst_pages,
)
from benchmarks.pdf_extraction.scoring import (
    RUN_FIELDS,
    load_table_truth,
    SCORE_FIELDS,
    collect_runs,
    read_csv,
    score_extraction,
    write_csv,
)


def _ok(out_root: Path, extractor: str, doc_id: str, markdown: str, pages: dict[int, str] | None = None, **run) -> None:
    pages = pages if pages is not None else {1: markdown}
    write_outputs(out_root / extractor / doc_id,
                  ExtractionResult(markdown=markdown, pages=[PageResult(n, t) for n, t in pages.items()]),
                  RunInfo(extractor=extractor, doc_id=doc_id, pages_processed=len(pages), seconds=2.0, **run))


@pytest.fixture(scope="module")
def truth_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("truth")
    docs = build_generated(root / "pdfs")
    write_truth(docs, root / "pdfs", root / "truth", root / "questions")
    return root, docs


ENTRIES = [
    ManifestEntry(id="gen_t2_simple", tier="T2", bucket="S", source="generated"),
    ManifestEntry(id="gen_t4_bar", tier="T4", bucket="S", source="generated"),
    ManifestEntry(id="scan_a", tier="T3", bucket="S", source="scan", derived_from="gen_t2_simple"),
]


TRUTH_TABLE = "<table><tr><td>Region</td><td>Sales</td></tr><tr><td>North</td><td>1,200</td></tr></table>"


# ── extraction scoring ───────────────────────────────────────────────────────


def test_table_and_figure_scores_use_exact_generated_truth(truth_corpus, tmp_path):
    root, docs = truth_corpus
    simple = next(d for d in docs if d.id == "gen_t2_simple")
    bar = next(d for d in docs if d.id == "gen_t4_bar")

    perfect = "Intro\n\n" + "\n".join(["| Region | Sales Q1 | Sales Q2 | Sales Q3 |", "|---|---|---|---|"])
    # Build the exact pipe table from the truth HTML rows so this test does not depend on the random data.
    from bs4 import BeautifulSoup
    rows = [[c.get_text() for c in tr.find_all("td")] for tr in BeautifulSoup(simple.tables[0], "html.parser").find_all("tr")]
    md = "\n".join(["| " + " | ".join(rows[0]) + " |", "|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows[1:]])
    _ok(tmp_path, "good", "gen_t2_simple", md)
    _ok(tmp_path, "flat", "gen_t2_simple", "Region Sales Q1 North 1")
    _ok(tmp_path, "good", "gen_t4_bar", " ".join(bar.figures[0]["facts"]))
    _ok(tmp_path, "flat", "gen_t4_bar", "nothing useful")

    scores = score_extraction(tmp_path, root / "truth", ENTRIES)
    value = {(r["extractor"], r["doc_id"], r["metric"]): r["value"] for r in scores}
    assert value[("good", "gen_t2_simple", "teds")] == pytest.approx(1.0)
    assert value[("good", "gen_t2_simple", "table_detection_recall")] == 1.0
    assert value[("flat", "gen_t2_simple", "teds")] == 0.0
    assert value[("good", "gen_t4_bar", "figure_fact_recall")] == 1.0
    assert value[("flat", "gen_t4_bar", "figure_fact_recall")] == 0.0
    assert {r["tier"] for r in scores if r["doc_id"] == "gen_t4_bar"} == {"T4"}


def test_failed_run_scores_zero_and_sampled_runs_only_count_processed_pages(truth_corpus, tmp_path):
    root, _ = truth_corpus
    write_run(tmp_path / "broken" / "gen_t2_simple", RunInfo(extractor="broken", doc_id="gen_t2_simple", status="oom"))
    scores = score_extraction(tmp_path, root / "truth", ENTRIES)
    assert {r["metric"]: r["value"] for r in scores if r["extractor"] == "broken"} == {"teds": 0.0, "table_detection_recall": 0.0}
    assert all(r["status"] == "oom" for r in scores if r["extractor"] == "broken")

    truth = tmp_path / "truth"
    write_page_truth(truth / "scan_a.pages.json", {1: "alpha beta", 2: "gamma delta"})
    _ok(tmp_path, "sampler", "scan_a", "alpha beta", pages={1: "alpha beta"}, page_range=[1, 1], sampled=True)
    text = {r["metric"]: r["value"] for r in score_extraction(tmp_path, truth, ENTRIES) if r["extractor"] == "sampler"}
    assert text["text_similarity"] == 1.0 and text["cer"] == 0.0  # page 2 was never requested

    _ok(tmp_path, "full", "scan_a", "alpha beta", pages={1: "alpha beta"})
    text = {r["metric"]: r["value"] for r in score_extraction(tmp_path, truth, ENTRIES) if r["extractor"] == "full"}
    assert text["text_similarity"] == pytest.approx(0.5)  # unprocessed page 2 counts as a miss


def test_run_statistics_and_csv_round_trip(tmp_path):
    _ok(tmp_path, "ex", "gen_t2_simple", "x", version="v1", peak_rss_mb=123.4)
    write_run(tmp_path / "ex" / "gen_t4_bar", RunInfo(extractor="ex", doc_id="gen_t4_bar", status="timeout", error="slow\nline"))
    rows = collect_runs(tmp_path, ENTRIES)
    by_doc = {r["doc_id"]: r for r in rows}
    assert by_doc["gen_t2_simple"]["seconds_per_page"] == 2.0 and by_doc["gen_t2_simple"]["peak_rss_mb"] == 123.4
    assert by_doc["gen_t4_bar"]["status"] == "timeout" and by_doc["gen_t4_bar"]["seconds_per_page"] == ""
    write_csv(tmp_path / "runs.csv", rows, RUN_FIELDS)
    assert len(read_csv(tmp_path / "runs.csv")) == 2 and read_csv(tmp_path / "missing.csv") == []
    assert "tier" in SCORE_FIELDS


def test_hidden_directories_are_not_treated_as_extractors(tmp_path):
    (tmp_path / ".run_state.json").write_text("[]", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    assert collect_runs(tmp_path, ENTRIES) == []
    assert collect_runs(tmp_path / "does_not_exist", ENTRIES) == []


# ── question verification ────────────────────────────────────────────────────


def test_verify_accepts_real_evidence_and_flags_wrong_page_missing_text_and_bad_pages(truth_corpus):
    root, _ = truth_corpus
    pdfs = root / "pdfs"
    good = Question("gen_t2_simple", "q", "a", ["Region"], 1, "lookup", id="g#1")
    wrong_text = Question("gen_t2_simple", "q", "a", ["definitely not in the document"], 1, "lookup", id="g#2")
    bad_page = Question("gen_t2_simple", "q", "a", ["Region"], 9, "lookup", id="g#3")
    no_pdf = Question("nope", "q", "a", ["x"], 1, "lookup", id="g#4")
    visual = Question("gen_t4_bar", "q", "a", ["Alpha"], 1, "figure", evidence_source="visual", id="g#5")
    unanswerable = Question("gen_t2_simple", "q", "n/a", [], 1, "unanswerable", id="g#6")

    result = verify_questions([good, wrong_text, bad_page, no_pdf, visual, unanswerable], pdfs)
    assert result.verified == 1
    assert len(result.problems) == 3
    assert any("not found on page 1" in p for p in result.problems)
    assert any("beyond" in p for p in result.problems)
    assert any("does not exist" in p for p in result.problems)
    assert len(result.needs_human) == 1 and "g#5" in result.needs_human[0]


def test_verify_sends_scans_without_a_text_layer_to_a_human(truth_corpus, tmp_path):
    root, _ = truth_corpus
    rasterize_pdf(root / "pdfs" / "gen_t2_simple.pdf", tmp_path / "scan_a.pdf", dpi=72)
    q = Question("scan_a", "q", "a", ["Region"], 1, "lookup", id="scan_a#1")
    result = verify_questions([q], tmp_path)
    assert result.problems == [] and len(result.needs_human) == 1


def test_cross_page_evidence_may_appear_anywhere_in_the_document(truth_corpus):
    root, _ = truth_corpus
    q = Question("gen_t2_long", "q", "a", ["Detailed sales ledger"], 2, "cross_page", id="l#1")
    assert verify_questions([q], root / "pdfs").verified == 1
    q_page = Question("gen_t2_long", "q", "a", ["Detailed sales ledger"], 2, "lookup", id="l#2")
    assert verify_questions([q_page], root / "pdfs").problems  # title is on page 1, not 2


def test_scans_inherit_their_sources_questions_unless_they_have_their_own():
    base = [Question("gen_t2_simple", "q1", "a", ["x"], 1, "lookup", id="gen_t2_simple#1")]
    inherited = inherit_scan_questions(base, ENTRIES)
    scan_questions = [q for q in inherited if q.doc_id == "scan_a"]
    assert [q.id for q in scan_questions] == ["scan_a#1"] and scan_questions[0].question == "q1"
    own = base + [Question("scan_a", "own", "a", ["y"], 1, "lookup", id="scan_a#1")]
    assert [q.question for q in inherit_scan_questions(own, ENTRIES) if q.doc_id == "scan_a"] == ["own"]


# ── report ───────────────────────────────────────────────────────────────────


def _r(extractor, tier, hit5, mode="m", bucket="S"):
    return {"extractor": extractor, "tier": tier, "bucket": bucket, "chunk_mode": mode, "hit5": str(hit5)}


def test_pivot_averages_and_skips_blank_cells():
    rows = [{"e": "a", "t": "T1", "v": "1"}, {"e": "a", "t": "T1", "v": "0"}, {"e": "a", "t": "T2", "v": ""}, {"e": "b", "t": "T1", "v": "1"}]
    assert pivot(rows, "e", "t", "v") == {"a": {"T1": 0.5}, "b": {"T1": 1.0}}
    assert md_table(["a", "b"], [["1", "2"]]).splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_recommendation_picks_best_per_tier_and_estimates_router_gain():
    retrieval = [_r("x", "T1", 1), _r("x", "T1", 1), _r("x", "T2", 0), _r("x", "T2", 0),
                 _r("y", "T1", 0), _r("y", "T1", 1), _r("y", "T2", 1), _r("y", "T2", 1)]
    runs = [{"extractor": "x", "seconds_per_page": "0.1"}, {"extractor": "y", "seconds_per_page": "5"}]
    rec = recommend(retrieval, runs, "m")
    assert rec["best_by_tier"] == {"T1": "x", "T2": "y"}
    assert rec["best_single"] == ("y", pytest.approx(0.75))  # y: (0 + 1 + 1 + 1) / 4
    assert rec["router_estimate"] == pytest.approx(1.0)  # x's T1 rows and y's T2 rows are all hits
    assert rec["router_gain"] == pytest.approx(0.25)


def test_recommendation_breaks_ties_by_speed_and_handles_no_data():
    rows = [_r("slow", "T1", 1), _r("fast", "T1", 1)]
    runs = [{"extractor": "slow", "seconds_per_page": "9"}, {"extractor": "fast", "seconds_per_page": "1"}]
    assert recommend(rows, runs, "m")["best_by_tier"] == {"T1": "fast"}
    empty = recommend([], [], "m")
    assert empty["best_by_tier"] == {} and empty["best_single"] == (None, None) and empty["router_estimate"] is None


def test_worst_pages_and_gallery_render(truth_corpus, tmp_path):
    root, _ = truth_corpus
    truth = tmp_path / "truth"
    write_page_truth(truth / "gen_t2_simple.pages.json", {1: "Regional sales summary and figures"})
    _ok(tmp_path, "poor", "gen_t2_simple", "garbled", pages={1: "garbled"})
    _ok(tmp_path, "great", "gen_t2_simple", "x", pages={1: "Regional sales summary and figures"})
    items = worst_pages(tmp_path, truth, per_extractor=1)
    assert {i["extractor"]: round(i["similarity"], 1) for i in items} == {"poor": pytest.approx(0.0, abs=0.3), "great": 1.0}
    page = render_gallery(items, root / "pdfs")
    assert "data:image/png;base64," in page and "garbled" in page and "<script" not in page


def test_gallery_escapes_html_in_extracted_text(tmp_path):
    page = render_gallery([{"extractor": "e<x>", "doc_id": "d", "page": 1, "similarity": 0.1, "pred": "<script>alert(1)</script>", "truth": "t"}], tmp_path)
    assert "<script>alert" not in page and "&lt;script&gt;" in page


def test_build_report_end_to_end_from_csvs(truth_corpus, tmp_path):
    root, docs = truth_corpus
    results, out_root = tmp_path / "results", tmp_path / "out"
    _ok(out_root, "ex", "gen_t2_simple", "Regional sales summary")
    write_csv(results / "extraction_scores.csv",
              [{"extractor": "ex", "doc_id": "gen_t2_simple", "tier": "T2", "bucket": "S", "status": "ok", "sampled": False, "metric": "teds", "value": 0.8}], SCORE_FIELDS)
    write_csv(results / "runs.csv", collect_runs(out_root, ENTRIES), RUN_FIELDS)
    write_csv(results / "retrieval_scores.csv",
              [{"extractor": "ex", "doc_id": "gen_t2_simple", "tier": "T2", "bucket": "S", "chunk_mode": "baseline", "question_id": "q", "qtype": "lookup", "n_chunks": 3, "hit5": 1, "hit50": 1}],
              ["extractor", "doc_id", "tier", "bucket", "chunk_mode", "question_id", "qtype", "n_chunks", "hit5", "hit50"])
    report = build_report(results, tmp_path / "report", out_root, root / "truth", root / "pdfs")
    text = report.read_text(encoding="utf-8")
    for heading in ("Capability matrix", "Extraction quality by tier", "Retrieval hit@5", "Speed", "Recommendation"):
        assert heading in text
    assert (tmp_path / "report" / "failure_gallery.html").exists()
    assert (tmp_path / "report" / "plots" / "time_vs_pages.png").exists()


def test_build_report_with_no_results_still_produces_a_report(tmp_path):
    report = build_report(tmp_path / "r", tmp_path / "rep", tmp_path / "o", tmp_path / "t", tmp_path / "p")
    assert "PDF extraction benchmark report" in report.read_text(encoding="utf-8")


# ── page-aware table truth (sampled runs) ────────────────────────────────────


def test_table_truth_loader_accepts_legacy_strings_and_page_dicts(tmp_path):
    legacy = tmp_path / "a.tables.json"
    legacy.write_text(json.dumps([TRUTH_TABLE]), encoding="utf-8")
    paged = tmp_path / "b.tables.json"
    paged.write_text(json.dumps([{"page": 3, "html": TRUTH_TABLE}, {"html": TRUTH_TABLE}]), encoding="utf-8")
    assert load_table_truth(legacy) == [(None, TRUTH_TABLE)]
    assert load_table_truth(paged) == [(3, TRUTH_TABLE), (None, TRUTH_TABLE)]


def test_sampled_runs_are_scored_only_on_tables_within_their_page_range(tmp_path):
    truth = tmp_path / "truth"
    truth.mkdir()
    table_p1 = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
    table_p9 = "<table><tr><td>Zeta</td><td>Eta</td></tr><tr><td>40</td><td>50</td></tr></table>"
    (truth / "doc.tables.json").write_text(json.dumps([{"page": 1, "html": table_p1}, {"page": 9, "html": table_p9}]), encoding="utf-8")
    md = "| A | B |\n|---|---|\n| 1 | 2 |"  # only the page-1 table was extracted
    _ok(tmp_path, "sampler", "doc", md, page_range=[1, 3], sampled=True)
    _ok(tmp_path, "full", "doc", md)
    entries = [ManifestEntry(id="doc", tier="T2", bucket="L", source="generated")]
    scores = {(r["extractor"], r["metric"]): r["value"] for r in score_extraction(tmp_path, truth, entries)}
    assert scores[("sampler", "teds")] == pytest.approx(1.0)  # the page-9 table was never requested
    assert scores[("full", "teds")] == pytest.approx(0.5)  # the full run missed it
    assert scores[("sampler", "table_detection_recall")] == 1.0 and scores[("full", "table_detection_recall")] == 0.5


def test_documents_whose_pages_have_no_text_truth_get_no_text_score(tmp_path):
    truth = tmp_path / "truth"
    truth.mkdir()
    write_page_truth(truth / "blank.pages.json", {1: "", 2: ""})
    _ok(tmp_path, "ex", "blank", "", pages={1: ""})
    metrics = [r["metric"] for r in score_extraction(tmp_path, truth, [ManifestEntry(id="blank", tier="T3", bucket="S", source="scan", derived_from="x")])]
    assert "text_similarity" not in metrics and "cer" not in metrics


# ── per-kind speed, sampled separation, T1 relabel ───────────────────────────


def _rr(extractor, tier, bucket, source, pages, seconds, sampled="False", status="ok"):
    return {"extractor": extractor, "doc_id": f"{tier}{pages}", "tier": tier, "bucket": bucket, "source": source, "status": status,
            "sampled": sampled, "pages": str(pages), "seconds": str(seconds), "seconds_per_page": str(round(seconds / pages, 3)),
            "peak_rss_mb": "100", "version": "v", "error": ""}


def test_speed_fit_separates_fixed_startup_from_per_page_cost():
    from benchmarks.pdf_extraction.report.build_report import fit_speed

    runs = [_rr("slowstart", "T1", "S", "real", p, 30 + 0.5 * p) for p in (1, 4, 20, 100)] + [_rr("tiny", "T1", "S", "real", 2, 1.0)]
    fit = fit_speed(runs)
    assert fit["slowstart"][0] == pytest.approx(30, abs=0.01) and fit["slowstart"][1] == pytest.approx(0.5, abs=0.001)
    assert fit["tiny"] == (0.0, pytest.approx(0.5), 1)  # too few documents: mean seconds/page, no fixed cost


def test_sampled_only_extractor_is_shown_but_never_recommended(tmp_path):
    """Review Focus 4."""
    from benchmarks.pdf_extraction.report.build_report import label_sampled

    retrieval = [_r("full", "T1", 1), _r("full", "T1", 0), {**_r("samp", "T1", 1), "sampled": "True"}, {**_r("samp", "T1", 1), "sampled": "True"}]
    labelled = label_sampled(retrieval)
    assert {r["extractor"] for r in labelled} == {"full", "samp (sampled)"}
    rec = recommend(labelled, [], "m")
    assert rec["best_single"][0] == "full" and rec["best_by_tier"] == {"T1": "full"}


def test_report_has_per_kind_speed_sections_and_relabels_t1(tmp_path, truth_corpus):
    root, _ = truth_corpus
    results, out_root = tmp_path / "results", tmp_path / "out"
    _ok(out_root, "ex", "gen_t2_simple", "x")
    runs = [_rr("ex", "T1", "S", "real", 3, 3.0), _rr("ex", "T3", "M", "scan", 10, 40.0), _rr("ex", "T2", "S", "generated", 1, 0.5)]
    write_csv(results / "runs.csv", runs, RUN_FIELDS)
    write_csv(results / "extraction_scores.csv",
              [{"extractor": "ex", "doc_id": "rfc", "tier": "T1", "bucket": "S", "source": "real", "status": "ok", "sampled": False, "metric": "text_similarity", "value": 0.9}], SCORE_FIELDS)
    text = build_report(results, tmp_path / "report", out_root, root / "truth", root / "pdfs").read_text(encoding="utf-8")
    assert "Speed by kind of PDF" in text and "| extractor | T1 | T2 | T3 | T4 |" in text
    assert "by source" in text.lower() and "real" in text and "scan" in text
    assert "agreement with PyMuPDF" in text and "headline" in text
    assert "total_ram_gb" in text or "RAM" in text


def test_capability_matrix_lists_every_config_entry_including_variants():
    from benchmarks.pdf_extraction.config import BenchConfig
    from benchmarks.pdf_extraction.report.build_report import capability_matrix

    cfg = BenchConfig(extractors={"pymupdf": {"enabled": True}, "marker": {"enabled": False},
                                  "marker_noocr": {"enabled": True, "adapter": "marker", "disable_ocr": True}})
    rows = {line.split(" | ")[0].strip("| "): line for line in capability_matrix(cfg).splitlines()[2:]}
    assert set(rows) == {"pymupdf", "marker (disabled)", "marker_noocr (marker variant)"}  # unconfigured adapters are left out
    assert rows["marker_noocr (marker variant)"].rstrip(" |").endswith("no")  # ocr column: OCR turned off
    assert rows["marker (disabled)"].rstrip(" |").endswith("yes")


def test_report_labels_retrieval_rows_of_sampled_runs_via_runs_csv(tmp_path, truth_corpus):
    """retrieval_scores.csv has no sampled column; the label comes from the matching run."""
    root, _ = truth_corpus
    results = tmp_path / "results"
    runs = [_rr("full", "T1", "S", "real", 3, 3.0), {**_rr("samp", "T1", "S", "real", 3, 3.0), "sampled": "True"}]
    write_csv(results / "runs.csv", runs, RUN_FIELDS)
    fields = ["extractor", "doc_id", "tier", "bucket", "chunk_mode", "question_id", "qtype", "n_chunks", "hit5", "hit50"]
    write_csv(results / "retrieval_scores.csv",
              [{"extractor": e, "doc_id": "T13", "tier": "T1", "bucket": "S", "chunk_mode": "baseline", "question_id": "q",
                "qtype": "lookup", "n_chunks": 3, "hit5": h, "hit50": h} for e, h in (("full", 0), ("samp", 1))], fields)
    text = build_report(results, tmp_path / "report", tmp_path / "out", root / "truth", root / "pdfs").read_text(encoding="utf-8")
    assert "| samp (sampled) |" in text and "Best single extractor overall: **full**" in text


def test_failure_matrix_lists_only_failed_runs_by_document():
    from benchmarks.pdf_extraction.report.build_report import failure_matrix

    runs = [
        {"extractor": "a", "doc_id": "d1", "status": "ok", "pages": 4},
        {"extractor": "a", "doc_id": "d2", "status": "oom", "pages": 487},
        {"extractor": "b", "doc_id": "d2", "status": "timeout", "pages": 487},
        {"extractor": "b", "doc_id": "d1", "status": "ok", "pages": 4},
    ]
    table = failure_matrix(runs)
    assert "d2" in table and "d1" not in table  # documents that every extractor handled are not listed
    assert "| d2 (487 pp) | oom | timeout |" in table
    assert failure_matrix([{"extractor": "a", "doc_id": "d1", "status": "ok", "pages": 1}]) == "No failed runs."


def test_run_rows_carry_the_machine_that_produced_them(tmp_path):
    """Runs from different machines (a laptop CPU, a Kaggle GPU) must stay distinguishable once their out/ trees are merged."""
    from benchmarks.pdf_extraction.scoring import machine_label

    assert machine_label({"cpus": 12, "gpu": None}) == "cpu"
    assert machine_label({"cpus": 4, "gpu": "Tesla T4, 15360 MiB"}) == "gpu:Tesla T4"
    assert machine_label({}) == "cpu"
    assert "machine" in RUN_FIELDS

    write_run(tmp_path / "pymupdf" / "gen_t2_simple",
              RunInfo(extractor="pymupdf", doc_id="gen_t2_simple", status="ok", pages_processed=1, seconds=1.0,
                      hardware={"cpus": 12, "total_ram_gb": 22.7, "gpu": None}))
    write_run(tmp_path / "docling" / "gen_t2_simple",
              RunInfo(extractor="docling", doc_id="gen_t2_simple", status="ok", pages_processed=1, seconds=9.0,
                      hardware={"cpus": 4, "total_ram_gb": 31.0, "gpu": "Tesla T4, 15360 MiB"}))
    rows = {r["extractor"]: r for r in collect_runs(tmp_path, ENTRIES)}
    assert rows["pymupdf"]["machine"] == "cpu"
    assert rows["docling"]["machine"] == "gpu:Tesla T4"


def test_report_labels_speed_rows_by_machine_when_runs_come_from_several(truth_corpus, tmp_path):
    root, _docs = truth_corpus
    results, out_root = tmp_path / "results", tmp_path / "out"
    write_run(out_root / "ex" / "gen_t2_simple",
              RunInfo(extractor="ex", doc_id="gen_t2_simple", status="ok", pages_processed=1, seconds=1.0, peak_rss_mb=100,
                      hardware={"cpus": 12, "total_ram_gb": 22.7, "gpu": None}))
    write_run(out_root / "ex" / "gen_t4_bar",
              RunInfo(extractor="ex", doc_id="gen_t4_bar", status="ok", pages_processed=1, seconds=0.1, peak_rss_mb=900,
                      hardware={"cpus": 4, "total_ram_gb": 31.0, "gpu": "Tesla T4, 15360 MiB"}))
    write_csv(results / "runs.csv", collect_runs(out_root, ENTRIES), RUN_FIELDS)
    write_csv(results / "extraction_scores.csv", [], SCORE_FIELDS)
    text = build_report(results, tmp_path / "report", out_root, root / "truth", root / "pdfs").read_text(encoding="utf-8")
    assert "| ex @cpu |" in text and "| ex @gpu:Tesla T4 |" in text  # speed and memory rows keep the machines apart
    assert "2 machines" in text and "gpu:Tesla T4" in text.split("## Capability matrix")[0]  # and the header says so
