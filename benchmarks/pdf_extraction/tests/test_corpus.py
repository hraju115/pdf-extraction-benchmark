"""Tests for manifest validation, question files, generated documents and synthetic scans."""

from __future__ import annotations

import json

import pymupdf
import pytest
import yaml

from benchmarks.pdf_extraction.corpus.generated import (
    Cell,
    build_generated,
    layout_grid,
    table_to_html,
    write_truth,
)
from benchmarks.pdf_extraction.corpus.make_scans import (
    has_text_layer,
    native_page_truth,
    rasterize_pdf,
    read_page_truth,
    write_page_truth,
)
from benchmarks.pdf_extraction.corpus.manifest import (
    ManifestEntry,
    bucket_for_pages,
    check_built_pages,
    load_manifest,
    validate_entry,
)
from benchmarks.pdf_extraction.corpus.questions import load_all_questions, load_question_file
from benchmarks.pdf_extraction.metrics.tables import teds

SHA = "a" * 64

# ── manifest ─────────────────────────────────────────────────────────────────


def _real(**overrides) -> ManifestEntry:
    base = dict(id="doc_a", tier="T1", bucket="S", source="real", pages=3, url="https://example.org/a.pdf",
                license="public-domain", sha256=SHA)
    base.update(overrides)
    return ManifestEntry(**base)


@pytest.mark.parametrize("pages,bucket", [(1, "S"), (5, "S"), (6, "M"), (30, "M"), (31, "L"), (100, "L"), (101, "XL"), (900, "XL")])
def test_bucket_boundaries(pages, bucket):
    assert bucket_for_pages(pages) == bucket


def test_bucket_rejects_zero_pages():
    with pytest.raises(ValueError):
        bucket_for_pages(0)


def test_valid_real_entry_has_no_problems():
    assert validate_entry(_real()) == []


def test_real_entry_without_verified_license_is_rejected():
    assert any("license" in p for p in validate_entry(_real(license="")))
    assert any("license" in p for p in validate_entry(_real(license="all-rights-reserved")))


def test_real_entry_needs_url_sha_and_matching_bucket():
    problems = validate_entry(_real(url="", sha256="xyz", bucket="M"))
    assert any("url" in p for p in problems)
    assert any("sha256" in p for p in problems)
    assert any("bucket" in p for p in problems)


def test_scan_must_be_tier_t3_and_name_its_source():
    bad = ManifestEntry(id="scan_a", tier="T1", bucket="S", source="scan")
    problems = validate_entry(bad)
    assert any("derived_from" in p for p in problems)
    assert any("T3" in p for p in problems)


def test_load_manifest_reports_every_problem_at_once(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump({"documents": [
            {"id": "a", "tier": "T1", "bucket": "S", "source": "real", "pages": 3, "url": "u", "license": "nope", "sha256": SHA},
            {"id": "a", "tier": "T9", "bucket": "S", "source": "real", "pages": 3},
            {"id": "s", "tier": "T3", "bucket": "S", "source": "scan", "derived_from": "missing"},
        ]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_manifest(path)
    message = str(exc.value)
    assert "duplicate id" in message and "tier" in message and "derived_from" in message and "license" in message


def test_load_manifest_accepts_scan_derived_from_generated_and_from_manifest(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump({"documents": [
            {"id": "doc_a", "tier": "T1", "bucket": "S", "source": "real", "pages": 3, "url": "u", "license": "cc0", "sha256": SHA},
            {"id": "doc_a_scan", "tier": "T3", "bucket": "S", "source": "scan", "derived_from": "doc_a"},
            {"id": "gen_scan", "tier": "T3", "bucket": "S", "source": "scan", "derived_from": "gen_t2_simple"},
        ]}),
        encoding="utf-8",
    )
    assert [e.id for e in load_manifest(path)] == ["doc_a", "doc_a_scan", "gen_scan"]


def test_empty_manifest_is_valid(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text("documents: []\n", encoding="utf-8")
    assert load_manifest(path) == []


def test_built_page_count_must_fit_bucket():
    entry = ManifestEntry(id="g", tier="T2", bucket="S", source="generated")
    assert check_built_pages(entry, 3) == []
    assert check_built_pages(entry, 12) != []


# ── questions ────────────────────────────────────────────────────────────────


def test_question_file_round_trip_and_validation(tmp_path):
    good = tmp_path / "doc_a.yaml"
    good.write_text(yaml.safe_dump({"questions": [
        {"question": "Q?", "answer": "A", "evidence": ["needle"], "page": 2, "type": "lookup"},
        {"question": "Unknown?", "answer": "n/a", "evidence": [], "page": 1, "type": "unanswerable"},
    ]}), encoding="utf-8")
    questions = load_question_file(good)
    assert [q.id for q in questions] == ["doc_a#1", "doc_a#2"]
    assert questions[1].evidence == []

    for bad_item, fragment in [
        ({"question": "Q?", "answer": "A", "evidence": ["x"], "page": 1, "type": "weird"}, "type"),
        ({"question": "Q?", "answer": "A", "evidence": [], "page": 1, "type": "lookup"}, "evidence"),
        ({"question": "Q?", "answer": "A", "evidence": ["x"], "page": 0, "type": "lookup"}, "page"),
        ({"question": "Q?", "answer": "A", "page": 1, "type": "lookup"}, "missing"),
    ]:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"questions": [bad_item]}), encoding="utf-8")
        with pytest.raises(ValueError, match=fragment):
            load_question_file(path)


def test_load_all_questions_skips_missing_directories(tmp_path):
    assert load_all_questions(tmp_path / "nope") == []


# ── generated documents ──────────────────────────────────────────────────────


def test_layout_grid_honors_rowspan_and_colspan():
    spec = [
        [Cell("Product", rowspan=2), Cell("Units", colspan=2)],
        [Cell("2023"), Cell("2024")],
        [Cell("Widget"), Cell("1"), Cell("2")],
    ]
    grid, spans = layout_grid(spec)
    assert grid == [["Product", "Units", ""], ["", "2023", "2024"], ["Widget", "1", "2"]]
    assert ("SPAN", (0, 0), (0, 1)) in spans and ("SPAN", (1, 0), (2, 0)) in spans


def test_table_to_html_carries_spans_and_escapes():
    html = table_to_html([[Cell("a<b", colspan=2)], [Cell("x"), Cell("y")]])
    assert 'colspan="2"' in html and "a&lt;b" in html


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("gen")
    docs = build_generated(root)
    write_truth(docs, root, root / "truth", root / "questions")
    return root, docs


def test_all_eight_generated_documents_are_written(generated):
    root, docs = generated
    assert len(docs) == 8
    for doc in docs:
        assert (root / f"{doc.id}.pdf").exists()
        with pymupdf.open(root / f"{doc.id}.pdf") as pdf:
            assert 1 <= len(pdf) <= 5, doc.id  # all fall in bucket S


def test_long_table_spans_multiple_pages(generated):
    root, _ = generated
    with pymupdf.open(root / "gen_t2_long.pdf") as pdf:
        assert len(pdf) >= 2


def test_rotated_document_really_is_rotated(generated):
    root, _ = generated
    with pymupdf.open(root / "gen_t2_rotated.pdf") as pdf:
        assert all(page.rotation == 90 for page in pdf)


def test_table_documents_have_a_text_layer_that_contains_every_cell(generated):
    root, docs = generated
    simple = next(d for d in docs if d.id == "gen_t2_simple")
    with pymupdf.open(root / "gen_t2_simple.pdf") as pdf:
        text = pdf[0].get_text()
    for cell in ("Region", "Sales Q1", "North"):
        assert cell in text
    assert teds(simple.tables[0], simple.tables[0]) == pytest.approx(1.0)


def test_no_figure_fact_leaks_into_any_text_layer(generated):
    """A text-layer tool must score ~0 on figure facts; any leak would reward tools for the wrong reason."""
    from benchmarks.pdf_extraction.metrics.facts import fact_present

    root, docs = generated
    figure_docs = [d for d in docs if d.figures]
    assert len(figure_docs) == 4
    for doc in figure_docs:
        with pymupdf.open(root / f"{doc.id}.pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
        assert text.strip()  # the page does have real text (title, table), just not the chart's
        leaked = [f for f in doc.figures[0]["facts"] if fact_present(text, f)]
        assert leaked == [], f"{doc.id} leaks {leaked}"


def test_generation_is_deterministic(tmp_path):
    first = build_generated(tmp_path / "a")
    second = build_generated(tmp_path / "b")
    assert [d.tables for d in first] == [d.tables for d in second]
    assert [d.figures for d in first] == [d.figures for d in second]
    assert [d.question_seeds for d in first] == [d.question_seeds for d in second]


def test_generated_truth_and_questions_are_written_and_loadable(generated):
    root, _ = generated
    assert json.loads((root / "truth" / "gen_t2_merged.tables.json").read_text())[0]["html"].startswith("<table>")
    questions = load_all_questions(root / "questions")
    assert len(questions) >= 8
    figure_questions = [q for q in questions if q.type == "figure"]
    assert figure_questions and all(q.evidence_source == "visual" for q in figure_questions)
    for q in questions:
        if q.type == "table_cell":
            assert q.page >= 1 and q.evidence


def test_table_question_pages_point_at_the_right_page(generated):
    root, _ = generated
    questions = {q.id: q for q in load_all_questions(root / "questions") if q.doc_id == "gen_t2_long"}
    with pymupdf.open(root / "gen_t2_long.pdf") as pdf:
        for q in questions.values():
            assert q.evidence[0] in pdf[q.page - 1].get_text()


# ── synthetic scans ──────────────────────────────────────────────────────────


def test_scan_has_no_text_layer_and_exact_truth(generated, tmp_path):
    root, _ = generated
    src = root / "gen_t2_simple.pdf"
    scan = tmp_path / "scan.pdf"
    truth = rasterize_pdf(src, scan, dpi=100)
    assert has_text_layer(src) and not has_text_layer(scan)
    assert set(truth) == {1} and "Regional sales summary" in truth[1]
    with pymupdf.open(scan) as pdf:
        assert len(pdf) == 1 and len(pdf[0].get_images()) == 1


def test_degraded_scan_differs_from_clean_and_is_reproducible(generated, tmp_path):
    root, _ = generated
    src = root / "gen_t2_simple.pdf"
    rasterize_pdf(src, tmp_path / "clean.pdf", dpi=100)
    rasterize_pdf(src, tmp_path / "bad1.pdf", dpi=100, degrade=True, seed=7)
    rasterize_pdf(src, tmp_path / "bad2.pdf", dpi=100, degrade=True, seed=7)
    clean, bad1, bad2 = ((tmp_path / n).read_bytes() for n in ("clean.pdf", "bad1.pdf", "bad2.pdf"))
    assert clean != bad1
    with pymupdf.open(tmp_path / "bad1.pdf") as a, pymupdf.open(tmp_path / "bad2.pdf") as b:
        assert a[0].get_pixmap(dpi=50).samples == b[0].get_pixmap(dpi=50).samples


def test_page_truth_round_trip_and_native_truth(generated, tmp_path):
    root, _ = generated
    truth = native_page_truth(root / "gen_t2_long.pdf")
    assert len(truth) >= 2
    write_page_truth(tmp_path / "t.json", truth)
    assert read_page_truth(tmp_path / "t.json") == truth


def test_generated_table_truth_carries_the_page_the_table_starts_on(generated):
    root, docs = generated
    tables = json.loads((root / "truth" / "gen_t2_long.tables.json").read_text(encoding="utf-8"))
    assert tables and all(isinstance(t, dict) and t["page"] == 1 and t["html"].startswith("<table>") for t in tables)
    mixed = json.loads((root / "truth" / "gen_t4_mixed.tables.json").read_text(encoding="utf-8"))
    assert mixed[0]["page"] == 1


def test_chart_values_never_coincide_with_axis_ticks(generated):
    """A tool that only reads tick labels (multiples of 5/10) must get no credit for chart values."""
    _, docs = generated
    checked = 0
    for doc in docs:
        for figure in doc.figures:
            numbers = [int(f) for f in figure["facts"] if f.isdigit()]
            if not numbers:  # the flowchart has no numeric facts
                continue
            checked += 1
            assert all(n % 5 != 0 for n in numbers), (doc.id, numbers)
    assert checked == 3  # gen_t4_bar, gen_t4_line, gen_t4_mixed


def test_manifest_t3_xl_scan_derives_from_a_born_digital_source():
    from benchmarks.pdf_extraction.paths import MANIFEST_PATH

    entries = {e.id: e for e in load_manifest(MANIFEST_PATH)}
    assert "scan_nasa_tm104114" not in entries  # its source has only an old OCR text layer: not exact truth
    scan = entries["scan_rfc9110"]
    assert scan.tier == "T3" and scan.bucket == "XL" and scan.derived_from == "rfc9110"
    assert entries["rfc9110"].truth_from_native  # so the scan's page truth is the RFC's own text layer
