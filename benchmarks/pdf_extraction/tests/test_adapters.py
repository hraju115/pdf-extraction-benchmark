"""Adapter tests: pure helpers, real behaviour of the lightweight tools, a contract test for every
installed adapter (no network, no API keys)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from benchmarks.pdf_extraction.corpus.generated import build_generated
from benchmarks.pdf_extraction.corpus.make_scans import rasterize_pdf
from benchmarks.pdf_extraction.extractors import known_extractors, load_extractor
from benchmarks.pdf_extraction.extractors.base import rows_to_markdown, validate_outputs
from benchmarks.pdf_extraction.extractors.easyocr_adapter import group_into_lines
from benchmarks.pdf_extraction.extractors.marker_adapter import _PAGE_MARK
from benchmarks.pdf_extraction.extractors.tesseract_adapter import parse_tsv_lines
from benchmarks.pdf_extraction.metrics.tables import extract_tables, score_tables
from benchmarks.pdf_extraction.paths import ENVS_DIR
from benchmarks.pdf_extraction.run import Doc, run_one

LOCAL = known_extractors()
IN_PROCESS = {"pymupdf": "pymupdf", "pymupdf4llm": "pymupdf4llm", "pdfplumber": "pdfplumber"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("adapter_corpus")
    build_generated(root)
    rasterize_pdf(root / "gen_t2_simple.pdf", root / "scan_simple.pdf", dpi=200)
    return root


def _available(name: str) -> bool:
    return importlib.util.find_spec(IN_PROCESS[name]) is not None if name in IN_PROCESS else False


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_rows_to_markdown_escapes_pipes_and_pads_short_rows():
    md = rows_to_markdown([["A", "B"], ["x|y", None], ["only"]])
    assert md.splitlines() == ["| A | B |", "|---|---|", "| x\\|y |  |", "| only |  |"]
    assert rows_to_markdown([]) == ""


def test_tesseract_tsv_lines_are_grouped_and_scaled_to_points():
    tsv = "\n".join([
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "5\t1\t1\t1\t1\t1\t100\t200\t50\t20\t95\tHello",
        "5\t1\t1\t1\t1\t2\t160\t200\t60\t20\t95\tWorld",
        "5\t1\t1\t1\t2\t1\t100\t300\t40\t20\t90\tNext",
        "5\t1\t1\t1\t2\t2\t150\t300\t10\t20\t-1\t ",
        "4\t1\t1\t1\t1\t0\t0\t0\t9\t9\t-1\t",
    ])
    blocks = parse_tsv_lines(tsv, dpi=144)  # 144 dpi -> scale 0.5
    assert [b.text for b in blocks] == ["Hello World", "Next"]
    assert blocks[0].bbox == [50.0, 100.0, 110.0, 110.0]


def test_easyocr_detections_are_clustered_into_ordered_lines():
    box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]  # noqa: E731
    detections = [
        (box(200, 12, 260, 32), "world"),
        (box(10, 10, 80, 30), "hello"),
        (box(10, 110, 90, 130), "second"),
    ]
    lines = group_into_lines(detections)
    assert [text for _, text in lines] == ["hello world", "second"]
    assert lines[0][0] == [10, 10, 260, 32]
    assert group_into_lines([]) == []


def test_marker_page_markers_split_paginated_output():
    text = "{0}" + "-" * 48 + "\n\nfirst page\n\n{1}" + "-" * 48 + "\n\nsecond page"
    marks = list(_PAGE_MARK.finditer(text))
    assert [m.group(1) for m in marks] == ["0", "1"]
    assert _PAGE_MARK.sub("", text).strip().startswith("first page")


# ── real behaviour of the lightweight tools (skipped where not installed) ────


@pytest.mark.skipif(not _available("pymupdf"), reason="pymupdf not installed")
def test_pymupdf_reads_born_digital_text_but_returns_nothing_for_a_scan(corpus):
    extractor = load_extractor("pymupdf")
    assert "Regional sales summary" in extractor.extract(corpus / "gen_t2_simple.pdf", None).markdown
    assert extractor.extract(corpus / "scan_simple.pdf", None).markdown.strip() == ""


@pytest.mark.skipif(not _available("pymupdf"), reason="pymupdf not installed")
def test_pymupdf_page_range_reports_absolute_page_numbers(corpus):
    result = load_extractor("pymupdf").extract(corpus / "gen_t2_long.pdf", (2, 2))
    assert [p.page for p in result.pages] == [2]


@pytest.mark.skipif(not _available("pymupdf4llm"), reason="pymupdf4llm not installed")
def test_pymupdf4llm_recovers_a_simple_table_structure(corpus):
    from benchmarks.pdf_extraction.corpus.generated import table_to_html  # noqa: F401  (truth comes from the builder)

    result = load_extractor("pymupdf4llm").extract(corpus / "gen_t2_simple.pdf", None)
    tables = extract_tables(result.markdown)
    assert len(tables) == 1
    assert any(b.kind == "table" for b in result.pages[0].blocks)


@pytest.mark.skipif(not _available("pdfplumber"), reason="pdfplumber not installed")
def test_pdfplumber_finds_the_table_and_keeps_text_outside_it(corpus):
    result = load_extractor("pdfplumber").extract(corpus / "gen_t2_simple.pdf", None)
    assert "Regional sales summary" in result.markdown
    assert len(extract_tables(result.markdown)) == 1
    assert result.markdown.count("North") == 1  # table text is not duplicated in the prose


@pytest.mark.skipif(not (_available("pdfplumber") and _available("pymupdf")), reason="tools not installed")
def test_table_scoring_separates_structured_from_flat_extractors(corpus):
    docs = build_generated(corpus / "regen")  # truth for the same seeded documents
    truth = next(d for d in docs if d.id == "gen_t2_simple").tables
    flat = load_extractor("pymupdf").extract(corpus / "gen_t2_simple.pdf", None).markdown
    structured = load_extractor("pdfplumber").extract(corpus / "gen_t2_simple.pdf", None).markdown
    assert score_tables(extract_tables(flat), truth)["teds"] == 0.0
    assert score_tables(extract_tables(structured), truth)["teds"] > 0.9


# ── contract test: every adapter whose environment exists must satisfy the schema ──


def _python_for(name: str) -> str | None:
    env_python = ENVS_DIR / name / "bin" / "python"
    if (ENVS_DIR / name / ".ready").exists() and env_python.exists():
        return str(env_python)
    if _available(name):
        return sys.executable
    return None


def _skip_if_blocked(name: str, run) -> None:
    """Skip, with the reason visible under -rs, when this machine cannot run the tool at all.

    Only two known cases, both recorded in the ledger: Marker's OCR engine needs llama.cpp's llama-server,
    and PaddleOCR-VL exceeds the memory this machine can spare. Every other failure still fails the test.
    """
    if name == "marker" and run.status == "error" and "llama-server binary not found" in run.error:
        pytest.skip("marker: its OCR engine needs llama.cpp's llama-server, which is not installed (see the ledger)")
    if name == "paddleocr_vl" and run.status == "oom":
        pytest.skip(f"paddleocr_vl: exceeds the memory cap ({run.error}); recorded as an oom result, not verified here")


@pytest.mark.parametrize("name", LOCAL)
def test_adapter_contract_on_a_one_page_document(name, corpus, tmp_path):
    python = _python_for(name)
    if python is None:
        pytest.skip(f"no environment for {name}: run `python -m benchmarks.pdf_extraction.run --setup-only --extractors {name}`")
    doc = Doc(id="gen_t2_simple", path=corpus / "gen_t2_simple.pdf", pages=1, tier="T2", bucket="S")
    run = run_one(name, doc, out_root=tmp_path, python=python, options={}, max_rss_gb=8, timeout_min=20)
    _skip_if_blocked(name, run)
    assert run.status == "ok", run.error
    assert validate_outputs(tmp_path / name / "gen_t2_simple") == []
    assert run.pages_processed == 1 and run.version
    assert "Region" in (tmp_path / name / "gen_t2_simple" / "result.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", [n for n in LOCAL if n in ("tesseract", "easyocr", "docling", "marker", "mineru", "paddleocr_vl")])
def test_ocr_capable_adapters_read_an_image_only_page(name, corpus, tmp_path):
    python = _python_for(name)
    if python is None:
        pytest.skip(f"no environment for {name}")
    doc = Doc(id="scan_simple", path=corpus / "scan_simple.pdf", pages=1, tier="T3", bucket="S")
    run = run_one(name, doc, out_root=tmp_path, python=python, options={}, max_rss_gb=8, timeout_min=20)
    _skip_if_blocked(name, run)
    assert run.status == "ok", run.error
    assert "sales" in (tmp_path / name / "scan_simple" / "result.md").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("name", LOCAL)
def test_adapters_honor_a_page_range_and_report_absolute_page_numbers(name, corpus, tmp_path):
    """Sampled runs (config `sampling`) depend on this: asking for page 2 must return only page 2."""
    python = _python_for(name)
    if python is None:
        pytest.skip(f"no environment for {name}")
    doc = Doc(id="gen_t2_long", path=corpus / "gen_t2_long.pdf", pages=2, tier="T2", bucket="S")
    run = run_one(name, doc, out_root=tmp_path, python=python, options={}, max_rss_gb=8, timeout_min=20, page_range=(2, 2))
    _skip_if_blocked(name, run)
    assert run.status == "ok", run.error
    assert run.sampled and run.page_range == [2, 2]
    pages = json.loads((tmp_path / name / "gen_t2_long" / "pages.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in pages] == [2]


def test_cli_failures_surface_the_tools_own_error_output():
    """A failing external CLI must explain itself in run.json, not just say 'exit status 3'."""
    import sys as _sys

    from benchmarks.pdf_extraction.extractors.mineru_adapter import run_cli

    command = [_sys.executable, "-c", "import sys; sys.stderr.write('ModuleNotFoundError: No module named six'); sys.exit(3)"]
    with pytest.raises(RuntimeError) as excinfo:
        run_cli(command)
    assert "exit status 3" in str(excinfo.value) and "No module named six" in str(excinfo.value)
    run_cli([_sys.executable, "-c", "pass"])  # success returns quietly


# ── tesseract: one recognition pass per page, version first, empty pages survive ──


def _fake_tesseract(commands: list[list[str]], words_per_page: int = 2):
    def run(command):
        commands.append(command)
        if command[1:2] == ["--version"]:
            return "tesseract 5.5.0\n leptonica-1.84\n"
        base = Path(command[2])
        rows = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"]
        rows += [f"5\t1\t1\t1\t1\t{i + 1}\t{100 * i}\t50\t40\t20\t95\tword{i}" for i in range(words_per_page)]
        base.with_suffix(".txt").write_text(" ".join(f"word{i}" for i in range(words_per_page)) + "\n", encoding="utf-8")
        base.with_suffix(".tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return ""
    return run


def test_tesseract_runs_one_recognition_per_page_and_reads_the_version_first(corpus, monkeypatch, tmp_path):
    from benchmarks.pdf_extraction.extractors import tesseract_adapter as ta

    import pymupdf

    monkeypatch.chdir(tmp_path)  # the pre-fix adapter passes "stdout" as the output base; keep any stray write out of the repo
    commands: list[list[str]] = []
    monkeypatch.setattr(ta, "_run", _fake_tesseract(commands))
    monkeypatch.setattr(ta.shutil, "which", lambda name: "/usr/bin/tesseract")
    with pymupdf.open(corpus / "gen_t2_long.pdf") as pdf:
        page_count = len(pdf)  # gen_t2_long has 4 pages
    result = load_extractor("tesseract").extract(corpus / "gen_t2_long.pdf", None)
    recognitions = [c for c in commands if c[1:2] != ["--version"]]
    assert commands[0][1] == "--version"  # version is read before any page work
    assert page_count > 1 and len(recognitions) == page_count  # one call per page
    assert all(c[-2:] == ["txt", "tsv"] for c in recognitions)  # both outputs from a single pass
    assert result.version.endswith("tesseract 5.5.0") and len(result.pages) == page_count
    assert result.pages[0].blocks and result.pages[0].blocks[0].text == "word0 word1"


def test_tesseract_handles_a_page_with_no_words(corpus, monkeypatch, tmp_path):
    from benchmarks.pdf_extraction.extractors import tesseract_adapter as ta

    monkeypatch.chdir(tmp_path)  # the pre-fix adapter passes "stdout" as the output base; keep any stray write out of the repo
    monkeypatch.setattr(ta, "_run", _fake_tesseract([], words_per_page=0))
    monkeypatch.setattr(ta.shutil, "which", lambda name: "/usr/bin/tesseract")
    result = load_extractor("tesseract").extract(corpus / "gen_t2_simple.pdf", None)
    assert [p.page for p in result.pages] == [1] and result.pages[0].text == "" and result.pages[0].blocks == []


def test_tesseract_version_failure_does_not_discard_the_extraction(monkeypatch):
    from benchmarks.pdf_extraction.extractors import tesseract_adapter as ta

    def boom(command):
        raise RuntimeError("no tesseract")

    monkeypatch.setattr(ta, "_run", boom)
    assert "unknown" in ta.version_string()


# ── pdfplumber releases page caches ──────────────────────────────────────────


@pytest.mark.skipif(not _available("pdfplumber"), reason="pdfplumber not installed")
def test_pdfplumber_flushes_each_page_cache(corpus, monkeypatch):
    import pdfplumber.page
    import pdfplumber.pdf

    # pdfplumber's own PDF.close() flushes every page at the end, and page.filter() flushes the derived
    # FilteredPage; neither releases memory during the loop. Count only real pages flushed while the PDF is open.
    calls, closing = [], []
    original_flush, original_close = pdfplumber.page.Page.flush_cache, pdfplumber.pdf.PDF.close

    def counting(self, *args, **kwargs):
        if type(self) is pdfplumber.page.Page and not closing:
            calls.append(self.page_number)
        return original_flush(self, *args, **kwargs)

    def marking_close(self):
        closing.append(True)
        return original_close(self)

    monkeypatch.setattr(pdfplumber.page.Page, "flush_cache", counting)
    monkeypatch.setattr(pdfplumber.pdf.PDF, "close", marking_close)
    load_extractor("pdfplumber").extract(corpus / "gen_t2_long.pdf", None)
    assert calls == [1, 2, 3, 4]  # gen_t2_long has 4 pages; each flushed in page order


# ── marker honours config options ────────────────────────────────────────────


def test_marker_config_includes_options_and_zero_based_page_list():
    from benchmarks.pdf_extraction.extractors.marker_adapter import MarkerExtractor

    config = MarkerExtractor({"disable_ocr": True}).build_config([2, 3], (2, 3))
    assert config["disable_ocr"] is True and config["paginate_output"] is True
    assert config["page_range"] == [1, 2] and config["output_format"] == "markdown"
    assert "page_range" not in MarkerExtractor().build_config([1, 2], None)


# ── marker_noocr: the variant the benchmark actually runs ────────────────────


def test_marker_noocr_variant_is_not_ocr_capable_but_finishes_an_image_only_page(corpus, tmp_path):
    """The benchmark runs Marker only as marker_noocr; on a scan it must finish (empty text is fine), not fail on llama-server.

    Real tool: the name contains ``ocr_capable`` so the light test command deselects it.
    """
    python = _python_for("marker")
    if python is None:
        pytest.skip("no environment for marker")
    doc = Doc(id="scan_simple", path=corpus / "scan_simple.pdf", pages=1, tier="T3", bucket="S")
    run = run_one("marker_noocr", doc, adapter="marker", out_root=tmp_path, python=python,
                  options={"disable_ocr": True}, max_rss_gb=8, timeout_min=20)
    assert run.status == "ok", run.error
    assert (tmp_path / "marker_noocr" / "scan_simple" / "run.json").exists()


def test_easyocr_uses_gpu_only_when_cuda_is_available_unless_overridden():
    from benchmarks.pdf_extraction.extractors.easyocr_adapter import use_gpu

    assert use_gpu({}, cuda_available=True) is True
    assert use_gpu({}, cuda_available=False) is False
    assert use_gpu({"gpu": False}, cuda_available=True) is False  # explicit option wins
    assert use_gpu({"gpu": True}, cuda_available=False) is True
