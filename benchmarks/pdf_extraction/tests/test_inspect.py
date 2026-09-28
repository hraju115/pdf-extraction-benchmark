"""Tests for the manifest-entry inspector."""

from __future__ import annotations

import pymupdf

from benchmarks.pdf_extraction.corpus.generated import build_generated
from benchmarks.pdf_extraction.corpus.inspect_pdf import inspect_pdf, main, manifest_entry
from benchmarks.pdf_extraction.corpus.make_scans import rasterize_pdf
from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry, validate_entry


def test_born_digital_and_scanned_documents_are_told_apart(tmp_path):
    build_generated(tmp_path)
    born = inspect_pdf(tmp_path / "gen_t2_simple.pdf")
    assert born["pages"] == 1 and born["bucket"] == "S" and born["pages_with_text_layer"] == 1
    assert "born-digital" in born["hint"] and len(born["sha256"]) == 64

    rasterize_pdf(tmp_path / "gen_t2_simple.pdf", tmp_path / "scan.pdf", dpi=72)
    scan = inspect_pdf(tmp_path / "scan.pdf")
    assert scan["pages_with_text_layer"] == 0 and scan["pages_with_images"] == 1
    assert "T3" in scan["hint"]


def test_mixed_document_is_flagged(tmp_path):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "This page has a real text layer with plenty of characters.")
        pdf.new_page()
        pdf.save(tmp_path / "mixed.pdf")
    assert "mixed" in inspect_pdf(tmp_path / "mixed.pdf")["hint"]


def test_license_is_never_guessed_so_an_unverified_entry_fails_validation(tmp_path):
    build_generated(tmp_path)
    info = inspect_pdf(tmp_path / "gen_t2_simple.pdf")
    entry = manifest_entry(info, "doc_a", "https://example.org/a.pdf", "", "T1")
    assert entry["license"] == ""
    assert any("license" in p for p in validate_entry(ManifestEntry(**entry)))
    verified = manifest_entry(info, "doc_a", "https://example.org/a.pdf", "cc0", "T1")
    assert validate_entry(ManifestEntry(**verified)) == []


def test_cli_prints_a_manifest_entry(tmp_path, capsys):
    build_generated(tmp_path)
    assert main([str(tmp_path / "gen_t2_simple.pdf"), "--id", "x", "--license", "cc0", "--url", "u"]) == 0
    out = capsys.readouterr().out
    assert '"id": "x"' in out and '"license": "cc0"' in out
