"""Tests for the uniform extractor output schema."""

from __future__ import annotations

import json

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    PageResult,
    RunInfo,
    read_run,
    resolve_pages,
    validate_outputs,
    write_outputs,
    write_run,
)


def _result() -> ExtractionResult:
    return ExtractionResult(
        markdown="# Title\n\nHello",
        pages=[PageResult(page=1, text="Title Hello", blocks=[Block("heading", "Title", [0, 0, 10, 10])])],
    )


def test_valid_outputs_pass_validation(tmp_path):
    write_outputs(tmp_path, _result(), RunInfo(extractor="x", doc_id="d", pages_processed=1))
    assert validate_outputs(tmp_path) == []


def test_missing_run_json_is_reported(tmp_path):
    assert validate_outputs(tmp_path) == ["run.json missing"]


def test_invalid_block_kind_and_bbox_are_reported(tmp_path):
    result = ExtractionResult(
        markdown="x",
        pages=[PageResult(page=1, text="x", blocks=[Block("banana", "x", [1, 2, 3])])],
    )
    write_outputs(tmp_path, result, RunInfo(extractor="x", doc_id="d"))
    problems = validate_outputs(tmp_path)
    assert any("kind invalid" in p for p in problems)
    assert any("bbox must have 4 numbers" in p for p in problems)


def test_page_number_must_be_positive_int(tmp_path):
    result = ExtractionResult(markdown="x", pages=[PageResult(page=0, text="x")])
    write_outputs(tmp_path, result, RunInfo(extractor="x", doc_id="d"))
    assert any("page must be an int" in p for p in validate_outputs(tmp_path))


def test_failed_run_needs_no_result_files(tmp_path):
    write_run(tmp_path, RunInfo(extractor="x", doc_id="d", status="timeout", error="too slow"))
    assert validate_outputs(tmp_path) == []


def test_unknown_status_is_reported(tmp_path):
    write_run(tmp_path, RunInfo(extractor="x", doc_id="d", status="weird"))
    assert any("invalid status" in p for p in validate_outputs(tmp_path))


def test_read_run_round_trips_and_tolerates_garbage(tmp_path):
    write_run(tmp_path, RunInfo(extractor="x", doc_id="d", seconds=1.5, page_range=[1, 3], sampled=True))
    run = read_run(tmp_path)
    assert run is not None and run.seconds == 1.5 and run.page_range == [1, 3] and run.sampled
    (tmp_path / "run.json").write_text("{not json", encoding="utf-8")
    assert read_run(tmp_path) is None
    assert read_run(tmp_path / "nope") is None


def test_read_run_ignores_keys_older_versions_wrote(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"extractor": "x", "doc_id": "d", "cost_usd": 0.0}), encoding="utf-8")
    run = read_run(tmp_path)
    assert run is not None and run.extractor == "x" and run.doc_id == "d"


def test_run_json_contains_required_keys(tmp_path):
    write_run(tmp_path, RunInfo(extractor="x", doc_id="d"))
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    for key in ("extractor", "doc_id", "status", "seconds", "peak_rss_mb", "pages_processed"):
        assert key in data


def test_resolve_pages_clamps_range_to_document():
    assert resolve_pages(5, None) == [1, 2, 3, 4, 5]
    assert resolve_pages(5, (2, 3)) == [2, 3]
    assert resolve_pages(5, (4, 99)) == [4, 5]
    assert resolve_pages(5, (0, 2)) == [1, 2]
    assert resolve_pages(5, (9, 12)) == []
