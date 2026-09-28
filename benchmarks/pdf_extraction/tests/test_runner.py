"""Tests for config, environments, the worker/runner, and corpus building."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest
import yaml

from benchmarks.pdf_extraction.config import load_config
from benchmarks.pdf_extraction.corpus.build import all_entries, build_corpus, copy_annotations, fetch_real, sha256_of
from benchmarks.pdf_extraction.corpus.generated import GENERATED_TIERS, build_generated
from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry
from benchmarks.pdf_extraction.envs import ensure_env
from benchmarks.pdf_extraction.extractors import known_extractors
from benchmarks.pdf_extraction.extractors.base import validate_outputs
from benchmarks.pdf_extraction.paths import CONFIG_PATH
from benchmarks.pdf_extraction.run import (
    Doc,
    collect_docs,
    compute_input_hash,
    effective_page_range,
    run_all,
    run_one,
    select,
)


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pdfs") / "doc.pdf"
    with pymupdf.open() as pdf:
        for number in range(1, 5):
            pdf.new_page().insert_text((72, 72), f"page {number}")
        pdf.save(path)
    return path


def _doc(pdf: Path) -> Doc:
    return Doc(id="doc", path=pdf, pages=4, tier="T1", bucket="S")


def _run(pdf, out_root, options=None, **kwargs):
    defaults = dict(out_root=out_root, python=sys.executable, options=options or {}, max_rss_gb=5, timeout_min=1)
    defaults.update(kwargs)
    return run_one("fake", _doc(pdf), **defaults)


# ── config ───────────────────────────────────────────────────────────────────


def test_shipped_config_loads_with_the_expected_extractors_enabled():
    cfg = load_config(CONFIG_PATH)
    enabled = cfg.enabled_extractors()
    assert "pymupdf" in enabled and "docling" in enabled
    assert "paddleocr_vl" not in enabled  # exceeds this machine's memory; see the ledger
    assert {cfg.adapter_for(name) for name in cfg.extractors} <= set(known_extractors())  # variants point at real adapters


def test_config_rejects_unknown_extractor(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"extractors": {"nonsense": {"enabled": True}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        load_config(path)


def test_options_exclude_enabled_and_max_pages_precedence(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({
        "sampling": {"default_max_pages": 50, "per_extractor": {"docling": 7}},
        "extractors": {"docling": {"enabled": True, "ocr": True}, "pymupdf": {"enabled": True}},
    }), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.options_for("docling") == {"ocr": True}
    assert cfg.max_pages_for("docling") == 7
    assert cfg.max_pages_for("pymupdf") == 50
    assert cfg.max_pages_for("docling", cli_override=3) == 3


# ── environments ─────────────────────────────────────────────────────────────


def test_ensure_env_installs_once_and_writes_a_lock_file(tmp_path):
    reqs, envs = tmp_path / "reqs", tmp_path / "envs"
    reqs.mkdir()
    (reqs / "toolx.txt").write_text("toolx\n", encoding="utf-8")
    calls: list[list[str]] = []

    class Done:
        stdout = "toolx==1.2.3\n"

    def fake_run(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        return Done()

    python = ensure_env("toolx", reqs, envs, run=fake_run)
    assert python == envs / "toolx" / "bin" / "python"
    assert any(c[-2:] == ["-r", str(reqs / "toolx.txt")] for c in calls)
    assert (reqs / "toolx.lock.txt").read_text(encoding="utf-8") == "toolx==1.2.3\n"

    calls.clear()
    ensure_env("toolx", reqs, envs, run=fake_run)
    assert calls == []  # already ready: nothing reinstalled


def test_ensure_env_prefers_lock_file_and_rejects_missing_requirements(tmp_path):
    reqs, envs = tmp_path / "reqs", tmp_path / "envs"
    reqs.mkdir()
    (reqs / "toolx.txt").write_text("toolx\n", encoding="utf-8")
    (reqs / "toolx.lock.txt").write_text("toolx==9.9.9\n", encoding="utf-8")
    calls: list[list[str]] = []
    ensure_env("toolx", reqs, envs, run=lambda cmd, **kw: calls.append([str(c) for c in cmd]))
    assert any(c[-2:] == ["-r", str(reqs / "toolx.lock.txt")] for c in calls)
    with pytest.raises(FileNotFoundError):
        ensure_env("missing", reqs, envs, run=lambda *a, **k: None)


# ── runner ───────────────────────────────────────────────────────────────────


def test_successful_run_writes_valid_outputs(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path)
    assert run.status == "ok" and run.pages_processed == 4 and run.peak_rss_mb > 0
    assert validate_outputs(tmp_path / "fake" / "doc") == []
    assert "fake page 3" in (tmp_path / "fake" / "doc" / "result.md").read_text(encoding="utf-8")


def test_crash_is_recorded_as_data_and_leaves_no_result_files(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path, {"mode": "crash"})
    assert run.status == "error" and "crashed on purpose" in run.error
    assert not (tmp_path / "fake" / "doc" / "result.md").exists()
    assert validate_outputs(tmp_path / "fake" / "doc") == []


def test_timeout_kills_the_worker(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path, {"mode": "sleep", "seconds": 60}, timeout_min=0.03)
    assert run.status == "timeout" and "timeout" in run.error
    assert run.seconds < 30


def test_memory_cap_kills_the_worker(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path, {"mode": "memory", "megabytes": 400, "seconds": 60}, max_rss_gb=0.25)
    assert run.status == "oom" and run.peak_rss_mb > 200


def test_resume_skips_ok_runs_and_force_reruns(tmp_path, sample_pdf):
    marker = tmp_path / "invocations.txt"
    options = {"marker_file": str(marker)}
    _run(sample_pdf, tmp_path / "out", options)
    _run(sample_pdf, tmp_path / "out", options)
    assert marker.read_text().count("run") == 1  # second call was a cache hit
    _run(sample_pdf, tmp_path / "out", options, force=True)
    assert marker.read_text().count("run") == 2


def test_changed_options_invalidate_the_cache(tmp_path, sample_pdf):
    marker = tmp_path / "invocations.txt"
    _run(sample_pdf, tmp_path / "out", {"marker_file": str(marker)})
    _run(sample_pdf, tmp_path / "out", {"marker_file": str(marker), "extra": 1})
    assert marker.read_text().count("run") == 2


def test_failed_runs_are_cached_until_retry_is_asked_or_a_cap_changes(tmp_path, sample_pdf):
    """Spec 5.3: a run is skipped when run.json exists for the same input AND config; failures are data (5.4)."""
    marker = tmp_path / "invocations.txt"
    options = {"mode": "crash", "marker_file": str(marker)}
    first = _run(sample_pdf, tmp_path / "out", options)
    assert first.status == "error"
    second = _run(sample_pdf, tmp_path / "out", options)
    assert second.status == "error" and marker.read_text().count("run") == 1  # cached failure, not re-paid
    _run(sample_pdf, tmp_path / "out", options, retry_failed=True)
    assert marker.read_text().count("run") == 2
    _run(sample_pdf, tmp_path / "out", options, max_rss_gb=6)  # a raised cap is a new config: re-run
    assert marker.read_text().count("run") == 3


def test_chatty_worker_output_does_not_deadlock_and_is_kept_in_a_log(tmp_path, sample_pdf):
    """A tool printing more than the 64 KiB pipe buffer (progress bars) must still finish, and its output must be saved."""
    run = _run(sample_pdf, tmp_path, {"mode": "chatty", "kilobytes": 1024}, timeout_min=2)
    assert run.status == "ok", run.error
    assert run.seconds < 60
    log = tmp_path / "fake" / "doc" / "worker.log"
    assert log.exists() and log.stat().st_size > 1024 * 1024


def test_crash_error_includes_the_workers_log_tail(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path, {"mode": "crash"})
    assert run.status == "error" and "crashed on purpose" in run.error
    assert (tmp_path / "fake" / "doc" / "worker.log").exists()


def test_every_secret_like_env_var_is_stripped_from_local_workers(tmp_path, sample_pdf, monkeypatch):
    for name in ("LLM_API_KEY", "SOME_VENDOR_API_KEY", "HF_TOKEN", "DB_PASSWORD", "MY_SECRET_THING"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")  # ordinary variables must still pass through
    run = _run(sample_pdf, tmp_path, {"report_env": ["LLM_API_KEY", "SOME_VENDOR_API_KEY", "HF_TOKEN", "DB_PASSWORD", "MY_SECRET_THING", "HF_HOME"]})
    text = (tmp_path / "fake" / "doc" / "result.md").read_text(encoding="utf-8")
    for name in ("LLM_API_KEY", "SOME_VENDOR_API_KEY", "HF_TOKEN", "DB_PASSWORD", "MY_SECRET_THING"):
        assert f"{name}=absent" in text, name
    assert "HF_HOME=present" in text


def test_environment_setup_failure_is_recorded_and_the_next_extractor_still_runs(tmp_path, sample_pdf, monkeypatch):
    import subprocess

    import benchmarks.pdf_extraction.run as run_module

    def fake_ensure_env(name):
        if name == "docling":
            raise subprocess.CalledProcessError(1, ["pip", "install"], stderr="no matching distribution")
        return sys.executable

    monkeypatch.setattr(run_module, "ensure_env", fake_ensure_env)
    cfg = load_config(CONFIG_PATH)
    results = run_all(cfg, [_doc(sample_pdf)], ["docling", "fake"], out_root=tmp_path)
    by = {r.extractor: r for r in results}
    assert by["docling"].status == "error" and "environment" in by["docling"].error
    assert (tmp_path / "docling" / "doc" / "run.json").exists()
    assert by["fake"].status == "ok"


def test_hardware_info_reports_total_ram():
    from benchmarks.pdf_extraction.extractors.base import hardware_info

    info = hardware_info()
    assert info["total_ram_gb"] > 0 and info["cpus"] > 0


def test_page_range_sampling_is_recorded(tmp_path, sample_pdf):
    run = _run(sample_pdf, tmp_path, page_range=(1, 2))
    assert run.sampled and run.page_range == [1, 2] and run.pages_processed == 2
    assert effective_page_range(4, 2) == (1, 2)
    assert effective_page_range(4, 4) is None and effective_page_range(4, None) is None


def test_local_workers_never_receive_api_keys(tmp_path, sample_pdf, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    run = _run(sample_pdf, tmp_path, {"report_env": ["LLM_API_KEY"]})
    assert run.status == "ok"
    assert "LLM_API_KEY=absent" in (tmp_path / "fake" / "doc" / "result.md").read_text(encoding="utf-8")


def test_input_hash_depends_on_file_options_and_range(tmp_path, sample_pdf):
    base = compute_input_hash(sample_pdf, None, {}, "fake")
    assert base == compute_input_hash(sample_pdf, None, {}, "fake")
    assert base != compute_input_hash(sample_pdf, (1, 2), {}, "fake")
    assert base != compute_input_hash(sample_pdf, None, {"a": 1}, "fake")
    assert base != compute_input_hash(sample_pdf, None, {}, "other")


def test_select_filters_by_id_tier_and_bucket(sample_pdf):
    docs = [Doc("a", sample_pdf, 1, "T1", "S"), Doc("b", sample_pdf, 9, "T2", "M"), Doc("c", sample_pdf, 9, "T2", "S")]
    assert [d.id for d in select(docs, ["a"], None, None)] == ["a"]
    assert [d.id for d in select(docs, None, ["T2"], ["S"])] == ["c"]
    assert len(select(docs, None, None, None)) == 3


# ── corpus build ─────────────────────────────────────────────────────────────


def test_all_entries_includes_every_generated_document(tmp_path):
    entries = all_entries(tmp_path / "missing_manifest.yaml")
    assert {e.id for e in entries} == set(GENERATED_TIERS)
    assert all(e.source == "generated" and e.bucket == "S" for e in entries)


def test_generated_ids_match_what_the_builder_produces(tmp_path):
    assert {d.id: d.tier for d in build_generated(tmp_path)} == GENERATED_TIERS


def _real_entry(data: bytes, pages: int) -> ManifestEntry:
    return ManifestEntry(id="doc_a", tier="T1", bucket="S", source="real", pages=pages,
                         url="https://example.org/a.pdf", license="cc0", sha256=sha256_of(data))


def _pdf_bytes(tmp_path, pages=2) -> bytes:
    path = tmp_path / "x.pdf"
    with pymupdf.open() as pdf:
        for _ in range(pages):
            pdf.new_page()
        pdf.save(path)
    return path.read_bytes()


def test_fetch_downloads_verifies_and_skips_existing(tmp_path):
    data = _pdf_bytes(tmp_path)
    entry = _real_entry(data, 2)
    calls: list[str] = []

    def download(url):
        calls.append(url)
        return data

    assert fetch_real([entry], tmp_path / "pdfs", download) == []
    assert (tmp_path / "pdfs" / "doc_a.pdf").read_bytes() == data
    assert fetch_real([entry], tmp_path / "pdfs", download) == []
    assert len(calls) == 1  # the second call found a verified file


def test_fetch_rejects_hash_mismatch_and_keeps_no_file(tmp_path):
    data = _pdf_bytes(tmp_path)
    entry = _real_entry(data, 2)
    problems = fetch_real([entry], tmp_path / "pdfs", lambda url: b"tampered")
    assert any("sha256 mismatch" in p for p in problems)
    assert not (tmp_path / "pdfs" / "doc_a.pdf").exists()


def test_fetch_reports_download_failures_and_wrong_page_counts(tmp_path):
    data = _pdf_bytes(tmp_path)

    def boom(url):
        raise OSError("network down")

    assert any("download failed" in p for p in fetch_real([_real_entry(data, 2)], tmp_path / "a", boom))
    assert any("manifest says 5 pages" in p for p in fetch_real([_real_entry(data, 5)], tmp_path / "b", lambda u: data))


def test_build_corpus_creates_generated_docs_scans_and_truth(tmp_path):
    entries = all_entries(tmp_path / "none.yaml") + [
        ManifestEntry(id="gen_scan_simple", tier="T3", bucket="S", source="scan", derived_from="gen_t2_simple", dpi=100),
        ManifestEntry(id="gen_scan_bad", tier="T3", bucket="S", source="scan", derived_from="gen_t2_simple", dpi=100, degrade=True),
        ManifestEntry(id="orphan_scan", tier="T3", bucket="S", source="scan", derived_from="gen_missing"),
    ]
    pdfs, truth = tmp_path / "pdfs", tmp_path / "truth"
    problems = build_corpus(entries, pdfs, truth, tmp_path / "q", skip_download=True)
    assert problems == ["orphan_scan: source gen_missing is not built"]
    assert (pdfs / "gen_scan_simple.pdf").exists() and (pdfs / "gen_scan_bad.pdf").exists()
    pages = json.loads((truth / "gen_scan_simple.pages.json").read_text(encoding="utf-8"))
    assert "Regional sales summary" in pages["1"]


# ── review-focus cases ───────────────────────────────────────────────────────


def test_corrupt_pdf_is_skipped_with_a_warning_instead_of_aborting_the_run(tmp_path, sample_pdf, capsys):
    (tmp_path / "good.pdf").write_bytes(sample_pdf.read_bytes())
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf at all")
    (tmp_path / "empty.pdf").write_bytes(b"")
    entries = [ManifestEntry(id=i, tier="T1", bucket="S", source="generated") for i in ("good", "broken", "empty", "absent")]
    docs = collect_docs(tmp_path, entries)
    assert [d.id for d in docs] == ["good"]
    warnings = capsys.readouterr().err
    assert "broken" in warnings and "unreadable" in warnings and "empty" in warnings and "absent" in warnings


def test_partial_output_from_an_interrupted_run_is_cleaned_and_rerun(tmp_path, sample_pdf):
    stale = tmp_path / "out" / "fake" / "doc"
    stale.mkdir(parents=True)
    (stale / "result.md").write_text("half-written output from a killed run", encoding="utf-8")  # no run.json
    run = _run(sample_pdf, tmp_path / "out")
    assert run.status == "ok"
    assert "half-written" not in (stale / "result.md").read_text(encoding="utf-8")


def test_half_built_environment_without_ready_marker_is_installed_again(tmp_path):
    reqs, envs = tmp_path / "reqs", tmp_path / "envs"
    reqs.mkdir()
    (reqs / "toolx.txt").write_text("toolx\n", encoding="utf-8")
    (envs / "toolx").mkdir(parents=True)  # directory exists from a crashed earlier setup, but no .ready
    calls: list[list[str]] = []

    class Done:
        stdout = "toolx==1.0\n"

    ensure_env("toolx", reqs, envs, run=lambda cmd, **kw: (calls.append([str(c) for c in cmd]), Done())[1])
    assert any("install" in c for c in calls) and (envs / "toolx" / ".ready").exists()


def test_hand_annotated_tables_are_copied_into_the_truth_directory(tmp_path):
    annotations, truth = tmp_path / "annotations", tmp_path / "truth"
    annotations.mkdir()
    (annotations / "real_doc.tables.json").write_text(json.dumps(["<table><tr><td>A</td></tr></table>"]), encoding="utf-8")
    assert copy_annotations(annotations, truth) == []
    assert json.loads((truth / "real_doc.tables.json").read_text(encoding="utf-8")) == ["<table><tr><td>A</td></tr></table>"]


def test_malformed_annotations_are_reported_and_not_copied(tmp_path):
    annotations, truth = tmp_path / "annotations", tmp_path / "truth"
    annotations.mkdir()
    (annotations / "not_json.tables.json").write_text("{oops", encoding="utf-8")
    (annotations / "not_html.tables.json").write_text(json.dumps(["just text"]), encoding="utf-8")
    (annotations / "not_list.tables.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    problems = copy_annotations(annotations, truth)
    assert len(problems) == 3 and not list(truth.glob("*.json"))
    assert copy_annotations(tmp_path / "missing_dir", truth) == []


def test_scans_inherit_the_table_and_figure_truth_of_their_source_document(tmp_path):
    """OCR on an image-only copy of a table page is scored against the same exact table truth."""
    entries = all_entries(tmp_path / "none.yaml") + [
        ManifestEntry(id="scan_of_table", tier="T3", bucket="S", source="scan", derived_from="gen_t2_simple", dpi=100),
        ManifestEntry(id="scan_of_chart", tier="T3", bucket="S", source="scan", derived_from="gen_t4_bar", dpi=100),
        ManifestEntry(id="scan_with_own", tier="T3", bucket="S", source="scan", derived_from="gen_t2_merged", dpi=100),
    ]
    truth = tmp_path / "truth"
    truth.mkdir(parents=True)
    own = json.dumps(["<table><tr><td>hand-annotated</td></tr></table>"])
    annotations = tmp_path / "ann"
    annotations.mkdir()
    (annotations / "scan_with_own.tables.json").write_text(own, encoding="utf-8")
    problems = build_corpus(entries, tmp_path / "pdfs", truth, tmp_path / "q", skip_download=True, annotations_dir=annotations)
    assert problems == []
    assert json.loads((truth / "scan_of_table.tables.json").read_text(encoding="utf-8")) == json.loads(
        (truth / "gen_t2_simple.tables.json").read_text(encoding="utf-8"))
    assert (truth / "scan_of_chart.figures.json").exists() and not (truth / "scan_of_chart.tables.json").exists()
    assert (truth / "scan_with_own.tables.json").read_text(encoding="utf-8") == own  # never overwritten


def test_inherited_scan_truth_is_refreshed_when_the_source_truth_changes(tmp_path):
    entries = all_entries(tmp_path / "none.yaml") + [
        ManifestEntry(id="scan_of_table", tier="T3", bucket="S", source="scan", derived_from="gen_t2_simple", dpi=72),
    ]
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "scan_of_table.tables.json").write_text(json.dumps(["<table><tr><td>stale legacy copy</td></tr></table>"]), encoding="utf-8")
    assert build_corpus(entries, tmp_path / "pdfs", truth, tmp_path / "q", skip_download=True, annotations_dir=tmp_path / "none") == []
    refreshed = json.loads((truth / "scan_of_table.tables.json").read_text(encoding="utf-8"))
    assert refreshed == json.loads((truth / "gen_t2_simple.tables.json").read_text(encoding="utf-8"))
    assert isinstance(refreshed[0], dict) and refreshed[0]["page"] == 1


def test_retry_failed_never_repeats_successful_runs(tmp_path, sample_pdf):
    marker = tmp_path / "invocations.txt"
    options = {"marker_file": str(marker)}
    assert _run(sample_pdf, tmp_path / "out", options).status == "ok"
    _run(sample_pdf, tmp_path / "out", options, retry_failed=True)
    assert marker.read_text().count("run") == 1  # retry_failed only touches failures


# ── variants: one adapter, several config entries ────────────────────────────


def test_config_variants_resolve_to_their_adapter(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"extractors": {
        "marker": {"enabled": False},
        "marker_noocr": {"enabled": True, "adapter": "marker", "disable_ocr": True},
    }}), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.enabled_extractors() == ["marker_noocr"]
    assert cfg.adapter_for("marker_noocr") == "marker" and cfg.adapter_for("marker") == "marker"
    assert cfg.options_for("marker_noocr") == {"disable_ocr": True}


def test_config_rejects_a_variant_of_an_unknown_adapter(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"extractors": {"x_fast": {"enabled": True, "adapter": "nonsense"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        load_config(path)


def test_variants_of_one_adapter_get_their_own_output_and_hash(tmp_path, sample_pdf):
    """Review Focus 1: two variants must never share out/ or a hash, or one silently overwrites the other."""
    a = run_one("fake_a", _doc(sample_pdf), adapter="fake", out_root=tmp_path, python=sys.executable, options={"mode": "ok"}, max_rss_gb=5, timeout_min=1)
    b = run_one("fake_b", _doc(sample_pdf), adapter="fake", out_root=tmp_path, python=sys.executable, options={"mode": "ok"}, max_rss_gb=5, timeout_min=1)
    assert a.status == b.status == "ok"
    assert (tmp_path / "fake_a" / "doc" / "result.md").exists() and (tmp_path / "fake_b" / "doc" / "result.md").exists()
    assert a.input_hash != b.input_hash and a.extractor == "fake_a" and b.extractor == "fake_b"


def test_shipped_config_runs_marker_only_as_the_no_ocr_variant():
    cfg = load_config(CONFIG_PATH)
    assert "marker" not in cfg.enabled_extractors() and "marker_noocr" in cfg.enabled_extractors()
    assert cfg.adapter_for("marker_noocr") == "marker" and cfg.options_for("marker_noocr") == {"disable_ocr": True}
    assert cfg.max_rss_gb == 4


def test_changing_a_variants_adapter_invalidates_its_cached_result(tmp_path, sample_pdf):
    h1 = compute_input_hash(sample_pdf, None, {}, "v", caps=(5, 1), adapter="fake")
    h2 = compute_input_hash(sample_pdf, None, {}, "v", caps=(5, 1), adapter="pymupdf")
    assert h1 != h2 and h1 == compute_input_hash(sample_pdf, None, {}, "v", caps=(5, 1), adapter="fake")


def test_hardware_info_records_the_gpu_probe_result():
    from benchmarks.pdf_extraction.extractors.base import hardware_info

    assert hardware_info(gpu_probe=lambda: "Tesla T4, 15360 MiB")["gpu"] == "Tesla T4, 15360 MiB"
    assert hardware_info(gpu_probe=lambda: None)["gpu"] is None
    assert "gpu" in hardware_info()  # the default probe never raises, even without nvidia-smi


def test_ensure_env_falls_back_to_virtualenv_when_venv_cannot_bootstrap_pip(tmp_path):
    """Some images (Kaggle's) ship a Python whose ``venv`` fails at ensurepip; virtualenv bundles its own pip."""
    import subprocess

    reqs, envs = tmp_path / "reqs", tmp_path / "envs"
    reqs.mkdir()
    (reqs / "toolx.txt").write_text("toolx\n", encoding="utf-8")
    (reqs / "toolx.lock.txt").write_text("toolx==1.0\n", encoding="utf-8")
    calls: list[list[str]] = []

    class Done:
        stdout = ""

    def fake_run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[1:3] == ["-m", "venv"]:
            (envs / "toolx").mkdir(parents=True, exist_ok=True)  # venv leaves a half-built directory behind
            raise subprocess.CalledProcessError(1, cmd)
        return Done()

    python = ensure_env("toolx", reqs, envs, run=fake_run)
    assert python == envs / "toolx" / "bin" / "python"
    assert any(c[1:3] == ["-m", "virtualenv"] for c in calls)
    assert any(c[-2:] == ["-r", str(reqs / "toolx.lock.txt")] for c in calls)
    assert (envs / "toolx" / ".ready").exists()
