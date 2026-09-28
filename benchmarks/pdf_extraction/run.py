"""Run every enabled extractor over the corpus, one extractor at a time.

    python -m benchmarks.pdf_extraction.run [--extractors a,b] [--docs id,id] [--tiers T1,T3]
        [--buckets S,M] [--max-pages N] [--force] [--setup-only] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pymupdf

from benchmarks.pdf_extraction.config import BenchConfig, load_config
from benchmarks.pdf_extraction.envs import ensure_env
from benchmarks.pdf_extraction.extractors.base import RunInfo, hardware_info, read_run, write_run
from benchmarks.pdf_extraction.paths import CONFIG_PATH, OUT_DIR, PDF_DIR, REPO_ROOT

#: Environment variables that look like credentials are never handed to extractors (no tool can leak a secret).
_SECRET_NAME = re.compile(r"(API_?KEY|_TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def is_secret_env_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


@dataclass
class Doc:
    id: str
    path: Path
    pages: int
    tier: str = ""
    bucket: str = ""


def count_pages(pdf: Path) -> int:
    with pymupdf.open(pdf) as document:
        return len(document)


def effective_page_range(total_pages: int, max_pages: int | None) -> tuple[int, int] | None:
    """(1, max_pages) when the document is longer than the cap, else None (whole document)."""
    if max_pages is None or max_pages >= total_pages:
        return None
    return (1, max_pages)


def compute_input_hash(
    pdf: Path,
    page_range: tuple[int, int] | None,
    options: dict[str, Any],
    extractor: str,
    caps: tuple[float, float] | None = None,
    adapter: str | None = None,
) -> str:
    """Identity of a run: the file, the page range, the options, the extractor and (spec 5.3) the resource caps.

    ``adapter`` (default: ``extractor``) is the registry adapter a variant runs; re-pointing a variant at another
    adapter must invalidate its cached result. A plain entry (adapter == extractor) keeps its earlier hash.
    """
    identity: list[Any] = [extractor, page_range, options, caps]
    if adapter and adapter != extractor:
        identity.append({"adapter": adapter})
    digest = hashlib.sha256()
    digest.update(pdf.read_bytes())
    digest.update(json.dumps(identity, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:24]


def _kill_tree(process: psutil.Process) -> None:
    for child in process.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass


def _tree_rss_bytes(process: psutil.Process) -> int:
    total = 0
    for proc in [process, *process.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def run_one(
    name: str,
    doc: Doc,
    *,
    adapter: str | None = None,
    out_root: Path,
    python: Path | str,
    options: dict[str, Any],
    max_rss_gb: float,
    timeout_min: float,
    page_range: tuple[int, int] | None = None,
    force: bool = False,
    retry_failed: bool = False,
) -> RunInfo:
    """Run one extractor on one document in a subprocess, enforcing memory and time caps.

    Resume policy (spec 5.3/5.4): a run whose input, options and caps are unchanged is not repeated, whether it
    succeeded or failed; ``retry_failed`` re-runs failures, ``force`` re-runs everything.

    ``name`` is the config entry (it owns the output directory, the hash and ``run.json``); ``adapter`` is the
    registry extractor the subprocess runs, defaulting to ``name``. Variants of one adapter never share output.
    """
    adapter = adapter or name
    out_dir = out_root / name / doc.id
    input_hash = compute_input_hash(doc.path, page_range, options, name, caps=(max_rss_gb, timeout_min), adapter=adapter)

    existing = read_run(out_dir)
    if existing and existing.input_hash == input_hash and not force:
        if existing.status == "ok" or not retry_failed:
            return existing

    if out_dir.exists():
        shutil.rmtree(out_dir)  # never leave a stale result.md next to a failed run
    out_dir.mkdir(parents=True)

    command = [
        str(python), "-m", "benchmarks.pdf_extraction.extractors.worker",
        adapter, str(doc.path), str(out_dir),
        "--doc-id", doc.id, "--options", json.dumps(options, default=str), "--input-hash", input_hash,
        "--run-name", name,
    ]
    if page_range:
        command += ["--pages", f"{page_range[0]}-{page_range[1]}"]

    env = {k: v for k, v in os.environ.items() if not is_secret_env_name(k)}
    env["PYTHONPATH"] = str(REPO_ROOT)

    started = time.monotonic()
    log_path = out_dir / "worker.log"  # a file, never a pipe: a chatty tool must not block on a full pipe buffer
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ps = psutil.Process(process.pid)
    peak = 0
    killed: str | None = None
    while process.poll() is None:
        peak = max(peak, _tree_rss_bytes(ps))
        if peak > max_rss_gb * 1024**3:
            killed = "oom"
        elif time.monotonic() - started > timeout_min * 60:
            killed = "timeout"
        if killed:
            _kill_tree(ps)
            break
        time.sleep(0.1)
    process.wait()
    elapsed = time.monotonic() - started
    output = _tail(log_path)
    peak_mb = peak / 1024**2

    if killed:
        run = RunInfo(
            extractor=name, doc_id=doc.id, status=killed, seconds=round(elapsed, 1), peak_rss_mb=round(peak_mb, 1),
            page_range=list(page_range) if page_range else None, sampled=page_range is not None,
            input_hash=input_hash, hardware=hardware_info(),
            error=f"killed: {'exceeded ' + str(max_rss_gb) + ' GB memory cap' if killed == 'oom' else 'exceeded ' + str(timeout_min) + ' min timeout'}",
        )
        write_run(out_dir, run)
        return run

    run = read_run(out_dir)
    if run is None:  # worker died before writing run.json (e.g. import error, hard crash)
        run = RunInfo(
            extractor=name, doc_id=doc.id, status="error", seconds=round(elapsed, 1),
            page_range=list(page_range) if page_range else None, sampled=page_range is not None,
            input_hash=input_hash, hardware=hardware_info(), error=output or "worker exited without output",
        )
    run.peak_rss_mb = round(max(run.peak_rss_mb, peak_mb), 1)
    write_run(out_dir, run)
    return run


def _tail(path: Path, limit: int = 1500) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def collect_docs(pdf_dir: Path, entries: list[Any]) -> list[Doc]:
    """Corpus documents whose PDF exists on disk; entries without a file are skipped with a warning."""
    docs: list[Doc] = []
    for entry in entries:
        path = pdf_dir / f"{entry.id}.pdf"
        if not path.exists():
            print(f"[skip] {entry.id}: {path} not built yet (run corpus.build)", file=sys.stderr)
            continue
        try:
            pages = count_pages(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the whole run
            print(f"[skip] {entry.id}: unreadable PDF ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        docs.append(Doc(id=entry.id, path=path, pages=pages, tier=entry.tier, bucket=entry.bucket))
    return docs


def select(docs: list[Doc], ids: list[str] | None, tiers: list[str] | None, buckets: list[str] | None) -> list[Doc]:
    chosen = docs
    if ids:
        chosen = [d for d in chosen if d.id in ids]
    if tiers:
        chosen = [d for d in chosen if d.tier in tiers]
    if buckets:
        chosen = [d for d in chosen if d.bucket in buckets]
    return chosen


def run_all(
    cfg: BenchConfig,
    docs: list[Doc],
    extractor_names: list[str],
    *,
    out_root: Path = OUT_DIR,
    force: bool = False,
    max_pages_override: int | None = None,
    python_override: str | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
) -> list[RunInfo]:
    results: list[RunInfo] = []

    for name in extractor_names:  # strictly sequential: RAM is the scarce resource
        options = cfg.options_for(name)
        adapter = cfg.adapter_for(name)
        python: Path | str = python_override or ""
        for doc in docs:
            page_range = effective_page_range(doc.pages, cfg.max_pages_for(name, max_pages_override))
            pages = (page_range[1] - page_range[0] + 1) if page_range else doc.pages
            if dry_run:
                print(f"[plan] {name} x {doc.id} ({pages} pages{', sampled' if page_range else ''})")
                continue
            if not python:
                try:
                    python = ensure_env(adapter)  # variants share their adapter's virtualenv
                except Exception as exc:  # noqa: BLE001 - a broken environment is a result for this extractor
                    detail = getattr(exc, "stderr", "") or str(exc)
                    run = RunInfo(extractor=name, doc_id=doc.id, status="error", hardware=hardware_info(),
                                  error=f"environment setup failed: {type(exc).__name__}: {str(detail)[-800:]}")
                    write_run(out_root / name / doc.id, run)
                    print(f"[error] {name}: environment setup failed; skipping this extractor", file=sys.stderr)
                    results.append(run)
                    break
            run = run_one(
                name, doc, adapter=adapter, out_root=out_root, python=python, options=options,
                max_rss_gb=cfg.max_rss_gb, timeout_min=cfg.timeout_min,
                page_range=page_range, force=force, retry_failed=retry_failed,
            )
            print(f"[{run.status}] {name} x {doc.id}: {run.seconds}s, {run.peak_rss_mb} MB")
            results.append(run)
    return results


def main(argv: list[str] | None = None) -> int:
    from benchmarks.pdf_extraction.corpus.build import all_entries

    parser = argparse.ArgumentParser(description="Run PDF extractors over the benchmark corpus")
    parser.add_argument("--extractors", help="comma-separated; default: every enabled extractor in config.yaml")
    parser.add_argument("--docs")
    parser.add_argument("--tiers")
    parser.add_argument("--buckets")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--force", action="store_true", help="re-run everything, even successful runs")
    parser.add_argument("--retry-failed", action="store_true", help="re-run failed runs (by default failures are kept as results)")
    parser.add_argument("--setup-only", action="store_true", help="create the virtualenvs and stop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", help="development only: use this interpreter instead of the per-extractor virtualenvs")
    args = parser.parse_args(argv)

    def split(value: str | None) -> list[str] | None:
        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    cfg = load_config(CONFIG_PATH)
    names = split(args.extractors) or cfg.enabled_extractors()
    for name in names:
        if name not in cfg.extractors:
            print(f"unknown or unconfigured extractor: {name}", file=sys.stderr)
            return 2

    if args.setup_only:
        for name in names:
            print(f"setting up {name} ...")
            ensure_env(cfg.adapter_for(name))
        return 0

    docs = select(collect_docs(PDF_DIR, all_entries()), split(args.docs), split(args.tiers), split(args.buckets))
    if not docs:
        print("no documents to run; build the corpus first: python -m benchmarks.pdf_extraction.corpus.build", file=sys.stderr)
        return 2
    run_all(cfg, docs, names, force=args.force, max_pages_override=args.max_pages, dry_run=args.dry_run,
            python_override=args.python, retry_failed=args.retry_failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
