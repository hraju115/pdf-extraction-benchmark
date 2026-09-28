"""Build the whole corpus: download real PDFs, generate documents, rasterize scans, write ground truth.

    python -m benchmarks.pdf_extraction.corpus.build [--skip-download]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

import httpx
import pymupdf

from benchmarks.pdf_extraction.corpus.generated import GENERATED_TIERS, build_generated, write_truth
from benchmarks.pdf_extraction.corpus.make_scans import native_page_truth, rasterize_pdf, write_page_truth
from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry, check_built_pages, load_manifest
from benchmarks.pdf_extraction.paths import (
    ANNOTATIONS_DIR,
    GENERATED_QUESTIONS_DIR,
    MANIFEST_PATH,
    PDF_DIR,
    TRUTH_DIR,
)

Downloader = Callable[[str], bytes]


def all_entries(manifest_path: Path = MANIFEST_PATH) -> list[ManifestEntry]:
    """Manifest documents (real + scans) plus every generated document."""
    entries = load_manifest(manifest_path) if manifest_path.exists() else []
    entries += [
        ManifestEntry(id=doc_id, tier=tier, bucket="S", source="generated", license="cc0")
        for doc_id, tier in GENERATED_TIERS.items()
    ]
    return entries


def default_download(url: str) -> bytes:
    response = httpx.get(url, follow_redirects=True, timeout=120, headers={"User-Agent": "DocuMind-Benchmark/1.0"})
    response.raise_for_status()
    return response.content


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_real(entries: list[ManifestEntry], pdf_dir: Path, download: Downloader = default_download) -> list[str]:
    """Download every real document, verifying sha256. Returns a list of problems (empty = all good)."""
    problems: list[str] = []
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for entry in (e for e in entries if e.source == "real"):
        target = pdf_dir / f"{entry.id}.pdf"
        if target.exists() and sha256_of(target.read_bytes()) == entry.sha256:
            continue  # already present and verified
        try:
            data = download(entry.url)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{entry.id}: download failed: {exc}")
            continue
        if sha256_of(data) != entry.sha256:
            problems.append(f"{entry.id}: sha256 mismatch (expected {entry.sha256[:12]}..., got {sha256_of(data)[:12]}...)")
            continue  # never keep an unverified file
        target.write_bytes(data)
        with pymupdf.open(target) as pdf:
            if len(pdf) != entry.pages:
                problems.append(f"{entry.id}: manifest says {entry.pages} pages, file has {len(pdf)}")
    return problems


def build_scans(entries: list[ManifestEntry], pdf_dir: Path, truth_dir: Path) -> list[str]:
    problems: list[str] = []
    for entry in (e for e in entries if e.source == "scan"):
        source = pdf_dir / f"{entry.derived_from}.pdf"
        if not source.exists():
            problems.append(f"{entry.id}: source {entry.derived_from} is not built")
            continue
        truth = rasterize_pdf(source, pdf_dir / f"{entry.id}.pdf", dpi=entry.dpi, degrade=entry.degrade, seed=hash_seed(entry.id))
        write_page_truth(truth_dir / f"{entry.id}.pages.json", truth)
        problems.extend(check_built_pages(entry, len(truth)))
    return problems


def hash_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def write_native_truth(entries: list[ManifestEntry], pdf_dir: Path, truth_dir: Path) -> None:
    for entry in (e for e in entries if e.source == "real" and e.truth_from_native):
        path = pdf_dir / f"{entry.id}.pdf"
        if path.exists():
            write_page_truth(truth_dir / f"{entry.id}.pages.json", native_page_truth(path))


def _valid_table_item(item: object) -> bool:
    if isinstance(item, str):
        return "<table" in item.lower()
    return (isinstance(item, dict) and isinstance(item.get("html"), str) and "<table" in item["html"].lower()
            and (item.get("page") is None or (isinstance(item["page"], int) and item["page"] >= 1)))


def copy_annotations(annotations_dir: Path, truth_dir: Path) -> list[str]:
    """Copy hand-annotated ``<doc_id>.tables.json`` files (a JSON list of HTML tables) into the truth directory."""
    problems: list[str] = []
    if not annotations_dir.exists():
        return problems
    truth_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(annotations_dir.glob("*.tables.json")):
        try:
            tables = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: not valid JSON ({exc})")
            continue
        if not isinstance(tables, list) or not tables or not all(_valid_table_item(t) for t in tables):
            problems.append(f"{path.name}: must be a non-empty JSON list of HTML <table> strings or {{page, html}} objects")
            continue
        shutil.copyfile(path, truth_dir / path.name)
    return problems


def annotated_ids(annotations_dir: Path) -> set[str]:
    """Document ids that have their own hand-annotated table truth."""
    if not annotations_dir.exists():
        return set()
    return {path.name.removesuffix(".tables.json") for path in annotations_dir.glob("*.tables.json")}


def inherit_scan_truth(entries: list[ManifestEntry], truth_dir: Path, own: set[str] | frozenset[str] = frozenset()) -> None:
    """A scan is an image-only copy of its source, so it shares the source's table and figure truth.

    The copy is refreshed on every build (a stale copy must never outlive a change to the source's truth), unless
    the scan has its own hand annotation (``own``).
    """
    for entry in (e for e in entries if e.source == "scan" and e.id not in own):
        for suffix in ("tables", "figures"):
            source = truth_dir / f"{entry.derived_from}.{suffix}.json"
            target = truth_dir / f"{entry.id}.{suffix}.json"
            if source.exists():
                shutil.copyfile(source, target)


def build_corpus(
    entries: list[ManifestEntry],
    pdf_dir: Path = PDF_DIR,
    truth_dir: Path = TRUTH_DIR,
    questions_dir: Path = GENERATED_QUESTIONS_DIR,
    download: Downloader = default_download,
    skip_download: bool = False,
    annotations_dir: Path = ANNOTATIONS_DIR,
) -> list[str]:
    problems: list[str] = []
    if not skip_download:
        problems += fetch_real(entries, pdf_dir, download)
    docs = build_generated(pdf_dir)
    write_truth(docs, pdf_dir, truth_dir, questions_dir)
    write_native_truth(entries, pdf_dir, truth_dir)
    problems += copy_annotations(annotations_dir, truth_dir)
    problems += build_scans(entries, pdf_dir, truth_dir)
    inherit_scan_truth(entries, truth_dir, own=annotated_ids(annotations_dir))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the PDF benchmark corpus")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)
    problems = build_corpus(all_entries(), skip_download=args.skip_download)
    for problem in problems:
        print(f"[problem] {problem}", file=sys.stderr)
    print(f"corpus built in {PDF_DIR} ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
