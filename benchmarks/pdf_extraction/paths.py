"""Filesystem locations shared by every stage. All are relative to the repository root."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks" / "pdf_extraction"

PDF_DIR = REPO_ROOT / "data" / "benchmark_pdfs"  # downloaded + generated + scanned PDFs (gitignored)
TRUTH_DIR = PDF_DIR / "truth"  # per-document ground truth (gitignored, rebuilt by corpus.build)
GENERATED_QUESTIONS_DIR = PDF_DIR / "questions_generated"  # auto-written for generated documents
QUESTIONS_DIR = BENCH_DIR / "corpus" / "questions"  # hand-authored, committed
MANIFEST_PATH = BENCH_DIR / "corpus" / "manifest.yaml"
ANNOTATIONS_DIR = BENCH_DIR / "corpus" / "truth_annotations"  # hand-annotated table truth for real documents (committed)
CONFIG_PATH = BENCH_DIR / "config.yaml"
REQUIREMENTS_DIR = BENCH_DIR / "requirements"

OUT_DIR = BENCH_DIR / "out"  # extractor outputs: out/<extractor>/<doc_id>/ (gitignored)
RESULTS_DIR = BENCH_DIR / "results"  # CSVs written by the scoring stages (gitignored)
ENVS_DIR = REPO_ROOT / ".envs"  # one virtualenv per extractor (gitignored)


def pdf_path(doc_id: str) -> Path:
    return PDF_DIR / f"{doc_id}.pdf"
