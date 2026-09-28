#!/usr/bin/env bash
# Build the upload bundle for the Kaggle run: harness + app source + the built corpus (PDFs, truth, questions).
# Excludes extractor outputs, results, report output, caches and virtualenvs. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-$ROOT/dist/documind-pdfbench-kaggle.zip}"
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT"
cd "$ROOT"
zip -qr "$OUT" pyproject.toml src benchmarks/__init__.py benchmarks/pdf_extraction data/benchmark_pdfs \
  -x '*/__pycache__/*' '*.pyc' '*/.pytest_cache/*' \
     'benchmarks/pdf_extraction/out/*' 'benchmarks/pdf_extraction/results/*' 'benchmarks/pdf_extraction/report/out/*' 'benchmarks/pdf_extraction/merged/*'
echo "$OUT ($(du -h "$OUT" | cut -f1)), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files"
