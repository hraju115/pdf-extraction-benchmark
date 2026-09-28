#!/usr/bin/env bash
# Run the PDF-extraction benchmark on a Kaggle GPU notebook, from a DISPOSABLE COPY of the repository.
#
# This script rewrites the copy it runs in: it replaces .envs with a symlink, edits config.yaml's caps, strips the CPU
# torch index from the requirement files and deletes their lock files. It therefore refuses to run unless the copy is
# under /tmp or /kaggle, or PDFBENCH_DISPOSABLE_COPY=1 says you know what you are doing. Never point it at a checkout
# you care about.
#
# Idempotent: re-running resumes by input hash, so a second "Run All" only does what is missing.
# Every knob is an environment variable so the notebook stays a thin wrapper:
#   PDFBENCH_ROOT         repo copy to run from            (default /kaggle/working/documind)
#   PDFBENCH_EXTRACTORS   comma-separated, run in order    (default: all eight local extractors)
#   PDFBENCH_MAX_RSS_GB   per-worker RSS cap               (default 12; the laptop used 4)
#   PDFBENCH_TIMEOUT_MIN  per-document time cap            (default 30, same as the laptop)
#   PDFBENCH_CLEAN_ENVS   1 = delete each heavy virtualenv after its run (disk)   (default 1)
#   PDFBENCH_DOWNSTREAM   1 = run the retrieval stage on the GPU afterwards        (default 1)
#   PDFBENCH_ENVS_DIR     where virtualenvs live (large scratch disk)  (default /tmp/pdfbench-envs)
#   PDFBENCH_RESULT_ZIP   results archive, rewritten after every extractor (default /kaggle/working/pdfbench_results.zip)
#   PDFBENCH_DOCS         optional comma-separated document ids to restrict the run to
set -uo pipefail  # deliberately not -e: one failing extractor must not stop the others

ROOT="${PDFBENCH_ROOT:-/kaggle/working/documind}"
EXTRACTORS="${PDFBENCH_EXTRACTORS:-pymupdf,pdfplumber,pymupdf4llm,easyocr,docling,marker_noocr,mineru,tesseract}"
MAX_RSS_GB="${PDFBENCH_MAX_RSS_GB:-12}"
TIMEOUT_MIN="${PDFBENCH_TIMEOUT_MIN:-30}"
CLEAN_ENVS="${PDFBENCH_CLEAN_ENVS:-1}"
RUN_DOWNSTREAM="${PDFBENCH_DOWNSTREAM:-1}"
ENVS="${PDFBENCH_ENVS_DIR:-/tmp/pdfbench-envs}"
RESULT_ZIP="${PDFBENCH_RESULT_ZIP:-/kaggle/working/pdfbench_results.zip}"
DOCS="${PDFBENCH_DOCS:-}"  # optional comma-separated document ids to restrict the run to (passed as --docs)
HEAVY="easyocr docling marker mineru paddleocr_vl"  # adapters whose virtualenvs are worth deleting

cd "$ROOT" || { echo "no repo at $ROOT"; exit 2; }
case "$(pwd -P)" in
  /tmp/*|/kaggle/*) ;;
  *) [ "${PDFBENCH_DISPOSABLE_COPY:-0}" = "1" ] || { echo "REFUSING: $ROOT is not under /tmp or /kaggle. This script deletes .envs, edits config.yaml and the requirement files; run it in a disposable copy, or set PDFBENCH_DISPOSABLE_COPY=1 if this copy really is disposable."; exit 5; } ;;
esac
export PYTHONPATH="$ROOT:$ROOT/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg  # the notebook kernel exports an inline backend that only exists in its own environment
BENCH=benchmarks/pdf_extraction
REQ=$BENCH/requirements
log() { echo "=== $(date +%T) $*"; }

log "environment"
{ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU visible"
  python -V; nproc; free -g | sed -n 2p; df -h /tmp /kaggle/working 2>/dev/null | tail -n +2; } | tee kaggle_env.txt

# 1. Virtualenvs on the large scratch disk; the harness expects them at <repo>/.envs
mkdir -p "$ENVS"
[ -L .envs ] || { rm -rf .envs; ln -s "$ENVS" .envs; }

# 2. Harness environment (runner, corpus, metrics, scoring) in its own virtualenv, not Kaggle's base image.
#    Kaggle's Python cannot run `python -m venv` (its ensurepip is broken), so virtualenv creates every environment;
#    the harness itself falls back to virtualenv when venv fails.
python -m pip install -q virtualenv 2>&1 | grep -v "^WARNING" || true
if [ ! -x .envs/core/bin/python ]; then
  log "core virtualenv"
  python -m virtualenv --quiet .envs/core && .envs/core/bin/pip install -q -r $REQ/core.txt virtualenv wrapt
fi
CORE=.envs/core/bin/python
$CORE -c "import psutil, yaml, numpy, pymupdf, rapidfuzz" || { echo "FATAL: core virtualenv is missing packages; see errors above"; exit 4; }

# 2b. Kaggle's sitecustomize imports wrapt and complains in every interpreter that lacks it: give each environment one
for f in $REQ/*.txt; do grep -q "^wrapt" "$f" || echo "wrapt" >> "$f"; done

# 3. GPU torch: drop the CPU wheel index and the CPU lock files of the four torch-based extractors
sed -i '/download.pytorch.org\/whl\/cpu/d' $REQ/docling.txt $REQ/marker.txt $REQ/mineru.txt $REQ/easyocr.txt
rm -f $REQ/docling.lock.txt $REQ/marker.lock.txt $REQ/mineru.lock.txt $REQ/easyocr.lock.txt

# 4. Resource caps for this machine (they enter the run hash, so results are labelled with them)
sed -i -E "s/^(\s*max_rss_gb:)\s*[0-9.]+/\1 $MAX_RSS_GB/; s/^(\s*timeout_min:)\s*[0-9.]+/\1 $TIMEOUT_MIN/" $BENCH/config.yaml
grep -E "max_rss_gb|timeout_min" $BENCH/config.yaml

# 5. Tesseract binary. The image's own package is 4.1.1; the laptop measured 5.5.0, and PyMuPDF4LLM's OCR uses whatever
#    Tesseract it finds, so 5.x is required whenever tesseract or pymupdf4llm is in the list. Ubuntu 22.04 gets it from
#    the alex-p/tesseract-ocr5 PPA (added by hand: the image has no add-apt-repository).
tess_major() { command -v tesseract >/dev/null && tesseract --version 2>&1 | head -1 | grep -oE "[0-9]+" | head -1 || echo 0; }
if [[ ",$EXTRACTORS," == *,tesseract,* ]] || [[ ",$EXTRACTORS," == *,pymupdf4llm,* ]]; then
  if [ "$(tess_major)" -lt 5 ]; then
    log "installing Tesseract 5 from the PPA"
    . /etc/os-release 2>/dev/null; codename="${VERSION_CODENAME:-jammy}"
    fp=$(curl -fsSL "https://api.launchpad.net/1.0/~alex-p/+archive/ubuntu/tesseract-ocr5" | python3 -c "import sys,json; print(json.load(sys.stdin)['signing_key_fingerprint'])" 2>/dev/null || true)
    if [ -n "$fp" ]; then
      curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$fp" | gpg --dearmor > /etc/apt/trusted.gpg.d/tesseract5.gpg 2>/dev/null \
        && echo "deb https://ppa.launchpadcontent.net/alex-p/tesseract-ocr5/ubuntu $codename main" > /etc/apt/sources.list.d/tesseract5.list \
        && apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq tesseract-ocr tesseract-ocr-eng >/dev/null 2>&1 || echo "PPA install failed"
    else
      echo "could not read the PPA signing key from Launchpad"
    fi
  fi
  tver=$(tesseract --version 2>&1 | head -1); echo "$tver" | tee -a kaggle_env.txt
  if [ "$(tess_major)" -lt 5 ] && [ "${PDFBENCH_ALLOW_OLD_TESSERACT:-0}" != "1" ]; then
    echo "FATAL: need Tesseract 5.x (have: ${tver:-none}); refusing to produce numbers with a different OCR engine than the rest of the benchmark. Set PDFBENCH_ALLOW_OLD_TESSERACT=1 to override."; exit 6
  fi
  # The engine alone is useless: the English model must load, or every OCR call fails instantly (seen once: 45 error runs).
  if ! tesseract --list-langs 2>/dev/null | grep -qx eng; then
    apt-get install -y -qq tesseract-ocr-eng >/dev/null 2>&1 || true
    tesseract --list-langs 2>/dev/null | grep -qx eng || { echo "FATAL: Tesseract has no 'eng' language data (tesseract-ocr-eng missing)"; exit 7; }
  fi
  echo "tesseract languages: $(tesseract --list-langs 2>/dev/null | tail -n +2 | paste -sd,)" | tee -a kaggle_env.txt
fi

# 6. Corpus check: the bundle carries every PDF and its truth; nothing is downloaded or regenerated here
n_pdf=$(ls data/benchmark_pdfs/*.pdf 2>/dev/null | wc -l)
log "corpus: $n_pdf PDFs"; [ "$n_pdf" -ge 27 ] || { echo "corpus incomplete (expected 27 PDFs)"; exit 3; }

snapshot() {  # rewrite the results archive so a killed session still leaves everything so far
  # The GPU lock files go under kaggle_env/, never under benchmarks/, so unpacking the archive can never overwrite
  # the CPU pins of the checkout it is unpacked next to.
  rm -rf kaggle_env; mkdir -p kaggle_env/requirements; cp $REQ/*.lock.txt kaggle_env/requirements/ 2>/dev/null; cp kaggle_env.txt kaggle_env/
  rm -f "$RESULT_ZIP"
  zip -qr "$RESULT_ZIP" $BENCH/out $BENCH/results kaggle_env -x '*.npz' '*/__pycache__/*' 2>/dev/null
  echo "snapshot: $RESULT_ZIP ($(du -h "$RESULT_ZIP" | cut -f1))"
}

summary() {
  $CORE - "$1" <<'EOF'
import collections, glob, json, sys
name = sys.argv[1]
runs = [json.load(open(p)) for p in glob.glob(f"benchmarks/pdf_extraction/out/{name}/*/run.json")]
counts = collections.Counter(r["status"] for r in runs)
pages = sum(r.get("pages_processed") or 0 for r in runs if r["status"] == "ok")
secs = sum(r.get("seconds") or 0 for r in runs if r["status"] == "ok")
print(f"{name}: {dict(counts)}; {pages} pages ok in {secs/60:.1f} min ({secs/max(pages,1):.2f} s/page)")
for r in runs:
    if r["status"] != "ok":
        print(f"  {r['status']:8s} {r['doc_id']}: {(r.get('error') or '')[:160]}")
EOF
}

# 7. Extractors, strictly one at a time (the runner already runs documents sequentially)
IFS=, read -ra LIST <<< "$EXTRACTORS"
for name in "${LIST[@]}"; do
  log "extractor $name"
  $CORE -m benchmarks.pdf_extraction.run --extractors "$name" ${DOCS:+--docs "$DOCS"} 2>&1 | grep -vE "^\s*$" | tail -n 60 || true
  summary "$name"
  adapter=$($CORE -c "from benchmarks.pdf_extraction.config import load_config; print(load_config().adapter_for('$name'))" 2>/dev/null || echo "$name")
  if [ "$CLEAN_ENVS" = "1" ] && [[ " $HEAVY " == *" $adapter "* ]]; then
    rm -rf ".envs/$adapter"; log "removed virtualenv $adapter ($(df -h /tmp | tail -1 | awk '{print $4}') free)"
  fi
  snapshot
done

# 8. Scoring (seconds, CPU)
log "scoring"
$CORE -m benchmarks.pdf_extraction.scoring 2>&1 | tail -n 5

# 9. Retrieval stage on the GPU: DocuMind's chunker + bge-large + bge-reranker over every extractor's output
if [ "$RUN_DOWNSTREAM" = "1" ]; then
  log "downstream virtualenv (documind + CUDA torch)"
  if [ ! -x .envs/downstream/bin/python ]; then
    python -m virtualenv --quiet .envs/downstream \
      && .envs/downstream/bin/pip install -q -e "$ROOT" -r $REQ/core.txt wrapt \
      || echo "downstream install failed"
  fi
  if .envs/downstream/bin/python -c "import documind, sentence_transformers, torch, rapidfuzz, yaml; print('downstream ok, cuda:', torch.cuda.is_available())"; then
    log "retrieval scoring"
    EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda .envs/downstream/bin/python -m benchmarks.pdf_extraction.downstream.score_retrieval 2>&1 | tail -n 20
  else
    echo "SKIPPED retrieval stage: downstream environment unusable (run it locally instead)"
  fi
fi

snapshot
log "ALL DONE"
