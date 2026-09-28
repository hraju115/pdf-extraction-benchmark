#!/usr/bin/env bash
# Run ONLY the retrieval stage (DocuMind chunker + bge-large + bge-reranker) on a GPU, over extraction outputs that
# were produced elsewhere (e.g. the laptop run). Expects:
#   PDFBENCH_ROOT   repo copy (harness + app source + corpus)          default /tmp/documind
#   PDFBENCH_OUT    directory holding out/<extractor>/<doc>/ to score  default: already inside $PDFBENCH_ROOT
#   PDFBENCH_MODES  chunking modes                                      default baseline,structure_preserving
#   PDFBENCH_RESULT_ZIP                                                 default /kaggle/working/pdfbench_retrieval.zip
set -uo pipefail
ROOT="${PDFBENCH_ROOT:-/tmp/documind}"
MODES="${PDFBENCH_MODES:-baseline,structure_preserving}"
RESULT_ZIP="${PDFBENCH_RESULT_ZIP:-/kaggle/working/pdfbench_retrieval.zip}"
ENVS="${PDFBENCH_ENVS_DIR:-/tmp/pdfbench-envs}"
cd "$ROOT" || { echo "no repo at $ROOT"; exit 2; }
case "$(pwd -P)" in  # this script replaces .envs and, with PDFBENCH_OUT, the out/ directory: disposable copies only
  /tmp/*|/kaggle/*) ;;
  *) [ "${PDFBENCH_DISPOSABLE_COPY:-0}" = "1" ] || { echo "REFUSING: $ROOT is not under /tmp or /kaggle; run this in a disposable copy (or set PDFBENCH_DISPOSABLE_COPY=1)."; exit 5; } ;;
esac
export PYTHONPATH="$ROOT:$ROOT/src" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
BENCH=benchmarks/pdf_extraction; REQ=$BENCH/requirements
log() { echo "=== $(date +%T) $*"; }

log "environment"; { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU"; python -V; } | tee kaggle_env.txt
if [ -n "${PDFBENCH_OUT:-}" ]; then rm -rf $BENCH/out; cp -r "$PDFBENCH_OUT" $BENCH/out; fi
n=$(find $BENCH/out -name run.json | wc -l); log "extraction runs to score: $n"; [ "$n" -gt 0 ] || { echo "no runs found"; exit 3; }

mkdir -p "$ENVS"; [ -L .envs ] || { rm -rf .envs; ln -s "$ENVS" .envs; }
python -m pip install -q virtualenv 2>&1 | grep -v "^WARNING" || true
if [ ! -x .envs/downstream/bin/python ]; then
  log "downstream virtualenv (documind + CUDA torch + harness deps)"
  python -m virtualenv --quiet .envs/downstream && .envs/downstream/bin/pip install -q -e "$ROOT" -r $REQ/core.txt wrapt
fi
.envs/downstream/bin/python -c "import documind, sentence_transformers, torch, rapidfuzz, yaml; print('downstream ok, cuda:', torch.cuda.is_available())" \
  || { echo "FATAL: downstream environment unusable"; exit 4; }

# Keep the per-document row cache in the notebook's persisted output, so a session killed by the quota or the 12 h
# limit still leaves every scored document behind and a re-run resumes instead of starting over.
if [ -d /kaggle/working ]; then
  mkdir -p /kaggle/working/results
  # Carry over any shipped row cache (from a previous scoring of the same texts) BEFORE the results dir is replaced,
  # otherwise every document is re-scored from scratch (this once cost a full 4 h GPU pass).
  for src in "$BENCH/results/retrieval_cache" "${PDFBENCH_OUT:-/nonexistent}/../results/retrieval_cache"; do
    [ -d "$src" ] && [ ! -e /kaggle/working/results/retrieval_cache ] && cp -r "$src" /kaggle/working/results/ && echo "reusing row cache from $src ($(find "$src" -name '*.json' | wc -l) files)"
  done
  rm -rf $BENCH/results; ln -s /kaggle/working/results $BENCH/results
fi

snapshot() { rm -f "$RESULT_ZIP"; zip -qr "$RESULT_ZIP" $BENCH/results/ kaggle_env.txt -x '*.npz'; log "snapshot $RESULT_ZIP ($(du -h "$RESULT_ZIP" | cut -f1))"; }
IFS=, read -ra MODE_LIST <<< "$MODES"
for mode in "${MODE_LIST[@]}"; do  # one mode at a time, snapshotting between, so a partial run is still usable
  log "retrieval scoring (mode: $mode)"
  EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda .envs/downstream/bin/python -m benchmarks.pdf_extraction.downstream.score_retrieval --modes "$mode" 2>&1 \
    | grep -vE "reranking_complete|embedding_chunks|chunking_complete|parsing_document" | tail -n 20
  cp $BENCH/results/retrieval_scores.csv "$BENCH/results/retrieval_scores.$mode.csv" 2>/dev/null
  snapshot
done
# score_retrieval rewrites retrieval_scores.csv per invocation; merge the per-mode files into one
.envs/downstream/bin/python - <<'EOF'
import csv, glob
rows, fields = [], None
for f in sorted(glob.glob("benchmarks/pdf_extraction/results/retrieval_scores.*.csv")):
    with open(f, newline="", encoding="utf-8") as h:
        r = csv.DictReader(h); fields = fields or r.fieldnames; rows += list(r)
if fields:
    with open("benchmarks/pdf_extraction/results/retrieval_scores.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"merged {len(rows)} rows into retrieval_scores.csv")
EOF
snapshot; log "ALL DONE"
