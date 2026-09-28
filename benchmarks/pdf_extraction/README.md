# PDF extraction benchmark

Measures how well each PDF extraction strategy turns documents into text DocuMind can retrieve from.
Results and discussion: `docs/results.md` at the repository root. The retrieval stage imports the DocuMind modules
vendored under `src/documind` (chunker, parser, embedder, reranker, models, config); nothing else in this package
depends on them, and the extractors' heavy dependencies live only in their own virtualenvs.

## One-time setup

```bash
python -m venv .envs/core && .envs/core/bin/pip install -r benchmarks/pdf_extraction/requirements/core.txt
sudo apt install tesseract-ocr        # only needed for the tesseract extractor
# retrieval scoring needs the vendored DocuMind modules plus sentence-transformers (CPU-only torch keeps it small):
python -m venv .envs/downstream && .envs/downstream/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -e . -r benchmarks/pdf_extraction/requirements/core.txt
```

Run everything below from the repository root: `.envs/core/bin/python` with `PYTHONPATH=.`, and
`.envs/downstream/bin/python` with `PYTHONPATH=.:src`. Each extractor gets its own virtualenv under `.envs/`,
created on first use (or with `run --setup-only`).

## Tests

```bash
PYTHONPATH=.:src .envs/core/bin/python -m pytest benchmarks/pdf_extraction/tests -q
```

The core environment does not install the vendored package (that would pull in torch); `tests/conftest.py` puts
`src` on the path, and the tests only exercise the chunker, parser and models, whose dependencies (pydantic,
tiktoken, structlog) are in `requirements/core.txt`. Retrieval is tested with fake embedders and rerankers.

## Pipeline

```bash
export PYTHONPATH=.:src PY=.envs/core/bin/python
$PY -m benchmarks.pdf_extraction.corpus.build                     # download, generate, rasterize scans, write truth
$PY -m benchmarks.pdf_extraction.corpus.verify                    # check hand-written questions against the PDFs
$PY -m benchmarks.pdf_extraction.run --setup-only --extractors pymupdf,pymupdf4llm,pdfplumber
$PY -m benchmarks.pdf_extraction.run --dry-run                    # show what would run
$PY -m benchmarks.pdf_extraction.run --extractors pymupdf         # one extractor at a time (RAM is scarce)
$PY -m benchmarks.pdf_extraction.scoring                          # extraction quality + run statistics -> results/*.csv
.envs/downstream/bin/python -m benchmarks.pdf_extraction.downstream.score_retrieval   # needs the downstream env
$PY -m benchmarks.pdf_extraction.report.build_report              # -> report/out/report.md
```

## Running extractors

- **One at a time, 4 GB cap.** Extractors run strictly one after another, each document in its own subprocess.
  A worker whose process tree passes `limits.max_rss_gb` (4 GB) is killed and recorded as `oom`; one that
  passes `limits.timeout_min` is recorded as `timeout`.
- **Failures are results.** A finished run (success or failure) with unchanged input, options and caps is not
  repeated. `--retry-failed` re-runs only the failures; `--force` re-runs everything.
- **Logs.** Each run's combined stdout/stderr is in `out/<extractor>/<doc_id>/worker.log`, next to `run.json`.
- **Variants.** A config entry can run another entry's adapter with different options via `adapter:`, e.g.
  `marker_noocr: {enabled: true, adapter: marker, disable_ocr: true}`. The variant has its own output directory
  and its own row in the report; it shares the adapter's virtualenv.
- **Sampled runs** (`--max-pages` or `sampling.per_extractor`) are labelled `(sampled)` in the report, judged
  only on the pages they read, never averaged with full runs, and never recommended.

## Local extractors only

The benchmark covers extractors that run on the local machine. Workers never receive API keys or other
secret-looking environment variables.

## Adding a real document

1. Download the PDF and read its license on the source page.
2. `$PY -m benchmarks.pdf_extraction.corpus.inspect_pdf file.pdf --id my_doc --url URL --license cc0 --tier T2`
3. Paste the printed entry into `corpus/manifest.yaml`, then write `corpus/questions/my_doc.yaml`
   from the **PDF itself** (never from an extractor's output) and run `corpus.verify`.

## Measured setup cost (this machine: 12 CPUs, ~22 GB RAM of which 4-7 GB free, no GPU)

| Extractor | Environment | Notes |
|---|---|---|
| core (harness) | 0.5 GB | |
| easyocr | 1.4 GB | torch 2.14.0+cpu; contract tests 49 s (models download on first run) |
| docling | 1.8 GB | docling 2.130.0; contract tests 104 s |
| marker | 1.7 GB | marker-pdf 2.0.0; born-digital works. **Its OCR of image-only pages needs llama.cpp's `llama-server`** (Surya OCR backend), which is not installed, so `marker` is disabled and Marker runs as the `marker_noocr` variant (text layer only; scans come out empty) |
| mineru | 1.7 GB | pinned to 3.x (`mineru[pipeline]>=3.4,<4`): MinerU 4 is a different product (cloud-service client, local parsing disabled by default). Also needs `six`. Contract tests 77-175 s |
| paddleocr_vl | 1.7 GB + 2.0 GB `~/.paddlex` | **Exceeds the memory cap on this machine** (killed above 8 GB on a one-page document); unverified |
| tesseract | system `tesseract-ocr` + small venv | Tesseract 5.5.0; contract tests pass |

Shared model caches: `~/.cache/huggingface` 1.9 GB, `~/.paddlex` 2.0 GB.

## Running on a GPU (Kaggle)

The four torch-based extractors are impractical on a CPU (fitted 32 s/page for EasyOCR and 40 s/page for Docling on
this machine), and the retrieval stage's cross-encoder took about 2 s per candidate pair here. `kaggle/` holds a
bundle builder, two notebooks (full extraction; retrieval-only over outputs produced elsewhere) and their drivers for a
free Kaggle T4. The drivers rewrite the copy they run in and refuse to start outside `/tmp` or `/kaggle`; results come
back as a zip to unpack into a directory of its own and report with `build_report(results_dir=..., out_root=...)`.
`runs.csv` carries a `machine` column (from each run's recorded `hardware.gpu`) so outputs from several machines can be
scored together with their speed, memory and failure rows kept apart. See `kaggle/README.md`.

## Chunk budgets are measured in the embedder's tokenizer

`downstream/chunking.py` counts its 450-token budget with tiktoken by default; the retrieval stage passes the
embedder's own tokenizer (`DocuMindEmbedder.count_tokens`) so chunks never exceed the 512-token window that
`bge-large-en-v1.5` and `bge-reranker-large` truncate at. The two tokenizers disagree most on numbers and punctuation,
which is exactly what tables are made of; measured with tiktoken, 13–44 % of the table-aware tools' chunks overflowed.
