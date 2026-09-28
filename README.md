# PDF extraction benchmark

Which local PDF extractor should feed a RAG pipeline? This repository holds the harness, corpus definition, scores and
reports behind that question: **8 local extractors** run over **27 public documents** (1 to 487 pages), sorted into
**4 difficulty tiers** and **4 page-count buckets**, and every output is scored both on its own and through a real
retrieval pipeline.

- **Extractors**: PyMuPDF, PyMuPDF4LLM, pdfplumber, Tesseract 5, EasyOCR, Docling, Marker (text layer only) and
  MinerU, each in its own virtualenv. The benchmark covers local extractors only.
- **Tiers**: T1 clean born-digital text, T2 tables and multi-column layouts, T3 image-only scans, T4 diagrams and
  figures. **Buckets**: S 1-5 pages, M 6-30, L 31-100, XL 100+.
- **Metrics**: agreement with the text layer (T1), TEDS against hand-annotated table HTML (T2), character error rate
  (T3), recovered chart facts (T4), time per page, peak memory, and retrieval hit@5 over 233 questions through a
  bge-large embedder and bge-reranker cross-encoder.
- **Failures are data.** Every extractor ran every document whole under the same caps (12 GB peak RSS per worker,
  30 minutes per document). A run that is killed is recorded as `oom` or `timeout` and scores 0; nothing is silently
  dropped. Under 12 GB no run failed; an earlier laptop run with a 4 GB cap, where four tools failed on long documents,
  is kept as an appendix.

The full write-up, with every table, the failure analysis, threats to validity and recommendations, is
**[docs/results.md](docs/results.md)**.

## Headline results

Every number in this README comes from [docs/results.md](docs/results.md) (section 1 and the tables it summarises).
Machines: the four CPU tools ran on a Kaggle CPU-only notebook, the four torch-based tools (EasyOCR, Docling, Marker,
MinerU) on a Kaggle T4 notebook, both with 4 vCPUs and 31 GB RAM, under a 12 GB per-worker cap and a 30 minute
per-document cap. **All 216 runs (8 extractors × 27 documents) finished; no failures.** The retrieval stage ran on the
T4. Hit@5 is with DocuMind's production chunker / the structure-preserving chunker, over 233 questions; seconds per
page are fitted (fixed start-up removed).

| Extractor | Machine | Retrieval hit@5 (production / structure-preserving) | s/page | Notable |
|---|---|---|---|---|
| Tesseract 5.5 | CPU | 0.82 / 0.80 | 3.1 | best OCR here: CER 0.17 on the clean scan, 0.32 mean over five scans; 65% of chart facts |
| EasyOCR | T4 | 0.82 / 0.76 | 3.7 | recovers 84% of chart facts |
| MinerU | T4 | 0.81 / 0.80 | 1.0 | real-table TEDS 0.80; generated tables 1.00; peak 10.8 GB |
| Docling | T4 | 0.80 / 0.79 | 2.0 | best real-table TEDS, 0.83; drops chart text (3% of facts); peak 7.9 GB |
| PyMuPDF4LLM | CPU | 0.79 / 0.78 | 1.1 | real-table TEDS 0.76; OCRs through the system Tesseract |
| PyMuPDF | CPU | 0.64-0.65 | 0.008 | text layer only |
| pdfplumber | CPU | 0.58-0.59 | 0.16 | real-table TEDS 0.11 |
| Marker (no OCR) | T4 | 0.50-0.52 | 0.28 | real-table TEDS 0.74; empty on scans |

The short version: given 12 GB every tool finishes every document; five tools land within 0.04 hit@5 of each other
and are separated by cost, memory and table structure rather than retrieval; table structure (TEDS) and retrieval
disagree (on the table tier plain PyMuPDF text scores hit@5 0.82); and layout-analysis tools throw chart text away.

On the laptop's 4 GB cap Docling, MinerU, EasyOCR and Marker failed on 8, 9, 7 and 1 of the 27 documents; that run is
the appendix, section 11 of [docs/results.md](docs/results.md).

## Repository layout

```
benchmarks/pdf_extraction/     the harness: corpus builder, runner, extractor adapters, metrics, scoring, report
  corpus/                      manifest.yaml (sources), questions/, truth_annotations/ (hand-annotated tables)
  extractors/                  one adapter per tool
  downstream/                  retrieval stage (chunking, embedding, reranking, hit@5)
  requirements/                one requirements file per extractor, plus the *.lock.txt pins of the laptop run
  kaggle/                      bundle builder, notebooks and drivers for the GPU run
  results/                     laptop run (appendix): runs.csv, extraction_scores.csv, retrieval_scores.csv
  report/out/                  laptop run (appendix): generated report.md
  merged/results, merged/report   the reported set (Kaggle CPU + T4): scores, generated report and plots
  tests/                       unit tests
src/documind/                  the DocuMind modules the retrieval stage uses (vendored, see below)
docs/results.md                the write-up
docs/generated/                the generated reports: all-Kaggle (reported), laptop run, Kaggle T4 run, and the
                               earlier laptop CPU + T4 merge
docs/plots/                    time and memory against page count (all-Kaggle set)
docs/article/charts/           charts for the write-up
```

## Reproduce

Run everything from the repository root. Python 3.12 is recommended: the Kaggle runs used 3.12.13, the laptop run
3.12.6, and the lock files were frozen with the latter (the harness and unit tests also run on 3.14). Tesseract 5 must
be installed system-wide for the `tesseract` extractor, and PyMuPDF4LLM uses it for OCR when present
(`sudo apt install tesseract-ocr tesseract-ocr-eng`; Ubuntu's `ppa:alex-p/tesseract-ocr5` provides 5.x where the
distro ships 4.x).

### Environments

One virtualenv per role, all under `.envs/` (gitignored):

```bash
# Harness: corpus builder, runner, metrics, scoring, report, unit tests
python3 -m venv .envs/core && .envs/core/bin/pip install -q -r benchmarks/pdf_extraction/requirements/core.txt

# Retrieval stage: the vendored DocuMind modules + sentence-transformers, with CPU-only torch to keep it small
python3 -m venv .envs/downstream && .envs/downstream/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu -e . -r benchmarks/pdf_extraction/requirements/core.txt
```

Each extractor gets its own `.envs/<name>` automatically on first use (or with `run --setup-only`), installed from
`requirements/<name>.lock.txt` when it exists (the exact CPU environments the laptop numbers came from) and from
`requirements/<name>.txt` otherwise.

The core environment does **not** install the vendored package, because that would pull in torch. Put `src` on
`PYTHONPATH` instead (the test suite's `conftest.py` also does this); the unit tests only need the chunker, parser
and models, whose dependencies are in `core.txt`.

### Tests

```bash
PYTHONPATH=.:src .envs/core/bin/python -m pytest benchmarks/pdf_extraction/tests -q
```

The adapter contract tests run each real extractor whose virtualenv exists (and skip the rest), loading its models;
to keep the suite to fast unit tests, deselect them with
`-k "not (contract_on_a_one_page or ocr_capable or honor_a_page_range or marker_noocr_variant)"`.

### The five stages

```bash
export PYTHONPATH=.:src
.envs/core/bin/python -m benchmarks.pdf_extraction.corpus.build              # download PDFs, generate, rasterise scans, write truth
.envs/core/bin/python -m benchmarks.pdf_extraction.run                       # every enabled extractor, one at a time
.envs/core/bin/python -m benchmarks.pdf_extraction.scoring                   # extraction quality + run stats -> results/*.csv
.envs/downstream/bin/python -m benchmarks.pdf_extraction.downstream.score_retrieval   # chunk, embed, rerank, hit@5
.envs/core/bin/python -m benchmarks.pdf_extraction.report.build_report       # -> report/out/report.md, plots, failure gallery
```

`run --extractors pymupdf` runs one extractor, `run --dry-run` shows what would run, `--retry-failed` re-runs only
failures. Runs resume by input hash (file bytes, page range, options, caps, extractor), so a second invocation only
does what is missing. `corpus.verify` checks the hand-written questions against the PDFs. See
[benchmarks/pdf_extraction/README.md](benchmarks/pdf_extraction/README.md) for variants, sampling and adding
documents.

### Kaggle path (how the reported numbers were produced)

`benchmarks/pdf_extraction/kaggle/` holds a bundle builder (`make_bundle.sh`, which zips the harness, `src/`,
`pyproject.toml` and the built corpus), two notebooks and their drivers. The extraction notebook runs twice: once
CPU-only for PyMuPDF, pdfplumber, PyMuPDF4LLM and Tesseract, once on a T4 for EasyOCR, Docling, Marker and MinerU;
the retrieval-only notebook then scores the combined outputs on the T4. The extraction driver installs Tesseract 5
with its English data from the `alex-p/tesseract-ocr5` PPA and stops if it cannot (the image ships 4.1.1), and
`PDFBENCH_DOCS` restricts a run to chosen documents. The drivers rewrite the copy they run in and refuse to start
outside `/tmp` or `/kaggle`. Results come back as a zip to unpack into a separate directory and report with
`build_report(results_dir=..., out_root=...)`; `runs.csv` carries a `machine` column so outputs from several machines
can be merged with their speed and memory rows kept apart, and the retrieval row cache is keyed on the extracted text,
questions and models, so the same text scored on another machine reuses its rows. See
[benchmarks/pdf_extraction/kaggle/README.md](benchmarks/pdf_extraction/kaggle/README.md).

## Data policy

- **PDFs are not in this repository.** `corpus.build` downloads the real documents from their public sources (IETF
  RFC Editor, govinfo.gov, census.gov, nasa.gov, NTRS; URLs and licences in `corpus/manifest.yaml`) into
  `data/benchmark_pdfs/`, which is gitignored.
- **Synthetic documents and scans are rebuilt locally, not redistributed.** The 8 generated table and chart documents
  and the 5 scans (rasterised from the born-digital originals, two deliberately degraded) are produced by
  `corpus.build` along with their truth files.
- **Extractor outputs are not included** (`out/`, `merged/out/`, the retrieval row cache and the failure
  galleries, which embed page images). The scores, run statistics and generated reports derived from them are.
- Committed truth: the hand-authored questions (`corpus/questions/`) and the hand-annotated table HTML
  (`corpus/truth_annotations/`).

## Hardware caveats

- Two Kaggle machines, not one: the CPU tools ran on a CPU-only notebook, the torch tools on a T4 notebook (both
  4 vCPUs, 31.3 GB RAM; the T4 has 15 GB). The T4 tools' per-page costs are GPU-assisted and the CPU tools' are not,
  so Tesseract's 3.1 s/page and Docling's 2.0 s/page compare a CPU-only tool with a GPU-assisted one.
- Kaggle's CPU speed varies between sessions (Tesseract averaged 4.6 s/page in the full session and 2.4 s/page in a
  later solo session on the 487-page report), so CPU-side per-page figures carry session noise; the ranking of the
  tools does not depend on it.
- One run used a different cap: Tesseract on the 487-page report hit the 30 minute cap in the full CPU session and was
  rerun alone with a 60 minute cap, finishing in 19.8 minutes.
- The 12 GB cap is a policy, not the tools' limit. The laptop appendix (Intel i5-1235U, 22.7 GB RAM, no GPU, 4 GB cap,
  torch tools two at a time) shows what a 4 GB budget does: the layout tools are absent on long documents.

## What is vendored

`src/documind/` is a verbatim copy (plus a one-line header) of the modules of the private DocuMind RAG application
that the retrieval stage imports, so extractor output is scored through the app's own chunking and ranking:

| Module | Why |
|---|---|
| `documind/ingestion/parser.py` | cleans extracted text before chunking |
| `documind/ingestion/chunker.py` | the production chunker (450-token budget in the embedder's tokenizer) |
| `documind/ingestion/embedder.py` | bge-large-en-v1.5 via sentence-transformers |
| `documind/retrieval/reranker.py` | bge-reranker-large cross-encoder |
| `documind/models.py` | `Document`, `Chunk`, `ScoredChunk` models |
| `documind/config.py` | settings (model names, devices, chunk sizes) read from environment variables |
| `documind/logging.py` | structlog setup, imported by the modules above |

Nothing else from the application (API, ingestion service, vector store, generation, evaluation, observability) is
included. `pyproject.toml` packages only these modules; their dependencies are pydantic, pydantic-settings, structlog,
tiktoken, sentence-transformers and torch.

## Licence

MIT, see [LICENSE](LICENSE). The benchmark documents belong to their publishers and are downloaded from the source
at build time.
