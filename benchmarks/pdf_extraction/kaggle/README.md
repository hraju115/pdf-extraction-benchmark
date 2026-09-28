# Running the benchmark on Kaggle

The reported results come from two free Kaggle notebooks with the same caps (12 GB peak RSS per worker, 30 minutes
per document): a CPU-only notebook for the four CPU tools (PyMuPDF, pdfplumber, PyMuPDF4LLM, Tesseract) and a T4
notebook for the four torch-based tools (EasyOCR, Docling, Marker, MinerU) and the retrieval stage. Nothing in the
harness changes; this folder holds the drivers, the notebooks and the bundle builder.

## Files

| File | Purpose |
|---|---|
| `make_bundle.sh` | Builds `dist/documind-pdfbench-kaggle.zip`: harness, `src/`, `pyproject.toml` and the built corpus (PDFs, truth, questions). ~140 MB. Excludes `out/`, `results/`, `report/out/` and `merged/`. |
| `pdfbench_kaggle.ipynb` | The extraction notebook: parameters, bundle discovery, one call to the driver, a summary. Run it twice (see below). |
| `run_kaggle.sh` | The extraction driver: virtualenvs on the scratch disk, CUDA torch instead of the CPU wheels, caps, one extractor at a time, scoring, optional retrieval stage, results zip after every step. |
| `pdfbench_downstream_kaggle.ipynb` | The retrieval-only notebook: scores extraction outputs produced elsewhere (a second dataset holding `out/`) through the chunker, embedder and reranker on the GPU. |
| `run_downstream_only.sh` | Its driver. A row cache shipped with the outputs (`results/retrieval_cache`) is carried over before the results directory is replaced, so texts already scored are not scored again. |

**Both drivers rewrite the copy they run in** (they replace `.envs` with a symlink, edit `config.yaml`, strip the CPU
torch index and delete lock files) and refuse to start unless that copy is under `/tmp` or `/kaggle`, or
`PDFBENCH_DISPOSABLE_COPY=1` is set. Never point them at a checkout you care about.

## The three notebooks

1. **Extraction, CPU-only.** `pdfbench_kaggle.ipynb` with **Accelerator: None**, `PDFBENCH_EXTRACTORS` set to
   `pymupdf,pdfplumber,pymupdf4llm,tesseract` and `PDFBENCH_DOWNSTREAM=0`.
2. **Extraction, T4.** The same notebook with **Accelerator: GPU T4 x2** and `PDFBENCH_EXTRACTORS` set to
   `easyocr,docling,marker_noocr,mineru`.
3. **Retrieval only.** `pdfbench_downstream_kaggle.ipynb` on a T4, over the combined `out/` trees of the two
   extraction runs, uploaded as a second dataset.

Tesseract: the Kaggle image ships Tesseract 4.1.1. Whenever `tesseract` or `pymupdf4llm` (which OCRs through the
system Tesseract) is in the list, `run_kaggle.sh` installs Tesseract 5 with its English data (`tesseract-ocr-eng`)
from the `alex-p/tesseract-ocr5` PPA, and stops with an error if 5.x or the `eng` data is not available afterwards
(`PDFBENCH_ALLOW_OLD_TESSERACT=1` overrides the version check). The version and languages are written to
`kaggle_env.txt`.

`PDFBENCH_DOCS` (comma-separated document ids, passed to `run --docs`) restricts a run to some documents, e.g. to
rerun one document with a different cap.

## Steps

1. Locally: `bash benchmarks/pdf_extraction/kaggle/make_bundle.sh`.
2. On kaggle.com (phone-verified account, needed for internet access): **Datasets → New Dataset**, upload
   `dist/documind-pdfbench-kaggle.zip`, keep it private, name it `documind-pdfbench-kaggle`.
3. **Code → New Notebook → File → Import Notebook**, choose `pdfbench_kaggle.ipynb`; edit the parameter cell for the
   CPU or the T4 run as above.
4. Right-hand panel: **Input → Add Input → your dataset**; **Settings → Accelerator** (None or GPU T4 x2);
   **Internet: On**.
5. **Save Version → Save & Run All (Commit)**. Close the tab if you like; the run continues on Kaggle for up to 12 h.
6. When the version finishes: **Output → `pdfbench_results.zip` → Download**. Unzip it into a directory of its own,
   never over the repo root: it holds `benchmarks/pdf_extraction/out`, `benchmarks/pdf_extraction/results` and
   `kaggle_env/` (the lock files and environment facts). Build its report with
   `build_report(results_dir=<dir>/benchmarks/pdf_extraction/results, out_root=<dir>/benchmarks/pdf_extraction/out, out_dir=...)`.
7. To combine the two runs, copy both `out/<extractor>/` trees under one scratch `out_root`, re-run scoring on it and
   run the retrieval-only notebook over it. `runs.csv` carries a `machine` column and the report labels speed, memory
   and failure rows `extractor @machine`. The retrieval row cache is keyed on the extracted text, the questions and
   the model names, so identical text scored on another machine reuses its rows.

## What differs from the laptop run

| Setting | Laptop (appendix) | Kaggle |
|---|---|---|
| torch | CPU wheels, pinned lock files | CUDA wheels from PyPI; the run writes fresh lock files into the results zip |
| Memory cap per worker | 4 GB | 12 GB (`PDFBENCH_MAX_RSS_GB`) |
| Time cap per document | 30 min | 30 min (`PDFBENCH_TIMEOUT_MIN`) |
| Concurrency | heavy extractors two at a time | strictly one at a time |
| Tesseract | 5.5.0 (apt) | 5.5.1 from the PPA (the image's own 4.1.1 is refused) |
| EasyOCR device | CPU (`gpu` auto-detect finds no CUDA) | GPU |
| Retrieval stage | CPU (hours) | GPU (`EMBEDDING_DEVICE=cuda RERANKER_DEVICE=cuda`) |

The run records the GPU name in every `run.json` (`hardware.gpu`), so the report can label which numbers came from
which machine. PaddleOCR-VL stays disabled: its requirement pins the CPU `paddlepaddle` wheel and enabling the GPU
build needs Paddle's own index, which this run does not attempt. Marker's OCR mode stays off (needs `llama-server`).
