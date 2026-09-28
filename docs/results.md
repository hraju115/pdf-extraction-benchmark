# PDF extraction benchmark: results (final, 2026-09-28)

Eight local PDF extractors, 27 documents (1 to 487 pages), four difficulty tiers, one machine family. Every extractor
ran every document whole under a 12 GB memory cap and a 30 minute time cap on Kaggle: the four CPU-bound tools on a
CPU notebook (4 vCPUs, 31 GB RAM), the four torch-based tools on a T4 notebook (4 vCPUs, 31 GB RAM, Tesla T4 15 GB).
No run failed. The retrieval stage over all of it ran on the T4. The generated report with every table is
`benchmarks/pdf_extraction/merged/report/report.md` (copy: `docs/generated/2026-09-28-report-all-kaggle.md`); the data
is `benchmarks/pdf_extraction/merged/results`. In the public repository the `out/` directories (extractor outputs) and
the failure galleries are not included; the `results/*.csv` files and the generated reports are.

An earlier pass ran the same corpus on a laptop with a 4 GB cap. It is kept as an appendix (section 11) because its
failures are a finding about budgets; none of its numbers are used in sections 1 to 10 unless the text says so.

## 1. Headline findings

1. **Given 12 GB, every tool finishes every document.** On the laptop's 4 GB, Docling, MinerU, EasyOCR and Marker had
   failed on 8, 9, 7 and 1 of 27 documents. The peaks explain why: MinerU reaches 10.8 GB, Docling 7.9 GB, Marker
   5.3 GB on the long documents. EasyOCR stays under 2 GB and had only been too slow.
2. **Docling is the most accurate table extractor, MinerU second.** On the hand-annotated real tables (five tables on
   three documents) TEDS is Docling 0.83, MinerU 0.80, PyMuPDF4LLM 0.76, Marker 0.74, pdfplumber 0.11; on the 487-page
   Economic Report Docling scores 1.00. MinerU alone reproduces the four generated tables perfectly (1.00) and has
   perfect table detection.
3. **Tesseract 5 is the best OCR here, on a CPU.** Character error rate 0.17 on the clean 194-page scan and 0.32 on
   average over the five scans, against MinerU 0.36, PyMuPDF4LLM 0.37 (which is Tesseract underneath), Docling 0.39
   and EasyOCR 0.42, all of those on the T4.
4. **Layout-analysis tools throw chart text away.** EasyOCR recovers 84% of the facts printed in the four generated
   charts, Tesseract 65%, PyMuPDF4LLM 55%; MinerU 7%, Docling 3%, Marker and the text-layer tools 0%.
5. **Through the retrieval pipeline five tools land within 0.04 of each other and the text-layer tools are well
   behind.** Production chunker: Tesseract 0.82, EasyOCR 0.82, MinerU 0.81, Docling 0.80, PyMuPDF4LLM 0.79.
   Structure-preserving chunker: Tesseract 0.80, MinerU 0.80, Docling 0.79, PyMuPDF4LLM 0.78, EasyOCR 0.76. PyMuPDF
   0.64 to 0.65, pdfplumber 0.58 to 0.59, Marker 0.50 to 0.52. Inside the top group the differences are a few
   questions out of 233; what separates them is cost, memory and table structure.
6. **Table structure does not buy retrieval on this corpus.** On the table tier plain PyMuPDF text scores hit@5 0.82
   against 0.60 to 0.66 for the table-aware tools (Docling 0.77 with the production chunker), while the same tools win
   on table structure. Retrieval finds the chunk; structure is what the generator needs to read the right cell.
7. **Speed spans three orders of magnitude on the same hardware.** Fitted per-page cost: PyMuPDF 0.008 s, pdfplumber
   0.16, Marker 0.28 (T4), MinerU 1.0 (T4), PyMuPDF4LLM 1.1, Docling 2.0 (T4), Tesseract 3.1, EasyOCR 3.7 (T4), with
   12 to 49 s of fixed start-up for the model-based tools.
8. **PyMuPDF4LLM 1.28 runs the system Tesseract.** Its worker logs say "Using Tesseract for OCR processing" on
   image-only pages and pictures. Its scan and figure numbers are Tesseract's text under PyMuPDF's layout analysis, and
   they depend on a system Tesseract being present; its per-page cost rises from 0.7 s on born-digital pages to 5.1 s
   on scans for the same reason.

## 2. What was run

### Corpus

Four tiers by difficulty, four buckets by page count (S 1–5, M 6–30, L 31–100, XL 100+).

| Tier | Meaning | Documents (pages) |
|---|---|---|
| T1 | clean born-digital text | RFC 9158 (4), RFC 9457 (16), RFC 9112 (46), RFC 9110 (194) |
| T2 | tables, multi-column | 4 generated table docs (1, 1, 1, 4), ERP 2024 table 1 (2) and table 3 (2), Census ACSBR-023 (9), Census P60-282 (59), Economic Report of the President 2024 (487) |
| T3 | image-only scans | scans of RFC 9158 (4, degraded), ACSBR-023 (9), RFC 9457 (16, degraded), P60-282 (59), RFC 9110 (194) |
| T4 | diagrams and figures | 4 generated chart docs (1 page each: bar, line, flowchart, mixed), NASA Webb factsheet (2), NASA TM X-9471 (7), NASA TM-100396 (46), NASA TM-104114 (103), NASA Systems Engineering Handbook (297) |

Sources: IETF RFC Editor, govinfo.gov, census.gov, nasa.gov and NTRS. The 8 generated documents (`corpus/generated.py`)
carry exact truth: table HTML, figure facts and questions written by the generator. The scans were rasterised from the
born-digital originals and inherit their text truth; two are degraded (blur, noise, skew). Seven real tables were
hand-annotated as HTML truth; the five on the T2 documents feed the table scores. 176 hand-authored questions cover the
real documents (155 machine-verified against the PDF text, 7 figure questions checked by eye, 14
deliberately unanswerable and not scored); with the questions the scans inherit and the generated ones, each extractor
faces 233 scored questions.

### Extractors and machines

| Extractor | Version recorded | Machine | Notes |
|---|---|---|---|
| pymupdf | pymupdf 1.28.2 | Kaggle CPU | text layer only; also the T1 reference |
| pdfplumber | pdfplumber 0.11.10 | Kaggle CPU | text layer + table finder |
| pymupdf4llm | pymupdf4llm 1.28.2 + pymupdf-layout | Kaggle CPU | Markdown with tables; OCRs images through the system Tesseract |
| tesseract | Tesseract 5.5.1 (PPA) via pymupdf rasterisation at 300 dpi | Kaggle CPU | one pass producing text and TSV boxes |
| easyocr | easyocr 1.7.2 | Kaggle T4 | 200 dpi, English, GPU |
| docling | docling 2.130.0 | Kaggle T4 | layout model + TableFormer on the GPU; RapidOCR on the CPU |
| marker_noocr | marker-pdf 2.0.0 | Kaggle T4 | `disable_ocr: true`; Marker's OCR path needs `llama-server` |
| mineru | mineru 3.4.5 | Kaggle T4 | pipeline backend; 4.x is a cloud-service client and was pinned away |

Both Kaggle machines have 4 vCPUs and 31.3 GB RAM; the T4 notebook adds a Tesla T4 with 15 GB. Python 3.12.13. Caps:
12 GB peak RSS per worker (`oom` if exceeded) and 30 minutes per document (`timeout`). One exception: Tesseract on the
487-page report hit the 30 minute cap in the full CPU session (that session averaged 4.6 s/page) and was rerun alone
with a 60 minute cap, where it finished in 19.8 minutes at 2.4 s/page; Kaggle's CPU speed varies between sessions.
Built but not run: `paddleocr_vl` (exceeded 8 GB on one page on the laptop; not retried).

The extraction outputs of PyMuPDF, pdfplumber and PyMuPDF4LLM are byte-identical to the laptop run's; Tesseract 5.5.1
differs from the laptop's 5.5.0 on 22 documents by a handful of characters with the same error rates to two decimals.

## 3. Setup and verification

Each extractor has its own virtualenv under `.envs/<name>` built from `requirements/<name>.txt` and frozen to a lock
file; on Kaggle the four torch-based tools get the CUDA build of torch and the CPU notebook installs Tesseract 5 and
its English data from the `alex-p/tesseract-ocr5` PPA, refusing to run with the image's 4.1.1. Each extractor was
verified on a generated one-page document and its scan before any run (`tests/test_adapters.py`). What the
verification caught, and what stayed blocked:

- Marker 2.x requires `page_range` as a list, not a string (caught by the page-range contract test).
- Marker's OCR mode needs a running `llama-server`; only the no-OCR variant runs.
- MinerU 4.x is a client for a cloud service; 3.4.5 pinned, plus a missing `six` dependency. MinerU emitted 45 of the
  46 pages of NASA TM-100396.
- PaddleOCR-VL exceeded 8 GB on a single page on the laptop and is disabled.
- Docling returned `<!-- image -->` for sparse synthetic pages; the generator was made to produce realistic pages
  (full-width tables, prose framing) so the benchmark does not penalise a tool for the corpus's artificiality.
- A 64 KiB pipe buffer would have turned chatty tools into fake timeouts; worker output goes to a file.
- Kaggle's Python cannot create virtualenvs (`ensurepip` is broken); the harness falls back to `virtualenv`. The
  notebook's matplotlib backend leaks into subprocesses; the driver forces `Agg`.

## 4. Extraction quality

All scores run from 0 to 1 and are quoted as decimals; character error rate is the one where lower is better. No run
failed, so every cell averages every document in its bucket.

### T1: agreement with PyMuPDF's text layer

There is no independent page truth for the real born-digital documents, so T1 measures agreement with PyMuPDF's own
text layer. PyMuPDF scores 1.00 by construction; a lower score can mean worse text or merely a different reading
order, dropped running headers, or de-hyphenation. The T1 headline is retrieval hit@5 (section 5), not this table.

| extractor | S | M | L | XL |
|---|---|---|---|---|
| marker_noocr | 0.79 | 0.82 | 0.87 | 0.92 |
| docling | 0.73 | 0.70 | 0.84 | 0.84 |
| pdfplumber | 0.70 | 0.70 | 0.82 | 0.83 |
| pymupdf4llm | 0.69 | 0.69 | 0.83 | 0.84 |
| tesseract | 0.69 | 0.70 | 0.83 | 0.83 |
| easyocr | 0.67 | 0.68 | 0.81 | 0.82 |
| mineru | 0.63 | 0.73 | 0.80 | 0.78 |

### T2: tables (TEDS, tree edit distance similarity, higher is better)

TEDS turns the extracted table and the annotated truth into HTML trees and scores one minus the normalised number of
cell edits between them: 1.00 is a perfect reconstruction, 0.00 shares nothing. "Real tables" are the five
hand-annotated tables on the three T2 documents (ACSBR-023, P60-282, ERP 2024), averaged per document; the two
annotated RFC tables sit on T1 documents and feed no table here.

| extractor | real tables (3 docs) | of which ERP 2024, 487 pages | generated tables (4 docs) | table detection recall, T2 | s/page |
|---|---|---|---|---|---|
| docling | **0.83** | 1.00 | 0.56 | 0.86 | 2.0 (T4) |
| mineru | 0.80 | 0.94 | **1.00** | **1.00** | 1.0 (T4) |
| pymupdf4llm | 0.76 | 0.91 | 0.57 | 0.86 | 1.1 |
| marker_noocr | 0.74 | 0.58 | 0.54 | 0.86 | 0.28 (T4) |
| pdfplumber | 0.11 | 0.20 | 0.57 | 0.48 | 0.16 |
| pymupdf, tesseract, easyocr | 0.00 | 0.00 | 0.00 | 0.00 | – |

By page-count bucket: Docling 0.56 / 0.72 / 0.76 / 1.00 (S / M / L / XL), MinerU 1.00 / 0.77 / 0.69 / 0.94,
PyMuPDF4LLM 0.57 / 0.66 / 0.71 / 0.91, Marker 0.54 / 0.84 / 0.80 / 0.58, pdfplumber 0.57 / 0.09 / 0.03 / 0.20. The
generated tables are a plain table, one with merged cells, a rotated page and a table running over four pages; every
tool but MinerU gets the plain one right (1.00), most of the merged-cell one (0.79 to 0.84), a third of the four-page
one (0.34) and almost nothing of the rotated page (0.00 to 0.10), which is why they all land near 0.55. pdfplumber
finds many tables but its cell structure rarely matches the truth on the real statistical tables.

### T3: scans (character error rate, lower is better)

| extractor | S (degraded, 4 pages) | M (9–16 pages, one degraded) | L (59 pages) | XL (194 pages) | mean of the 5 scans |
|---|---|---|---|---|---|
| tesseract | 0.29 | 0.36 | 0.44 | 0.17 | **0.32** |
| mineru (T4) | 0.37 | 0.39 | 0.40 | 0.25 | 0.36 |
| pymupdf4llm (Tesseract underneath) | 0.31 | 0.41 | 0.57 | 0.17 | 0.37 |
| docling (T4) | 0.42 | 0.43 | 0.48 | 0.21 | 0.39 |
| easyocr (T4) | 0.34 | 0.48 | 0.62 | 0.18 | 0.42 |
| marker_noocr, pdfplumber, pymupdf | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

0.17 means 17 characters in every 100 are wrong, missing or extra against the original text, and part of that on the
clean XL scan is reading-order and header differences rather than OCR error. The S and M buckets hold the deliberately
degraded scans; the hardest scan is an undegraded one, the 59-page Census report whose dense statistical tables push
every engine above 0.40. The three text-layer tools return nothing on image-only pages, as expected; Marker without
its OCR engine behaves the same way.

### T4: figures (fraction of the facts printed in the charts that appear in the output)

| extractor | bar | line | flowchart | mixed | mean |
|---|---|---|---|---|---|
| easyocr (T4) | 0.92 | 0.62 | 1.00 | 0.83 | **0.84** |
| tesseract | 0.58 | 0.50 | 1.00 | 0.50 | 0.65 |
| pymupdf4llm (Tesseract on pictures) | 0.50 | 0.88 | 0.00 | 0.83 | 0.55 |
| mineru (T4) | 0.08 | 0.12 | 0.00 | 0.08 | 0.07 |
| docling (T4) | 0.00 | 0.12 | 0.00 | 0.00 | 0.03 |
| marker_noocr, pdfplumber, pymupdf | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The charts are embedded as raster images, so the text layer contains none of their labels or values (PyMuPDF 0.00 is
correct behaviour). The layout tools detect the chart as a picture and emit a placeholder. Only the OCR tools read the
values, and only EasyOCR reads most of them. The real NASA documents have no figure truth, so this tier's quality
number rests on four one-page generated documents.

## 5. Retrieval hit@5 through DocuMind's pipeline

Each extractor's output for each document was chunked in two modes, embedded with `bge-large-en-v1.5`, searched
(dense top 50, in memory, one index per document), reranked with `bge-reranker-large` to 5, and the question counts as
a hit when one of the 5 chunks contains its evidence string. 233 scored questions per extractor per mode (T1 48,
T2 65, T3 57, T4 63; types: 141 lookup, 65 table cell, 12 multi-row table, 11 figure, 4 cross-page). `baseline` is
DocuMind's production chunker (450-token budget, 60 overlap); `structure_preserving` is a benchmark-only chunker that
never splits a table or paragraph mid-way. Both measure the budget in the embedder's own tokenizer (a first pass had
used tiktoken; measured on this data set with the embedder's tokenizer, that budget left 13 to 56% of the table-aware
tools' chunks over 512 tokens with the structure-preserving chunker (PyMuPDF4LLM 13%, Marker 25%, MinerU 30%, Docling
56%) and 17 to 30% with the production one, against 0.4 to 6% for Tesseract and PyMuPDF; the effect of fixing it,
isolated on tools whose text did not change, was up to 0.07 on the table tier and 0.02 overall). The stage ran 4 h 14 min on the T4.

### hit@5 by tier

| extractor | production chunker: T1 | T2 | T3 | T4 | **all** | structure-preserving: T1 | T2 | T3 | T4 | **all** |
|---|---|---|---|---|---|---|---|---|---|---|
| tesseract | 0.98 | 0.65 | 0.84 | 0.87 | **0.82** | 0.92 | 0.65 | 0.81 | 0.87 | **0.80** |
| easyocr | 0.92 | 0.62 | 0.84 | 0.92 | **0.82** | 0.92 | 0.55 | 0.72 | 0.90 | 0.76 |
| mineru | 0.94 | 0.65 | 0.84 | 0.84 | 0.81 | 0.96 | 0.60 | 0.86 | 0.83 | **0.80** |
| docling | 0.88 | 0.77 | 0.84 | 0.75 | 0.80 | 0.94 | 0.66 | 0.84 | 0.78 | 0.79 |
| pymupdf4llm | 0.94 | 0.69 | 0.86 | 0.73 | 0.79 | 0.94 | 0.66 | 0.84 | 0.71 | 0.78 |
| pymupdf | 0.96 | 0.82 | 0.00 | 0.83 | 0.65 | 0.90 | 0.82 | 0.00 | 0.83 | 0.64 |
| pdfplumber | 0.94 | 0.65 | 0.00 | 0.81 | 0.59 | 0.94 | 0.62 | 0.00 | 0.81 | 0.58 |
| marker_noocr | 0.83 | 0.55 | 0.00 | 0.63 | 0.50 | 0.83 | 0.60 | 0.00 | 0.67 | 0.52 |

Each cell is the share of that tier's questions for which a chunk holding the evidence made the top five; a 0.00 on
T3 means the tool produced no text for the scans. By page-count bucket (structure-preserving): Tesseract 0.68 / 0.74 /
0.91 / 0.90 (S / M / L / XL), MinerU 0.68 / 0.88 / 0.78 / 0.86, Docling 0.60 / 0.91 / 0.78 / 0.90, PyMuPDF4LLM 0.63 /
0.83 / 0.89 / 0.78, EasyOCR 0.73 / 0.76 / 0.72 / 0.84. The S bucket is the hardest for everyone because it holds the
generated table and figure documents.

### By question type (structure-preserving)

| extractor | lookup (141) | table cell (65) | multi-row table (12) | figure (11) | cross-page (4) |
|---|---|---|---|---|---|
| docling | 0.95 | 0.68 | 0.50 | 0.00 | 0.25 |
| mineru | 0.92 | 0.66 | 0.67 | 0.36 | 0.25 |
| tesseract | 0.93 | 0.63 | 0.67 | 0.55 | 0.25 |
| pymupdf | 0.70 | 0.63 | 0.58 | 0.09 | 0.25 |
| pymupdf4llm | 0.94 | 0.60 | 0.42 | 0.36 | 0.25 |
| easyocr | 0.86 | 0.52 | 0.92 | 1.00 | 0.25 |
| pdfplumber | 0.71 | 0.46 | 0.33 | 0.09 | 0.25 |
| marker_noocr | 0.63 | 0.42 | 0.33 | 0.00 | 0.25 |

Figure questions are answerable only from OCR text (EasyOCR 1.00, Tesseract 0.55). Cross-page questions fail for every
extractor at the same rate, which points at chunking, not extraction. Table-cell questions top out at 0.68.

### Chunking mode and the reranker

The production chunker scores higher than the structure-preserving one for every tool except Marker (Tesseract 0.82
versus 0.80, EasyOCR 0.82 versus 0.76, MinerU 0.81 versus 0.80, Docling 0.80 versus 0.79, PyMuPDF4LLM 0.79 versus
0.78). Keeping tables whole did not help retrieval on these questions. Dense search alone puts the evidence in the top
50 more often than the reranker keeps it in the top 5 on the table tier: PyMuPDF 0.89 against 0.82, PyMuPDF4LLM 0.77
against 0.66, Docling 0.74 against 0.66, Tesseract 0.74 against 0.65. The cut from 50 to 5 is severe by design, but
table-cell questions are where the cross-encoder loses the right chunk most often.

### Recommendation from hit@5 (generated report, structure-preserving)

Best per tier: T1 MinerU (0.96), T2 PyMuPDF (0.82), T3 MinerU (0.86), T4 EasyOCR (0.90). Best single extractor:
**Tesseract, 0.80**, with MinerU 0.80 and Docling 0.79 one and two questions behind; under the production chunker
Tesseract and EasyOCR 0.82, MinerU 0.81, Docling 0.80. A per-tier router using the structure-preserving picks would
score 0.88, +0.08 over the best single extractor. That is an estimate from tier-level picks assuming the tier of each
document is known, and its T1 pick beats four other tools by one question; it is not a measured router.

## 6. Speed: time per page by kind of PDF

Least-squares fit of seconds on pages over all 27 documents per tool, separating fixed start-up (model loading,
mostly) from per-page cost. A 100-page document costs start-up plus 100 times the per-page figure.

| extractor | machine | fixed s | s/page | 100-page document |
|---|---|---|---|---|
| pymupdf | Kaggle CPU | 0.1 | 0.008 | 1 s |
| pdfplumber | Kaggle CPU | 0.0 | 0.16 | 16 s |
| marker_noocr | T4 | 12.0 | 0.28 | 40 s |
| mineru | T4 | 33.7 | 1.02 | 2.3 min |
| pymupdf4llm | Kaggle CPU | 35.6 | 1.06 | 2.4 min |
| docling | T4 | 15.7 | 1.98 | 3.6 min |
| tesseract | Kaggle CPU | 49.3 | 3.10 | 6.0 min |
| easyocr | T4 | 43.0 | 3.69 | 6.9 min |

Fitted seconds per page by kind of document:

| extractor | T1 text | T2 tables | T3 scans | T4 figures | real | scan | generated |
|---|---|---|---|---|---|---|---|
| pymupdf | 0.006 | 0.008 | 0.001 | 0.014 | 0.009 | 0.001 | 0.001 |
| pdfplumber | 0.12 | 0.20 | 0.002 | 0.15 | 0.18 | 0.002 | 0.05 |
| marker_noocr | 0.20 | 0.28 | 0.28 | 0.29 | 0.27 | 0.28 | 0.22 |
| mineru | 0.59 | 1.2 | 0.80 | 0.71 | 1.0 | 0.80 | 2.1 |
| pymupdf4llm | 0.36 | 0.87 | 5.1 | 0.49 | 0.74 | 5.1 | 0.29 |
| docling | 1.2 | 2.3 | 2.4 | 1.1 | 2.0 | 2.4 | 1.9 |
| tesseract | 3.9 | 2.4 | 4.9 | 4.1 | 2.9 | 4.9 | 2.2 |
| easyocr | 4.2 | 3.1 | 4.3 | 5.0 | 3.6 | 4.3 | 2.4 |

What the by-kind numbers say:

- PyMuPDF4LLM is the one text-layer tool sensitive to kind: 5.1 s/page on scans against 0.7 on born-digital pages,
  because it is running Tesseract page by page there.
- Tesseract's cost is 2.4 to 4.9 s/page across kinds; scans cost about 70% more than born-digital pages.
- Docling and EasyOCR spend 20% more per page on scans; MinerU spends less. Marker is flat.
- On the T4 the GPU accelerates only the model forward passes; page rendering, OCR pre- and post-processing (RapidOCR
  inside Docling runs on the CPU) and Markdown assembly run on the four vCPUs, which is why per-page costs there are
  seconds rather than tenths.
- Kaggle's CPU speed varies between sessions: Tesseract averaged 4.6 s/page in the full session and 2.4 s/page in the
  later solo session on the 487-page report.

Plots: `docs/plots/time_vs_pages.png` (seconds per document against pages, log-log, one panel per tier) and
`docs/plots/memory_vs_pages.png`.

## 7. Peak memory

| extractor | S | M | L | XL | maximum |
|---|---|---|---|---|---|
| pymupdf / pdfplumber | 171 MB | 120 MB | 125 MB | 137–142 MB | 205 MB |
| tesseract | 208 | 286 | 370 | 382 | 535 MB |
| pymupdf4llm | 342 | 480 | 699 | 757 | 1.2 GB |
| easyocr (T4) | 1,592 | 1,749 | 1,845 | 1,822 | 2.0 GB |
| marker_noocr (T4) | 1,099 | 1,096 | 2,910 | 3,637 | 5.3 GB |
| docling (T4) | 2,374 | 2,773 | 3,590 | 6,003 | 7.9 GB |
| mineru (T4) | 4,042 | 4,505 | 6,544 | 9,577 | 10.8 GB |

Cells are means per bucket; the last column is the largest single run. Memory grows with page count for Marker,
Docling and MinerU; MinerU's 10.8 GB is the 194-page scan, Docling's 7.9 GB and Marker's 5.3 GB the 487-page report.
These are the peaks that a 4 GB cap turns into failures (section 11).

## 8. Failures

None. All 216 runs (8 extractors × 27 documents) completed under the 12 GB and 30 minute caps, with the one
documented exception of Tesseract on the 487-page report, rerun alone with a 60 minute cap and finished in 19.8
minutes. No run ended with status `error` on any machine: no tool crashed on the content itself.

## 9. Failure gallery highlights

`benchmarks/pdf_extraction/merged/report/failure_gallery.html` (not included in the public repository; the report
builder regenerates it) shows the three worst pages per extractor with the page image, the truth and the output.

- **Docling drops table-of-contents pages.** RFC 9457 page 2 comes back as the 17 characters "Table of Contents"
  (similarity 0.02); the same happens on the scan of that page.
- **Marker without OCR, pdfplumber and PyMuPDF return empty pages on scans**, as their capability rows predict.
- **Dense statistical tables in the Census P60-282 scan** are the hardest OCR pages for every engine (Tesseract
  similarity 0.04 to 0.18 on pages 26, 27 and 35).
- **MinerU's worst T1 page is not wrong, just different:** RFC 9158 page 4 is the author's address block, and MinerU
  reorders the running header and footer relative to PyMuPDF, which the agreement metric punishes.

## 10. Recommendation

| If your PDFs are... | Use | Expect | Why |
|---|---|---|---|
| mixed, one tool, CPU only | PyMuPDF4LLM with Tesseract installed | 1.1 s/page + 36 s start, up to 1.2 GB | hit@5 0.78 / 0.79, TEDS 0.76 on real tables, table detection 0.86 |
| mixed, CPU only, tables do not matter | Tesseract 5 on every page | 3.1 s/page + 49 s, up to 535 MB | hit@5 0.80 / 0.82, the best retrieval score in either mode, no table structure |
| mixed, a GPU and 8+ GB | Docling | 2.0 s/page + 16 s, up to 7.9 GB | TEDS 0.83 on real tables (1.00 on the 487-page report), hit@5 0.79 / 0.80, drops chart text |
| born-digital text, speed first | PyMuPDF | 0.008 s/page, up to 205 MB | hit@5 0.90 on T1 and 0.82 on tables, complete text layer |
| tables where structure matters most | MinerU or Docling, on a GPU | MinerU 1.0 s/page + 34 s, up to 10.8 GB | MinerU TEDS 1.00 generated / 0.80 real, table detection 1.00, hit@5 0.80 / 0.81 |
| image-only scans | Tesseract 5 | 3.1 s/page + 49 s, up to 535 MB | CER 0.17 on clean scans, 0.32 mean, hit@5 0.81 / 0.84 on scans |
| charts and diagrams whose numbers you need | EasyOCR on the page image | 3.7 s/page + 43 s on a T4, up to 2 GB | figure facts 0.84 on four generated charts, figure questions hit@5 1.00; layout tools drop chart text |
| long documents on a 4 GB budget | PyMuPDF4LLM or Tesseract | see above | the only tools with structure or OCR that finished every 194+ page document under 4 GB (section 11) |

The per-tier router estimate is 0.88 hit@5 against 0.80 for the best single extractor: MinerU on clean text and scans,
PyMuPDF on table documents, EasyOCR on figure-heavy documents. It assumes the tier of a document is known in advance
and its clean text pick wins by one question. For DocuMind the practical conclusion is: on a CPU deployment,
PyMuPDF4LLM with Tesseract available (structure plus OCR); with a GPU and 8 GB or more in the ingestion path,
Docling; either way, OCR of figure images if chart values must be retrievable.

## 11. Appendix: the laptop run (4 GB cap)

The same corpus first ran on a laptop (Intel i5-1235U, 22.7 GB RAM, no GPU, on the "balanced" power profile with the
`powersave` governor, cores near 1.5 GHz under load) with a 4 GB cap per worker and the same 30 minute cap; to finish
in a day the four torch-based tools ran two at a time. Its generated report is `docs/generated/2026-09-26-report-laptop-run.md`.

| extractor | documents finished | failures | fitted s/page (under contention for the torch tools) |
|---|---|---|---|
| pymupdf, pdfplumber, pymupdf4llm, tesseract | 27 / 27 | 0 | 0.004, 0.10, 0.67, 2.9 |
| marker_noocr | 26 / 27 | 1 oom (487 pages) | 1.1 |
| easyocr | 20 / 27 | 7 timeouts (59 pages and up) | 32 |
| docling | 19 / 27 | 5 oom, 3 timeouts (46 pages and up) | 40 |
| mineru | 18 / 27 | 9 oom (46 pages and up) | 18 |

Every failure was a cap, not a crash: MinerU exceeded 4 GB within 15 to 40 seconds on four of the five longest
documents and after 111 to 395 seconds on the shorter ones, Docling within 27 to 100 seconds, Marker only on the
487-page report after 9 minutes; EasyOCR never exceeded 2 GB and timed out on every document of 59 pages or more.
Scored with failures as zero, those four tools' retrieval hit@5 fell to 0.40 to 0.52, and their table scores to 0.24
to 0.55. That is the finding this appendix exists for: on a 4 GB budget the layout tools are absent on long documents,
and the "safe" choice is whichever tool never fails.

The laptop's extraction text for PyMuPDF, pdfplumber and PyMuPDF4LLM is byte-identical to Kaggle's; its Tesseract 5.5.0
text differs from Kaggle's 5.5.1 on 22 documents with the same error rates to two decimals; its per-page costs for the
CPU tools were 0.4 to 0.6 of Kaggle's because its cores are faster even throttled.

## 12. Threats to validity

- **T1 has no truth.** Agreement with PyMuPDF's text layer rewards tools that reproduce PyMuPDF's reading order and
  headers. Retrieval hit@5 is the intended T1 measure.
- **Two Kaggle machines, not one.** The CPU tools ran on a CPU notebook and the torch tools on a T4 notebook; both
  have 4 vCPUs and 31 GB, but the T4 tools' per-page costs are GPU-assisted and the CPU tools' are not. Comparing
  Tesseract's 3.1 s/page with Docling's 2.0 compares a CPU-only tool with a GPU-assisted one.
- **Kaggle's CPU speed varies between sessions** (Tesseract 4.6 versus 2.4 s/page on the same document), so the
  CPU-side per-page figures carry perhaps ±50% session noise; the ranking of the tools does not depend on it.
- **One document ran with a different cap** (Tesseract on the 487-page report, 60 minutes); it finished inside 30.
- **Retrieval numbers mix two Tesseract builds only in the sense that** the laptop and Kaggle produced slightly
  different Tesseract text; all numbers in sections 4 to 10 are Kaggle's.
- **PyMuPDF4LLM's scan and figure numbers depend on a system Tesseract.** Without it the tool would score like
  PyMuPDF on those tiers.
- **Marker ran without its OCR engine**, so its scan numbers are those of a text-layer tool. PaddleOCR-VL was not
  run.
- **Different render resolutions for the OCR engines**: Tesseract 300 dpi, EasyOCR 200 dpi.
- **Scans are synthetic.** They were rasterised from the born-digital originals; two were degraded deliberately. The
  XL scan's truth is RFC text only.
- **Small tiers.** Figure truth exists for four one-page generated documents; real table truth is five tables on three
  documents. Differences of a few hundredths are noise at this size.
- **Questions were author-written**, machine-verified against the PDF text and hand spot-checked (20 sampled questions
  plus the 10 figure questions the verifier defers to a human, all correct; see `question-spot-check.md`), but not
  reviewed by a second person.
- **The retrieval stage's first pass measured the chunk budget in the wrong tokenizer**; the corrected pass is what
  this document reports (overflow rates in section 5). The re-score did not reuse its row cache because of a driver defect (now fixed), which cost
  a full GPU pass but not correctness.

## 13. Reproduce

```
PYTHONPATH=. .envs/core/bin/python -m benchmarks.pdf_extraction.corpus.build
PYTHONPATH=. .envs/core/bin/python -m benchmarks.pdf_extraction.run
PYTHONPATH=. .envs/core/bin/python -m benchmarks.pdf_extraction.scoring
PYTHONPATH=.:src .envs/downstream/bin/python -m benchmarks.pdf_extraction.downstream.score_retrieval
PYTHONPATH=. .envs/core/bin/python -m benchmarks.pdf_extraction.report.build_report
```

Runs resume by input hash (file bytes, page range, options, caps, extractor), so a second invocation only does what is
missing. `README.md` in the benchmark folder documents the environments and the variants; `kaggle/README.md`
documents the Kaggle path (a CPU notebook for the CPU tools, a T4 notebook for the torch tools, a retrieval-only
notebook that scores any set of outputs). Results from another machine are unpacked into a separate directory and
reported with `build_report(results_dir=..., out_root=...)`; when outputs from several machines are merged, `runs.csv`
carries a `machine` column and the report labels speed, memory and failure rows by it. The retrieval row cache is keyed
on the extracted text, the questions and the model names, so identical text scored on another machine reuses its rows.
