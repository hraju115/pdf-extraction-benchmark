# PDF extraction benchmark report
Hardware (recorded with the runs): 4 CPUs, 31.3 GB RAM, no GPU unless stated (`{"cpus": 4, "total_ram_gb": 31.3, "platform": "Linux-6.12.90+-x86_64-with-glibc2.35", "python": "3.12.13", "gpu": "Tesla T4, 15360 MiB"}`).
Sampled runs (first pages only, not comparable to full runs): none.

## Capability matrix
| extractor | page_numbers | bboxes | tables | figures | ocr |
|---|---|---|---|---|---|
| pymupdf | yes | yes | no | no | no |
| pymupdf4llm | yes | yes | yes | yes | yes |
| pdfplumber | yes | yes | yes | no | no |
| docling | yes | yes | yes | yes | yes |
| marker (disabled) | yes | no | yes | no | yes |
| marker_noocr (marker variant) | yes | no | yes | no | no |
| mineru | yes | yes | yes | no | yes |
| tesseract | yes | yes | no | no | yes |
| easyocr | yes | yes | no | no | yes |
| paddleocr_vl (disabled) | yes | no | yes | no | yes |

## Extraction quality by tier
### T1: agreement with PyMuPDF's text layer (reference, not truth — pymupdf is the reference itself)
text_similarity by page-count bucket (higher is closer to PyMuPDF). Retrieval hit@5 below is the T1 headline.
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.73 | 0.70 | 0.84 | 0.84 |
| easyocr | 0.67 | 0.68 | 0.81 | 0.82 |
| marker_noocr | 0.79 | 0.82 | 0.87 | 0.92 |
| mineru | 0.63 | 0.73 | 0.80 | 0.78 |
| pdfplumber | 0.70 | 0.70 | 0.82 | 0.83 |
| pymupdf | 1.00 | 1.00 | 1.00 | 1.00 |
| pymupdf4llm | 0.69 | 0.69 | 0.83 | 0.84 |

### T2: teds (higher is better), by page-count bucket
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.56 | 0.72 | 0.76 | 1.00 |
| easyocr | 0.00 | 0.00 | 0.00 | 0.00 |
| marker_noocr | 0.54 | 0.84 | 0.80 | 0.58 |
| mineru | 1.00 | 0.77 | 0.69 | 0.94 |
| pdfplumber | 0.57 | 0.09 | 0.03 | 0.20 |
| pymupdf | 0.00 | 0.00 | 0.00 | 0.00 |
| pymupdf4llm | 0.57 | 0.66 | 0.71 | 0.91 |

### T3: cer (lower is better), by page-count bucket
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.42 | 0.43 | 0.48 | 0.21 |
| easyocr | 0.34 | 0.48 | 0.62 | 0.18 |
| marker_noocr | 1.00 | 1.00 | 1.00 | 1.00 |
| mineru | 0.37 | 0.39 | 0.40 | 0.25 |
| pdfplumber | 1.00 | 1.00 | 1.00 | 1.00 |
| pymupdf | 1.00 | 1.00 | 1.00 | 1.00 |
| pymupdf4llm | 0.31 | 0.41 | 0.57 | 0.17 |

### T4: figure_fact_recall (higher is better), by page-count bucket
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.03 | — | — | — |
| easyocr | 0.84 | — | — | — |
| marker_noocr | 0.00 | — | — | — |
| mineru | 0.07 | — | — | — |
| pdfplumber | 0.00 | — | — | — |
| pymupdf | 0.00 | — | — | — |
| pymupdf4llm | 0.55 | — | — | — |

### Table detection recall by tier
| extractor | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| docling | 1.00 | 0.86 | 1.00 | 1.00 |
| easyocr | 0.00 | 0.00 | 0.00 | 0.00 |
| marker_noocr | 1.00 | 0.86 | 0.00 | 1.00 |
| mineru | 1.00 | 1.00 | 1.00 | 1.00 |
| pdfplumber | 1.00 | 0.48 | 0.00 | 1.00 |
| pymupdf | 0.00 | 0.00 | 0.00 | 0.00 |
| pymupdf4llm | 1.00 | 0.86 | 1.00 | 1.00 |

## Retrieval hit@5 — chunking: baseline
### By tier
| extractor | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| docling | 0.88 | 0.68 | 0.79 | 0.76 |
| easyocr | 0.92 | 0.60 | 0.82 | 0.92 |
| marker_noocr | 0.83 | 0.58 | 0.00 | 0.63 |
| mineru | 0.94 | 0.68 | 0.86 | 0.83 |
| pdfplumber | 0.94 | 0.65 | 0.00 | 0.81 |
| pymupdf | 0.96 | 0.82 | 0.00 | 0.83 |
| pymupdf4llm | 0.96 | 0.71 | 0.88 | 0.73 |
### By page-count bucket
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.65 | 0.83 | 0.74 | 0.86 |
| easyocr | 0.73 | 0.81 | 0.81 | 0.88 |
| marker_noocr | 0.38 | 0.53 | 0.67 | 0.47 |
| mineru | 0.70 | 0.84 | 0.87 | 0.86 |
| pdfplumber | 0.51 | 0.48 | 0.65 | 0.74 |
| pymupdf | 0.65 | 0.50 | 0.69 | 0.76 |
| pymupdf4llm | 0.63 | 0.95 | 0.87 | 0.79 |

## Retrieval hit@5 — chunking: structure_preserving
### By tier
| extractor | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| docling | 0.94 | 0.71 | 0.86 | 0.79 |
| easyocr | 0.92 | 0.57 | 0.72 | 0.87 |
| marker_noocr | 0.90 | 0.60 | 0.00 | 0.65 |
| mineru | 0.96 | 0.62 | 0.86 | 0.83 |
| pdfplumber | 0.92 | 0.60 | 0.00 | 0.81 |
| pymupdf | 0.90 | 0.75 | 0.00 | 0.81 |
| pymupdf4llm | 0.96 | 0.72 | 0.84 | 0.71 |
### By page-count bucket
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 0.63 | 0.93 | 0.78 | 0.93 |
| easyocr | 0.71 | 0.72 | 0.74 | 0.86 |
| marker_noocr | 0.40 | 0.53 | 0.69 | 0.52 |
| mineru | 0.68 | 0.88 | 0.76 | 0.90 |
| pdfplumber | 0.51 | 0.48 | 0.63 | 0.69 |
| pymupdf | 0.60 | 0.47 | 0.65 | 0.74 |
| pymupdf4llm | 0.65 | 0.86 | 0.93 | 0.78 |

## Speed
Per-document seconds include interpreter start and model loading; the fit separates that fixed cost. The fit is least squares of seconds on pages over full (not sampled) successful runs; with fewer than 3 documents it falls back to mean seconds per page and no fixed cost.
| extractor | fixed s | s/page | docs |
|---|---|---|---|
| docling | 15.7 | 1.981 | 27 |
| easyocr | 43.0 | 3.686 | 27 |
| marker_noocr | 12.0 | 0.279 | 27 |
| mineru | 33.7 | 1.018 | 27 |
| pdfplumber | 0.0 | 0.138 | 27 |
| pymupdf | 0.1 | 0.007 | 27 |
| pymupdf4llm | 28.5 | 0.933 | 27 |

### Seconds per page by page-count bucket (raw, includes fixed cost)
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 12.808 | 3.504 | 3.124 | 1.780 |
| easyocr | 8.592 | 5.812 | 6.402 | 4.073 |
| marker_noocr | 7.447 | 1.305 | 0.624 | 0.339 |
| mineru | 22.140 | 3.960 | 2.705 | 0.947 |
| pdfplumber | 0.162 | 0.076 | 0.096 | 0.091 |
| pymupdf | 0.081 | 0.015 | 0.008 | 0.006 |
| pymupdf4llm | 2.452 | 2.440 | 2.501 | 1.284 |

## Speed by kind of PDF
Fitted seconds per page (fixed cost removed where at least 3 documents allow a fit).
### Seconds per page by tier
| extractor | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| docling | 1.199 | 2.343 | 2.373 | 1.104 |
| easyocr | 4.239 | 3.080 | 4.261 | 4.975 |
| marker_noocr | 0.204 | 0.283 | 0.276 | 0.291 |
| mineru | 0.587 | 1.204 | 0.800 | 0.714 |
| pdfplumber | 0.099 | 0.163 | 0.001 | 0.126 |
| pymupdf | 0.005 | 0.006 | 0.001 | 0.013 |
| pymupdf4llm | 0.304 | 0.808 | 4.091 | 0.437 |

### Seconds per page by source
| extractor | real | scan | generated |
|---|---|---|---|
| docling | 1.960 | 2.373 | 1.860 |
| easyocr | 3.556 | 4.261 | 2.414 |
| marker_noocr | 0.271 | 0.276 | 0.224 |
| mineru | 1.033 | 0.800 | 2.127 |
| pdfplumber | 0.151 | 0.001 | 0.044 |
| pymupdf | 0.008 | 0.001 | 0.000 |
| pymupdf4llm | 0.680 | 4.091 | 0.259 |

## Peak memory (MB)
| extractor | S | M | L | XL |
|---|---|---|---|---|
| docling | 2374 | 2773 | 3590 | 6003 |
| easyocr | 1592 | 1749 | 1845 | 1822 |
| marker_noocr | 1099 | 1096 | 2910 | 3637 |
| mineru | 4042 | 4505 | 6544 | 9577 |
| pdfplumber | 171 | 120 | 125 | 137 |
| pymupdf | 171 | 120 | 125 | 141 |
| pymupdf4llm | 349 | 484 | 706 | 799 |

## Peak memory by kind
### Peak memory (MB) by tier
| extractor | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| docling | 3604 | 3211 | 3340 | 3232 |
| easyocr | 1698 | 1617 | 1855 | 1701 |
| marker_noocr | 2641 | 1720 | 1520 | 1772 |
| mineru | 6297 | 5005 | 6245 | 5298 |
| pdfplumber | 117 | 156 | 145 | 157 |
| pymupdf | 117 | 158 | 145 | 158 |
| pymupdf4llm | 466 | 499 | 700 | 436 |

### Peak memory (MB) by source
| extractor | real | scan | generated |
|---|---|---|---|
| docling | 3867 | 3340 | 2282 |
| easyocr | 1733 | 1855 | 1549 |
| marker_noocr | 2462 | 1520 | 941 |
| mineru | 6141 | 6245 | 3993 |
| pdfplumber | 118 | 145 | 204 |
| pymupdf | 120 | 145 | 204 |
| pymupdf4llm | 574 | 700 | 280 |

## Failures
| extractor | failed runs |
|---|---|
| docling | 0 |
| easyocr | 0 |
| marker_noocr | 0 |
| mineru | 0 |
| pdfplumber | 0 |
| pymupdf | 0 |
| pymupdf4llm | 0 |

### Failed runs by document
A failed run scores 0 on every quality metric above (oom = exceeded the memory cap, timeout = exceeded the time cap, error = the tool raised). Empty cells are successful runs.
No failed runs.

_(The T4 run's own plots were not preserved; see the merged report's plots in `docs/plots/`.)_

## Recommendation (from hit@5, chunking: structure_preserving)
Best extractor per tier: **T1** → pymupdf4llm, **T2** → pymupdf, **T3** → docling, **T4** → easyocr
Best single extractor overall: **docling** (hit@5 0.82).
A per-tier router using those picks would score hit@5 0.85 (+0.04 vs the best single extractor). This is an estimate from tier-level picks, not a measured per-page router.

Failure gallery: not included in the public repository (it embeds page images and exceeds 5 MB); `python -m benchmarks.pdf_extraction.report.build_report` regenerates it from a local run.
