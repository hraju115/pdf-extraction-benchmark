"""Baseline: PyMuPDF's raw text layer. No OCR, no table or figure understanding."""

from __future__ import annotations

from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
    resolve_pages,
)


class PyMuPDFExtractor(Extractor):
    name = "pymupdf"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": False, "figures": False, "ocr": False}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf

        pages: list[PageResult] = []
        with pymupdf.open(pdf_path) as pdf:
            for number in resolve_pages(len(pdf), page_range):
                page = pdf[number - 1]
                blocks = [
                    Block("text", b[4].strip(), [b[0], b[1], b[2], b[3]])
                    for b in page.get_text("blocks")
                    if b[6] == 0 and b[4].strip()
                ]
                pages.append(PageResult(page=number, text=page.get_text("text"), blocks=blocks))
        markdown = "\n\n".join(p.text for p in pages)
        return ExtractionResult(markdown=markdown, pages=pages, version=package_version("pymupdf"))
