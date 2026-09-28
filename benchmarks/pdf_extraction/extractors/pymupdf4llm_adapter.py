"""PyMuPDF4LLM: Markdown from the text layer, with headings and (born-digital) tables."""

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


_KIND = {"table": "table", "picture": "figure", "figure": "figure", "section-header": "heading", "title": "heading"}


class PyMuPDF4LLMExtractor(Extractor):
    name = "pymupdf4llm"
    # pymupdf4llm 1.28 (with pymupdf-layout) OCRs image-only pages and pictures through the system Tesseract when
    # one is installed ("Using Tesseract for OCR processing" in worker.log), so its scan and figure numbers are
    # PyMuPDF's layout analysis on top of Tesseract's text, not a text-layer-only result.
    capabilities = {"page_numbers": True, "bboxes": True, "tables": True, "figures": True, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf
        import pymupdf4llm

        with pymupdf.open(pdf_path) as pdf:
            numbers = resolve_pages(len(pdf), page_range)
            chunks = pymupdf4llm.to_markdown(pdf, pages=[n - 1 for n in numbers], page_chunks=True)

        pages: list[PageResult] = []
        for number, chunk in zip(numbers, chunks):
            text = chunk.get("text", "")
            if chunk.get("page_boxes"):  # layout-aware releases: classified boxes with offsets into the text
                blocks = [
                    Block(_KIND.get(box.get("class", ""), "text"), text[box["pos"][0]:box["pos"][1]], [float(v) for v in box["bbox"]])
                    for box in chunk["page_boxes"]
                ]
            else:  # older releases: a "tables" list with boxes
                blocks = [Block("table", "", [float(v) for v in t["bbox"]]) for t in chunk.get("tables") or [] if "bbox" in t]
                blocks.append(Block("text", text))
            pages.append(PageResult(page=number, text=text, blocks=blocks))
        markdown = "\n\n".join(p.text for p in pages)
        return ExtractionResult(markdown=markdown, pages=pages, version=package_version("pymupdf4llm", "pymupdf"))
