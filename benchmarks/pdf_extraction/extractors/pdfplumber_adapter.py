"""pdfplumber: text layer plus rule/whitespace-based table detection (a table-focused baseline)."""

from __future__ import annotations

from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
    resolve_pages,
    rows_to_markdown,
)


def _outside(obj: dict, boxes: list[tuple[float, float, float, float]]) -> bool:
    """Keep non-character objects and characters whose centre is not inside any table box."""
    if obj.get("object_type") != "char":
        return True
    x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)


class PDFPlumberExtractor(Extractor):
    name = "pdfplumber"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": True, "figures": False, "ocr": False}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pdfplumber

        pages: list[PageResult] = []
        with pdfplumber.open(pdf_path) as pdf:
            for number in resolve_pages(len(pdf.pages), page_range):
                page = pdf.pages[number - 1]
                tables = page.find_tables()
                boxes = [tuple(t.bbox) for t in tables]
                outside = page.filter(lambda obj, b=boxes: _outside(obj, b)).extract_text() or ""

                parts = [outside] if outside.strip() else []
                blocks = [Block("text", outside)] if outside.strip() else []
                for table in tables:
                    markdown = rows_to_markdown(table.extract())
                    if markdown:
                        parts.append(markdown)
                        blocks.append(Block("table", markdown, [float(v) for v in table.bbox]))
                pages.append(PageResult(page=number, text="\n\n".join(parts), blocks=blocks))
                page.flush_cache()  # release the page's object cache: a 487-page document otherwise peaks above 3 GB
        markdown = "\n\n".join(p.text for p in pages)
        return ExtractionResult(markdown=markdown, pages=pages, version=package_version("pdfplumber"))
