"""Marker: surya-based layout + OCR + table recognition, exported as paginated Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
    resolve_pages,
)

# With paginate_output, Marker inserts "{<page index>}" followed by a run of dashes before each page.
_PAGE_MARK = re.compile(r"\{(\d+)\}-{20,}\n*")


class MarkerExtractor(Extractor):
    name = "marker"
    capabilities = {"page_numbers": True, "bboxes": False, "tables": True, "figures": False, "ocr": True}

    def build_config(self, numbers: list[int], page_range: tuple[int, int] | None) -> dict:
        """Marker's own config keys. Every adapter option (e.g. ``disable_ocr``) is passed through unchanged."""
        config: dict = {"output_format": "markdown", "paginate_output": True}
        if page_range:
            config["page_range"] = [n - 1 for n in numbers]  # Marker 2.x wants a list of 0-based page ids
        config.update(self.options)
        return config

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        with pymupdf.open(pdf_path) as pdf:
            numbers = resolve_pages(len(pdf), page_range)
        converter = PdfConverter(artifact_dict=create_model_dict(), config=self.build_config(numbers, page_range))
        markdown, _, _ = text_from_rendered(converter(str(pdf_path)))

        marks = list(_PAGE_MARK.finditer(markdown))
        if marks:
            pages = []
            for i, mark in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(markdown)
                pages.append(PageResult(page=int(mark.group(1)) + 1, text=markdown[mark.end():end].strip()))
        else:  # pagination markers not found: keep the text, attribute it to the first processed page
            pages = [PageResult(page=numbers[0] if numbers else 1, text=markdown)]
        return ExtractionResult(markdown=_PAGE_MARK.sub("", markdown), pages=pages, version=package_version("marker-pdf"))
