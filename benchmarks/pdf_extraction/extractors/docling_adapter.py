"""Docling: layout model + table-structure model + OCR, exported as Markdown."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
)

_KIND = {"table": "table", "picture": "figure", "chart": "figure", "section_header": "heading", "title": "heading"}


class DoclingExtractor(Extractor):
    """Options: ocr (bool, default true)."""

    name = "docling"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": True, "figures": True, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = bool(self.options.get("ocr", True))
        options.do_table_structure = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        kwargs = {"page_range": page_range} if page_range else {}
        document = converter.convert(str(pdf_path), **kwargs).document

        texts: dict[int, list[str]] = defaultdict(list)
        blocks: dict[int, list[Block]] = defaultdict(list)
        for item, _level in document.iterate_items():
            provenance = getattr(item, "prov", None)
            if not provenance:
                continue
            page_no = provenance[0].page_no
            label = str(getattr(getattr(item, "label", ""), "value", getattr(item, "label", "")))
            kind = _KIND.get(label, "text")
            text = item.export_to_markdown(document) if kind == "table" else (getattr(item, "text", "") or "")
            box = provenance[0].bbox
            height = document.pages[page_no].size.height if page_no in document.pages else None
            bbox = None
            if height is not None:
                top_left = box.to_top_left_origin(height)
                bbox = [float(top_left.l), float(top_left.t), float(top_left.r), float(top_left.b)]
            blocks[page_no].append(Block(kind, text, bbox))
            if text:
                texts[page_no].append(text)

        page_numbers = sorted(set(document.pages) | set(texts))
        pages = [PageResult(page=n, text="\n\n".join(texts[n]), blocks=blocks[n]) for n in page_numbers]
        return ExtractionResult(
            markdown=document.export_to_markdown(), pages=pages, version=package_version("docling")
        )
