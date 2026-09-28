"""PaddleOCR-VL (0.9B document-parsing VLM). Slow on CPU: config.yaml samples the first pages only."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
    resolve_pages,
)


class PaddleOCRVLExtractor(Extractor):
    """Options: dpi (default 200)."""

    name = "paddleocr_vl"
    capabilities = {"page_numbers": True, "bboxes": False, "tables": True, "figures": False, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf
        from paddleocr import PaddleOCRVL

        dpi = int(self.options.get("dpi", 200))
        pipeline = PaddleOCRVL()
        pages: list[PageResult] = []
        with pymupdf.open(pdf_path) as pdf, tempfile.TemporaryDirectory() as tmp:
            for number in resolve_pages(len(pdf), page_range):
                image = Path(tmp) / f"page{number}.png"
                pdf[number - 1].get_pixmap(dpi=dpi, alpha=False).save(image)
                out_dir = Path(tmp) / f"out{number}"
                for result in pipeline.predict(str(image)):
                    result.save_to_markdown(save_path=str(out_dir))
                markdown_files = sorted(out_dir.rglob("*.md"))
                text = "\n\n".join(f.read_text(encoding="utf-8") for f in markdown_files)
                pages.append(PageResult(page=number, text=text.strip()))
        return ExtractionResult(
            markdown="\n\n".join(p.text for p in pages), pages=pages, version=package_version("paddleocr", "paddlepaddle")
        )
