"""Tesseract OCR on rasterized pages (needs the system binary: sudo apt install tesseract-ocr)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
    resolve_pages,
)


def _run(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def version_string() -> str:
    """First line of ``tesseract --version``; never lets a version probe break an extraction."""
    try:
        return _run(["tesseract", "--version"]).splitlines()[0]
    except Exception:  # noqa: BLE001 - version is metadata only
        return "tesseract (version unknown)"


def parse_tsv_lines(tsv: str, dpi: int) -> list[Block]:
    """Group Tesseract TSV words into line blocks with boxes converted from pixels to PDF points."""
    scale = 72 / dpi
    lines: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        lines[(cols[2], cols[3], cols[4])].append(
            {"x0": int(cols[6]), "y0": int(cols[7]), "x1": int(cols[6]) + int(cols[8]), "y1": int(cols[7]) + int(cols[9]), "text": cols[11]}
        )
    blocks = []
    for words in lines.values():
        bbox = [min(w["x0"] for w in words) * scale, min(w["y0"] for w in words) * scale,
                max(w["x1"] for w in words) * scale, max(w["y1"] for w in words) * scale]
        blocks.append(Block("text", " ".join(w["text"] for w in words), bbox))
    return blocks


class TesseractExtractor(Extractor):
    """Options: dpi (default 300), lang (default 'eng')."""

    name = "tesseract"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": False, "figures": False, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf

        if shutil.which("tesseract") is None:
            raise RuntimeError("tesseract binary not found; install it with: sudo apt install tesseract-ocr")
        dpi, lang = int(self.options.get("dpi", 300)), self.options.get("lang", "eng")
        version = package_version("pymupdf") + "; " + version_string()

        pages: list[PageResult] = []
        with pymupdf.open(pdf_path) as pdf, tempfile.TemporaryDirectory() as tmp:
            for number in resolve_pages(len(pdf), page_range):
                image = Path(tmp) / f"page{number}.png"
                pdf[number - 1].get_pixmap(dpi=dpi, alpha=False).save(image)
                base = Path(tmp) / f"page{number}"
                # ONE recognition pass producing both outputs (base.txt and base.tsv); two passes doubled the timing.
                _run(["tesseract", str(image), str(base), "-l", lang, "--psm", "3", "txt", "tsv"])
                text = base.with_suffix(".txt").read_text(encoding="utf-8", errors="replace") if base.with_suffix(".txt").exists() else ""
                tsv = base.with_suffix(".tsv").read_text(encoding="utf-8", errors="replace") if base.with_suffix(".tsv").exists() else ""
                pages.append(PageResult(page=number, text=text.strip(), blocks=parse_tsv_lines(tsv, dpi)))
        return ExtractionResult(markdown="\n\n".join(p.text for p in pages), pages=pages, version=version)
