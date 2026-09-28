"""EasyOCR on rasterized pages; detections are re-assembled into reading-order lines.

Runs on the GPU when CUDA is available (or when the ``gpu`` option forces it), on the CPU otherwise.
"""

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


def group_into_lines(detections: list[tuple[list[list[float]], str]]) -> list[tuple[list[float], str]]:
    """Cluster OCR boxes into lines by vertical centre, then order each line left to right.

    ``detections`` are (four-corner box in pixels, text). Returns ([x0, y0, x1, y1], line text).
    """
    boxes = []
    for corners, text in detections:
        xs, ys = [p[0] for p in corners], [p[1] for p in corners]
        boxes.append((min(xs), min(ys), max(xs), max(ys), text))
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    tolerance = 0.6 * heights[len(heights) // 2]

    lines: list[list[tuple[float, float, float, float, str]]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        centre = (box[1] + box[3]) / 2
        if lines and abs(centre - sum((b[1] + b[3]) / 2 for b in lines[-1]) / len(lines[-1])) <= tolerance:
            lines[-1].append(box)
        else:
            lines.append([box])

    result = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        result.append(([min(b[0] for b in line), min(b[1] for b in line), max(b[2] for b in line), max(b[3] for b in line)],
                       " ".join(b[4] for b in line)))
    return result


def use_gpu(options: dict, cuda_available: bool) -> bool:
    """An explicit ``gpu`` option wins; otherwise use the GPU exactly when CUDA is available."""
    if "gpu" in options:
        return bool(options["gpu"])
    return cuda_available


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # torch missing or broken: EasyOCR itself would fall back to the CPU
        return False


class EasyOCRExtractor(Extractor):
    """Options: dpi (default 200), languages (default ['en']), gpu (default: auto-detect CUDA)."""

    name = "easyocr"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": False, "figures": False, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import easyocr
        import numpy as np
        import pymupdf

        dpi = int(self.options.get("dpi", 200))
        reader = easyocr.Reader(self.options.get("languages", ["en"]), gpu=use_gpu(self.options, _cuda_available()),
                                verbose=False)
        scale = 72 / dpi

        pages: list[PageResult] = []
        with pymupdf.open(pdf_path) as pdf:
            for number in resolve_pages(len(pdf), page_range):
                pixmap = pdf[number - 1].get_pixmap(dpi=dpi, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
                detections = [(box, text) for box, text, _conf in reader.readtext(image)]
                lines = group_into_lines(detections)
                blocks = [Block("text", text, [v * scale for v in bbox]) for bbox, text in lines]
                pages.append(PageResult(page=number, text="\n".join(t for _, t in lines), blocks=blocks))
        return ExtractionResult(markdown="\n\n".join(p.text for p in pages), pages=pages, version=package_version("easyocr"))
