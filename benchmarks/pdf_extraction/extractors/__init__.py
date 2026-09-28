"""Extractor registry. Adapters are imported lazily so a missing tool only breaks its own adapter."""

from __future__ import annotations

import importlib

from benchmarks.pdf_extraction.extractors.base import Extractor

_PKG = "benchmarks.pdf_extraction.extractors"

#: name -> "module:ClassName". ``fake`` is test-only and never appears in config.yaml.
_REGISTRY: dict[str, str] = {
    "fake": f"{_PKG}.fake_adapter:FakeExtractor",
    "pymupdf": f"{_PKG}.pymupdf_adapter:PyMuPDFExtractor",
    "pymupdf4llm": f"{_PKG}.pymupdf4llm_adapter:PyMuPDF4LLMExtractor",
    "pdfplumber": f"{_PKG}.pdfplumber_adapter:PDFPlumberExtractor",
    "docling": f"{_PKG}.docling_adapter:DoclingExtractor",
    "marker": f"{_PKG}.marker_adapter:MarkerExtractor",
    "mineru": f"{_PKG}.mineru_adapter:MinerUExtractor",
    "tesseract": f"{_PKG}.tesseract_adapter:TesseractExtractor",
    "easyocr": f"{_PKG}.easyocr_adapter:EasyOCRExtractor",
    "paddleocr_vl": f"{_PKG}.paddleocr_vl_adapter:PaddleOCRVLExtractor",
}


def known_extractors() -> list[str]:
    return [n for n in _REGISTRY if n != "fake"]


def load_extractor(name: str, options: dict | None = None) -> Extractor:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown extractor {name!r}. Known: {', '.join(sorted(_REGISTRY))}")
    module_name, class_name = _REGISTRY[name].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(options)
