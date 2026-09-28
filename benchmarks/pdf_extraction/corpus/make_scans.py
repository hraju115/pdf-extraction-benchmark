"""Turn born-digital PDFs into image-only PDFs (no text layer) with exact text ground truth."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image


def _degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    """Skew slightly and add noise, imitating a poor scan."""
    angle = rng.uniform(-1.5, 1.5)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    pixels = np.asarray(rotated).astype(np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 12, pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def rasterize_pdf(
    src: Path,
    dst: Path,
    dpi: int = 200,
    degrade: bool = False,
    seed: int = 0,
) -> dict[int, str]:
    """Write an image-only copy of ``src`` to ``dst``; return {page number: exact source text}.

    The output has one full-page JPEG per page and no text layer, so extractors must OCR it.
    """
    rng = random.Random(seed)
    truth: dict[int, str] = {}
    out = pymupdf.open()
    with pymupdf.open(src) as source:
        for number, page in enumerate(source, start=1):
            truth[number] = page.get_text("text")
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            if degrade:
                image = _degrade(image, rng)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=40 if degrade else 85)
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buffer.getvalue())
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst)
    out.close()
    return truth


def has_text_layer(pdf_path: Path) -> bool:
    with pymupdf.open(pdf_path) as pdf:
        return any(page.get_text().strip() for page in pdf)


def write_page_truth(path: Path, truth: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in truth.items()}, indent=1), encoding="utf-8")


def read_page_truth(path: Path) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def native_page_truth(pdf_path: Path) -> dict[int, str]:
    """Approximate text truth from a born-digital PDF's own text layer."""
    with pymupdf.open(pdf_path) as pdf:
        return {number: page.get_text("text") for number, page in enumerate(pdf, start=1)}
