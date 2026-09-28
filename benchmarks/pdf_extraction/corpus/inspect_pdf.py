"""Describe a candidate PDF before adding it to the manifest.

    python -m benchmarks.pdf_extraction.corpus.inspect_pdf path/to/file.pdf [--id my_doc_id --url URL --license LICENSE --tier T2]

Prints page count, bucket, sha256, how much of the document has a text layer, and a manifest entry to paste.
The license field is never guessed: pass the license you verified on the source page, or it stays blank
and the manifest validator will reject the entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from benchmarks.pdf_extraction.corpus.manifest import bucket_for_pages

MIN_TEXT_CHARS = 20  # a page with fewer characters than this counts as having no text layer


def inspect_pdf(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pages = len(pdf)
        text_pages = sum(1 for page in pdf if len(page.get_text().strip()) >= MIN_TEXT_CHARS)
        image_pages = sum(1 for page in pdf if page.get_images())
    ratio = text_pages / pages if pages else 0.0
    if ratio < 0.1:
        hint = "image-only: a T3 (scanned) document"
    elif ratio < 0.9:
        hint = "mixed: some pages are image-only; consider whether it suits T3 or T2/T4"
    else:
        hint = "born-digital: T1, T2 or T4 depending on tables and figures"
    return {
        "pages": pages,
        "bucket": bucket_for_pages(pages),
        "sha256": hashlib.sha256(data).hexdigest(),
        "pages_with_text_layer": text_pages,
        "pages_with_images": image_pages,
        "hint": hint,
    }


def manifest_entry(info: dict[str, Any], doc_id: str, url: str, license_: str, tier: str) -> dict[str, Any]:
    return {"id": doc_id, "tier": tier, "bucket": info["bucket"], "source": "real", "pages": info["pages"],
            "url": url, "license": license_, "sha256": info["sha256"], "notes": ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--id", default="")
    parser.add_argument("--url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--tier", default="T1")
    args = parser.parse_args(argv)
    info = inspect_pdf(args.pdf)
    print(json.dumps(info, indent=2))
    print("\nmanifest entry:")
    print(json.dumps(manifest_entry(info, args.id or args.pdf.stem, args.url, args.license, args.tier), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
