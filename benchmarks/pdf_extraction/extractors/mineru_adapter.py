"""MinerU (pipeline backend, CPU): layout + OCR + table/formula recognition via its CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    package_version,
)


def run_cli(command: list[str]) -> None:
    """Run an external tool; on failure raise with the tool's own output so run.json explains itself."""
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        tail = ((done.stderr or "") + "\n" + (done.stdout or "")).strip()[-1500:]
        raise RuntimeError(f"{Path(command[0]).name} failed with exit status {done.returncode}:\n{tail}")


def _cli() -> str:
    """The ``mineru`` script installed beside the running interpreter (the virtualenv is not activated)."""
    candidate = Path(sys.executable).parent / "mineru"
    return str(candidate) if candidate.exists() else "mineru"


class MinerUExtractor(Extractor):
    """Options: backend (default 'pipeline'). Page flags -s/-e are 0-based, inclusive."""

    name = "mineru"
    capabilities = {"page_numbers": True, "bboxes": True, "tables": True, "figures": False, "ocr": True}

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            command = [_cli(), "-p", str(pdf_path), "-o", tmp, "-b", self.options.get("backend", "pipeline")]
            if page_range:
                command += ["-s", str(page_range[0] - 1), "-e", str(page_range[1] - 1)]
            run_cli(command)

            offset = page_range[0] - 1 if page_range else 0
            markdown_files = sorted(Path(tmp).rglob("*.md"))
            if not markdown_files:
                raise RuntimeError("mineru produced no Markdown file")
            markdown = markdown_files[0].read_text(encoding="utf-8")

            sizes: dict[int, tuple[float, float]] = {}
            with pymupdf.open(pdf_path) as pdf:
                for index, page in enumerate(pdf):
                    sizes[index] = (page.rect.width, page.rect.height)

            texts: dict[int, list[str]] = defaultdict(list)
            blocks: dict[int, list[Block]] = defaultdict(list)
            for content_file in sorted(Path(tmp).rglob("*_content_list.json")):
                for item in json.loads(content_file.read_text(encoding="utf-8")):
                    # MinerU numbers pages relative to the requested range; make them absolute (0-based)
                    index = offset + item.get("page_idx", 0)
                    kind = {"table": "table", "image": "figure"}.get(item.get("type"), "heading" if item.get("text_level") else "text")
                    text = item.get("table_body") or item.get("text") or ""
                    bbox = None
                    if item.get("bbox") and index in sizes:  # MinerU boxes are normalized to 0-1000
                        width, height = sizes[index]
                        x0, y0, x1, y1 = item["bbox"]
                        bbox = [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height]
                    blocks[index + 1].append(Block(kind, text, bbox))
                    if text:
                        texts[index + 1].append(text)
        pages = [PageResult(page=n, text="\n\n".join(texts[n]), blocks=blocks[n]) for n in sorted(set(texts) | set(blocks))]
        return ExtractionResult(markdown=markdown, pages=pages, version=package_version("mineru"))
