"""Test-only extractor used to exercise the runner. Not listed in config.yaml."""

from __future__ import annotations

import os
import time
from pathlib import Path

from benchmarks.pdf_extraction.extractors.base import (
    Block,
    ExtractionResult,
    Extractor,
    PageResult,
    resolve_pages,
)


class FakeExtractor(Extractor):
    """Options: mode (ok|crash|sleep|memory|chatty), seconds, megabytes, kilobytes, marker_file, report_env (list of env var names)."""

    name = "fake"

    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        mode = self.options.get("mode", "ok")
        marker = self.options.get("marker_file")
        if marker:
            with open(marker, "a", encoding="utf-8") as handle:
                handle.write("run\n")

        if mode == "chatty":  # imitate a tool that streams progress bars: more than a pipe buffer holds
            import sys

            line = "progress " * 12 + "\n"
            for _ in range(int(float(self.options.get("kilobytes", 256)) * 1024 // len(line)) + 1):
                sys.stderr.write(line)
            sys.stderr.flush()
        if mode == "crash":
            raise RuntimeError("fake extractor crashed on purpose")
        if mode == "sleep":
            time.sleep(float(self.options.get("seconds", 30)))
        if mode == "memory":
            hog = bytearray(int(float(self.options.get("megabytes", 300)) * 1024 * 1024))
            time.sleep(float(self.options.get("seconds", 30)))
            del hog

        import pymupdf  # available in the benchmark core environment

        with pymupdf.open(pdf_path) as pdf:
            pages = resolve_pages(len(pdf), page_range)
        env_report = ", ".join(
            f"{var}={'present' if os.environ.get(var) else 'absent'}" for var in self.options.get("report_env", [])
        )
        results = [PageResult(page=n, text=f"fake page {n}", blocks=[Block("text", f"fake page {n}")]) for n in pages]
        markdown = "\n\n".join(p.text for p in results) + (f"\n\n{env_report}" if env_report else "")
        return ExtractionResult(markdown=markdown, pages=results, version="fake-1")
