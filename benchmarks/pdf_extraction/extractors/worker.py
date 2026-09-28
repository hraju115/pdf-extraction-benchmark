"""Subprocess entry point: run ONE extractor on ONE document and write the uniform outputs.

    python -m benchmarks.pdf_extraction.extractors.worker <extractor> <pdf> <out_dir> --doc-id ID
        [--pages FIRST-LAST] [--options JSON] [--input-hash HASH] [--run-name NAME]

Runs inside the extractor's own virtualenv. Always writes run.json, even on failure.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import traceback
from pathlib import Path

from benchmarks.pdf_extraction.extractors import load_extractor
from benchmarks.pdf_extraction.extractors.base import RunInfo, hardware_info, write_outputs, write_run


def _peak_rss_mb() -> float:
    """Peak resident memory of this process and its finished children, in MB (Linux reports KB)."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(own, children) / 1024


def _parse_pages(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    first, last = value.split("-")
    return int(first), int(last)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extractor")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--pages")
    parser.add_argument("--options", default="{}")
    parser.add_argument("--input-hash", default="")
    parser.add_argument("--run-name", default=None, help="config entry name recorded in run.json (a variant of <extractor>)")
    args = parser.parse_args(argv)

    page_range = _parse_pages(args.pages)
    run = RunInfo(
        extractor=args.run_name or args.extractor,
        doc_id=args.doc_id,
        page_range=list(page_range) if page_range else None,
        sampled=page_range is not None,
        input_hash=args.input_hash,
        hardware=hardware_info(),
    )

    try:
        extractor = load_extractor(args.extractor, json.loads(args.options))
        started = time.perf_counter()
        result = extractor.extract(args.pdf, page_range)
        run.seconds = round(time.perf_counter() - started, 3)
        run.version = result.version
        run.pages_processed = len(result.pages)
        run.peak_rss_mb = round(_peak_rss_mb(), 1)
        write_outputs(args.out_dir, result, run)
        return 0
    except MemoryError:
        run.status, run.error = "oom", "MemoryError"
    except Exception as exc:  # noqa: BLE001 - failures are data, not crashes
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    run.peak_rss_mb = round(_peak_rss_mb(), 1)
    write_run(args.out_dir, run)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
