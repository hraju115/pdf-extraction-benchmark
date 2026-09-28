"""Uniform extractor interface and output schema.

Stdlib only: adapters import this inside their own virtualenvs, which do not
contain the benchmark's core dependencies.
"""

from __future__ import annotations

import abc
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

BLOCK_KINDS = ("text", "heading", "table", "figure")
RUN_STATUSES = ("ok", "error", "timeout", "oom")
RUN_REQUIRED_KEYS = ("extractor", "doc_id", "status", "seconds", "peak_rss_mb", "pages_processed")


@dataclass
class Block:
    kind: str  # one of BLOCK_KINDS
    text: str
    bbox: list[float] | None = None  # [x0, y0, x1, y1] in PDF points, origin top-left


@dataclass
class PageResult:
    page: int  # 1-based page number in the *source* PDF
    text: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ExtractionResult:
    markdown: str
    pages: list[PageResult]
    version: str = ""


@dataclass
class RunInfo:
    extractor: str
    doc_id: str
    status: str = "ok"
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    pages_processed: int = 0
    page_range: list[int] | None = None  # [first, last], 1-based inclusive, when sampled
    sampled: bool = False
    version: str = ""
    input_hash: str = ""
    hardware: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def nvidia_gpu_name() -> str | None:
    """'<name>, <memory>' of the first NVIDIA GPU via nvidia-smi, or None when there is none (never raises)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else ""
    return first or None


def hardware_info(gpu_probe: Callable[[], str | None] = nvidia_gpu_name) -> dict[str, Any]:
    try:
        total_ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        total_ram_gb = 0.0
    try:
        gpu = gpu_probe()
    except Exception:
        gpu = None
    return {"cpus": os.cpu_count(), "total_ram_gb": total_ram_gb, "platform": platform.platform(),
            "python": platform.python_version(), "gpu": gpu}


class Extractor(abc.ABC):
    """One extraction strategy. Subclasses import their heavy libraries lazily inside ``extract``."""

    name: str = ""
    #: What the strategy can provide; surfaced in the report's capability matrix.
    capabilities: dict[str, bool] = {
        "page_numbers": True,
        "bboxes": False,
        "tables": False,
        "figures": False,
        "ocr": False,
    }

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}

    @abc.abstractmethod
    def extract(self, pdf_path: Path, page_range: tuple[int, int] | None) -> ExtractionResult:
        """Extract ``pdf_path``. ``page_range`` is (first, last), 1-based inclusive; None means all pages."""


def resolve_pages(total: int, page_range: tuple[int, int] | None) -> list[int]:
    """Return the 1-based page numbers to process."""
    if page_range is None:
        return list(range(1, total + 1))
    first, last = page_range
    first = max(1, first)
    last = min(total, last)
    return list(range(first, last + 1))


def write_outputs(out_dir: Path, result: ExtractionResult, run: RunInfo) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.md").write_text(result.markdown, encoding="utf-8")
    (out_dir / "pages.json").write_text(
        json.dumps([asdict(p) for p in result.pages], indent=1), encoding="utf-8"
    )
    write_run(out_dir, run)


def write_run(out_dir: Path, run: RunInfo) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(asdict(run), indent=1), encoding="utf-8")


def read_run(out_dir: Path) -> RunInfo | None:
    path = out_dir / "run.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Ignore keys this version no longer records (e.g. cost_usd in run.json files written by older versions).
        known = {f.name for f in fields(RunInfo)}
        return RunInfo(**{k: v for k, v in raw.items() if k in known})
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def validate_outputs(out_dir: Path) -> list[str]:
    """Return a list of schema problems; empty means the outputs are valid."""
    problems: list[str] = []

    run_path = out_dir / "run.json"
    if not run_path.exists():
        return ["run.json missing"]
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"run.json is not valid JSON: {exc}"]
    for key in RUN_REQUIRED_KEYS:
        if key not in run:
            problems.append(f"run.json missing key '{key}'")
    if run.get("status") not in RUN_STATUSES:
        problems.append(f"run.json has invalid status {run.get('status')!r}")

    if run.get("status") != "ok":
        return problems  # failed runs legitimately have no result files

    if not (out_dir / "result.md").exists():
        problems.append("result.md missing")

    pages_path = out_dir / "pages.json"
    if not pages_path.exists():
        problems.append("pages.json missing")
        return problems
    try:
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return problems + [f"pages.json is not valid JSON: {exc}"]
    if not isinstance(pages, list):
        return problems + ["pages.json must be a list"]
    for i, page in enumerate(pages):
        if not isinstance(page.get("page"), int) or page["page"] < 1:
            problems.append(f"pages[{i}].page must be an int >= 1")
        if not isinstance(page.get("text"), str):
            problems.append(f"pages[{i}].text must be a string")
        for j, block in enumerate(page.get("blocks", [])):
            if block.get("kind") not in BLOCK_KINDS:
                problems.append(f"pages[{i}].blocks[{j}].kind invalid: {block.get('kind')!r}")
            bbox = block.get("bbox")
            if bbox is not None and len(bbox) != 4:
                problems.append(f"pages[{i}].blocks[{j}].bbox must have 4 numbers")
    return problems


def package_version(*names: str) -> str:
    """'pkg==1.2.3, other==4.5' for the installed distributions among ``names``."""
    from importlib.metadata import PackageNotFoundError, version

    parts = []
    for name in names:
        try:
            parts.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            continue
    return ", ".join(parts)


def rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render table rows (first row = header) as a Markdown pipe table."""
    cleaned = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in row] for row in rows if row]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    lines = ["| " + " | ".join(cleaned[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(lines)
