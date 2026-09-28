"""Corpus manifest: one entry per PDF, validated before anything is downloaded or built."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

TIERS = ("T1", "T2", "T3", "T4")
SOURCES = ("real", "scan", "generated")

#: (name, min pages, max pages or None)
BUCKETS = (("S", 1, 5), ("M", 6, 30), ("L", 31, 100), ("XL", 101, None))

#: Licenses under which a real document may enter the corpus.
ALLOWED_LICENSES = frozenset(
    {"public-domain", "us-gov-work", "cc0", "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "ietf-trust"}
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def bucket_for_pages(pages: int) -> str:
    for name, low, high in BUCKETS:
        if pages >= low and (high is None or pages <= high):
            return name
    raise ValueError(f"page count must be >= 1, got {pages}")


@dataclass
class ManifestEntry:
    id: str
    tier: str
    bucket: str
    source: str  # real | scan | generated
    pages: int = 0  # required for real; filled in when a scan/generated document is built
    url: str = ""
    license: str = ""
    sha256: str = ""
    derived_from: str = ""  # scans: the id of the born-digital source document
    dpi: int = 200  # scans
    degrade: bool = False  # scans: add skew, noise and JPEG artifacts
    truth_from_native: bool = False  # real: use the PDF's own text layer as approximate text truth
    notes: str = ""


def validate_entry(entry: ManifestEntry) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", entry.id):
        problems.append(f"{entry.id!r}: id must be lowercase letters, digits and underscores")
    if entry.tier not in TIERS:
        problems.append(f"{entry.id}: tier must be one of {TIERS}")
    if entry.bucket not in {b[0] for b in BUCKETS}:
        problems.append(f"{entry.id}: bucket must be one of S, M, L, XL")
    if entry.source not in SOURCES:
        problems.append(f"{entry.id}: source must be one of {SOURCES}")
        return problems

    if entry.source == "real":
        if not entry.url:
            problems.append(f"{entry.id}: real documents need a url")
        if entry.license not in ALLOWED_LICENSES:
            problems.append(
                f"{entry.id}: license {entry.license!r} is not a verified license; "
                f"allowed: {sorted(ALLOWED_LICENSES)}"
            )
        if not _SHA256.match(entry.sha256):
            problems.append(f"{entry.id}: real documents need a 64-hex sha256")
        if entry.pages < 1:
            problems.append(f"{entry.id}: real documents need pages >= 1")
        elif entry.bucket in {b[0] for b in BUCKETS} and bucket_for_pages(entry.pages) != entry.bucket:
            problems.append(
                f"{entry.id}: bucket {entry.bucket} does not match {entry.pages} pages "
                f"({bucket_for_pages(entry.pages)})"
            )
    elif entry.source == "scan":
        if not entry.derived_from:
            problems.append(f"{entry.id}: scans need derived_from")
        if entry.tier != "T3":
            problems.append(f"{entry.id}: scans belong to tier T3")
        if entry.dpi < 72 or entry.dpi > 600:
            problems.append(f"{entry.id}: dpi must be between 72 and 600")
    return problems


def check_built_pages(entry: ManifestEntry, actual_pages: int) -> list[str]:
    """After a scan/generated document is built, its real page count must fit the declared bucket."""
    if bucket_for_pages(actual_pages) != entry.bucket:
        return [f"{entry.id}: built {actual_pages} pages, which is bucket {bucket_for_pages(actual_pages)}, not {entry.bucket}"]
    return []


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Load and validate ``manifest.yaml``. Raises ValueError listing *every* problem found."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("documents", [])
    entries: list[ManifestEntry] = []
    problems: list[str] = []
    seen: set[str] = set()

    for item in items:
        try:
            entry = ManifestEntry(**item)
        except TypeError as exc:
            problems.append(f"bad manifest item {item!r}: {exc}")
            continue
        if entry.id in seen:
            problems.append(f"{entry.id}: duplicate id")
        seen.add(entry.id)
        problems.extend(validate_entry(entry))
        entries.append(entry)

    ids = {e.id for e in entries}
    for entry in entries:
        # A scan may derive from a manifest document or from a generated one (ids start with "gen_").
        if entry.source == "scan" and entry.derived_from:
            if entry.derived_from not in ids and not entry.derived_from.startswith("gen_"):
                problems.append(f"{entry.id}: derived_from {entry.derived_from!r} is not in the manifest")

    if problems:
        raise ValueError("Invalid manifest:\n  " + "\n  ".join(problems))
    return entries
