"""Benchmark configuration loaded from ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmarks.pdf_extraction.extractors import known_extractors


@dataclass
class BenchConfig:
    max_rss_gb: float = 5.0
    timeout_min: float = 30.0
    default_max_pages: int | None = None
    per_extractor_max_pages: dict[str, int] = field(default_factory=dict)
    extractors: dict[str, dict[str, Any]] = field(default_factory=dict)

    def enabled_extractors(self) -> list[str]:
        return [name for name, opts in self.extractors.items() if opts.get("enabled", False)]

    def adapter_for(self, name: str) -> str:
        """The registry adapter a config entry runs; a plain entry is its own adapter."""
        return self.extractors.get(name, {}).get("adapter", name)

    def options_for(self, name: str) -> dict[str, Any]:
        """Adapter options: the config entry without the ``enabled`` switch and the ``adapter`` pointer."""
        return {k: v for k, v in self.extractors.get(name, {}).items() if k not in ("enabled", "adapter")}

    def max_pages_for(self, name: str, cli_override: int | None = None) -> int | None:
        if cli_override is not None:
            return cli_override
        return self.per_extractor_max_pages.get(name, self.default_max_pages)


def load_config(path: Path) -> BenchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    limits = raw.get("limits", {})
    sampling = raw.get("sampling", {})
    extractors = raw.get("extractors", {}) or {}

    known = set(known_extractors())
    adapters = {name: (opts or {}).get("adapter", name) for name, opts in extractors.items()}
    # Name both the entry and the adapter it points at, so a typo in ``adapter:`` is findable.
    unknown = sorted(name if adapter == name else f"{name} -> {adapter}" for name, adapter in adapters.items() if adapter not in known)
    if unknown:
        raise ValueError(f"config.yaml lists unknown extractors/adapters: {unknown}")

    return BenchConfig(
        max_rss_gb=float(limits.get("max_rss_gb", 5)),
        timeout_min=float(limits.get("timeout_min", 30)),
        default_max_pages=sampling.get("default_max_pages"),
        per_extractor_max_pages={k: int(v) for k, v in (sampling.get("per_extractor") or {}).items()},
        extractors=extractors,
    )
