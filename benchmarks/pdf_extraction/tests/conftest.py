"""Make DocuMind's ``src`` importable for the downstream-stage tests (they reuse its chunker)."""

import sys

from benchmarks.pdf_extraction.paths import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "src"))
