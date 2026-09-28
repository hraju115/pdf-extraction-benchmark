"""One virtualenv per extractor, so conflicting torch/transformers pins never share an environment."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from benchmarks.pdf_extraction.paths import ENVS_DIR, REQUIREMENTS_DIR

Runner = Callable[..., subprocess.CompletedProcess]


def env_dir(name: str, envs_dir: Path = ENVS_DIR) -> Path:
    return envs_dir / name


def env_python(name: str, envs_dir: Path = ENVS_DIR) -> Path:
    return env_dir(name, envs_dir) / "bin" / "python"


def ensure_env(
    name: str,
    requirements_dir: Path = REQUIREMENTS_DIR,
    envs_dir: Path = ENVS_DIR,
    run: Runner = subprocess.run,
) -> Path:
    """Create ``.envs/<name>`` and install its requirements once; return its python.

    Installs from ``<name>.lock.txt`` when it exists (exact versions from an earlier run), otherwise from
    ``<name>.txt``, then writes ``<name>.lock.txt`` with ``pip freeze`` so later runs are reproducible.
    """
    directory = env_dir(name, envs_dir)
    python = env_python(name, envs_dir)
    ready = directory / ".ready"
    if ready.exists():
        return python

    requirements = requirements_dir / f"{name}.txt"
    lock = requirements_dir / f"{name}.lock.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"No requirements file for extractor {name!r}: {requirements}")

    try:
        run([sys.executable, "-m", "venv", str(directory)], check=True)
    except subprocess.CalledProcessError:
        # Some images (Kaggle's) ship a Python whose ensurepip is broken, so ``venv`` dies half-built.
        # virtualenv bundles its own pip and does not need ensurepip.
        shutil.rmtree(directory, ignore_errors=True)
        run([sys.executable, "-m", "virtualenv", "--quiet", str(directory)], check=True)
    run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    source = lock if lock.exists() else requirements
    run([str(python), "-m", "pip", "install", "--quiet", "-r", str(source)], check=True)

    if not lock.exists():
        frozen = run([str(python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        lock.write_text(frozen.stdout, encoding="utf-8")

    directory.mkdir(parents=True, exist_ok=True)
    ready.write_text("ok\n", encoding="utf-8")
    return python
