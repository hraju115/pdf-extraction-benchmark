# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Text cleaning and metadata extraction from raw documents."""

from __future__ import annotations

import re

from documind.logging import get_logger
from documind.models import Document

logger = get_logger(__name__)

_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_MULTIPLE_SPACES = re.compile(r"[ \t]{2,}")
_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)[^)]*\)")  # [text](url "title") -> text (url)
# A fenced code block: an opening ``` or ~~~ line through the matching closing fence (or the end).
_FENCED = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*\1[`~]*[ \t]*$|\Z)", re.M | re.S)


def _clean_prose(text: str) -> str:
    text = _LINK.sub(r"\1 (\2)", text)
    text = _MULTIPLE_SPACES.sub(" ", text)
    return _MULTIPLE_NEWLINES.sub("\n\n", text)


def clean_text(text: str) -> str:
    """Normalize whitespace outside fenced code; keep code verbatim and links as ``text (url)``."""
    parts: list[str] = []
    last = 0
    for fence in _FENCED.finditer(text):
        parts.append(_clean_prose(text[last : fence.start()]))
        parts.append(fence.group(0))
        last = fence.end()
    parts.append(_clean_prose(text[last:]))
    return "".join(parts).strip()


def extract_sections(content: str) -> list[tuple[str, str]]:
    """Split a Markdown-like document into (heading, body) pairs.

    Returns a list of tuples. If the document starts before any heading,
    the first tuple has heading="".
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Flush previous section
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = stripped.lstrip("#").strip()
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


def parse_document(doc: Document) -> Document:
    """Clean the document content in place and enrich metadata."""
    logger.info("parsing_document", doc_id=doc.id, title=doc.metadata.title)
    doc.content = clean_text(doc.content)
    return doc
