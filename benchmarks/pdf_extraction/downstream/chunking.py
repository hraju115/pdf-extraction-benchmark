"""Two chunking modes for extracted Markdown.

``baseline``            DocuMind's existing ``parse_document`` + ``chunk_document`` (the production behaviour).
``structure_preserving`` never splits a table or a paragraph mid-way; a table too large for one chunk is split
                        by rows with its header repeated in every piece.
"""

from __future__ import annotations

import re
from typing import Callable

import tiktoken

CHUNK_TOKENS = 450  # below bge-large's 512-token input limit; the production default (600) exceeds it
OVERLAP_TOKENS = 60

_ENCODER = tiktoken.get_encoding("cl100k_base")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_ROW = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    """Default budget counter (tiktoken cl100k). Pass the embedder's own tokenizer as ``count`` to measure the
    budget in the tokens the embedder and reranker actually see; cl100k packs digits and punctuation more tightly
    than WordPiece, so dense tables measured here can overflow a 512-token model window."""
    return len(_ENCODER.encode(text))


Counter = Callable[[str], int]


# ── baseline ─────────────────────────────────────────────────────────────────


def chunk_baseline(doc_id: str, markdown: str, count: Counter = count_tokens, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Run the exact ingestion path DocuMind uses today (needs ``src`` on sys.path)."""
    from documind.ingestion.chunker import chunk_document
    from documind.ingestion.parser import parse_document
    from documind.models import Document, DocumentMetadata

    document = parse_document(Document(content=markdown, metadata=DocumentMetadata(title=doc_id)))
    overlap = OVERLAP_TOKENS if OVERLAP_TOKENS < max_tokens else max_tokens // 5  # the app requires overlap < budget
    return [c.content for c in chunk_document(document, count_tokens=count, max_tokens=max_tokens, overlap=overlap)]


# ── structure preserving ─────────────────────────────────────────────────────


def split_blocks(markdown: str) -> list[tuple[str, str]]:
    """Split Markdown into ("table" | "text", block) pieces in document order."""
    blocks: list[tuple[str, str]] = []

    def add_text(segment: str) -> None:
        paragraph: list[str] = []
        table: list[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(("text", "\n".join(paragraph).strip()))
                paragraph.clear()

        def flush_table() -> None:
            if len(table) >= 2 and any(_SEPARATOR_ROW.match(row) for row in table):
                flush_paragraph()  # prose before the table is emitted first, keeping document order
                blocks.append(("table", "\n".join(table)))
            elif table:
                paragraph.extend(table)  # a lone pipe line is just text and stays in its paragraph
            table.clear()

        for line in segment.splitlines():
            if line.strip().startswith("|"):
                table.append(line)
                continue
            flush_table()
            if not line.strip():
                flush_paragraph()
            else:
                paragraph.append(line)
        flush_table()
        flush_paragraph()

    position = 0
    for match in _HTML_TABLE.finditer(markdown):
        add_text(markdown[position:match.start()])
        blocks.append(("table", match.group(0)))
        position = match.end()
    add_text(markdown[position:])
    return [(kind, text) for kind, text in blocks if text.strip()]


def split_table(table: str, max_tokens: int, count: Counter = count_tokens) -> list[str]:
    """Keep a table whole when it fits, else split by rows repeating the header."""
    if count(table) <= max_tokens:
        return [table]

    if table.lstrip().lower().startswith("<table"):
        rows = _HTML_ROW.findall(table)
        if len(rows) < 2:
            return [table]
        header, body, wrap = rows[0], rows[1:], ("<table>", "</table>")
        join = ""
    else:
        lines = table.splitlines()
        header_len = 2 if len(lines) > 1 and _SEPARATOR_ROW.match(lines[1]) else 1
        header, body, wrap, join = "\n".join(lines[:header_len]), lines[header_len:], ("", ""), "\n"

    def assemble(rows: list[str]) -> str:
        if wrap[0]:
            return wrap[0] + header + join.join(rows) + wrap[1]
        return header + "\n" + join.join(rows)

    pieces: list[str] = []
    current: list[str] = []
    for row in body:
        if current and count(assemble(current + [row])) > max_tokens:
            pieces.append(assemble(current))
            current = []
        current.append(row)
    if current:
        pieces.append(assemble(current))
    return pieces


def split_text(text: str, max_tokens: int, count: Counter = count_tokens) -> list[str]:
    """Split an oversized paragraph on sentence boundaries (and on words for a giant sentence)."""
    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(" ".join(current))
            current.clear()

    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        while count(sentence) > max_tokens and len(words) > 1:
            flush()
            half = len(words) // 2
            pieces.extend(split_text(" ".join(words[:half]), max_tokens, count))
            words, sentence = words[half:], " ".join(words[half:])
        if current and count(" ".join(current + [sentence])) > max_tokens:
            flush()
        current.append(sentence)
    flush()
    return pieces


def chunk_structure_preserving(markdown: str, max_tokens: int = CHUNK_TOKENS, count: Counter = count_tokens) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    buffered = 0

    def flush() -> None:
        nonlocal buffered
        if buffer:
            chunks.append("\n\n".join(buffer))
            buffer.clear()
            buffered = 0

    for kind, block in split_blocks(markdown):
        if kind == "table":
            flush()
            chunks.extend(split_table(block, max_tokens, count))
            continue
        tokens = count(block)
        if tokens > max_tokens:
            flush()
            chunks.extend(split_text(block, max_tokens, count))
            continue
        if buffered + tokens > max_tokens:
            flush()
        buffer.append(block)
        buffered += tokens
    flush()
    return chunks


CHUNKERS = {"baseline": chunk_baseline, "structure_preserving": lambda doc_id, md, **kw: chunk_structure_preserving(md, **kw)}
