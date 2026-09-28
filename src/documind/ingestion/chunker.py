# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Structure-aware chunking measured in the embedder's own tokens.

Blocks: fenced code (```...```), pipe tables, headings (outside fences only), paragraphs. Code
and tables are atomic when they fit; anything larger is split by lines (tables by rows, header
repeated), then by words, then by characters, so no piece ever exceeds the hard cap and no
content is dropped. Every chunk starts with a breadcrumb "Title > H1 > H2" so the embedding
carries its section context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from documind.logging import get_logger
from documind.models import Chunk, ChunkMetadata, Document

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

_FENCE_OPEN = re.compile(r"^(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Block:
    kind: str  # "code" | "table" | "text"
    text: str
    breadcrumb: list[str]
    heading: str


def split_blocks(content: str) -> list[Block]:
    """Split Markdown into code / table / text blocks, each tagged with its heading breadcrumb."""
    blocks: list[Block] = []
    crumbs: dict[int, str] = {}
    fence: list[str] | None = None
    fence_close: re.Pattern[str] | None = None
    para: list[str] = []
    table: list[str] = []

    def crumb_list() -> list[str]:
        return [crumbs[k] for k in sorted(crumbs)]

    def heading() -> str:
        return crumbs[max(crumbs)] if crumbs else ""

    def flush_para() -> None:
        if para:
            blocks.append(Block("text", "\n".join(para).strip(), crumb_list(), heading()))
            para.clear()

    def flush_table() -> None:
        if len(table) >= 2:
            blocks.append(Block("table", "\n".join(table), crumb_list(), heading()))
        else:
            para.extend(table)
        table.clear()

    for line in content.splitlines():
        if fence is not None:
            fence.append(line)
            if fence_close is not None and fence_close.fullmatch(line.strip()):
                blocks.append(Block("code", "\n".join(fence), crumb_list(), heading()))
                fence = None
            continue
        opener = _FENCE_OPEN.match(line.strip())
        if opener:
            flush_table()
            flush_para()
            mark = opener.group(1)
            # A closing fence uses the same character, at least as long, with no info string.
            fence_close = re.compile(re.escape(mark[0]) + "{" + str(len(mark)) + ",}")
            fence = [line]
            continue
        match = _HEADING.match(line)
        if match:
            flush_table()
            flush_para()
            level = len(match.group(1))
            crumbs = {k: v for k, v in crumbs.items() if k < level}
            crumbs[level] = match.group(2)
            continue
        if _PIPE_ROW.match(line):
            flush_para()
            table.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_para()
        else:
            para.append(line)
    if fence is not None:  # unterminated fence: keep it as code anyway
        blocks.append(Block("code", "\n".join(fence), crumb_list(), heading()))
    flush_table()
    flush_para()
    return [b for b in blocks if b.text.strip()]


def _pack(parts: list[str], sep: str, count: Callable[[str], int], cap: int) -> list[str]:
    """Greedily join consecutive parts with ``sep`` while the joined text fits ``cap``."""
    out: list[str] = []
    current: list[str] = []
    for part in parts:
        if current and count(sep.join([*current, part])) > cap:
            out.append(sep.join(current))
            current = []
        current.append(part)
    if current:
        out.append(sep.join(current))
    return out


def _split_word(word: str, count: Callable[[str], int], cap: int) -> list[str]:
    """Halve a single over-long token run (minified code, base64) until each piece fits."""
    if count(word) <= cap or len(word) < 2:
        return [word]
    mid = len(word) // 2
    return _split_word(word[:mid], count, cap) + _split_word(word[mid:], count, cap)


def _split_line(line: str, count: Callable[[str], int], cap: int) -> list[str]:
    if count(line) <= cap:
        return [line]
    words = [piece for w in line.split() for piece in _split_word(w, count, cap)]
    return _pack(words, " ", count, cap)


def _split_oversize(text: str, count: Callable[[str], int], cap: int) -> list[str]:
    """Split by lines, then words, then characters until every piece fits. Never drops content."""
    if count(text) <= cap:
        return [text]
    units = [piece for line in text.splitlines() for piece in _split_line(line, count, cap)]
    return _pack(units, "\n", count, cap)


def _split_table(text: str, count: Callable[[str], int], cap: int) -> list[str]:
    """Split a table by rows, repeating the header and separator rows in every piece."""
    if count(text) <= cap:
        return [text]
    rows = text.splitlines()
    header = "\n".join(rows[:2])
    body_cap = cap - count(header)
    if body_cap < cap // 2:  # a header that eats the budget is not worth repeating
        return _split_oversize(text, count, cap)
    return [f"{header}\n{piece}" for piece in _split_oversize("\n".join(rows[2:]), count, body_cap)]


@cache
def _cl100k() -> Callable[[str], int]:
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoder.encode(text))


def chunk_document(
    doc: Document,
    *,
    count_tokens: Callable[[str], int] | None = None,
    max_tokens: int = 450,
    overlap: int = 50,
    hard_cap: int = 512,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a document into chunks of about ``max_tokens`` and never more than ``hard_cap`` tokens.

    Tokens are counted with ``count_tokens`` (the embedder's tokenizer in production), including the
    breadcrumb line. Prose is packed sentence by sentence with an ``overlap``-token tail carried
    into the next chunk of the same section; code blocks and tables are emitted atomically when
    they fit.

    Compatibility shim: ``chunk_size`` / ``chunk_overlap`` are accepted as aliases of
    ``max_tokens`` / ``overlap`` and, when no ``count_tokens`` is given, tokens are counted with
    tiktoken ``cl100k_base`` (the pre-2026-09 behaviour). Kept for callers outside the service
    (benchmarks); the ingestion runner always passes the embedder's ``count_tokens``.
    """
    if chunk_size is not None:
        max_tokens = chunk_size
    if chunk_overlap is not None:
        overlap = chunk_overlap
    count = count_tokens if count_tokens is not None else _cl100k()
    if not 0 <= overlap < max_tokens <= hard_cap:
        raise ValueError(
            f"need 0 <= overlap < max_tokens <= hard_cap, got {overlap}, {max_tokens}, {hard_cap}"
        )

    title = doc.metadata.title or ""
    chunks: list[Chunk] = []

    def crumb_of(block: Block) -> str:
        crumb = " > ".join(c for c in [title, *block.breadcrumb] if c)
        if crumb and count(f"{crumb}\n\n") > hard_cap // 4:  # absurdly long headings: drop it
            return ""
        return crumb

    def prefix_tokens(block: Block) -> int:
        crumb = crumb_of(block)
        return count(f"{crumb}\n\n") if crumb else 0

    def emit(text: str, block: Block) -> None:
        crumb = crumb_of(block)
        body = f"{crumb}\n\n{text}" if crumb else text
        chunks.append(
            Chunk(
                content=body,
                metadata=ChunkMetadata(
                    document_id=doc.id,
                    document_title=title,
                    section_heading=block.heading,
                    source_url=doc.metadata.source_url,
                    chunk_index=len(chunks),
                ),
                token_count=count(body),
            )
        )

    def join(text: str, piece: str, sep: str) -> str:
        return f"{text}{sep}{piece}" if text else piece

    def overlap_tail(text: str) -> str:
        tail = text.split()
        keep: list[str] = []
        while tail and count(" ".join([tail[-1], *keep])) <= overlap:
            keep.insert(0, tail.pop())
        return " ".join(keep)

    buffer = ""  # prose waiting to be emitted; always <= budget tokens
    buffer_block: Block | None = None

    def flush() -> None:
        nonlocal buffer, buffer_block
        if buffer and buffer_block is not None:
            emit(buffer, buffer_block)
        buffer, buffer_block = "", None

    for block in split_blocks(doc.content):
        prefix = prefix_tokens(block)
        if block.kind in ("code", "table"):
            flush()
            split = _split_table if block.kind == "table" else _split_oversize
            for piece in split(block.text, count, hard_cap - prefix):
                emit(piece, block)
            continue
        if buffer_block is not None and buffer_block.breadcrumb != block.breadcrumb:
            flush()  # never mix sections (or carry overlap across them)
        # prefix <= hard_cap // 4, so prefix + budget <= hard_cap even when the floor applies
        budget = max(max_tokens - prefix, max_tokens // 2)
        sentences = _SENTENCE_END.split(block.text) if count(block.text) > budget else [block.text]
        sep = "\n\n"  # between paragraphs; sentences of one paragraph are joined with a space
        for sentence in sentences:
            for piece in _split_oversize(sentence, count, budget):
                candidate = join(buffer, piece, sep)
                if buffer and count(candidate) > budget:
                    emit(buffer, buffer_block or block)
                    carried = overlap_tail(buffer)
                    candidate = join(carried, piece, " ")
                    if count(candidate) > budget:  # the overlap would push the piece over
                        candidate = piece
                buffer = candidate
                buffer_block = block
                sep = " "
    flush()
    logger.info("chunking_complete", doc_id=doc.id, num_chunks=len(chunks))
    return chunks
