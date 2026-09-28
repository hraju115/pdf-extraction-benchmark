# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Domain models shared across all modules."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, Field

# ── Document & Chunk ─────────────────────────────────────────────────────────


class DocumentSource(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


class DocumentMetadata(BaseModel):
    title: str = ""
    source_url: str = ""
    source_type: DocumentSource = DocumentSource.MARKDOWN
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, str] = Field(default_factory=dict)


class Document(BaseModel):
    """A raw document before chunking."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    metadata: DocumentMetadata


class ChunkMetadata(BaseModel):
    document_id: str
    document_title: str = ""
    section_heading: str = ""
    source_url: str = ""
    chunk_index: int = 0


def chunk_id(source: str, index: int) -> str:
    """Deterministic point id for chunk ``index`` of ``source`` (hyphenated UUIDv5).

    Re-ingesting a source yields the same ids, so its points are overwritten, never duplicated.
    """
    return str(uuid5(NAMESPACE_URL, f"{source}#{index}"))


class Chunk(BaseModel):
    """An indexed chunk with embedding-ready text."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    metadata: ChunkMetadata
    token_count: int = 0


# ── Retrieval ────────────────────────────────────────────────────────────────


class ScoredChunk(BaseModel):
    """A chunk with a relevance score from retrieval or reranking."""

    chunk: Chunk
    score: float
    source: str = "hybrid"  # "vector", "hybrid", "reranker"


# ── Query & Response ─────────────────────────────────────────────────────────


COLLECTION_PATTERN = r"^[a-z0-9_-]{1,64}$"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field("documind", pattern=COLLECTION_PATTERN)
    top_k: int = Field(5, ge=1, le=20)
    session_id: str = ""


class Citation(BaseModel):
    chunk_id: str
    document_title: str
    section_heading: str
    source_url: str
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    is_grounded: bool = True
    trace_id: str = ""
    latency_ms: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)


# ── Feedback ─────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    trace_id: str
    score: float = Field(ge=0, le=1)
    comment: str = ""


class FeedbackResponse(BaseModel):
    status: str
    trace_id: str


# ── Ingestion ────────────────────────────────────────────────────────────────


class IngestionRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1, max_length=100)  # URLs or paths inside DATA_DIR
    collection: str = Field("documind", pattern=COLLECTION_PATTERN)
    source_type: DocumentSource = DocumentSource.HTML
    crawl: bool = False
    max_depth: int = Field(default=3, ge=1, le=10)
    max_pages: int = Field(100, ge=1, le=500)


class IngestionResult(BaseModel):
    documents_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = Field(default_factory=list)


class IngestionAccepted(BaseModel):
    """202 body of POST /ingest: poll GET /ingest/{job_id} for the outcome."""

    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    result: IngestionResult | None = None
    error: str | None = None


# ── Evaluation ───────────────────────────────────────────────────────────────


class GoldenExample(BaseModel):
    question: str
    ground_truth: str
    context_keywords: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_correctness: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    num_samples: int = 0
    passed: bool = False
