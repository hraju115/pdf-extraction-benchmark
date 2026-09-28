# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Input limit of the embedding model (bge-large-en-v1.5) and reranker, in their own tokens
MODEL_MAX_TOKENS = 512


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Embeddings ───────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cpu"

    # ── Reranker ─────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cpu"
    reranker_top_k: int = 5

    # ── Qdrant ───────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documind"
    qdrant_api_key: str = ""

    # ── Sparse Embeddings (F7) ────────────────────────
    sparse_embedding_model: str = "Qdrant/bm25"

    # ── Retrieval ────────────────────────────────────
    retrieval_top_k: int = 50

    # ── Chunking (in the embedder's tokens; hard cap 512 = bge's input limit) ──
    chunk_size: int = 450
    chunk_overlap: int = 50

    @model_validator(mode="after")
    def _chunks_fit_the_model_window(self) -> Self:
        """Fail at startup, not per document at ingestion, when chunks cannot fit the model."""
        if self.chunk_size + self.chunk_overlap > MODEL_MAX_TOKENS:
            raise ValueError(
                f"CHUNK_SIZE + CHUNK_OVERLAP must be <= {MODEL_MAX_TOKENS} tokens of the embedding "
                f"model (got {self.chunk_size} + {self.chunk_overlap}); "
                "set CHUNK_SIZE=450 and CHUNK_OVERLAP=50"
            )
        if self.chunk_size < 50:
            raise ValueError(f"CHUNK_SIZE must be >= 50 tokens (got {self.chunk_size})")
        return self

    # ── Langfuse ─────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "http://localhost:3000"
    langfuse_enabled: bool = True

    # ── Prompts ──────────────────────────────────────
    # Directory of prompt YAML overrides; empty = packaged documind/prompts
    prompt_dir: str = ""

    # ── Redis (F8) ────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_auth: str = ""
    cache_ttl: int = 3600
    cache_enabled: bool = False

    # ── API ──────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    log_level: str = "info"
    cors_origins: list[str] = ["http://localhost", "http://localhost:3000"]
    # Required for /ingest and /collections (X-API-Key); ingestion is disabled while empty
    api_key: str = ""
    # Serve /docs, /redoc and /openapi.json
    api_docs_enabled: bool = False

    # ── Ingestion ────────────────────────────────────
    # Local ingestion sources must resolve inside this directory
    data_dir: str = "data/raw"

    # ── Evaluation ───────────────────────────────────
    eval_golden_dataset_path: str = "data/golden_dataset/golden.json"
    eval_min_faithfulness: float = 0.8
    eval_min_answer_correctness: float = 0.7
    eval_min_context_precision: float = 0.75
    eval_min_context_recall: float = 0.75
    eval_dataset_name: str = "documind-golden"
    eval_use_langfuse_dataset: bool = False


def get_settings() -> Settings:
    """Factory cached at module level; override in tests via monkeypatch."""
    return Settings()
