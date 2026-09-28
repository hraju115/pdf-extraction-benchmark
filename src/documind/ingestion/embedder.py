# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Embedding generation using sentence-transformers (bge-large-en-v1.5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from documind.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

    from documind.models import Chunk

logger = get_logger(__name__)

# BGE models require a task-specific prefix for queries
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps a sentence-transformer model for encoding documents and queries."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading_embedding_model", model=model_name, device=device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
        """Embed chunk contents. No prefix for passage embeddings per BGE spec."""
        texts = [c.content for c in chunks]
        logger.info("embedding_chunks", count=len(texts), batch_size=batch_size)
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. BGE requires a task prefix for queries."""
        prefixed = _QUERY_PREFIX + query
        embedding: np.ndarray = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    def count_tokens(self, text: str) -> int:
        """Length of ``text`` in this model's tokens, [CLS]/[SEP] included (unit of the 512 cap)."""
        return len(self._model.tokenizer(text, add_special_tokens=True)["input_ids"])

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
