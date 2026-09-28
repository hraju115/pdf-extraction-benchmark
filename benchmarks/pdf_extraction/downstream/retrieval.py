"""In-memory dense retrieval + rerank over one document's chunks, with pluggable models.

This measures the *extraction's* effect on retrieval; it deliberately does not involve Qdrant.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    """May also define ``save()`` to persist a cache; callers check for it with ``hasattr``."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # (n, d), L2-normalized
    def embed_query(self, text: str) -> np.ndarray: ...  # (d,), L2-normalized


class TextReranker(Protocol):
    def rerank(self, query: str, texts: list[str], top_k: int) -> list[int]: ...  # indices into texts, best first


class DocuMindEmbedder:
    """DocuMind's own ``Embedder`` with an on-disk ``.npz`` cache keyed by sha1(model + text).

    The cache is written only by ``save()`` (atomically: temp file + ``os.replace``) and a missing or corrupt cache
    loads as empty, so an interrupted run never corrupts hours of embeddings. Each model gets its own file
    (``<stem>.<sha1(model)[:10]><suffix>`` next to ``cache_path``), so models with different dimensions never share
    one matrix.
    """

    def __init__(self, model_name: str, device: str = "cpu", cache_path: Path | None = None) -> None:
        self._embedder = self._load_model(model_name, device)
        self._model = model_name
        if cache_path:
            model_tag = hashlib.sha1(model_name.encode()).hexdigest()[:10]
            cache_path = cache_path.with_name(f"{cache_path.stem}.{model_tag}{cache_path.suffix}")
        self._cache_path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        if cache_path and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    self._cache = dict(zip(data["keys"].tolist(), data["vectors"]))
            except Exception:  # noqa: BLE001 - a corrupt cache is rebuilt, not fatal
                self._cache = {}

    def count_tokens(self, text: str) -> int:
        """Length of ``text`` in the embedding model's own tokenizer (special tokens included), so chunk budgets are
        measured in the tokens the model truncates at. Falls back to the default counter when no tokenizer is exposed."""
        tokenizer = getattr(getattr(self._embedder, "_model", None), "tokenizer", None)
        if tokenizer is None:
            from benchmarks.pdf_extraction.downstream.chunking import count_tokens

            return count_tokens(text)
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    @staticmethod
    def _load_model(model_name: str, device: str):
        from documind.ingestion.embedder import Embedder

        return Embedder(model_name=model_name, device=device)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self._model}\n{text}".encode()).hexdigest()

    def cached_count(self) -> int:
        return len(self._cache)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        from documind.models import Chunk, ChunkMetadata

        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        if missing:
            chunks = [Chunk(content=t, metadata=ChunkMetadata(document_id="bench")) for t in missing]
            for text, vector in zip(missing, self._embedder.embed_chunks(chunks, batch_size=16)):
                self._cache[self._key(text)] = np.asarray(vector, dtype=np.float32)
            self._dirty = True
        return np.stack([self._cache[self._key(t)] for t in texts]) if texts else np.zeros((0, 1), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._embedder.embed_query(text), dtype=np.float32)

    def save(self) -> None:
        if not self._cache_path or not self._dirty:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._cache.keys()))
        vectors = np.stack(list(self._cache.values())) if self._cache else np.zeros((0, 1), dtype=np.float32)
        tmp = self._cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vectors=vectors)
        os.replace(tmp, self._cache_path)
        self._dirty = False


class DocuMindReranker:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from documind.retrieval.reranker import Reranker

        self._reranker = Reranker(model_name=model_name, device=device)

    def rerank(self, query: str, texts: list[str], top_k: int) -> list[int]:
        from documind.models import Chunk, ChunkMetadata, ScoredChunk

        candidates = [
            ScoredChunk(chunk=Chunk(id=str(i), content=t, metadata=ChunkMetadata(document_id="bench")), score=0.0)
            for i, t in enumerate(texts)
        ]
        return [int(sc.chunk.id) for sc in self._reranker.rerank(query, candidates, top_k=top_k)]


class DenseIndex:
    def __init__(self, embedder: TextEmbedder, chunks: list[str]) -> None:
        self._embedder = embedder
        self.chunks = chunks
        self._matrix = embedder.embed_documents(chunks) if chunks else np.zeros((0, 1), dtype=np.float32)

    def search(self, query: str, k: int) -> list[int]:
        """Indices of the ``k`` most similar chunks, best first."""
        if not self.chunks:
            return []
        scores = self._matrix @ self._embedder.embed_query(query)
        return [int(i) for i in np.argsort(-scores)[:k]]
