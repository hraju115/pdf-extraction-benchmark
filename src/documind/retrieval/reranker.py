# Vendored from the DocuMind application on 2026-09-27; the benchmark's retrieval stage scores extractor output through the app's own chunker, embedder and reranker.
"""Cross-encoder reranker using bge-reranker-large."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from documind.logging import get_logger
from documind.models import ScoredChunk

logger = get_logger(__name__)


class Reranker:
    """Scores query-document pairs with a cross-encoder and returns the top-k."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
    ) -> None:
        logger.info("loading_reranker", model=model_name, device=device)
        # Truncate (query, passage) pairs at the model's 512-token input limit
        self._model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Score every candidate against the query and return the top-k.

        The cross-encoder processes (query, passage) pairs jointly, giving
        much higher accuracy than bi-encoder similarity alone.
        """
        if not candidates:
            return []

        pairs = [(query, c.chunk.content) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach reranker scores and sort
        reranked: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    source="reranker",
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "reranking_complete",
            candidates=len(candidates),
            top_k=top_k,
            top_score=reranked[0].score if reranked else 0.0,
        )

        return reranked[:top_k]
