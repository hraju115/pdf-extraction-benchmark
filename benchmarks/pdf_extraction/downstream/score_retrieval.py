"""Retrieval hit-rate per extractor: does the evidence for each question survive extraction, chunking,
search and reranking?  One index per document, so extractors are compared on extraction quality alone.

    python -m benchmarks.pdf_extraction.downstream.score_retrieval [--modes baseline,structure_preserving]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry
from benchmarks.pdf_extraction.corpus.questions import Question
from benchmarks.pdf_extraction.downstream.chunking import CHUNKERS
from benchmarks.pdf_extraction.downstream.retrieval import DenseIndex, TextEmbedder, TextReranker
from benchmarks.pdf_extraction.metrics.facts import evidence_hit
from benchmarks.pdf_extraction.scoring import iter_runs

RETRIEVAL_FIELDS = ["extractor", "doc_id", "tier", "bucket", "chunk_mode", "question_id", "qtype", "n_chunks", "hit5", "hit50"]
SEARCH_K = 50
RERANK_K = 5
SAVE_EVERY_DOCS = 10  # bound the embedding-cache rewrites (the whole .npz each time)


def score_retrieval(
    out_root: Path,
    questions: list[Question],
    entries: list[ManifestEntry],
    chunk_mode: str,
    embedder: TextEmbedder,
    reranker: TextReranker,
    chunker: Callable[[str, str], list[str]] | None = None,
    cache_dir: Path | None = None,
    cache_salt: str = "",
) -> list[dict[str, Any]]:
    """One row per (extractor, question). A missing, failed or empty extraction yields hit5 = hit50 = 0.

    With ``cache_dir``, each (mode, extractor, doc)'s rows are stored in ``cache_dir/<mode>/<extractor>/<doc>.json``
    and reused while the run's ``input_hash``, the bytes of its ``result.md`` and the document's questions are unchanged.
    ``cache_salt`` (the embedding and reranker model names) is part of that key, so a model change recomputes.
    The embedder's ``save()``, if any, runs every ``SAVE_EVERY_DOCS`` newly scored documents and at the end.
    """
    chunker = chunker or CHUNKERS[chunk_mode]
    by_id = {e.id: e for e in entries}
    by_doc: dict[str, list[Question]] = {}
    for q in questions:
        if q.type != "unanswerable":
            by_doc.setdefault(q.doc_id, []).append(q)

    rows: list[dict[str, Any]] = []
    scored_docs = 0
    for extractor, doc_id, doc_dir, run in iter_runs(out_root):
        if doc_id not in by_doc:
            continue
        markdown_path = doc_dir / "result.md"
        markdown_bytes = markdown_path.read_bytes() if run.status == "ok" and markdown_path.exists() else b""
        # Keyed on what retrieval actually depends on: the extracted text, the questions and the models. The run's
        # input hash is deliberately left out, so the same result.md produced on another machine (different caps,
        # hence a different hash) reuses its rows, while any re-extraction that changes the text recomputes them.
        key = {
            "cache_salt": cache_salt,
            "markdown_sha1": hashlib.sha1(markdown_bytes).hexdigest(),
            "questions_sha1": _questions_sha1(by_doc[doc_id]),
        }
        cache_file = cache_dir / chunk_mode / extractor / f"{doc_id}.json" if cache_dir else None
        if cache_file and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if all(cached.get(k) == v for k, v in key.items()):
                    rows.extend(cached["rows"])
                    continue
            except (json.JSONDecodeError, KeyError, AttributeError):
                pass  # an unreadable cache entry is recomputed
        markdown = markdown_bytes.decode("utf-8")
        # Measure the chunk budget in the embedder's own tokens when it can tell us (see chunking.count_tokens)
        chunk_kwargs = {"count": embedder.count_tokens} if hasattr(embedder, "count_tokens") else {}
        chunks = chunker(doc_id, markdown, **chunk_kwargs) if markdown.strip() else []
        index = DenseIndex(embedder, chunks)
        entry = by_id.get(doc_id)

        doc_rows: list[dict[str, Any]] = []
        for q in by_doc[doc_id]:
            if run.sampled and run.page_range and not run.page_range[0] <= q.page <= run.page_range[1]:
                continue  # a sampled run is only judged on questions whose page it was asked to read
            hit5 = hit50 = False
            if chunks:
                ranked = index.search(q.question, SEARCH_K)
                candidates = [chunks[i] for i in ranked]
                hit50 = evidence_hit(candidates, q.evidence, SEARCH_K)
                if hit50:  # the reranked top-5 is a subset of these 50: a miss here is a miss there
                    top = [candidates[i] for i in reranker.rerank(q.question, candidates, RERANK_K)]
                    hit5 = evidence_hit(top, q.evidence, RERANK_K)
            doc_rows.append({
                "extractor": extractor, "doc_id": doc_id, "tier": entry.tier if entry else "", "bucket": entry.bucket if entry else "",
                "chunk_mode": chunk_mode, "question_id": q.id, "qtype": q.type, "n_chunks": len(chunks),
                "hit5": int(hit5), "hit50": int(hit50),
            })
        rows.extend(doc_rows)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({**key, "rows": doc_rows}), encoding="utf-8")
        scored_docs += 1
        if scored_docs % SAVE_EVERY_DOCS == 0 and hasattr(embedder, "save"):
            embedder.save()
    if hasattr(embedder, "save"):
        embedder.save()
    return rows


def _questions_sha1(questions: list[Question]) -> str:
    payload = [[q.id, q.question, q.evidence, q.page, q.type] for q in questions]
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    from benchmarks.pdf_extraction.corpus.build import all_entries
    from benchmarks.pdf_extraction.corpus.questions import inherit_scan_questions, load_all_questions
    from benchmarks.pdf_extraction.downstream.retrieval import DocuMindEmbedder, DocuMindReranker
    from benchmarks.pdf_extraction.paths import GENERATED_QUESTIONS_DIR, OUT_DIR, QUESTIONS_DIR, RESULTS_DIR
    from benchmarks.pdf_extraction.scoring import write_csv
    from documind.config import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="baseline,structure_preserving")
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = DocuMindEmbedder(settings.embedding_model, settings.embedding_device, RESULTS_DIR / "embedding_cache.npz")
    reranker = DocuMindReranker(settings.reranker_model, settings.reranker_device)
    entries = all_entries()
    questions = inherit_scan_questions(load_all_questions(QUESTIONS_DIR, GENERATED_QUESTIONS_DIR), entries)

    rows: list[dict[str, Any]] = []
    for mode in args.modes.split(","):
        rows += score_retrieval(OUT_DIR, questions, entries, mode.strip(), embedder, reranker,
                                cache_dir=RESULTS_DIR / "retrieval_cache",
                                cache_salt=f"{settings.embedding_model}|{settings.reranker_model}|budget:embedder-tokens")
    embedder.save()
    write_csv(RESULTS_DIR / "retrieval_scores.csv", rows, RETRIEVAL_FIELDS)
    print(f"wrote {len(rows)} rows to {RESULTS_DIR}/retrieval_scores.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
