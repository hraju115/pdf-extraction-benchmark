"""Tests for chunking modes, in-memory retrieval, hit-rate scoring and the end-to-end aggregation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from benchmarks.pdf_extraction.corpus.manifest import ManifestEntry
from benchmarks.pdf_extraction.corpus.questions import Question
from benchmarks.pdf_extraction.downstream.chunking import (
    chunk_baseline,
    chunk_structure_preserving,
    count_tokens,
    split_blocks,
    split_table,
)
from benchmarks.pdf_extraction.downstream.retrieval import DenseIndex
from benchmarks.pdf_extraction.downstream.score_retrieval import score_retrieval
from benchmarks.pdf_extraction.extractors.base import ExtractionResult, PageResult, RunInfo, write_outputs, write_run


def pipe_table(rows: int) -> str:
    lines = ["| Region | Sales |", "|---|---|"] + [f"| Row{i:03d} | {i * 111} |" for i in range(rows)]
    return "\n".join(lines)


# ── chunking ─────────────────────────────────────────────────────────────────


def test_small_table_stays_in_one_chunk_with_its_neighbours_split_off():
    md = "Intro paragraph.\n\n" + pipe_table(3) + "\n\nOutro paragraph."
    chunks = chunk_structure_preserving(md, max_tokens=200)
    table_chunks = [c for c in chunks if "| Region |" in c]
    assert len(table_chunks) == 1 and "Row002" in table_chunks[0]
    assert "Intro paragraph." not in table_chunks[0]  # tables never share a chunk with prose


def test_large_pipe_table_splits_by_rows_and_repeats_the_header():
    table = pipe_table(80)
    chunks = chunk_structure_preserving(table, max_tokens=120)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("| Region | Sales |\n|---|---|")
        assert count_tokens(chunk) <= 120
    body = "\n".join(chunks)
    for i in range(80):
        assert body.count(f"Row{i:03d}") == 1  # every row exactly once


def test_large_html_table_splits_by_rows_and_repeats_the_header_row():
    rows = "".join(f"<tr><td>Row{i:03d}</td><td>{i}</td></tr>" for i in range(60))
    table = f"<table><tr><td>Region</td><td>Sales</td></tr>{rows}</table>"
    pieces = split_table(table, 100)
    assert len(pieces) > 1
    assert all(p.startswith("<table><tr><td>Region</td>") and p.endswith("</table>") for p in pieces)
    assert sum(p.count("Row0") for p in pieces) == 60


def test_paragraphs_are_packed_but_never_split_when_they_fit():
    paragraphs = [f"Paragraph number {i} has a few words in it." for i in range(6)]
    chunks = chunk_structure_preserving("\n\n".join(paragraphs), max_tokens=40)
    assert "".join(chunks).count("Paragraph number") == 6
    for p in paragraphs:
        assert any(p in c for c in chunks)  # each paragraph intact inside some chunk


def test_oversized_paragraph_is_split_on_sentences_within_the_limit():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = chunk_structure_preserving(text, max_tokens=60)
    assert len(chunks) > 1 and all(count_tokens(c) <= 60 for c in chunks)
    assert " ".join(chunks).count("Sentence number") == 60


def test_single_giant_row_is_emitted_alone_rather_than_dropped():
    giant = "| A | B |\n|---|---|\n| " + "word " * 300 + " | x |"
    chunks = split_table(giant, 50)
    assert any("word word" in c for c in chunks)


def test_empty_and_whitespace_markdown_produce_no_chunks():
    assert chunk_structure_preserving("") == []
    assert chunk_structure_preserving("  \n\n \n") == []


def test_lone_pipe_line_is_treated_as_text_not_a_table():
    blocks = split_blocks("before\n| not | a table\nafter")
    assert [k for k, _ in blocks] == ["text"]


def test_html_and_pipe_tables_are_both_detected_in_order():
    md = "text one\n\n<table><tr><td>A</td></tr></table>\n\ntext two\n\n" + pipe_table(1)
    assert [k for k, _ in split_blocks(md)] == ["text", "table", "text", "table"]


def test_baseline_chunker_runs_documinds_real_ingestion_path():
    chunks = chunk_baseline("doc", "# Title\n\n" + "A sentence about widgets. " * 200)
    assert len(chunks) >= 2 and all(c.strip() for c in chunks)
    assert max(count_tokens(c) for c in chunks) <= 520  # stays under the embedder's 512-token input limit


# ── retrieval with fake models ───────────────────────────────────────────────


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(128, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % 128] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, 128), dtype=np.float32)

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    def rerank(self, query, texts, top_k):
        q = set(re.findall(r"\w+", query.lower()))
        scored = sorted(range(len(texts)), key=lambda i: -len(q & set(re.findall(r"\w+", texts[i].lower()))))
        return scored[:top_k]


def test_dense_index_ranks_the_matching_chunk_first_and_handles_empty():
    index = DenseIndex(HashEmbedder(), ["apples and pears", "quarterly sales north region", "unrelated text here"])
    assert index.search("north region sales", 3)[0] == 1
    assert index.search("anything", 2) != []
    assert DenseIndex(HashEmbedder(), []).search("x", 5) == []


def _write_run(out_root: Path, extractor: str, doc_id: str, markdown: str | None, status: str = "ok") -> None:
    out_dir = out_root / extractor / doc_id
    if markdown is None:
        write_run(out_dir, RunInfo(extractor=extractor, doc_id=doc_id, status=status, error="boom"))
    else:
        write_outputs(out_dir, ExtractionResult(markdown=markdown, pages=[PageResult(1, markdown)]),
                      RunInfo(extractor=extractor, doc_id=doc_id, pages_processed=1))


def _q(doc_id: str, text: str, evidence: list[str], qtype: str = "table_cell") -> Question:
    return Question(doc_id=doc_id, question=text, answer="a", evidence=evidence, page=1, type=qtype, id=f"{doc_id}#{text[:6]}")


ENTRY = ManifestEntry(id="doc", tier="T2", bucket="S", source="generated")


def test_score_retrieval_separates_good_broken_and_missing_extractions(tmp_path):
    good = "Sales report\n\n| Region | Sales |\n|---|---|\n| North | 1,200 |\n| South | 900 |"
    broken = "Sales report North South Region Sales 1200"  # values flattened, but still text
    _write_run(tmp_path, "good", "doc", good)
    _write_run(tmp_path, "flat", "doc", broken)
    _write_run(tmp_path, "empty", "doc", "")
    _write_run(tmp_path, "crashed", "doc", None, status="error")

    questions = [_q("doc", "What are North sales?", ["North", "1200"]), _q("doc", "Anything unanswerable?", [], "unanswerable")]
    rows = score_retrieval(tmp_path, questions, [ENTRY], "structure_preserving", HashEmbedder(), OverlapReranker())

    by_extractor = {r["extractor"]: r for r in rows}
    assert len(rows) == 4  # the unanswerable question is not scored
    assert by_extractor["good"]["hit5"] == 1 and by_extractor["good"]["hit50"] == 1
    assert by_extractor["flat"]["hit5"] == 1  # evidence still present in one chunk
    assert by_extractor["empty"]["hit5"] == 0 and by_extractor["empty"]["n_chunks"] == 0
    assert by_extractor["crashed"]["hit5"] == 0 and by_extractor["crashed"]["hit50"] == 0
    assert all(r["tier"] == "T2" and r["bucket"] == "S" and r["chunk_mode"] == "structure_preserving" for r in rows)


def test_evidence_split_across_chunks_is_a_miss(tmp_path):
    # Evidence needs BOTH strings in ONE chunk; a chunker that separates them must score 0.
    _write_run(tmp_path, "ex", "doc", "North region text.\n\nSomewhere else 1200 units.")
    rows = score_retrieval(tmp_path, [_q("doc", "north 1200?", ["North", "1200"])], [ENTRY], "custom",
                           HashEmbedder(), OverlapReranker(), chunker=lambda doc, md: [p for p in md.split("\n\n")])
    assert rows[0]["hit5"] == 0 and rows[0]["n_chunks"] == 2


def test_documents_without_questions_are_ignored(tmp_path):
    _write_run(tmp_path, "ex", "other_doc", "text")
    assert score_retrieval(tmp_path, [_q("doc", "q?", ["x"])], [ENTRY], "structure_preserving", HashEmbedder(), OverlapReranker()) == []


# ── end to end ───────────────────────────────────────────────────────────────


def test_sampled_extractors_are_not_penalized_for_questions_on_unprocessed_pages(tmp_path):
    from benchmarks.pdf_extraction.extractors.base import write_outputs

    out_dir = tmp_path / "sampler" / "doc"
    write_outputs(out_dir, ExtractionResult(markdown="Page one text about widgets.", pages=[PageResult(1, "Page one text about widgets.")]),
                  RunInfo(extractor="sampler", doc_id="doc", pages_processed=1, page_range=[1, 3], sampled=True))
    early = Question(doc_id="doc", question="widgets?", answer="a", evidence=["widgets"], page=1, type="lookup", id="doc#early")
    late = Question(doc_id="doc", question="gizmos?", answer="a", evidence=["gizmos"], page=40, type="lookup", id="doc#late")
    rows = score_retrieval(tmp_path, [early, late], [ENTRY], "structure_preserving", HashEmbedder(), OverlapReranker())
    assert [r["question_id"] for r in rows] == ["doc#early"]  # the page-40 question is out of scope for this run
    assert rows[0]["hit5"] == 1


# ── efficiency and resume ────────────────────────────────────────────────────


class SpyReranker(OverlapReranker):
    def __init__(self):
        self.calls = 0

    def rerank(self, query, texts, top_k):
        self.calls += 1
        return super().rerank(query, texts, top_k)


class SpyEmbedder(HashEmbedder):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return super().embed_documents(texts)


def test_reranking_is_skipped_when_the_top_50_already_misses(tmp_path):
    _write_run(tmp_path, "ex", "doc", "nothing relevant here at all")
    spy = SpyReranker()
    rows = score_retrieval(tmp_path, [_q("doc", "north 1200?", ["North", "1200"])], [ENTRY], "structure_preserving", HashEmbedder(), spy)
    assert rows[0]["hit50"] == 0 and rows[0]["hit5"] == 0 and spy.calls == 0
    _write_run(tmp_path, "ex", "doc", "Sales: North 1,200 units")
    rows = score_retrieval(tmp_path, [_q("doc", "north 1200?", ["North", "1200"])], [ENTRY], "structure_preserving", HashEmbedder(), spy)
    assert rows[0]["hit50"] == 1 and spy.calls == 1


def test_retrieval_rows_are_persisted_per_document_and_reused(tmp_path):
    _write_run(tmp_path / "out", "ex", "doc", "Sales: North 1,200 units")
    questions = [_q("doc", "north 1200?", ["North", "1200"])]
    spy = SpyEmbedder()
    first = score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_dir=tmp_path / "cache")
    assert spy.document_calls == 1 and (tmp_path / "cache" / "baseline" / "ex" / "doc.json").exists()
    second = score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_dir=tmp_path / "cache")
    assert second == first and spy.document_calls == 1  # served from the per-document cache
    _write_run(tmp_path / "out", "ex", "doc", "Sales: North 1,200 units (re-extracted)")  # new input hash
    score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_dir=tmp_path / "cache")
    assert spy.document_calls == 2


def test_embedding_cache_is_written_atomically_and_tolerates_corruption(tmp_path, monkeypatch):
    """Review Focus 3: an interrupted write must not corrupt the cache; a corrupt cache must load as empty."""
    from benchmarks.pdf_extraction.downstream.retrieval import DocuMindEmbedder

    class FakeInner:  # stands in for documind.ingestion.embedder.Embedder
        def embed_chunks(self, chunks, batch_size=16):
            return [[float(len(c.content)), 1.0] for c in chunks]

        def embed_query(self, text):
            return [1.0, 0.0]

    monkeypatch.setattr(DocuMindEmbedder, "_load_model", lambda self, model_name, device: FakeInner())
    cache = tmp_path / "emb.npz"
    embedder = DocuMindEmbedder("fake-model", cache_path=cache)
    embedder.embed_documents(["abc", "de"])
    embedder.save()
    written = list(tmp_path.glob("emb.*.npz"))  # one file per model, derived from cache_path
    assert len(written) == 1 and not list(tmp_path.glob("*.tmp*"))
    again = DocuMindEmbedder("fake-model", cache_path=cache)
    assert again.cached_count() == 2
    written[0].write_bytes(b"not an npz")
    assert DocuMindEmbedder("fake-model", cache_path=cache).cached_count() == 0


def test_embedding_cache_is_saved_every_ten_documents_and_at_the_end(tmp_path):
    class SavingEmbedder(HashEmbedder):
        def __init__(self):
            self.saves = 0

        def save(self):
            self.saves += 1

    docs = [f"doc{i:02d}" for i in range(25)]
    for d in docs:
        _write_run(tmp_path, "ex", d, f"Sales: North 1,200 units in {d}")
    entries = [ManifestEntry(id=d, tier="T2", bucket="S", source="generated") for d in docs]
    questions = [_q(d, "north 1200?", ["North", "1200"]) for d in docs]
    embedder = SavingEmbedder()
    score_retrieval(tmp_path, questions, entries, "structure_preserving", embedder, OverlapReranker())
    assert embedder.saves == 3  # after documents 10 and 20, and once at the end


def test_row_cache_is_salted_with_the_model_names(tmp_path):
    _write_run(tmp_path / "out", "ex", "doc", "Sales: North 1,200 units")
    questions = [_q("doc", "north 1200?", ["North", "1200"])]
    spy = SpyEmbedder()
    kwargs = {"cache_dir": tmp_path / "cache"}
    score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_salt="emb-a|rr-a", **kwargs)
    score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_salt="emb-a|rr-a", **kwargs)
    assert spy.document_calls == 1  # same models: served from the cache
    score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_salt="emb-b|rr-a", **kwargs)
    assert spy.document_calls == 2  # a model change invalidates the cached rows


def test_two_models_with_different_dimensions_share_a_cache_path_without_clashing(tmp_path, monkeypatch):
    from benchmarks.pdf_extraction.downstream.retrieval import DocuMindEmbedder

    class Inner:
        def __init__(self, dim):
            self.dim = dim

        def embed_chunks(self, chunks, batch_size=16):
            return [[1.0] * self.dim for _ in chunks]

        def embed_query(self, text):
            return [1.0] * self.dim

    monkeypatch.setattr(DocuMindEmbedder, "_load_model", lambda self, name, device: Inner(1024 if "large" in name else 384))
    cache = tmp_path / "embedding_cache.npz"
    large = DocuMindEmbedder("bge-large", cache_path=cache)
    large.embed_documents(["a", "b"]); large.save()
    small = DocuMindEmbedder("bge-small", cache_path=cache)
    small.embed_documents(["a", "b", "c"]); small.save()  # must not raise
    assert DocuMindEmbedder("bge-large", cache_path=cache).cached_count() == 2
    assert DocuMindEmbedder("bge-small", cache_path=cache).cached_count() == 3
    assert len(list(tmp_path.glob("embedding_cache.*.npz"))) == 2


def test_structure_preserving_chunker_honours_an_injected_token_counter():
    """The budget must be measured in the embedder's own tokenizer, not a fixed encoding, or table chunks overflow it."""
    from benchmarks.pdf_extraction.downstream.chunking import CHUNKERS, chunk_structure_preserving

    text = "\n\n".join(f"Paragraph {i} " + " ".join(f"w{j}" for j in range(20)) for i in range(6))
    words = lambda t: len(t.split())  # noqa: E731  (a deliberately different tokenizer)
    default = chunk_structure_preserving(text, max_tokens=30)
    custom = chunk_structure_preserving(text, max_tokens=30, count=words)
    assert all(words(c) <= 30 for c in custom)
    assert custom != default  # the counter changed where the splits fall
    assert CHUNKERS["structure_preserving"]("doc", text, count=words, max_tokens=30) == custom
    assert all(words(c) <= 30 for c in CHUNKERS["baseline"]("doc", text, count=words, max_tokens=30))


def test_row_cache_is_keyed_on_the_extracted_text_not_on_how_it_was_produced(tmp_path):
    """The same result.md scored on another machine (different caps, hence a different input hash) must reuse the rows:
    retrieval depends only on the text and the questions."""
    import json

    _write_run(tmp_path / "out", "ex", "doc", "Sales: North 1,200 units")
    questions = [_q("doc", "north 1200?", ["North", "1200"])]
    spy = SpyEmbedder()
    first = score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_dir=tmp_path / "cache")
    run_json = tmp_path / "out" / "ex" / "doc" / "run.json"
    record = json.loads(run_json.read_text(encoding="utf-8"))
    record["input_hash"] = "another-machine-another-cap"
    run_json.write_text(json.dumps(record), encoding="utf-8")
    second = score_retrieval(tmp_path / "out", questions, [ENTRY], "baseline", spy, OverlapReranker(), cache_dir=tmp_path / "cache")
    assert second == first and spy.document_calls == 1
