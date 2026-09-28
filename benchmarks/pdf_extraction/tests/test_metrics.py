"""Tests for the deterministic extraction and retrieval metrics."""

from __future__ import annotations

import pytest

from benchmarks.pdf_extraction.metrics.facts import (
    evidence_hit,
    fact_present,
    fact_recall,
)
from benchmarks.pdf_extraction.metrics.tables import cell_overlap, extract_tables, score_tables, teds
from benchmarks.pdf_extraction.metrics.text import (
    cer,
    edit_similarity,
    mean_page_cer,
    mean_page_similarity,
    normalize_text,
)

# ── text ─────────────────────────────────────────────────────────────────────


def test_normalize_strips_markup_and_case():
    md = "# Title\n\n**Bold** [link](http://x.io) ![img](a.png)\n\n| a | b |"
    assert normalize_text(md) == "title bold link a b"


def test_identical_text_scores_one_and_different_scores_lower():
    assert edit_similarity("Hello World", "hello   world") == 1.0
    assert edit_similarity("hello world", "goodbye moon") < 0.5


def test_empty_vs_empty_is_perfect_and_empty_vs_text_is_zero():
    assert edit_similarity("", "") == 1.0
    assert edit_similarity("", "some text") == 0.0


def test_cer_counts_character_edits():
    assert cer("abcd", "abcd") == 0.0
    assert cer("abxd", "abcd") == pytest.approx(0.25)
    assert cer("anything", "") == 1.0
    assert cer("", "") == 0.0


def test_missing_page_scores_zero_in_page_means():
    truth = {1: "alpha beta", 2: "gamma delta"}
    pred = {1: "alpha beta"}
    assert mean_page_similarity(pred, truth) == pytest.approx(0.5)
    assert mean_page_cer(pred, truth) == pytest.approx((0.0 + 1.0) / 2)
    assert mean_page_similarity(pred, {}) is None  # nothing to score


# ── tables ───────────────────────────────────────────────────────────────────

TRUTH = "<table><tr><td>Region</td><td>Sales</td></tr><tr><td>North</td><td>1,200</td></tr></table>"


def test_identical_tables_score_one():
    assert teds(TRUTH, TRUTH) == pytest.approx(1.0)


def test_wrong_cell_text_lowers_score_partially():
    wrong = TRUTH.replace("1,200", "1,900")
    score = teds(wrong, TRUTH)
    assert 0.5 < score < 1.0


def test_missing_row_lowers_score_more_than_a_typo():
    missing_row = "<table><tr><td>Region</td><td>Sales</td></tr></table>"
    typo = TRUTH.replace("North", "Nort")
    assert teds(missing_row, TRUTH) < teds(typo, TRUTH) < 1.0


def test_merged_cell_structure_is_penalized():
    truth = "<table><tr><td colspan='2'>Header</td></tr><tr><td>a</td><td>b</td></tr></table>"
    flat = "<table><tr><td>Header</td><td></td></tr><tr><td>a</td><td>b</td></tr></table>"
    assert teds(flat, truth) < 1.0


def test_unparsable_input_scores_zero():
    assert teds("no table here", TRUTH) == 0.0
    assert teds(TRUTH, "") == 0.0


def test_extract_tables_finds_pipe_and_html_tables():
    md = (
        "Intro\n\n"
        "| Region | Sales |\n|---|---|\n| North | 1,200 |\n\n"
        "text between\n\n"
        "<table><tr><td>A</td></tr></table>\n"
    )
    tables = extract_tables(md)
    assert len(tables) == 2
    assert any("<td>North</td>" in t for t in tables)
    assert any("<td>A</td>" in t for t in tables)


def test_extract_tables_ignores_lone_pipe_lines_without_separator():
    assert extract_tables("| just | a | line |\n\nplain") == []


def test_pipe_table_scores_perfectly_against_equivalent_truth():
    md = "| Region | Sales |\n|---|---|\n| North | 1,200 |"
    result = score_tables(extract_tables(md), [TRUTH])
    assert result["teds"] == pytest.approx(1.0)
    assert result["detection_recall"] == 1.0


def test_score_tables_missing_prediction_is_zero_and_matching_is_one_to_one():
    assert score_tables([], [TRUTH]) == {"teds": 0.0, "detection_recall": 0.0}
    # One predicted table cannot satisfy two identical truth tables.
    result = score_tables([TRUTH], [TRUTH, TRUTH])
    assert result["teds"] == pytest.approx(0.5)
    assert result["detection_recall"] == pytest.approx(0.5)
    assert score_tables([TRUTH], []) == {"teds": 0.0, "detection_recall": 0.0}


# ── facts and evidence ───────────────────────────────────────────────────────


def test_numeric_facts_ignore_thousands_separators():
    assert fact_present("Revenue was 1,234 in Q1", "1234")
    assert fact_present("Revenue was 1234 in Q1", "1,234")
    assert not fact_present("Revenue was 1,235 in Q1", "1234")


def test_number_does_not_match_inside_longer_number():
    assert not fact_present("total 12345", "123")


def test_short_text_facts_need_exact_match_but_long_ones_tolerate_ocr_noise():
    assert fact_present("the Ingest stage", "ingest")
    assert not fact_present("the lngest stage", "ingest")
    assert fact_present("Quarterly Revenue by Reglon", "quarterly revenue by region")


def test_fact_recall_fraction_and_empty_facts():
    text = "Bar chart: Alpha 10, Beta 20"
    assert fact_recall(text, ["alpha", "beta", "gamma", "20"]) == pytest.approx(0.75)
    assert fact_recall(text, []) == 0.0


def test_evidence_must_be_in_one_chunk_within_top_k():
    chunks = ["North region", "sales 1,200 units", "North 1,200"]
    assert not evidence_hit(chunks, ["north", "1200"], k=2)  # split across chunks 1 and 2
    assert evidence_hit(chunks, ["north", "1200"], k=3)
    assert not evidence_hit(chunks, [], k=3)
    assert not evidence_hit([], ["x"], k=3)


def test_non_ascii_text_is_handled_consistently():
    assert edit_similarity("Café société — naïve", "café société — naïve") == 1.0
    assert edit_similarity("日本語のテキスト", "日本語のテキスト") == 1.0
    assert 0.0 < edit_similarity("Café", "Cafe") < 1.0  # a dropped accent is a small error, not a crash
    assert cer("résumé", "résumé") == 0.0
    assert fact_present("Total: 1 234,5 €", "€")
    assert fact_present("Überblick über die Ergebnisse der Studie", "überblick über die ergebnisse")


# ── unicode normalization, empty pages, table pre-filter ─────────────────────


def test_normalize_maps_ligatures_dashes_quotes_and_drops_separator_rows():
    assert normalize_text("ﬁnancial ﬂow") == "financial flow"  # NFKC ligatures
    assert normalize_text("–8 — “quoted” ‘x’") == "-8 - \"quoted\" 'x'"
    assert normalize_text("| a | b |\n|---|---|\n| 1 | 2 |") == "a b 1 2"


def test_dash_variants_match_in_facts_and_evidence():
    assert fact_present("change –8 points", "-8")
    assert fact_present("change -8 points", "–8")
    assert fact_present("Q4 − Q1", "Q4 - Q1")


def test_pages_with_empty_truth_are_not_scored():
    truth = {1: "alpha beta", 2: "", 3: "   "}
    assert mean_page_similarity({1: "alpha beta"}, truth) == 1.0  # page 2/3 have nothing to read
    assert mean_page_similarity({1: "alpha beta", 2: "ocr noise"}, truth) == 1.0
    assert mean_page_cer({1: "alpha beta"}, truth) == 0.0
    assert mean_page_similarity({}, {1: "", 2: ""}) is None  # nothing scorable at all
    assert mean_page_cer({}, {1: ""}) is None


def test_cell_overlap_and_prefilter_exclude_text_free_matches():
    same_shape = "<table><tr><td>Fruit</td><td>Kg</td></tr><tr><td>Apple</td><td>7</td></tr></table>"
    assert cell_overlap(same_shape, TRUTH) == 0.0
    assert cell_overlap(TRUTH.replace("North", "Nort"), TRUTH) >= 0.5
    # A same-shaped table with entirely different text is not a detection of the truth table.
    result = score_tables([same_shape], [TRUTH])
    assert result["teds"] == 0.0 and result["detection_recall"] == 0.0


def test_score_tables_is_fast_with_many_unrelated_candidates():
    import time

    junk = [f"<table><tr><td>k{i}</td><td>v{i}</td></tr><tr><td>x{i}</td><td>{i}</td></tr></table>" for i in range(300)]
    big_truth = "<table>" + "".join(f"<tr><td>row {r}</td>" + "".join(f"<td>{r * c}</td>" for c in range(1, 8)) + "</tr>" for r in range(40)) + "</table>"
    started = time.perf_counter()
    result = score_tables(junk + [TRUTH], [TRUTH, big_truth, big_truth])
    assert time.perf_counter() - started < 3.0
    assert result["detection_recall"] == pytest.approx(1 / 3)
