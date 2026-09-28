"""Figure-fact recall and retrieval-evidence matching (deterministic, no LLM judge)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from benchmarks.pdf_extraction.metrics.text import fold_unicode

#: Fuzzy substring threshold for long strings (tolerates OCR noise). Short strings need an exact match.
FUZZY_THRESHOLD = 90
FUZZY_MIN_LEN = 12

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_SPACES = re.compile(r"\s+")


def normalize_fact(text: str) -> str:
    return _SPACES.sub(" ", fold_unicode(text).replace(" ", " ")).strip().lower()


def _parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for token in _NUMBER.findall(text):
        value = _parse_number(token.rstrip(".,"))
        if value is not None:
            found.add(value)
    return found


def fact_present(text: str, fact: str) -> bool:
    """True when ``fact`` appears in ``text``.

    Numeric facts match any number in the text with the same value ("1,234" == "1234").
    Long text facts allow a fuzzy match to tolerate OCR noise; short ones must match exactly.
    """
    norm_text, norm_fact = normalize_fact(text), normalize_fact(fact)
    if not norm_fact:
        return True

    fact_number = _parse_number(norm_fact) if _NUMBER.fullmatch(norm_fact) else None
    if fact_number is not None:
        return fact_number in _numbers_in(norm_text)

    if norm_fact in norm_text:
        return True
    if len(norm_fact) >= FUZZY_MIN_LEN:
        return fuzz.partial_ratio(norm_fact, norm_text) >= FUZZY_THRESHOLD
    return False


def fact_recall(text: str, facts: list[str]) -> float:
    """Fraction of ``facts`` present in ``text``. No facts scores 0.0 (nothing to recall)."""
    if not facts:
        return 0.0
    return sum(1 for fact in facts if fact_present(text, fact)) / len(facts)


def evidence_in_chunk(chunk: str, evidence: list[str]) -> bool:
    """True when *every* evidence string appears in this single chunk."""
    return all(fact_present(chunk, item) for item in evidence)


def evidence_hit(chunks: list[str], evidence: list[str], k: int) -> bool:
    """True when some chunk among the first ``k`` contains all evidence strings."""
    if not evidence:
        return False
    return any(evidence_in_chunk(chunk, evidence) for chunk in chunks[:k])
