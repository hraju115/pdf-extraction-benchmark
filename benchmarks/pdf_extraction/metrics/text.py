"""Text-fidelity metrics: normalized edit similarity and character error rate."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_PUNCT = re.compile(r"[#*_`>|]")
_SPACES = re.compile(r"\s+")
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = {**dict.fromkeys(map(ord, "\u201c\u201d\u201e\u00ab\u00bb"), '"'), **dict.fromkeys(map(ord, "\u2018\u2019\u201a"), "'")}


def fold_unicode(text: str) -> str:
    """NFKC (ligatures, full-width forms) plus one ASCII form for the many dashes and quotes PDFs use."""
    return unicodedata.normalize("NFKC", text).translate(_DASHES).translate(_QUOTES)


def normalize_text(text: str) -> str:
    """Strip Markdown/HTML markup, fold Unicode variants and collapse whitespace so formats compare fairly."""
    text = fold_unicode(text)
    text = _SEPARATOR_ROW.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _MD_PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def edit_similarity(pred: str, truth: str) -> float:
    """1.0 for identical normalized text, 0.0 for completely different. Two empty texts score 1.0."""
    p, t = normalize_text(pred), normalize_text(truth)
    if not p and not t:
        return 1.0
    return Levenshtein.normalized_similarity(p, t)


def cer(pred: str, truth: str) -> float:
    """Character error rate: edit distance / truth length. Can exceed 1.0 when pred is much longer."""
    p, t = normalize_text(pred), normalize_text(truth)
    if not t:
        return 0.0 if not p else 1.0
    return Levenshtein.distance(p, t) / len(t)


def _scorable(truth_pages: dict[int, str]) -> dict[int, str]:
    """Pages with no truth text (blank or pure-image pages) cannot be scored and are left out."""
    return {page: text for page, text in truth_pages.items() if text.strip()}


def mean_page_similarity(pred_pages: dict[int, str], truth_pages: dict[int, str]) -> float | None:
    """Average edit similarity over every scorable truth page; a page the extractor never produced scores 0.

    Returns None when no page has truth text.
    """
    truth = _scorable(truth_pages)
    if not truth:
        return None
    scores = [edit_similarity(pred_pages.get(page, ""), text) for page, text in truth.items()]
    return sum(scores) / len(scores)


def mean_page_cer(pred_pages: dict[int, str], truth_pages: dict[int, str]) -> float | None:
    truth = _scorable(truth_pages)
    if not truth:
        return None
    scores = [cer(pred_pages.get(page, ""), text) for page, text in truth.items()]
    return sum(scores) / len(scores)
