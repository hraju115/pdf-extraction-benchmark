"""Table metrics: TEDS (tree-edit-distance similarity) and table extraction from Markdown output."""

from __future__ import annotations

import html
import re

from apted import APTED, Config
from bs4 import BeautifulSoup
from rapidfuzz.distance import Levenshtein

from benchmarks.pdf_extraction.metrics.text import fold_unicode

#: A truth table counts as "detected" when its best-matching predicted table scores at least this.
DETECTION_THRESHOLD = 0.2
#: TEDS is only computed for pairs sharing at least this fraction of the truth table's cell texts. Cheap, and it
#: stops a same-shaped table with unrelated content from counting as a detection.
OVERLAP_MIN = 0.2


class _Node:
    def __init__(self, tag: str, colspan: int = 1, rowspan: int = 1, content: str = "") -> None:
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children: list[_Node] = []

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)


class _TedsConfig(Config):
    def rename(self, node1: _Node, node2: _Node) -> float:
        if node1.tag != node2.tag or node1.colspan != node2.colspan or node1.rowspan != node2.rowspan:
            return 1.0
        if node1.tag == "td" and (node1.content or node2.content):
            return Levenshtein.normalized_distance(node1.content, node2.content)
        return 0.0

    def children(self, node: _Node) -> list[_Node]:
        return node.children


def _span(value: object) -> int:
    try:
        return max(1, int(str(value)))
    except ValueError:
        return 1


def _cell_text(cell) -> str:
    return re.sub(r"\s+", " ", fold_unicode(cell.get_text(" ", strip=True))).strip().lower()


def _to_tree(table_html: str) -> _Node | None:
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return None
    root = _Node("table")
    for row in table.find_all("tr"):
        row_node = _Node("tr")
        for cell in row.find_all(["td", "th"], recursive=False):
            row_node.children.append(
                _Node("td", _span(cell.get("colspan", 1)), _span(cell.get("rowspan", 1)), _cell_text(cell))
            )
        root.children.append(row_node)
    return root


def teds(pred_html: str, truth_html: str) -> float:
    """TEDS similarity in [0, 1]. 1.0 means identical structure and cell text; unparsable input scores 0."""
    pred, truth = _to_tree(pred_html), _to_tree(truth_html)
    if pred is None or truth is None:
        return 0.0
    distance = APTED(pred, truth, _TedsConfig()).compute_edit_distance()
    return max(0.0, 1.0 - distance / max(pred.size(), truth.size()))


# ── Table extraction from Markdown ────────────────────────────────────────────

_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_PIPE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_pipe_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _pipe_table_to_html(lines: list[str]) -> str:
    rows = [_split_pipe_row(line) for line in lines if not _PIPE_SEPARATOR.match(line)]
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table>{body}</table>"


def extract_tables(markdown: str) -> list[str]:
    """Return every table in ``markdown`` (HTML tables and Markdown pipe tables) as HTML strings."""
    tables: list[str] = []
    for match in _HTML_TABLE.finditer(markdown):
        tables.append(match.group(0))
    without_html = _HTML_TABLE.sub("\n", markdown)

    block: list[str] = []
    for line in without_html.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 2 and any(_PIPE_SEPARATOR.match(row) for row in block):
            tables.append(_pipe_table_to_html(block))
        block = []
    return tables


def _cells(table_html: str) -> list[str]:
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return []
    return [t for t in (_cell_text(c) for c in table.find_all(["td", "th"])) if t]


def cell_overlap(pred_html: str, truth_html: str) -> float:
    """Fraction of the truth table's distinct cell texts that occur among the predicted table's cells."""
    truth_cells = set(_cells(truth_html))
    if not truth_cells:
        return 0.0
    pred_text = " | ".join(_cells(pred_html))
    return sum(1 for cell in truth_cells if cell in pred_text) / len(truth_cells)


def score_tables(pred_tables: list[str], truth_tables: list[str]) -> dict[str, float]:
    """Greedy one-to-one matching of predicted tables to truth tables by TEDS.

    Only pairs whose cell texts overlap by at least OVERLAP_MIN are compared (see ``cell_overlap``).
    Returns ``teds`` (mean over truth tables; a truth table with no match scores 0) and
    ``detection_recall`` (fraction of truth tables matched at or above DETECTION_THRESHOLD).
    """
    if not truth_tables:
        return {"teds": 0.0, "detection_recall": 0.0}

    pred_cells = [" | ".join(_cells(p)) for p in pred_tables]
    truth_cells = [set(_cells(t)) for t in truth_tables]
    candidates = []
    for ti, truth in enumerate(truth_tables):
        cells = truth_cells[ti]
        for pi, pred in enumerate(pred_tables):
            overlap = sum(1 for c in cells if c in pred_cells[pi]) / len(cells) if cells else 0.0
            if overlap >= OVERLAP_MIN:
                candidates.append((teds(pred, truth), ti, pi))
    candidates.sort(reverse=True)

    best: dict[int, float] = {}
    used_pred: set[int] = set()
    for score, ti, pi in candidates:
        if ti in best or pi in used_pred:
            continue
        best[ti] = score
        used_pred.add(pi)

    scores = [best.get(ti, 0.0) for ti in range(len(truth_tables))]
    return {
        "teds": sum(scores) / len(scores),
        "detection_recall": sum(1 for s in scores if s >= DETECTION_THRESHOLD) / len(scores),
    }
