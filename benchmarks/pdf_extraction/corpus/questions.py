"""Question files: hand-authored (committed) and auto-generated (for generated documents)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

QUESTION_TYPES = ("lookup", "table_cell", "table_multi_row", "figure", "cross_page", "unanswerable")
EVIDENCE_SOURCES = ("text", "visual")  # visual = the fact exists only in image pixels, not in a text layer


@dataclass
class Question:
    doc_id: str
    question: str
    answer: str
    evidence: list[str]
    page: int
    type: str
    evidence_source: str = "text"
    id: str = ""


def _validate(doc_id: str, item: dict, index: int) -> Question:
    where = f"{doc_id} question #{index + 1}"
    missing = [k for k in ("question", "answer", "evidence", "page", "type") if k not in item]
    if missing:
        raise ValueError(f"{where}: missing {missing}")
    if item["type"] not in QUESTION_TYPES:
        raise ValueError(f"{where}: type {item['type']!r} not in {QUESTION_TYPES}")
    if item.get("evidence_source", "text") not in EVIDENCE_SOURCES:
        raise ValueError(f"{where}: evidence_source must be one of {EVIDENCE_SOURCES}")
    if item["type"] == "unanswerable":
        evidence: list[str] = []
    else:
        evidence = [str(e) for e in item["evidence"]]
        if not evidence:
            raise ValueError(f"{where}: answerable questions need at least one evidence string")
    if not isinstance(item["page"], int) or item["page"] < 1:
        raise ValueError(f"{where}: page must be an int >= 1")
    return Question(
        doc_id=doc_id,
        question=str(item["question"]),
        answer=str(item["answer"]),
        evidence=evidence,
        page=item["page"],
        type=item["type"],
        evidence_source=item.get("evidence_source", "text"),
        id=f"{doc_id}#{index + 1}",
    )


def load_question_file(path: Path) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    doc_id = path.stem
    return [_validate(doc_id, item, i) for i, item in enumerate(raw.get("questions", []))]


def write_question_file(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"questions": questions}, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_all_questions(*directories: Path) -> list[Question]:
    """Load every ``<doc_id>.yaml`` in the given directories (missing directories are skipped)."""
    questions: list[Question] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            questions.extend(load_question_file(path))
    return questions


def inherit_scan_questions(questions: list[Question], entries: list) -> list[Question]:
    """Scans reuse the questions of the document they were made from (unless they have their own)."""
    have_own = {q.doc_id for q in questions}
    inherited: list[Question] = []
    for entry in entries:
        if entry.source != "scan" or entry.id in have_own:
            continue
        for q in (q for q in questions if q.doc_id == entry.derived_from):
            inherited.append(replace(q, doc_id=entry.id, id=q.id.replace(entry.derived_from, entry.id, 1)))
    return questions + inherited
