"""Check hand-authored questions against the source PDFs, independently of every benchmarked extractor.

    python -m benchmarks.pdf_extraction.corpus.verify

Each text-evidence string must appear on the stated page of the PDF's own text layer. Questions whose
evidence lives only in pixels (evidence_source: visual) or whose document has no text layer are reported
as "needs human check" instead of being silently trusted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from benchmarks.pdf_extraction.corpus.questions import Question, load_all_questions
from benchmarks.pdf_extraction.metrics.facts import fact_present
from benchmarks.pdf_extraction.paths import GENERATED_QUESTIONS_DIR, PDF_DIR, QUESTIONS_DIR


@dataclass
class Verification:
    problems: list[str] = field(default_factory=list)
    needs_human: list[str] = field(default_factory=list)
    verified: int = 0


def verify_questions(questions: list[Question], pdf_dir: Path) -> Verification:
    result = Verification()
    for q in questions:
        if q.type == "unanswerable":
            continue
        pdf_path = pdf_dir / f"{q.doc_id}.pdf"
        if not pdf_path.exists():
            result.problems.append(f"{q.id}: {pdf_path.name} does not exist")
            continue
        with pymupdf.open(pdf_path) as pdf:
            if q.page > len(pdf):
                result.problems.append(f"{q.id}: page {q.page} is beyond the {len(pdf)}-page document")
                continue
            page_texts = [page.get_text() for page in pdf]
        if q.evidence_source == "visual" or not any(t.strip() for t in page_texts):
            result.needs_human.append(f"{q.id}: evidence is not in a text layer; check {q.evidence} on page {q.page} by eye")
            continue
        scope = "\n".join(page_texts) if q.type == "cross_page" else page_texts[q.page - 1]
        missing = [e for e in q.evidence if not fact_present(scope, e)]
        if missing:
            where = "the document" if q.type == "cross_page" else f"page {q.page}"
            result.problems.append(f"{q.id}: evidence {missing} not found on {where}")
        else:
            result.verified += 1
    return result


def main() -> int:
    questions = load_all_questions(QUESTIONS_DIR, GENERATED_QUESTIONS_DIR)
    result = verify_questions(questions, PDF_DIR)
    for line in result.problems:
        print(f"[problem] {line}", file=sys.stderr)
    for line in result.needs_human:
        print(f"[human] {line}")
    print(f"{result.verified} verified, {len(result.needs_human)} need a human check, {len(result.problems)} problems")
    return 1 if result.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
