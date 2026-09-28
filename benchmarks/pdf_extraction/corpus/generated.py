"""Programmatically generated PDFs with *exact* table and figure ground truth.

Tables come from a HTML-style cell spec, so merged cells are known precisely. Charts and diagrams are
rasterized to PNG before being placed in the PDF, so their labels and values exist only as pixels.
"""

from __future__ import annotations

import html
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from benchmarks.pdf_extraction.corpus.questions import write_question_file  # noqa: E402

SEED = 1234

#: Every generated document id and its tier. The build must produce exactly these (checked in tests).
GENERATED_TIERS: dict[str, str] = {
    "gen_t2_simple": "T2",
    "gen_t2_merged": "T2",
    "gen_t2_long": "T2",
    "gen_t2_rotated": "T2",
    "gen_t4_bar": "T4",
    "gen_t4_line": "T4",
    "gen_t4_flow": "T4",
    "gen_t4_mixed": "T4",
}
REGIONS = ["North", "South", "East", "West", "Central", "Coastal", "Highland", "Island"]
PRODUCTS = ["Widget", "Gadget", "Sprocket", "Gizmo", "Bracket", "Coupler", "Flange", "Valve"]
#: Body text that gives generated pages a realistic amount of content. It has no digits and none of the
#: chart vocabulary below, so it can never leak a figure fact (checked by a test).
PROSE = (
    "This summary reports figures for the period under review. Values are provisional and were supplied by "
    "regional offices. Differences between periods reflect seasonal demand, promotions and supply constraints, "
    "and the tables and charts that follow should be read together with the notes to this document."
)

#: Words that appear only inside chart images, never in a table or in page text (checked by a test).
CHART_CATEGORIES = ["Apex", "Bolt", "Crest", "Delta", "Ember"]


@dataclass
class Cell:
    text: str
    colspan: int = 1
    rowspan: int = 1


TableSpec = list[list[Cell]]  # rows of *originating* cells, exactly like HTML


def table_to_html(spec: TableSpec) -> str:
    rows = []
    for row in spec:
        cells = "".join(
            "<td{}{}>{}</td>".format(
                f' colspan="{c.colspan}"' if c.colspan > 1 else "",
                f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "",
                html.escape(c.text),
            )
            for c in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def layout_grid(spec: TableSpec) -> tuple[list[list[str]], list[tuple]]:
    """Turn a cell spec into a reportlab grid plus SPAN commands, honoring rowspan occupancy."""
    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[int, int, Cell]] = []
    width = 0
    for r, row in enumerate(spec):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            placements.append((r, c, cell))
            for dr in range(cell.rowspan):
                for dc in range(cell.colspan):
                    occupied.add((r + dr, c + dc))
            c += cell.colspan
            width = max(width, c)
    height = max(r for r, _ in occupied) + 1
    grid = [[""] * width for _ in range(height)]
    spans: list[tuple] = []
    for r, c, cell in placements:
        grid[r][c] = cell.text
        if cell.colspan > 1 or cell.rowspan > 1:
            spans.append(("SPAN", (c, r), (c + cell.colspan - 1, r + cell.rowspan - 1)))
    return grid, spans


def _reportlab_table(spec: TableSpec, repeat_header: bool = False) -> Table:
    grid, spans = layout_grid(spec)
    width = 17 * cm  # full text width: a table in the top-left corner of a blank page is not a realistic page
    table = Table(grid, colWidths=[width / len(grid[0])] * len(grid[0]), repeatRows=1 if repeat_header else 0)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                *spans,
            ]
        )
    )
    return table


# ── Table document data ──────────────────────────────────────────────────────


def _sales_table(rng: random.Random, rows: int, header_prefix: str = "Sales") -> TableSpec:
    header = [Cell("Region")] + [Cell(f"{header_prefix} {q}") for q in ("Q1", "Q2", "Q3")]
    body = []
    for i in range(rows):
        label = f"{REGIONS[i % len(REGIONS)]}-{i // len(REGIONS) + 1:02d}" if rows > len(REGIONS) else REGIONS[i]
        body.append([Cell(label)] + [Cell(f"{rng.randint(1000, 99999):,}") for _ in range(3)])
    return [header] + body


def _merged_table(rng: random.Random) -> TableSpec:
    return [
        [Cell("Product", rowspan=2), Cell("Units sold", colspan=3), Cell("Return rate", rowspan=2)],
        [Cell("2022"), Cell("2023"), Cell("2024")],
        *[
            [Cell(PRODUCTS[i])] + [Cell(f"{rng.randint(100, 9999):,}") for _ in range(3)] + [Cell(f"{rng.randint(1, 15)}%")]
            for i in range(5)
        ],
    ]


# ── Chart / diagram images ───────────────────────────────────────────────────


def chart_value(rng: random.Random, low: int = 11, high: int = 98) -> int:
    """A chart value that is never a multiple of 5, so it cannot coincide with an axis tick label."""
    return rng.choice([v for v in range(low, high + 1) if v % 5])


def _bar_chart(png: Path, rng: random.Random) -> dict:
    cats = CHART_CATEGORIES
    values = [chart_value(rng) for _ in cats]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(cats, values, color="#4477aa")
    ax.bar_label(bars)
    title, ylabel = "Throughput by team", "Crates (thousands)"
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    facts = [title, ylabel, *cats, *[str(v) for v in values]]
    return {"title": title, "categories": cats, "values": values, "facts": facts}


def _line_chart(png: Path, rng: random.Random) -> dict:
    years = [2020, 2021, 2022, 2023, 2024]
    series = {name: [chart_value(rng, 20, 90) for _ in years] for name in ("Alpha", "Beta", "Gamma")}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, ys in series.items():
        ax.plot(years, ys, marker="o", label=name)
        ax.annotate(str(ys[-1]), (years[-1], ys[-1]), textcoords="offset points", xytext=(6, 0))
    title, ylabel = "Adoption over time", "Score"
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    facts = [title, ylabel, *series, *[str(ys[-1]) for ys in series.values()]]
    return {"title": title, "series": {k: v[-1] for k, v in series.items()}, "facts": facts}


def _flowchart(png: Path, rng: random.Random) -> dict:
    steps = ["Collect", "Normalize", "Segment", "Vectorize", "Store"]
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.set_xlim(0, len(steps) * 2)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, step in enumerate(steps):
        x = i * 2 + 0.2
        ax.add_patch(plt.Rectangle((x, 0.6), 1.5, 0.8, fill=False, linewidth=1.5))
        ax.text(x + 0.75, 1.0, step, ha="center", va="center", fontsize=11)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.0, 1.0), xytext=(x + 1.5, 1.0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"title": "Stage sequence", "steps": steps, "facts": list(steps)}


# ── Document builders ────────────────────────────────────────────────────────


@dataclass
class GeneratedDoc:
    id: str
    tier: str
    bucket: str = "S"
    tables: list[str] = field(default_factory=list)  # ground-truth HTML
    figures: list[dict] = field(default_factory=list)  # {"id", "page", "facts", ...}
    question_seeds: list[dict] = field(default_factory=list)  # resolved to pages after rendering
    rotate: int = 0


def _story_styles():
    styles = getSampleStyleSheet()
    return styles["Title"], styles["BodyText"], styles["Italic"]


def _build_pdf(path: Path, story: list) -> None:
    """Render ``story`` (title first), framed by body text so the page looks like a real document page."""
    body = getSampleStyleSheet()["BodyText"]
    framed = [story[0], Paragraph(PROSE, body), Spacer(1, 12), *story[1:], Spacer(1, 12), Paragraph(PROSE, body)]
    SimpleDocTemplate(str(path), pagesize=A4, invariant=1, leftMargin=2 * cm, rightMargin=2 * cm).build(framed)


def _table_questions(doc_id: str, spec: TableSpec, rng: random.Random, count: int = 3) -> list[dict]:
    header = [c.text for c in spec[0]]
    body = spec[1:]
    picks = rng.sample(range(len(body)), k=min(count, len(body)))
    seeds = []
    for r in picks:
        col = rng.randint(1, len(header) - 1)
        label, value = body[r][0].text, body[r][col].text
        seeds.append(
            {
                "question": f"In the table, what is the value for {label} under '{header[col]}'?",
                "answer": value,
                "evidence": [label, value],
                "type": "table_cell",
                "find": label,
            }
        )
    return seeds


def build_generated(pdf_dir: Path, seed: int = SEED) -> list[GeneratedDoc]:
    """Write every generated PDF into ``pdf_dir`` and return its ground truth (tables, figures, question seeds)."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    assets = pdf_dir / "_assets"
    assets.mkdir(exist_ok=True)
    title_style, body_style, caption_style = _story_styles()
    docs: list[GeneratedDoc] = []

    def rng_for(doc_id: str) -> random.Random:
        return random.Random(f"{seed}:{doc_id}")

    # 1. simple table
    doc = GeneratedDoc("gen_t2_simple", "T2")
    rng = rng_for(doc.id)
    spec = _sales_table(rng, 6)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Regional sales summary", title_style), Paragraph("Quarterly figures by region.", body_style),
         Spacer(1, 12), _reportlab_table(spec)],
    )
    doc.tables = [table_to_html(spec)]
    doc.question_seeds = _table_questions(doc.id, spec, rng)
    docs.append(doc)

    # 2. merged cells
    doc = GeneratedDoc("gen_t2_merged", "T2")
    spec = _merged_table(rng_for(doc.id))
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Product performance", title_style), Spacer(1, 12), _reportlab_table(spec)],
    )
    doc.tables = [table_to_html(spec)]
    rng = rng_for(doc.id + ":q")
    product_row = rng.randint(2, len(spec) - 1)
    label = spec[product_row][0].text
    doc.question_seeds = [
        {
            "question": f"What was the return rate for {label}?",
            "answer": spec[product_row][-1].text,
            "evidence": [label, spec[product_row][-1].text],
            "type": "table_cell",
            "find": label,
        }
    ]
    docs.append(doc)

    # 3. long multi-page table with a repeated header
    doc = GeneratedDoc("gen_t2_long", "T2")
    rng = rng_for(doc.id)
    spec = _sales_table(rng, 70)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Detailed sales ledger", title_style), Spacer(1, 12), _reportlab_table(spec, repeat_header=True)],
    )
    doc.tables = [table_to_html(spec)]
    doc.question_seeds = _table_questions(doc.id, spec, rng, count=4)
    docs.append(doc)

    # 4. rotated pages (the page is rotated 90 degrees after rendering)
    doc = GeneratedDoc("gen_t2_rotated", "T2", rotate=90)
    rng = rng_for(doc.id)
    spec = _sales_table(rng, 5, header_prefix="Revenue")
    _build_pdf(pdf_dir / f"{doc.id}.pdf", [Paragraph("Rotated revenue table", title_style), Spacer(1, 12), _reportlab_table(spec)])
    with pymupdf.open(pdf_dir / f"{doc.id}.pdf") as src:
        for page in src:
            page.set_rotation(90)
        src.save(pdf_dir / f"{doc.id}.tmp.pdf")
    (pdf_dir / f"{doc.id}.tmp.pdf").replace(pdf_dir / f"{doc.id}.pdf")
    doc.tables = [table_to_html(spec)]
    doc.question_seeds = _table_questions(doc.id, spec, rng, count=2)
    docs.append(doc)

    # 5. bar chart (pixels only)
    doc = GeneratedDoc("gen_t4_bar", "T4")
    rng = rng_for(doc.id)
    fig = _bar_chart(assets / f"{doc.id}.png", rng)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Quarterly review", title_style), Image(str(assets / f"{doc.id}.png"), width=15 * cm, height=9 * cm),
         Paragraph("Figure 1: see the chart above.", caption_style)],
    )
    doc.figures = [{"id": "fig1", "page": 1, "facts": fig["facts"]}]
    pick = rng.randrange(len(fig["categories"]))
    doc.question_seeds = [
        {
            "question": f"In the chart '{fig['title']}', what is the value for {fig['categories'][pick]}?",
            "answer": str(fig["values"][pick]),
            "evidence": [fig["categories"][pick], str(fig["values"][pick])],
            "type": "figure",
            "evidence_source": "visual",
            "page": 1,
        }
    ]
    docs.append(doc)

    # 6. line chart (pixels only)
    doc = GeneratedDoc("gen_t4_line", "T4")
    rng = rng_for(doc.id)
    fig = _line_chart(assets / f"{doc.id}.png", rng)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Trend review", title_style), Image(str(assets / f"{doc.id}.png"), width=15 * cm, height=9 * cm),
         Paragraph("Figure 1: see the chart above.", caption_style)],
    )
    doc.figures = [{"id": "fig1", "page": 1, "facts": fig["facts"]}]
    name = "Beta"
    doc.question_seeds = [
        {
            "question": f"What is the final 2024 value of the {name} series in '{fig['title']}'?",
            "answer": str(fig["series"][name]),
            "evidence": [name, str(fig["series"][name])],
            "type": "figure",
            "evidence_source": "visual",
            "page": 1,
        }
    ]
    docs.append(doc)

    # 7. flowchart (pixels only)
    doc = GeneratedDoc("gen_t4_flow", "T4")
    rng = rng_for(doc.id)
    fig = _flowchart(assets / f"{doc.id}.png", rng)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("System overview", title_style), Image(str(assets / f"{doc.id}.png"), width=16 * cm, height=5 * cm)],
    )
    doc.figures = [{"id": "fig1", "page": 1, "facts": fig["facts"]}]
    doc.question_seeds = [
        {
            "question": "In the stage diagram, which stage comes directly after Segment?",
            "answer": "Vectorize",
            "evidence": ["Segment", "Vectorize"],
            "type": "figure",
            "evidence_source": "visual",
            "page": 1,
        }
    ]
    docs.append(doc)

    # 8. mixed page: chart plus table
    doc = GeneratedDoc("gen_t4_mixed", "T4")
    rng = rng_for(doc.id)
    fig = _bar_chart(assets / f"{doc.id}.png", rng)
    spec = _sales_table(rng, 4)
    _build_pdf(
        pdf_dir / f"{doc.id}.pdf",
        [Paragraph("Combined report", title_style), Image(str(assets / f"{doc.id}.png"), width=12 * cm, height=7.2 * cm),
         Spacer(1, 12), _reportlab_table(spec)],
    )
    doc.tables = [table_to_html(spec)]
    doc.figures = [{"id": "fig1", "page": 1, "facts": fig["facts"]}]
    doc.question_seeds = _table_questions(doc.id, spec, rng, count=1)
    docs.append(doc)

    return docs


def _first_label(table_html: str) -> str:
    """Text of the first cell in the second row (the first body row's label)."""
    rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S)
    cells = re.findall(r"<td[^>]*>(.*?)</td>", rows[1] if len(rows) > 1 else rows[0], flags=re.S)
    return html.unescape(cells[0]) if cells else ""


def page_of_text(pdf_path: Path, needle: str) -> int:
    """First 1-based page whose text layer contains ``needle``; 1 when the text is pixels only."""
    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf, start=1):
            if needle in page.get_text():
                return index
    return 1


def write_truth(docs: list[GeneratedDoc], pdf_dir: Path, truth_dir: Path, questions_dir: Path) -> None:
    """Write tables/figures truth JSON and auto-generated question files for each generated document."""
    truth_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        pdf = pdf_dir / f"{doc.id}.pdf"
        if doc.tables:
            # Every generated table starts on the page where its first body-row label is printed.
            items = [{"page": page_of_text(pdf, _first_label(html)), "html": html} for html in doc.tables]
            (truth_dir / f"{doc.id}.tables.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
        if doc.figures:
            (truth_dir / f"{doc.id}.figures.json").write_text(json.dumps(doc.figures, indent=1), encoding="utf-8")

        questions = []
        for seed in doc.question_seeds:
            item = {k: v for k, v in seed.items() if k != "find"}
            item["page"] = seed.get("page") or page_of_text(pdf, seed["find"])
            questions.append(item)
        if questions:
            write_question_file(questions_dir / f"{doc.id}.yaml", questions)
