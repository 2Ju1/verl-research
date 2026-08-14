#!/usr/bin/env python3
"""Render the bilingual offloading Markdown reports as styled PDFs."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline_markup(text: str) -> str:
    """Convert the small inline-Markdown subset used by the reports."""
    escaped = html.escape(text.strip())
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1f5f99">\1</link>', escaped)
    return escaped


def make_styles(korean: bool):
    base = getSampleStyleSheet()
    if korean:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
        regular, bold = "HYSMyeongJo-Medium", "HYGothic-Medium"
    else:
        regular, bold = "Helvetica", "Helvetica-Bold"

    styles = {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName=bold, fontSize=21,
            leading=27, textColor=colors.HexColor("#153A5B"), spaceAfter=12 * mm,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=bold, fontSize=15,
            leading=20, textColor=colors.HexColor("#153A5B"), spaceBefore=7 * mm,
            spaceAfter=3 * mm, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=bold, fontSize=12,
            leading=16, textColor=colors.HexColor("#245D87"), spaceBefore=5 * mm,
            spaceAfter=2.5 * mm, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName=bold, fontSize=10.5,
            leading=14, textColor=colors.HexColor("#333333"), spaceBefore=4 * mm,
            spaceAfter=2 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=regular, fontSize=8.8,
            leading=13.2, alignment=TA_JUSTIFY, textColor=colors.HexColor("#20252A"),
            spaceAfter=2.2 * mm, splitLongWords=True,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName=regular, fontSize=8.6,
            leading=12.8, leftIndent=3 * mm, firstLineIndent=0, spaceAfter=1 * mm,
        ),
        "quote": ParagraphStyle(
            "Quote", parent=base["BodyText"], fontName=bold, fontSize=9.3,
            leading=14, leftIndent=7 * mm, rightIndent=5 * mm,
            borderColor=colors.HexColor("#4C78A8"), borderWidth=1.5,
            borderPadding=(2 * mm, 3 * mm, 2 * mm, 4 * mm),
            backColor=colors.HexColor("#EEF4F8"), spaceBefore=2 * mm, spaceAfter=3 * mm,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier", fontSize=7.5,
            leading=10, leftIndent=4 * mm, rightIndent=4 * mm,
            backColor=colors.HexColor("#F3F5F7"), borderPadding=3 * mm,
            spaceBefore=2 * mm, spaceAfter=3 * mm,
        ),
        "table": ParagraphStyle(
            "TableCell", parent=base["BodyText"], fontName=regular, fontSize=6.7,
            leading=9.1, textColor=colors.HexColor("#20252A"),
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName=bold, fontSize=6.8,
            leading=9.2, textColor=colors.white, alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["BodyText"], fontName=regular, fontSize=7,
            leading=9, textColor=colors.HexColor("#666666"), alignment=TA_CENTER,
        ),
    }
    return styles, regular


def is_separator(row: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row)


def table_flowable(rows: list[list[str]], styles, width: float):
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        style = styles["table_head"] if row_index == 0 else styles["table"]
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(data, colWidths=[width / columns] * columns, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315E7D")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C4CC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def parse_markdown(text: str, styles, content_width: float):
    lines = text.splitlines()
    story = []
    index = 0
    first_heading = True

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            story.append(Preformatted("\n".join(code), styles["code"])); continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            key = "title" if first_heading and level == 1 else {1: "h1", 2: "h1", 3: "h2", 4: "h3"}[level]
            story.append(Paragraph(inline_markup(heading.group(2)), styles[key]))
            first_heading = False
            index += 1; continue

        if stripped.startswith("|") and index + 1 < len(lines):
            candidate = [cell.strip() for cell in stripped.strip("|").split("|")]
            separator = [cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")]
            if is_separator(separator):
                rows = [candidate]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                    index += 1
                story.extend([table_flowable(rows, styles, content_width), Spacer(1, 3 * mm)])
                continue

        if re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
            ordered = bool(re.match(r"^\d+\.\s+", stripped))
            items = []
            pattern = r"^\d+\.\s+" if ordered else r"^[-*]\s+"
            while index < len(lines) and re.match(pattern, lines[index].strip()):
                content = re.sub(pattern, "", lines[index].strip())
                items.append(ListItem(Paragraph(inline_markup(content), styles["bullet"])))
                index += 1
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet", leftIndent=7 * mm, bulletFontSize=7))
            story.append(Spacer(1, 1.5 * mm)); continue

        if stripped.startswith(">"):
            quote = stripped.lstrip(">").strip()
            story.append(Paragraph(inline_markup(quote), styles["quote"]))
            index += 1; continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt or nxt.startswith(("#", "|", "```", ">")) or re.match(r"^[-*]\s+|^\d+\.\s+", nxt):
                break
            paragraph.append(nxt)
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), styles["body"]))

    return story


def render(source: Path, destination: Path, korean: bool):
    styles, regular = make_styles(korean)
    doc = SimpleDocTemplate(
        str(destination), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=17 * mm, bottomMargin=16 * mm,
        title=source.stem, author="VERL Offloading Study",
    )

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5DC"))
        canvas.line(16 * mm, 12 * mm, A4[0] - 16 * mm, 12 * mm)
        canvas.setFont(regular, 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(A4[0] / 2, 7.5 * mm, f"{source.stem}  |  {document.page}")
        canvas.restoreState()

    story = parse_markdown(source.read_text(encoding="utf-8"), styles, doc.width)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--korean", action="store_true")
    args = parser.parse_args()
    render(args.source, args.destination, args.korean)
    print(f"Saved: {args.destination}")


if __name__ == "__main__":
    main()
