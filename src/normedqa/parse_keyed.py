"""Parse the published answer keys of the national joint final exam.

The keyed PDFs ("Revidert fasit") print every question in a fixed layout:

    <number>                         bold, left margin
    <case text and question>         left margin
    A X  <option text>               "X" marks the key, letter in bold at the left margin
        <explanation>                indented, usually in italics
    B    <option text>
    ...

When the key was changed at grading, the question starts with "NB Fasit ble
endret ...", the X marks the original key and a second column gives the points
awarded per option. All options at 0 points means the question was voided.

The parser reads positioned text lines with PyMuPDF and works as a small state
machine over rows (lines that share a baseline). It also cuts out the images
and assigns each to the question it appears in, before the options ("stem") or
among the explanations ("explanation").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

OPTION_LABELS = "ABCDEFGH"
# "NB Fasit ble endret ...", also "NB Fasit og begrunnelser ble endret ..." (2019V) and Nynorsk forms.
CHANGED_KEY = re.compile(r"NB\s*!?\s*Fasit\b.{0,40}?\b(ble|vart|blei|er)\s+endr", re.I)
# Note printed after the last question ("Testen har 140 oppgaver. På utskriftstidspunktet ...").
END_NOTE = re.compile(r"^(Testen|Prøven|Eksamen|Oppgavesettet) har \d+ oppg|På utskriftstidspunktet", re.I)
# Grading notes printed among the options ("jf. sensurmøtet" = according to the grading meeting).
GRADING_NOTE = re.compile(
    r"sensurm[øo]te|tas ut|tatt ut|vert teke ut|blir teke ut|korrekte svaralternativ|rette svaralternativ|"
    r"godkjent som riktig|godkjent som rett|godtatt som riktig|godteke som rett|gis poeng|gjev poeng",
    re.I,
)
VOIDED_NOTE = re.compile(r"tas ut|tatt ut|teke ut|tatt bort|strøke?t", re.I)
PAGE_NUMBER = re.compile(r"^-?\s*\d{1,3}\s*-?$")
MIN_IMAGE_W, MIN_IMAGE_H = 60, 40


@dataclass
class Row:
    page: int
    y: float
    x: float
    text: str
    bold_start: bool  # first visible span is bold
    italic: float  # share of characters in italic
    size: float
    red: float = 0.0  # share of characters in red: grading notes are printed in red
    main: str = ""  # text before an italic tail on the same line (option text followed by its explanation)
    tail: str = ""  # the italic tail


def _is_red(color: int) -> bool:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return r > 190 and g < 70 and b < 70


def _is_italic(span: dict) -> bool:
    return bool(span["flags"] & 2) or "Italic" in span["font"] or "Oblique" in span["font"]


@dataclass
class ParsedOption:
    label: str
    text: str = ""
    explanation: list[str] = field(default_factory=list)
    keyed: bool = False
    points: float | None = None


@dataclass
class ParsedImage:
    page: int
    y: float
    xref: int
    position: str  # "stem" or "explanation"
    width: int
    height: int


@dataclass
class ParsedQuestion:
    number: int
    pages: set[int] = field(default_factory=set)
    stem: list[str] = field(default_factory=list)
    options: list[ParsedOption] = field(default_factory=list)
    changed_key: bool = False
    notes: list[str] = field(default_factory=list)
    images: list[ParsedImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # layout problems a person should look at
    pending: dict[str, ParsedOption] = field(default_factory=dict)  # options found out of order
    skipping: bool = False  # inside a second print of the same question

    # Derived after parsing -------------------------------------------------
    @property
    def original_answer(self) -> list[str]:
        return [o.label for o in self.options if o.keyed]

    @property
    def voided(self) -> bool:
        if self.changed_key and self.options and all((o.points or 0) == 0 for o in self.options):
            return True
        return any(VOIDED_NOTE.search(n) for n in self.notes) and not self.answer

    @property
    def answer(self) -> list[str]:
        if self.changed_key and any(o.points is not None for o in self.options):
            return [o.label for o in self.options if (o.points or 0) > 0]
        return self.original_answer


SUPERSCRIPT = str.maketrans("0123456789+-−–=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁻⁼⁽⁾ⁿ")


def _span_text(span: dict) -> str:
    """Span text with superscripts kept: 10⁹/L, min⁻¹, m², 38 °C (flattening them changes the numbers)."""
    t = span["text"]
    if not span["flags"] & 1:
        return t
    core = t.strip()
    if core == "o":  # superscript o before C: degree sign
        return "°"
    if core and all(c in "0123456789+-−–=()n" for c in core):
        return t.replace(core, core.translate(SUPERSCRIPT))
    return t


def _rows(doc: pymupdf.Document) -> list[Row | tuple]:
    """Rows of text in reading order, plus image markers ("img", page, y, xref, w, h)."""
    out: list = []
    image_seen: dict[int, int] = {}
    for page in doc:
        for info in page.get_image_info(xrefs=True):
            image_seen[info["xref"]] = image_seen.get(info["xref"], 0) + 1
    for pno, page in enumerate(doc):
        lines = []
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for ln in block["lines"]:
                spans = [s for s in ln["spans"] if s["text"].strip()]
                if not spans or max(s["size"] for s in spans) < 6:  # footer hash in 3 pt
                    continue
                lines.append((ln["bbox"][1], ln["bbox"][0], ln["spans"], spans))
        lines.sort(key=lambda t: (round(t[0]), t[1]))
        rows: list[list] = []
        for y, x, all_spans, spans in lines:
            # Lines within 2.5 pt vertically belong to the same row (letter column + option text column).
            if rows and abs(rows[-1][0] - y) < 2.5:
                rows[-1][2].append((x, all_spans, spans))
            else:
                rows.append([y, x, [(x, all_spans, spans)]])
        for y, _, parts in rows:
            parts.sort(key=lambda p: p[0])
            # Cells of one row (table columns, letter + option text) are separated by a tab.
            cells = [re.sub(r"\s+", " ", "".join(_span_text(s) for s in all_spans)).strip() for _, all_spans, _ in parts]
            text = "\t".join(c for c in cells if c)
            if (PAGE_NUMBER.match(text) or text in ("-", "–")) and parts[0][0] > 200:
                continue  # centred page number (sometimes split into "-" and the number)
            first = parts[0][2][0]
            visible = [s for _, _, sp in parts for s in sp]
            chars = sum(len(s["text"]) for s in visible)
            ital = sum(len(s["text"]) for s in visible if _is_italic(s))
            red = sum(len(s["text"]) for s in visible if _is_red(s["color"]))
            # Split "option text + explanation" printed on one line. Explanations are italic and coloured;
            # italics in the body colour are species or drug names inside the option text, not explanations.
            main, tail = text, ""
            body_color = next((s["color"] for s in visible if not _is_italic(s)), None)
            cells_main, cells_tail, split = [], [], False
            for _, all_spans, _ in parts:
                head = []
                for s in all_spans:
                    if not split and s["text"].strip() and _is_italic(s) and body_color is not None \
                            and s["color"] != body_color and not _is_red(s["color"]) and (cells_main or head):
                        split = True
                    (cells_tail if split else head).append(_span_text(s))
                if head:
                    cells_main.append(re.sub(r"\s+", " ", "".join(head)).strip())
            if split:
                main = "\t".join(c for c in cells_main if c)
                tail = re.sub(r"\s+", " ", "".join(cells_tail)).strip()
            out.append(
                Row(pno, y, parts[0][0], text, bool(first["flags"] & 16) or "Bold" in first["font"],
                    ital / max(chars, 1), first["size"], red / max(chars, 1), main, tail)
            )
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info["bbox"]
            if x1 - x0 < MIN_IMAGE_W or y1 - y0 < MIN_IMAGE_H or info["xref"] == 0:
                continue
            if image_seen[info["xref"]] > 3:  # logo or header repeated on many pages
                continue
            out.append(("img", pno, y0, info["xref"], info["width"], info["height"]))
    out.sort(key=lambda r: (r.page, r.y) if isinstance(r, Row) else (r[1], r[2]))
    return out


OPTION_ROW = re.compile(r"^([A-H])(?:\s+(X))?(?:\s+(\d+(?:[.,]\d+)?)(?=\s|$))?(?:\s+(.*))?$")


def _page_margins(rows: list) -> dict[int, float]:
    """Left margin per page. Some PDFs shift single pages sideways (2017V pages 7-8 by 12 pt)."""
    first = next((r.x for r in rows if isinstance(r, Row) and r.text == "1" and r.bold_start), None)
    page_min: dict[int, float] = {}
    for r in rows:
        if isinstance(r, Row):
            page_min[r.page] = min(page_min.get(r.page, r.x), r.x)
    if first is None:
        return page_min
    # A page holding only indented explanation text has a larger minimum; fall back to the document margin.
    return {p: (m if -5 <= m - first <= 15 else first) for p, m in page_min.items()}


def _option(m: re.Match, changed_key: bool) -> ParsedOption:
    label, x, pts, rest = m.groups()
    opt = ParsedOption(label, keyed=bool(x))
    if pts is not None and changed_key:
        opt.points = float(pts.replace(",", "."))
    elif pts is not None:
        rest = f"{pts} {rest or ''}".strip()  # a number that belongs to the option text
    opt.text = re.sub(r"\s+", " ", rest or "").strip()
    return opt


def parse(path: Path) -> list[ParsedQuestion]:
    doc = pymupdf.open(path)
    rows = _rows(doc)
    margin = _page_margins(rows)
    questions: list[ParsedQuestion] = []
    q: ParsedQuestion | None = None
    for r in rows:
        if isinstance(r, tuple):  # image marker
            _, pno, y, xref, w, h = r
            if q is not None:
                q.images.append(ParsedImage(pno, y, xref, "explanation" if q.options else "stem", w, h))
                q.pages.add(pno)
            continue
        at_margin = abs(r.x - margin.get(r.page, r.x)) < 8
        expected = (questions[-1].number + 1) if questions else 1
        if r.text == str(expected) and at_margin:
            _finish(q)
            q = ParsedQuestion(expected)
            q.pages.add(r.page)
            questions.append(q)
            continue
        if q is None:
            continue  # cover page
        if r.text == str(q.number) and at_margin and r.bold_start and len(q.options) >= 3:
            q.skipping = True  # the PDF prints this question a second time (2017V Q86)
            q.warnings.append("question printed twice; second copy skipped")
            continue
        if q.skipping:
            continue
        q.pages.add(r.page)
        if END_NOTE.search(r.text) or (q.notes and END_NOTE.search(q.notes[-1]) and q.options):
            q.notes.append(r.text)  # the closing note and its continuation lines
            continue
        if CHANGED_KEY.search(r.text) or (q.changed_key and not q.stem and r.text.startswith("eksamen er gitt")):
            q.changed_key = True
            continue
        if r.red > 0.5 or (GRADING_NOTE.search(r.text) and re.search(r"sensurm|kommisjon", r.text, re.I)):
            q.notes.append(r.text)  # grading note, printed in red
            continue
        m = OPTION_ROW.match(r.main if r.tail else r.text)
        next_label = OPTION_LABELS[len(q.options)] if len(q.options) < len(OPTION_LABELS) else None
        if m and at_margin and r.bold_start and next_label and m.group(1) != next_label \
                and m.group(1) > next_label and len(r.text) < 120:
            # An option stored out of reading order in the PDF; hold it until its turn.
            q.pending[m.group(1)] = _option(m, q.changed_key)
            q.warnings.append(f"option {m.group(1)} out of reading order")
            continue
        while next_label in q.pending and not (m and m.group(1) == next_label):
            q.options.append(q.pending.pop(next_label))
            next_label = OPTION_LABELS[len(q.options)]
        if m and at_margin and m.group(1) == next_label and (r.bold_start or q.options or r.text[1:2] in (" ", "\t")):
            opt = _option(m, q.changed_key)
            if r.tail:
                opt.explanation.append(r.tail)
            if r.italic > 0.6 and EXPLANATION_START.match(opt.text):
                # The option's own text is missing in the source; the line is already its explanation.
                opt.explanation.insert(0, opt.text)
                opt.text = ""
            q.options.append(opt)
            continue
        if not q.options:
            q.stem.append(r.text)
            continue
        opt = q.options[-1]
        if opt.explanation or (r.italic > 0.5 and not r.tail):
            opt.explanation.append(r.text)  # explanations are printed in italics
        elif r.tail:
            _append(opt, r.main)
            opt.explanation.append(r.tail)
        elif not opt.text:
            opt.text = re.sub(r"\s+", " ", r.text)
        else:
            _append(opt, r.text)  # regular-font line right under an option: wrapped option text
    _finish(q)
    return questions


def _finish(q: ParsedQuestion | None) -> None:
    """Add options still held back, then record anything a person should check."""
    if q is None:
        return
    # A points column without the "NB Fasit ble endret" banner: every option starts with 0, 0.5 or 1.
    pts = [re.match(r"^(0|1|0[.,]5)\s+(\S.*)$", o.text) for o in q.options]
    values = {m.group(1) for m in pts if m}
    if not q.changed_key and len(q.options) >= 3 and all(pts) and "0" in values and len(values) > 1:
        for o, m in zip(q.options, pts):
            o.points, o.text = float(m.group(1).replace(",", ".")), m.group(2)
        q.changed_key = True
        q.warnings.append("points column without banner")
    for label in sorted(q.pending):
        if label == OPTION_LABELS[len(q.options)]:
            q.options.append(q.pending.pop(label))
    if q.pending:
        q.warnings.append(f"options never placed: {sorted(q.pending)}")
    labels = [o.label for o in q.options]
    if labels != list(OPTION_LABELS[: len(labels)]):
        q.warnings.append(f"option labels {labels}")
    if len(q.options) < 3:
        q.warnings.append(f"only {len(q.options)} options")
    empty = [o.label for o in q.options if not o.text]
    if empty:
        q.warnings.append(f"option text missing in source: {empty}")
    if not q.voided and not q.answer:
        q.warnings.append("no answer")


def _append(opt: ParsedOption, text: str) -> None:
    opt.text = re.sub(r"\s+", " ", f"{opt.text} {text}").strip()


BULLET = re.compile(r"^[•\-–▪◦·*]\s|^\d+[.)]\s")


def join_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines into paragraphs; keep table rows and list items on their own lines."""
    out: list[str] = []
    for line in lines:
        tabular = "\t" in line
        line = line.replace("\t", " | ") if tabular else line
        if out and not tabular and "|" not in out[-1] and not BULLET.match(line) \
                and not re.search(r"[.?!:]$", out[-1]):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out).strip()


EXPLANATION_START = re.compile(r"^(riktig|rett|feil|korrekt|galt)\b[.:!]?", re.I)


def save_images(path: Path, questions: list[ParsedQuestion], out_dir: Path) -> dict[int, list[dict]]:
    """Write each image once (original encoding) as <sha256[:16]>.<ext>; return records per question."""
    doc = pymupdf.open(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[int, list[dict]] = {}
    for q in questions:
        for im in q.images:
            data = doc.extract_image(im.xref)
            blob, ext = data["image"], data["ext"]
            if ext not in ("jpeg", "jpg", "png"):
                pix = pymupdf.Pixmap(doc, im.xref)
                if pix.n - pix.alpha >= 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                blob, ext = pix.tobytes("png"), "png"
            digest = hashlib.sha256(blob).hexdigest()
            name = f"{digest[:16]}.{'jpg' if ext == 'jpeg' else ext}"
            target = out_dir / name
            if not target.exists():
                target.write_bytes(blob)
            records.setdefault(q.number, []).append(
                {"file": f"images/{name}", "sha256": digest, "page": im.page + 1, "position": im.position,
                 "width": data["width"], "height": data["height"]}
            )
    return records
