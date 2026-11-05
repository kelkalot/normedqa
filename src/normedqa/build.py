"""Build release items from the parsed source documents."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import yaml

from .dedup import annotate
from .parse_keyed import ParsedQuestion, join_lines, parse, save_images
from .schema import Exam, Image, Item, Option, Source
from .sources import PUBLISHER, load, raw_path

# Frequent words that differ between the two written forms.
NN_WORDS = re.compile(r"\b(ikkje|kva|korleis|kvifor|eit|frå|ein|ei|dei|ho|han|me|vore|blitt|vert|ved|òg|mykje|berre|gong|sidan|no)\b", re.I)
NB_WORDS = re.compile(r"\b(ikke|hva|hvordan|hvorfor|et|fra|en|de|hun|vi|vært|blir|også|mye|bare|gang|siden|nå)\b", re.I)


def detect_written_form(texts: list[str]) -> str:
    joined = " ".join(texts)
    nn = len(re.findall(r"\b(ikkje|kva|korleis|kvifor|eit|frå|mykje|berre|sidan)\b", joined, re.I))
    nb = len(re.findall(r"\b(ikke|hva|hvordan|hvorfor|et|fra|mye|bare|siden)\b", joined, re.I))
    return "nn" if nn > nb else "nb"


# Splits by sitting, so both written forms of a question land in the same split.
# Test: the two newest sittings. Dev: the one before. Train: everything older.
TEST_SITTINGS = {"2025H", "2026V"}
DEV_SITTINGS = {"2025V"}


def split_for(sitting: str) -> str:
    return "test" if sitting in TEST_SITTINGS else "dev" if sitting in DEV_SITTINGS else "train"


def exam_name(sitting: str) -> str:
    year = int(sitting[:4])
    term = sitting[4]
    # The exam was called "Nasjonal delprøve" until H20 and "Felles avsluttende skriftlig deleksamen" from V21.
    if year < 2021:
        return "Nasjonal delprøve i medisin"
    return "Felles avsluttende skriftlig deleksamen i medisin"


def _option_explanation(lines: list[str]) -> str | None:
    text = join_lines(lines)
    return text or None


def to_items(doc: dict, questions: list[ParsedQuestion], images: dict[int, list[dict]], language: str,
             item_notes: dict) -> list[Item]:
    sitting = doc["sitting"]
    exam = Exam(code="felles", name=exam_name(sitting), sitting=sitting, year=int(sitting[:4]), term=sitting[4])
    items = []
    for q in questions:
        num = f"{q.number:03d}"
        iid = f"felles-{sitting}-{language}-{num}"
        # Drop the second line of the changed-key banner ("... poengiving til eksamen er gitt i 2. kolonne").
        notes = [n.replace("\t", " ") for n in q.notes if not re.search(r"eksamen er gitt i 2\. kolonne", n)]
        if q.changed_key:
            notes.insert(0, "Fasit endret ved sensur; poeng per alternativ fra sensurmøtet.")
        defects = [w for w in q.warnings if w.startswith("option text missing")]
        if defects:
            notes.append(f"Source defect: {'; '.join(defects)}.")
        manual = item_notes.get(iid) or item_notes.get(f"felles-{sitting}-{num}") or {}
        if manual.get("note"):
            notes.append(manual["note"])
        status = "voided" if q.voided else ("excluded" if defects or manual.get("status") == "excluded" else "ok")
        stem_imgs = [i for i in images.get(q.number, []) if i["position"] == "stem"]
        items.append(
            Item(
                id=iid,
                group_id=f"felles-{sitting}-{num}",
                task="exam",
                language=language,
                exam=exam,
                question_number=str(q.number),
                question=join_lines(q.stem),
                options=[
                    Option(label=o.label, text=o.text, explanation=_option_explanation(o.explanation), points=o.points)
                    for o in q.options
                ],
                answer=[] if q.voided else q.answer,
                original_answer=q.original_answer if q.changed_key else None,
                status=status,
                status_note=" ".join(notes) or None,
                images=[Image(**i) for i in images.get(q.number, [])],
                requires_image=bool(stem_imgs),
                source=Source(
                    url=doc["url"],
                    page_url=doc["page_url"],
                    title=doc["title"],
                    publisher=PUBLISHER,
                    sha256=doc["sha256"],
                    retrieved=doc["retrieved"],
                    pages=sorted(p + 1 for p in q.pages),
                ),
                method="points-key" if q.changed_key else "x-key",
                split=split_for(sitting),
            )
        )
    return items


def build_exams(sources: Path, raw_dir: Path, out_dir: Path) -> list[Item]:
    data = load(sources)
    notes_path = sources.parent / "item_notes.yaml"
    item_notes = (yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}) if notes_path.exists() else {}
    items: list[Item] = []
    warnings: list[str] = []
    for doc in data["documents"]:
        if doc.get("use") != "exam":
            continue
        path = raw_path(raw_dir, doc)
        questions = parse(path)
        language = doc["language"] or detect_written_form([" ".join(q.stem) for q in questions])
        images = save_images(path, questions, out_dir / "images")
        items.extend(to_items(doc, questions, images, language, item_notes))
        warnings += [f"{doc['id']} Q{q.number}: {w}" for q in questions for w in q.warnings]
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [it.model_dump(mode="json", exclude_none=True) for it in items]
    reused = annotate(records)
    with (out_dir / "exam.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            Item.model_validate(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / "parse_warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
    langs = Counter((i.exam.sitting, i.language) for i in items)
    status = Counter(i.status for i in items)
    print(f"{len(items)} exam items, {len({i.group_id for i in items})} distinct questions, {dict(status)}, "
          f"{sum(1 for i in items if i.requires_image)} with images, {reused} reused-question pairs")
    print("written forms per sitting:", dict(sorted(langs.items())))
    return items
