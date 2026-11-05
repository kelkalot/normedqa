"""Sámi versions of the questions: machine drafts, replaced by reviewed text as reviews come in.

Inputs (both in sources/):
- sami_drafts.jsonl   one machine draft per item: {"id", "target", "question", "options", "uncertain_terms", "drafted_by"}
- sami_reviews.jsonl  exports from the review app (one line per review; the latest per item counts):
                      {"id", "status": approved|corrected|flagged|skipped, "question", "options", "comment", "reviewer", ...}

Outputs (data/v3/):
- sami_reading_sami.jsonl  the reading questions with question and options in the passage's Sámi language
- exam_sami.jsonl          the sampled exam questions in North Sámi

Each output item copies its Bokmål source item (answer, passage, source, split) and records
translation = "machine_draft" or "reviewed" and translated_from = the source id.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Item, Option


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build(data_dir: Path, sources: Path) -> dict:
    reading = {it["id"]: it for it in _read(data_dir / "sami_reading.jsonl")}
    exam = {it["id"]: it for it in _read(data_dir / "exam.jsonl")}
    drafts = {d["id"]: d for d in _read(sources / "sami_drafts.jsonl")}
    reviews = {}
    for r in _read(sources / "sami_reviews.jsonl"):  # later lines win
        reviews[r["id"]] = r
    out = {"sami_reading_sami.jsonl": [], "exam_sami.jsonl": []}
    stats = {"machine_draft": 0, "reviewed": 0, "flagged": 0, "skipped_not_in_release": 0, "skipped_bad_draft": 0}
    for iid, d in sorted(drafts.items()):
        src = reading.get(iid) or exam.get(iid)
        if src is None:
            stats["skipped_not_in_release"] += 1
            continue
        if len(d["options"]) != len(src["options"]):
            stats["skipped_bad_draft"] += 1
            continue
        question, options, translation, status, note = d["question"], d["options"], "machine_draft", src["status"], src.get("status_note")
        rev = reviews.get(iid)
        if rev and rev["status"] in ("approved", "corrected"):
            translation = "reviewed"
            if rev.get("question"):
                question = rev["question"]
            if rev.get("options") and len(rev["options"]) == len(options):
                options = rev["options"]
        elif rev and rev["status"] == "flagged":
            status = "excluded"
            note = f"Flagged in Sámi review: {rev.get('comment') or 'no comment'}"
            stats["flagged"] += 1
        stats[translation] += 1
        item = dict(src)
        item.update(
            question=question,
            options=[{**o, "text": t, "explanation": None} for o, t in zip(src["options"], options)],
            question_language=d["target"],
            translation=translation,
            translated_from=iid,
            uncertain_terms=[] if translation == "reviewed" else d.get("uncertain_terms", []),
            status=status,
            status_note=note,
        )
        if src["task"] == "exam":
            # The exam questions are translated into North Sámi; the item becomes a North Sámi item.
            item.update(id=iid.replace(f"-{src['language']}-", "-se-", 1), language="se")
            item.pop("also_in", None)
            out_name = "exam_sami.jsonl"
        else:
            item["id"] = f"{iid}-sami"
            out_name = "sami_reading_sami.jsonl"
        out[out_name].append(Item.model_validate(item))
    for name, items in out.items():
        with (data_dir / name).open("w", encoding="utf-8") as f:
            for it in items:
                f.write(it.model_dump_json(exclude_none=True) + "\n")
        stats[name] = len(items)
    return stats
