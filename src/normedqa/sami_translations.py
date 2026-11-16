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
import re
from pathlib import Path

import yaml

from .schema import Item, Option


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


PASSAGE_FLAG = "sami_passage_differs"


def _keep_case(word: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if word[:1].isupper() else replacement


def apply_draft_fixes(iid: str, target: str, question: str, options: list[str], labels: list[str], fixes: dict) -> tuple[str, list[str], list[str]]:
    """Term rules and item corrections from sources/sami_draft_fixes.yaml; returns the text and the ids of applied fixes."""
    applied = []
    for rule in fixes.get("terms", []):
        if rule["language"] != target or iid not in set(rule.get("items", [])):
            continue
        rx = re.compile(rule["pattern"], re.I)

        def sub(text: str) -> str:
            return rx.sub(lambda m: _keep_case(m.group(0), rule["replacement"]), text)

        new_q, new_o = sub(question), [sub(o) for o in options]
        if (new_q, new_o) != (question, options):
            applied.append(rule["id"])
            question, options = new_q, new_o
    fx = (fixes.get("items") or {}).get(iid)
    if fx:
        question = fx.get("question") or question
        options = [fx.get("options", {}).get(lab, o) for lab, o in zip(labels, options)]
        applied.append(f"item:{iid}")
    return question, options, applied


def build(data_dir: Path, sources: Path) -> dict:
    reading = {it["id"]: it for it in _read(data_dir / "sami_reading.jsonl")}
    exam = {it["id"]: it for it in _read(data_dir / "exam.jsonl")}
    drafts = {d["id"]: d for d in _read(sources / "sami_drafts.jsonl")}
    reviews = {}
    for r in _read(sources / "sami_reviews.jsonl"):  # later lines win
        reviews[r["id"]] = r
    fixes_path = sources / "sami_draft_fixes.yaml"
    fixes = (yaml.safe_load(fixes_path.read_text(encoding="utf-8")) or {}) if fixes_path.exists() else {}
    # A reviewer can report that the Sámi passage itself says something else than the original text.
    passage_differs = {iid for iid, r in reviews.items() if r.get("passage_differs")}
    out = {"sami_reading_sami.jsonl": [], "exam_sami.jsonl": []}
    stats = {"machine_draft": 0, "reviewed": 0, "flagged": 0, "skipped_not_in_release": 0, "skipped_bad_draft": 0,
             "draft_fixes_applied": 0, PASSAGE_FLAG: 0}
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
        if not (rev and rev["status"] in ("approved", "corrected")):
            question, options, applied = apply_draft_fixes(iid, d["target"], question, options,
                                                           [o["label"] for o in src["options"]], fixes)
            stats["draft_fixes_applied"] += bool(applied)
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
        flags = [f for f in item.get("flags", []) if f != PASSAGE_FLAG]
        if iid in passage_differs:
            flags.append(PASSAGE_FLAG)
            item["status_note"] = "; ".join(x for x in (item.get("status_note"), f"Sámi passage differs from the original: {reviews[iid].get('comment') or ''}".strip()) if x)
            stats[PASSAGE_FLAG] += 1
        item["flags"] = flags
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
    # The passage is shared with the Bokmål-question version, so flag that item too (recomputed on every run).
    if reading:
        with (data_dir / "sami_reading.jsonl").open("w", encoding="utf-8") as f:
            for iid, it in reading.items():
                it["flags"] = [x for x in it.get("flags", []) if x != PASSAGE_FLAG] + ([PASSAGE_FLAG] if iid in passage_differs else [])
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return stats
