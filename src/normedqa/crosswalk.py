"""Map every v2 item to its v3 counterpart, so v2 results can be compared with v3.

v2 renamed its 16 source PDFs to 1.pdf ... 16.pdf. Their real identity was worked out from the per-file counts in
the v2 extraction log and from the question texts (see SOURCES below). For items from the national exam, the v3
item is found by question text within the same sitting and written form, and v2's answer is compared with the
official key.
"""

from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

# v2 source file -> (document, v3 sitting, v3 written form). None: a course exam that v3 does not include.
SOURCES = {
    "1.pdf": ("MED6800 V20, revidert fasit (national exam spring 2020)", "2020V", "nb"),
    "2.pdf": ("Felles avsluttende skriftlig deleksamen høst 2023, uten fasit (bokmål)", "2023H", "nb"),
    "3.pdf": ("MED2200 H23 kont, med fasit", None, None),
    "4.pdf": ("MED3300 V24 ord, korrigert", None, None),
    "5.pdf": ("MED4500R 2023 høst psykiatri, med fasit", None, None),
    "6.pdf": ("MED5600 2024 høst, med fasit (English)", None, None),
    "7.pdf": ("MED6800 H20 ORD, revidert fasit (bokmål)", "2020H", "nb"),
    "8.pdf": ("MED6800 H20, revidert fasit (nynorsk)", "2020H", "nn"),
    "9.pdf": ("MEDSEM2 2013 høst kont", None, None),
    "10.pdf": ("MEDSEM4A 2015 høst", None, None),
    "11.pdf": ("MFFAGPR høst 2022, revidert fasit (bokmål)", "2022H", "nb"),
    "12.pdf": ("MFFAGPR høst 2022, uten fasit (bokmål)", "2022H", "nb"),
    "13.pdf": ("MFFAGPR H20, ordinær fasit (bokmål)", "2020H", "nb"),
    "14.pdf": ("Modul 1 blokk 1 2025 vår kont", None, None),
    "15.pdf": ("Modul 1 blokk 3 2023 høst", None, None),
    "16.pdf": ("Felles avsluttande skriftleg deleksamen haust 2023, revidert fasit (nynorsk)", "2023H", "nn"),
}


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def lenient_load(path: Path) -> list[dict]:
    """v2's full file has one unescaped quote (question 88 of 12.pdf); escape it on the fly."""
    s = path.read_text(encoding="utf-8")
    for _ in range(20):
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            i = s.rfind('"', 0, e.pos)
            s = s[:i] + '\\"' + s[i + 1:]
    raise ValueError(f"cannot parse {path}")


def answer_status(v2_answer: str, v3: dict) -> str:
    if v3["status"] == "voided":
        return "voided at grading (not scored in v3)"
    a = norm(v2_answer)
    accepted = [norm(o["text"]) for o in v3["options"] if o["label"] in v3["answer"]]
    others = [norm(o["text"]) for o in v3["options"] if o["label"] not in v3["answer"]]

    def same(x: str, y: str) -> bool:
        # v2 answers sometimes have the explanation appended, so compare the option with the start of the answer.
        return bool(x) and (a.startswith(x) or x.startswith(a) or SequenceMatcher(None, a[:len(x) + 10], x).ratio() >= 0.85)

    if any(same(x, a) for x in accepted):
        return "same answer"
    if any(same(x, a) for x in others):
        return "different answer: v2 marks an option the official key does not accept"
    return "unclear: v2 answer matches no option text"


def build(v2_full: Path, v2_txt: Path, v3_exam: Path, out: Path) -> dict:
    full = lenient_load(v2_full)
    txt = json.loads(v2_txt.read_text(encoding="utf-8"))
    txt_index = {(x["source_file"], norm(x["question_text"])[:300]): i for i, x in enumerate(txt)}
    v3 = [json.loads(l) for l in v3_exam.read_text(encoding="utf-8").splitlines() if l.strip()]
    pools = {}
    for it in v3:
        pools.setdefault((it["exam"]["sitting"], it["language"]), []).append(it)
    rows, counts = [], {}
    for i, x in enumerate(full):
        doc, sitting, lang = SOURCES[x["source_file"]]
        row = {"v2_index": i, "v2_txt_only_index": txt_index.get((x["source_file"], norm(x["question_text"])[:300]), ""),
               "v2_source_file": x["source_file"], "v2_question_number": x.get("question_number"),
               "source_document": doc, "v3_id": "", "similarity": "", "status": ""}
        if sitting is None:
            row["status"] = "not in v3 (course exam)"
        else:
            q = norm(x["question_text"])[:400]
            best, score = None, 0.0
            for it in pools.get((sitting, lang), []):
                s = SequenceMatcher(None, q, norm(it["question"])[:400]).ratio()
                if s > score:
                    best, score = it, s
            if best is None or score < 0.6:
                row["status"] = "no match in v3"
            else:
                row.update(v3_id=best["id"], similarity=f"{score:.2f}", status=answer_status(x["answer_text"], best))
        rows.append(row)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return counts
