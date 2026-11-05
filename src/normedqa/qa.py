"""Quality checks for the built exam items.

Every check compares the items with something independent of the parser where
possible: the unkeyed copy of the same exam (text), Fagprøven's key for the
same paper (answers), the psychometric notes (voided questions and questions
with more than one accepted answer), and the other written form of the same
question (Bokmål vs Nynorsk).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

from .build import detect_written_form
from .dedup import norm, reused_pairs
from .parse_keyed import join_lines, parse
from .sources import load, raw_path

# Phrases that point at a picture shown with the question (not "ser du", which describes an examination finding).
IMAGE_WORDS = re.compile(
    r"\b(bildet viser|biletet viser|bilde(t)? under|bilete(t)? under|på bildet|på biletet|se bildet|sjå biletet|"
    r"vist (på|i) (bildet|figuren|biletet)|figuren (viser|under)|hva viser (bildet|røntgen|ekg|ct|mr|undersøkelsen|funnet)|"
    r"kva viser (biletet|røntgen|ekg|ct|mr|undersøkinga|funnet)|se figur|sjå figur|følgende (bilde|ekg)|følgjande (bilete|ekg)|"
    r"(ekg|røntgenbilde|ct-bilde|mr-bilde|ultralydbilde|bilde|bilete) (som )?(vises|vist) (under|nedenfor))\b",
    re.I,
)
ARTIFACTS = {
    "footer hash": re.compile(r"\b0{3,}[0-9a-f]{12,}\b"),
    "page number": re.compile(r"(^|\n)\s*-\s*\d{1,3}\s*-\s*($|\n)"),
    "changed-key banner": re.compile(r"NB\s*!?\s*Fasit", re.I),
    "explanation in option": re.compile(r"^(riktig|rett|feil) svar\b", re.I),
}


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_text(items: list[dict]) -> list[tuple[str, str]]:
    flags = []
    for it in items:
        q = it["question"]
        if len(q) < 40:
            flags.append((it["id"], f"short question ({len(q)} chars)"))
        if "?" not in q and not re.search(r":\s*$", q):
            flags.append((it["id"], "question text has no question mark"))
        texts = [o["text"] for o in it["options"]]
        if any(not t for t in texts):
            flags.append((it["id"], "empty option text"))
        if len(set(map(norm, texts))) < len(texts):
            flags.append((it["id"], "two options with the same text"))
        for name, rx in ARTIFACTS.items():
            fields = texts if name == "explanation in option" else [q] + texts
            if any(rx.search(f) for f in fields):
                flags.append((it["id"], name))
        has_stem_img = any(i["position"] == "stem" for i in it.get("images", []))
        if IMAGE_WORDS.search(q) and not has_stem_img:
            flags.append((it["id"], "refers to an image but has none"))
    return flags


def check_pairs(items: list[dict]) -> list[tuple[str, str]]:
    flags = []
    groups = defaultdict(dict)
    for it in items:
        groups[it["group_id"]][it["language"]] = it
    for gid, g in groups.items():
        if "nb" not in g or "nn" not in g:
            continue
        a, b = g["nb"], g["nn"]
        if len(a["options"]) != len(b["options"]):
            flags.append((gid, f"option count nb={len(a['options'])} nn={len(b['options'])}"))
        if a["answer"] != b["answer"] or a["status"] != b["status"]:
            flags.append((gid, f"answer nb={a['answer']}/{a['status']} nn={b['answer']}/{b['status']}"))
        ia = sorted(i["sha256"] for i in a.get("images", []) if i["position"] == "stem")
        ib = sorted(i["sha256"] for i in b.get("images", []) if i["position"] == "stem")
        if len(ia) != len(ib):
            flags.append((gid, f"stem images nb={len(ia)} nn={len(ib)}"))
    return flags


def check_unkeyed(items: list[dict], sources: Path, raw_dir: Path) -> tuple[list[tuple[str, str]], dict]:
    """Compare question and option text with the unkeyed copy of the same exam."""
    flags, stats = [], {"compared": 0, "identical_question": 0, "identical_options": 0}
    docs = [d for d in load(sources)["documents"] if d.get("use") == "textcheck"]
    for d in docs:
        path = raw_path(raw_dir, d)
        if not path.exists():
            continue
        other = parse(path)
        # Detect the written form from the content: UiO's links for the V23 unkeyed copies are swapped.
        lang = detect_written_form([" ".join(q.stem) + " " + " ".join(o.text for o in q.options) for q in other])
        mine = [it for it in items if it["exam"]["sitting"] == d["sitting"] and it["language"] == lang]
        if not mine:
            continue
        by_num = {q.number: q for q in other}

        def fingerprint(stem: str, options: list[str]) -> str:
            return norm(stem)[-300:] + " || " + " | ".join(sorted(norm(o) for o in options))

        prints = [(q, fingerprint(join_lines(q.stem), [o.text for o in q.options])) for q in other]
        for it in mine:
            # Same number first. Some unkeyed copies (2017V) order questions and options differently, and several
            # questions can share one case text, so fall back to matching the question and its options together.
            q = by_num.get(int(it["question_number"]))
            target = fingerprint(it["question"], [o["text"] for o in it["options"]])
            if q is None or SequenceMatcher(None, target, fingerprint(join_lines(q.stem), [o.text for o in q.options])).ratio() < 0.85:
                best = max(prints, key=lambda c: SequenceMatcher(None, target, c[1]).ratio(), default=None)
                q = best[0] if best and SequenceMatcher(None, target, best[1]).ratio() >= 0.6 else None
            if q is None:
                flags.append((it["id"], "missing in unkeyed copy"))
                continue
            stats["compared"] += 1
            s = sim(it["question"], join_lines(q.stem))
            if s >= 0.98:
                stats["identical_question"] += 1
            else:
                flags.append((it["id"], f"question text differs from unkeyed copy (similarity {s:.2f})"))
            if len(q.options) != len(it["options"]):
                flags.append((it["id"], f"unkeyed copy has {len(q.options)} options, we have {len(it['options'])}"))
                continue
            worst = min((sim(o["text"], p.text) for o, p in zip(it["options"], q.options)), default=1.0)
            if worst < 0.98:  # the unkeyed copy may list the same options in another order
                ours = sorted(norm(o["text"]) for o in it["options"])
                theirs = sorted(norm(p.text) for p in q.options)
                worst = min((sim(a, b) for a, b in zip(ours, theirs)), default=1.0)
            if worst >= 0.98:
                stats["identical_options"] += 1
            else:
                flags.append((it["id"], f"option text differs from unkeyed copy (worst similarity {worst:.2f})"))
    return flags, stats


def check_fagproven(items: list[dict], sources: Path, raw_dir: Path) -> tuple[list[tuple[str, str]], dict]:
    """Compare our answers with Fagprøven's key for the same paper (matched by question text)."""
    flags, stats = [], {"matched": 0, "same_answer": 0, "unmatched": 0}
    docs = [d for d in load(sources)["documents"] if d.get("use") == "crosscheck"]
    for d in docs:
        path = raw_path(raw_dir, d)
        mine = [it for it in items if it["exam"]["sitting"] == d["sitting"] and it["language"] == d["language"]]
        if not mine or not path.exists():
            continue
        other = parse(path)
        index = {norm(join_lines(q.stem))[:200]: q for q in other}
        for it in mine:
            q = index.get(norm(it["question"])[:200])
            if q is None:
                stats["unmatched"] += 1
                continue
            stats["matched"] += 1
            # Fagprøven prints the options in a different order, so compare the accepted option texts.
            theirs = sorted(norm(o.text) for o in q.options if not q.voided and o.label in q.answer)
            ours = sorted(norm(o["text"]) for o in it["options"] if o["label"] in it["answer"])
            if theirs == ours:
                stats["same_answer"] += 1
            else:
                flags.append((it["id"], f"Fagprøven accepts {theirs or 'nothing (voided)'}; we accept {ours or 'nothing (voided)'}"))
    return flags, stats


def psychometric_mentions(items: list[dict], sources: Path, raw_dir: Path) -> dict:
    """For manual review: the questions each psychometric note discusses, next to our voided/multi-answer lists.

    The notes are free text in varying formats, so this is not scored automatically; a person reads the
    listed questions in the note and compares them with our status.
    """
    out = {}
    docs = [d for d in load(sources)["documents"] if d.get("use") == "psychometrics"]
    by_sitting = defaultdict(list)
    for it in items:
        if it["language"] == "nb":
            by_sitting[it["exam"]["sitting"]].append(it)
    for d in docs:
        path = raw_path(raw_dir, d)
        mine = {int(it["question_number"]): it for it in by_sitting.get(d["sitting"], [])}
        if not mine or not path.exists():
            continue
        text = "\n".join(p.get_text() for p in pymupdf.open(path))
        mentioned = sorted({int(m.group(1)) for m in re.finditer(r"(?:Oppgave|Oppgåve|Spørsmål)\s*(\d{1,3})\b", text)
                            if int(m.group(1)) in mine})
        out[d["sitting"]] = {
            "discussed_in_note": mentioned,
            "voided_here": sorted(n for n, it in mine.items() if it["status"] == "voided"),
            "several_answers_here": sorted(n for n, it in mine.items() if len(it["answer"]) > 1),
        }
    return out


def run(items_path: Path, sources: Path, raw_dir: Path, out_dir: Path) -> None:
    items = load_items(items_path)
    sections = {}
    sections["text"] = check_text(items)
    sections["pairs"] = check_pairs(items)
    sections["unkeyed"], unkeyed_stats = check_unkeyed(items, sources, raw_dir)
    sections["fagproven"], fag_stats = check_fagproven(items, sources, raw_dir)
    psy = psychometric_mentions(items, sources, raw_dir)
    reuse = reused_pairs(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "items": len(items),
        "unkeyed_stats": unkeyed_stats,
        "fagproven_stats": fag_stats,
        "psychometrics": psy,
        "flags": {k: [list(f) for f in v] for k, v in sections.items()},
        "reused_questions": [list(p) for p in reuse],
    }
    (out_dir / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    status = defaultdict(int)
    for it in items:
        status[it["status"]] += 1
    print(f"items: {len(items)} {dict(status)}")
    print(f"text vs unkeyed copy: {unkeyed_stats}")
    print(f"answers vs Fagprøven: {fag_stats}")
    for k, v in sections.items():
        kinds = defaultdict(int)
        for _, msg in v:
            kinds[re.sub(r"\(.*?\)|\[.*?\]|\d+", "#", msg)] += 1
        print(f"[{k}] {len(v)} flags: {dict(kinds)}")
    print(f"[reuse] {len(reuse)} pairs of the same question in two sittings (same written form)")
