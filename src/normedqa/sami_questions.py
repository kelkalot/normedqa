"""Write and check reading-comprehension questions for the Sámi passages.

For each passage a model sees the Sámi text and the matching part of the original
document (Norwegian, Swedish or Finnish), quotes the corresponding original
text, and writes questions in Norwegian Bokmål with evidence quoted from both
versions. Checks then confirm the quotes exist, that a second model reaches the
same answer from the original text, and whether the question can be answered
without the passage at all.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from .sami import parse_doc, passages, publisher_for

LANG_NAME = {"se": "North Sámi", "smj": "Lule Sámi", "sma": "South Sámi",
             "nob": "Norwegian Bokmål", "nno": "Norwegian Nynorsk", "swe": "Swedish", "fin": "Finnish"}

PROMPT_ROUND1 = """You write reading-comprehension questions for a benchmark that tests whether language models
understand health texts in {lang}. Below is a passage in {lang} and the matching part of the same document in
{olang} (one is a translation of the other).

Step 1. Find the part of the {olang} text that corresponds to the {lang} passage and copy it verbatim as
"original_excerpt" (only what the passage covers).
Step 2. Write {k} multiple-choice questions in Norwegian Bokmål about the content of the passage.
- The answer must be stated in the passage. Do not require outside medical knowledge.
- Ask about specific content of this passage (who, what, when, how many, which advice, which condition), not
  facts any doctor would know without reading it.
- Four options, exactly one correct. Distractors must be plausible, similar in length and style to the correct
  option, and not supported by the passage.
- Paraphrase the correct option instead of copying a long phrase from the text.
- Each question must stand alone: name what it is about instead of saying "teksten" or "avsnittet" alone.
- For each question copy "evidence_original" (verbatim sentence(s) from the {olang} excerpt) and "evidence_sami"
  (verbatim sentence(s) from the {lang} passage) that support the answer.
If the passage has no substantive health content (only names, contact details, dates or meeting logistics),
return "questions": [].

Return JSON only:
{{"original_excerpt": "...", "questions": [{{"question": "...", "options": ["...", "...", "...", "..."],
  "answer": "A", "evidence_original": "...", "evidence_sami": "..."}}]}}

{lang} passage:
<<<
{passage}
>>>

{olang} text:
<<<
{original}
>>>"""

# Round 2 (stricter): only details a reader cannot guess, wrong options built from other details in the text.
PROMPT = """You write reading-comprehension questions for a benchmark that tests whether language models
understand health texts in {lang}. A model will see the {lang} passage and must answer your question. The
question is only useful if it CANNOT be answered without reading the passage. Below is the passage in {lang}
and the matching part of the same document in {olang} (one is a translation of the other).

Step 1. Find the part of the {olang} text that corresponds to the {lang} passage and copy it verbatim as
"original_excerpt" (only what the passage covers).
Step 2. Write {k} multiple-choice questions in Norwegian Bokmål about the passage.
- Target a specific detail stated in the passage that a knowledgeable doctor could not guess: a number, age,
  duration, time limit, dose, frequency, a named service, place, organisation or person, a condition or
  exception, the order of steps, who is responsible, or what the text advises against.
- Never ask about general health advice or facts that common sense or medical knowledge answers.
- Four options of the same type and length. All four must be plausible to someone who has not read the passage.
  Build at least two wrong options from other details in the passage or the surrounding text (another number,
  another service, another group of people), so that only careful reading identifies the right one.
- Paraphrase; do not copy a long phrase from the text into the correct option.
- The correct option can be any of the four; write the options in a random order and give its letter.
- Each question must stand alone: name what it is about instead of "teksten" or "avsnittet" alone.
- Before answering, check each question: if someone who has not read the passage could pick the right option by
  common sense or medical knowledge, rewrite it or leave it out. Fewer good questions are better than {k} easy ones.
- For each question copy "evidence_original" (verbatim sentence(s) from the {olang} excerpt) and "evidence_sami"
  (verbatim sentence(s) from the {lang} passage) that support the answer.
If the passage has no substantive health content (only names, contact details, dates or meeting logistics),
return "questions": [].

Return JSON only:
{{"original_excerpt": "...", "questions": [{{"question": "...", "options": ["...", "...", "...", "..."],
  "answer": "C", "evidence_original": "...", "evidence_sami": "..."}}]}}

{lang} passage:
<<<
{passage}
>>>

{olang} text:
<<<
{original}
>>>"""

CHECK = """Answer the multiple-choice question{ctx_hint}. Reply with the letter only.
{context}
{question}

A. {a}
B. {b}
C. {c}
D. {d}"""


ORIGINAL_REF = re.compile(r"\bden (norske|svenske|finske|nordiske|originale) (teksten|versjonen)\b", re.I)


FINNISH = re.compile(r"\b(on|ja|että|oli|ovat|mikä|mitä|kuka|miten|missä|mukaan|tekstin)\b", re.I)
SWEDISH = re.compile(r"\b(och|är|inte|enligt|vilken|vilket|vilka|hur|också)\b", re.I)
NORWEGIAN = re.compile(r"\b(og|er|ikke|ifølge|hvilken|hvilket|hvilke|hva|hvordan|hvem|hvor|også|kan|skal|du)\b", re.I)


def is_norwegian(text: str) -> bool:
    """The question writer occasionally answers in the original's language (Finnish, Swedish) instead of Bokmål."""
    no = len(NORWEGIAN.findall(text))
    return not ((len(FINNISH.findall(text)) >= 3 or len(SWEDISH.findall(text)) >= 3) and no < 2)


def refer_to_passage(text: str) -> str:
    """Questions are asked about the Sámi passage, so "den norske/svenske/finske teksten" becomes "teksten"."""
    return ORIGINAL_REF.sub("teksten", text)


def shuffled(q: dict, qid: str) -> tuple[list[str], str]:
    """Options in a fixed random order per question (generators put the correct answer first), and the new key."""
    order = list(range(4))
    random.Random(f"normedqa-sami-{qid}").shuffle(order)
    options = [q["options"][i] for i in order]
    key = "ABCD"[order.index("ABCD".index(q["answer"]))]
    return options, key


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def found_in(quote: str, text: str) -> float:
    """How well a quote matches somewhere in the text (1.0 = verbatim)."""
    q, t = norm(quote), norm(text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    sm = SequenceMatcher(None, t, q, autojunk=False)
    blocks = sum(b.size for b in sm.get_matching_blocks())
    return blocks / len(q)


def original_window(doc: dict, cache: Path, rel_start: float, rel_end: float, min_words: int = 350,
                    max_words: int = 1100) -> str:
    """The part of the original around the passage's relative position (the whole text if it is short)."""
    o = doc["original"]
    paras = parse_doc((cache / o["repo"] / o["commit"][:12] / o["path"]).read_text(encoding="utf-8"))["paragraphs"]
    total = sum(len(p.split()) for p in paras)
    if total <= max_words:
        return "\n".join(paras)
    lo, hi = max(0.0, rel_start - 0.2), min(1.0, rel_end + 0.2)
    out, pos = [], 0
    for p in paras:
        n = len(p.split())
        mid = (pos + n / 2) / total
        if lo <= mid <= hi:
            out.append(p)
        pos += n
    while sum(len(p.split()) for p in out) < min_words and len(out) < len(paras):
        lo, hi = max(0.0, lo - 0.05), min(1.0, hi + 0.05)
        out, pos = [], 0
        for p in paras:
            n = len(p.split())
            if lo <= (pos + n / 2) / total <= hi:
                out.append(p)
            pos += n
    return "\n".join(out)


def plan(sources: Path, cache: Path, *, per_medical_doc: int = 8, per_policy_doc: int = 1, seed: int = 3) -> list[dict]:
    """Choose passages: up to 8 per medical/patient document, 1 per health-policy document."""
    data = yaml.safe_load(sources.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    jobs = []
    for doc in data["documents"]:
        ps = passages(doc, cache)
        if not ps:
            continue
        cap = per_medical_doc if doc["topic"].startswith("medical") else per_policy_doc
        idx = list(range(len(ps)))
        if len(idx) > cap:
            idx = sorted(rng.sample(idx, cap)) if cap > 1 else [max(idx, key=lambda i: len(ps[i]))]
        total = sum(len(p) for p in ps)
        for i in idx:
            start = sum(len(p) for p in ps[:i]) / total
            end = (sum(len(p) for p in ps[:i + 1])) / total
            jobs.append({"doc": doc, "index": i, "passage": ps[i],
                         "original": original_window(doc, cache, start, end)})
    return jobs


def generate(jobs: list[dict], out: Path, model: str, *, workers: int = 12, round_: int = 2) -> None:
    """Round 1 wrote 2-3 questions per passage with PROMPT_ROUND1; round 2 wrote 3-4 with the stricter PROMPT."""
    from openai import OpenAI

    client = OpenAI()
    done = set()
    if out.exists():
        done = {json.loads(l)["job"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    lock = threading.Lock()

    def one(job: dict) -> None:
        d = job["doc"]
        key = f"{d['id']}#{job['index']}"
        if round_ == 1:
            k = 3 if len(job["passage"].split()) > 160 else 2
        else:
            k = 4 if len(job["passage"].split()) > 140 else 3
        prompt = (PROMPT_ROUND1 if round_ == 1 else PROMPT).format(lang=LANG_NAME[d["language"]], olang=LANG_NAME[d["original"]["language"]], k=k,
                               passage=job["passage"], original=job["original"])
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              response_format={"type": "json_object"}, reasoning_effort="low")
        rec = {"job": key, "doc_id": d["id"], "index": job["index"], "language": d["language"],
               "passage": job["passage"], "original_window": job["original"], "model": model,
               "result": json.loads(resp.choices[0].message.content),
               "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = [j for j in jobs if f"{j['doc']['id']}#{j['index']}" not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in todo]):
            try:
                f.result()
            except Exception as e:
                print("error:", str(e)[:160])


def ask(client, model: str, question: dict, context: str | None, hint: str) -> str | None:
    o = question["options"]
    msg = CHECK.format(ctx_hint=hint, context=(f"\n{context}\n" if context else ""), question=question["question"],
                       a=o[0], b=o[1], c=o[2], d=o[3])
    kwargs = {"reasoning_effort": "low"} if model.startswith("gpt-5") else {}
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], **kwargs)
    m = re.search(r"\b([ABCD])\b", (r.choices[0].message.content or "").upper())
    return m.group(1) if m else None


def verify(gen_path: Path, out: Path, model: str, *, workers: int = 16, ask_sami: bool = True) -> None:
    """Quote checks plus three answers per question: from the original excerpt, from the Sámi passage, and with no text."""
    from openai import OpenAI

    client = OpenAI()
    rows = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    lock = threading.Lock()
    tasks = []
    for r in rows:
        for k, q in enumerate(r["result"].get("questions", [])):
            qid = f"{r['job']}#q{k + 1}"
            if qid not in done and isinstance(q.get("options"), list) and len(q["options"]) == 4 and q.get("answer") in ("A", "B", "C", "D"):
                tasks.append((r, k, q, qid))

    def one(t):
        r, k, q0, qid = t
        options, key = shuffled(q0, qid)
        q = {**q0, "options": options, "answer": key}
        excerpt = r["result"].get("original_excerpt", "") or r["original_window"]
        rec = {
            "qid": qid,
            "evidence_sami_found": round(found_in(q.get("evidence_sami", ""), r["passage"]), 3),
            "evidence_original_found": round(found_in(q.get("evidence_original", ""), r["original_window"]), 3),
            "excerpt_found": round(found_in(excerpt[:400], r["original_window"]), 3),
            "answer_from_original": ask(client, model, q, excerpt, " using the text"),
            "answer_from_sami": ask(client, model, q, r["passage"], " using the text") if ask_sami else None,
            "answer_closed_book": ask(client, model, q, None, ""),
            "key": q.get("answer"),
        }
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in tasks]):
            try:
                f.result()
            except Exception as e:
                print("error:", str(e)[:160])


def build_items(gen_path: Path, check_path: Path, sources: Path, cache: Path, out: Path | None, *,
                min_quote: float = 0.85, id_letter: str = "q", only_passage_dependent: bool = False) -> dict:
    """Turn checked questions into release items. With only_passage_dependent, keep only questions the check
    model got wrong without the passage (used for the second, stricter round, ids ...-hN)."""
    import datetime as dt
    import hashlib

    from .schema import Item, Option, Source

    data = yaml.safe_load(sources.read_text(encoding="utf-8"))
    docs = {d["id"]: d for d in data["documents"]}
    checks = {json.loads(l)["qid"]: json.loads(l) for l in check_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    stats = {"generated": 0, "kept": 0, "dropped_quote": 0, "dropped_answer": 0, "dropped_format": 0, "not_used_licence": 0,
             "answerable_without_passage": 0, "dropped_duplicate_document": 0}
    # GiellaLT holds a few pages twice (hyphen and underscore file names) that got the same document id; their
    # questions share one key, so their checks cannot be told apart. Leave those out.
    job_counts: dict[str, int] = {}
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)["job"]
            job_counts[j] = job_counts.get(j, 0) + 1
    items = []
    today = dt.date.today().isoformat()
    for line in gen_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d = docs[r["doc_id"]]
        if job_counts[r["job"]] > 1:
            stats["dropped_duplicate_document"] += len(r["result"].get("questions", []))
            continue
        for k, q in enumerate(r["result"].get("questions", [])):
            stats["generated"] += 1
            c = checks.get(f"{r['job']}#q{k + 1}")
            if not c or not isinstance(q.get("options"), list) or len(q["options"]) != 4 or q.get("answer") not in "ABCD":
                stats["dropped_format"] += 1
                continue
            if not is_norwegian(q["question"] + " " + " ".join(q["options"])):
                stats["dropped_format"] += 1
                continue
            if c["evidence_sami_found"] < min_quote or c["evidence_original_found"] < min_quote:
                stats["dropped_quote"] += 1
                continue
            if c["answer_from_original"] != c["key"]:
                stats["dropped_answer"] += 1
                continue
            if not d.get("use"):
                stats["not_used_licence"] += 1
                continue
            options, key = shuffled(q, f"{r['job']}#q{k + 1}")
            flags = []
            if c["answer_closed_book"] == key:
                if only_passage_dependent:
                    stats["dropped_answerable_without_passage"] = stats.get("dropped_answerable_without_passage", 0) + 1
                    continue
                flags.append("answerable_without_passage")
                stats["answerable_without_passage"] += 1
            raw = (cache / d["repo"] / d["commit"][:12] / d["path"]).read_bytes()
            iid = f"sami-{r['doc_id']}-p{r['index']:02d}-{id_letter}{k + 1}"
            items.append(Item(
                id=iid, group_id=iid, task="sami_reading", language=d["language"],
                passage=r["passage"], passage_original=r["result"].get("original_excerpt") or None,
                passage_original_language=d["original"]["language"], question_language="nb",
                question=refer_to_passage(q["question"]),
                options=[Option(label="ABCD"[i], text=refer_to_passage(t)) for i, t in enumerate(options)],
                answer=[key], topic=d["topic"], flags=flags,
                source=Source(url=d["url"] or f"https://github.com/giellalt/{d['repo']}/blob/{d['commit']}/converted/{d['path']}",
                              title=d["title"] or d["path"], publisher=publisher_for(d)[0],
                              attribution=(f"Author: {publisher_for(d)[1]}. " if publisher_for(d)[1] else "")
                              + "Sámi text via the GiellaLT corpus (UiT Norgga árktalaš universitehta).",
                              sha256=hashlib.sha256(raw).hexdigest(), retrieved=today,
                              corpus_path=f"giellalt/{d['repo']}/converted/{d['path']}", corpus_commit=d["commit"],
                              licence=f"{d['licence']} (GiellaLT corpus metadata)"),
                method="drafted", verified=False, split="test",  # evaluation-only set
            ))
            stats["kept"] += 1
    stats["items"] = items
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(it.model_dump_json(exclude_none=True) + "\n")
    return stats


def build_all(build_dir: Path, sources: Path, cache: Path, out: Path) -> dict:
    """First round (all checked questions, flagged) plus the stricter second round (passage-dependent only)."""
    a = build_items(build_dir / "gen.jsonl", build_dir / "checks.jsonl", sources, cache, None)
    b = build_items(build_dir / "gen_hard.jsonl", build_dir / "checks_hard.jsonl", sources, cache, None,
                    id_letter="h", only_passage_dependent=True)
    items = a.pop("items") + b.pop("items")
    # Corrections found in review: drop an item or replace its question text (sources/sami_item_fixes.yaml).
    fixes_path = sources.parent / "sami_item_fixes.yaml"
    fixes = (yaml.safe_load(fixes_path.read_text(encoding="utf-8")) or {}) if fixes_path.exists() else {}
    kept = []
    for it in items:
        fx = fixes.get(it.id)
        if fx and fx.get("drop"):
            continue
        if fx and fx.get("question"):
            it.question = fx["question"]
        kept.append(it)
    dropped = len(items) - len(kept)
    items = kept
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json(exclude_none=True) + "\n")
    return {"round1": a, "round2": b, "dropped_in_review": dropped, "total": len(items)}
