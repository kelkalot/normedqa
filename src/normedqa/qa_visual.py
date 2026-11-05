"""Visual spot check: a vision model compares rendered PDF pages with the extracted item.

This check does not use the parser on the reference side, so it catches errors
that the text cross-checks cannot (they parse both documents the same way).
"""

from __future__ import annotations

import base64
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from .sources import load

PROMPT = """You are checking a dataset extracted from a Norwegian medical exam answer key (PDF).
The page image(s) show question {num}. In this document the correct option is marked with "X" after the
letter. If the question begins with "NB Fasit ble endret", a second column gives points per option and every
option with points > 0 is accepted; all options at 0 means the question was voided.

Extracted item (JSON):
{item}

Compare the extraction with what is printed for question {num} only. Ignore line breaks, hyphenation at
line ends, bullet characters, the formatting of tables (cells may be separated by " | "), the explanations
printed under each option (they are not part of the option text), and the "NB Fasit ble endret" banner and
any red grading notes (deliberately left out of the question text). The text was taken from the PDF's text
layer, so do not report single-letter spelling differences or typos that are in the printed text; report
missing, extra or wrong words, numbers, units and options.
Report JSON with these keys:
  "question_ok": true if the question text (case + question sentence) matches, with nothing missing or extra,
  "options_ok": true if every option's text matches and no option is missing,
  "answer_ok": true if "answer" lists exactly the accepted option letter(s) (empty if voided),
  "image_ok": true if the extraction's "has_image" matches whether the question itself shows a picture,
  "problems": list of short strings describing each mismatch (empty if all ok).
"""


def render(pdf: Path, pages: list[int], dpi: int = 110) -> list[str]:
    doc = pymupdf.open(pdf)
    out = []
    for p in pages[:3]:
        pix = doc[p - 1].get_pixmap(dpi=dpi)
        out.append(base64.b64encode(pix.tobytes("png")).decode())
    return out


def sample(items: list[dict], per_group: int, n_images: int, n_changed: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[(it["exam"]["sitting"], it["language"])].append(it)
    chosen = {}
    for g in sorted(groups):
        for it in rng.sample(groups[g], min(per_group, len(groups[g]))):
            chosen[it["id"]] = it
    imgs = [it for it in items if it["requires_image"] and it["id"] not in chosen]
    for it in rng.sample(imgs, min(n_images, len(imgs))):
        chosen[it["id"]] = it
    changed = [it for it in items if it.get("original_answer") is not None and it["id"] not in chosen]
    for it in rng.sample(changed, min(n_changed, len(changed))):
        chosen[it["id"]] = it
    return list(chosen.values())


def run(items_path: Path, sources: Path, raw_dir: Path, out_path: Path, model: str, *, per_group: int = 4,
        n_images: int = 40, n_changed: int = 20, seed: int = 7, workers: int = 8) -> None:
    from openai import OpenAI

    client = OpenAI()
    items = [json.loads(l) for l in items_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = {d["url"]: d for d in load(sources)["documents"]}
    todo = sample(items, per_group, n_images, n_changed, seed)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [it for it in todo if it["id"] not in done]

    def one(it: dict) -> dict:
        doc = docs[it["source"]["url"]]
        pdf = raw_dir / f"{doc['id']}.pdf"
        view = {
            "question": it["question"],
            "options": [{"label": o["label"], "text": o["text"]} for o in it["options"]],
            "answer": it["answer"] if it["status"] != "voided" else [],
            "has_image": it["requires_image"],
        }
        content = [{"type": "text", "text": PROMPT.format(num=it["question_number"], item=json.dumps(view, ensure_ascii=False, indent=1))}]
        for b64 in render(pdf, it["source"]["pages"]):
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"}, reasoning_effort="low",
        )
        verdict = json.loads(resp.choices[0].message.content)
        rec = {"id": it["id"], **verdict, "tokens_in": resp.usage.prompt_tokens, "tokens_out": resp.usage.completion_tokens}
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(one, it) for it in todo]:
            try:
                f.result()
            except Exception as e:
                print("error:", str(e)[:200])
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = ["question_ok", "options_ok", "answer_ok", "image_ok"]
    print(f"{len(rows)} items checked; " + ", ".join(f"{k}: {sum(1 for r in rows if r.get(k))}" for k in keys))
    print(f"tokens in {sum(r['tokens_in'] for r in rows):,}, out {sum(r['tokens_out'] for r in rows):,}")
    for r in rows:
        if not all(r.get(k) for k in keys):
            print(" ", r["id"], r.get("problems"))
