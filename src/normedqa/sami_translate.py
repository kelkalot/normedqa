"""Machine drafts of Sámi translations, for review by a Sámi-speaking clinician.

Which system drafts is decided by measurement: translate Norwegian sentences that
have an official Sámi translation and score the output against it with chrF.
Drafts are stored with translation status "machine_draft" and are not scored
until a reviewer has corrected and approved them.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LANG = {"se": "North Sámi (davvisámegiella)", "smj": "Lule Sámi (julevsámegiella)", "sma": "South Sámi (åarjelsaemien gïele)"}

SENTENCE_PROMPT = """Translate the following Norwegian text into {lang}. Use the standard written norm and established
Sámi health terminology. Reply with the translation only.

{text}"""

ITEM_PROMPT = """Translate this Norwegian multiple-choice question and its options into {lang} for a medical benchmark.
Use the standard written norm and established Sámi health terminology. Keep drug names, units, numbers and lab
values exactly as they are. Where no established Sámi term exists, use the term Sámi health services use, or keep
the Norwegian/Latin term. Keep the number and order of options.

Return JSON only: {{"question": "...", "options": ["...", ...], "uncertain_terms": ["terms you were unsure about"]}}

{passage_block}Question:
{question}

Options:
{options}"""


def chrf(hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2).corpus_score(hyps, [refs]).score  # chrF++


def translate_sentences(client, model: str, lang: str, texts: list[str], workers: int = 12) -> list[str]:
    out = [None] * len(texts)

    def one(i):
        kwargs = {"reasoning_effort": "low"} if model.startswith("gpt-5") else {}
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": SENTENCE_PROMPT.format(lang=LANG[lang], text=texts[i])}], **kwargs)
        out[i] = (r.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(one, i) for i in range(len(texts))]))
    return out


def draft_items(items: list[dict], out: Path, model: str, *, workers: int = 12, include_passage: bool = False) -> None:
    """Translate question + options of each item into its target Sámi language (item["target"])."""
    from openai import OpenAI

    client = OpenAI()
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    lock = threading.Lock()

    def one(it):
        passage_block = f"(Context, already in {LANG[it['target']]}; do not translate it:)\n{it['passage']}\n\n" if include_passage and it.get("passage") else ""
        prompt = ITEM_PROMPT.format(lang=LANG[it["target"]], passage_block=passage_block, question=it["question"],
                                    options="\n".join(f"{chr(65 + k)}. {o}" for k, o in enumerate(it["options"])))
        kwargs = {"reasoning_effort": "low"} if model.startswith("gpt-5") else {}
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           response_format={"type": "json_object"}, **kwargs)
        res = json.loads(r.choices[0].message.content)
        if len(res.get("options", [])) != len(it["options"]):
            raise ValueError(f"{it['id']}: option count changed")
        rec = {"id": it["id"], "target": it["target"], "model": model, "question": res["question"],
               "options": res["options"], "uncertain_terms": res.get("uncertain_terms", []),
               "tokens_in": r.usage.prompt_tokens, "tokens_out": r.usage.completion_tokens}
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    todo = [it for it in items if it["id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, it) for it in todo]):
            try:
                f.result()
            except Exception as e:
                print("error:", str(e)[:160])
