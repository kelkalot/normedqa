"""Multiple-choice evaluation against an OpenAI-compatible chat API.

Each question is shown with its options (and its images, unless ``--no-images``)
and the model is asked for the letter of the correct option. A prediction is
correct if it is one of the accepted answers. Voided questions are skipped.
Results are cached per (model, condition, item) so a run can be resumed.
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INSTRUCTION = {
    "nb": "Svar med bokstaven til det riktigste alternativet. Skriv bare bokstaven.",
    "nn": "Svar med bokstaven til det mest rette alternativet. Skriv berre bokstaven.",
    "se": "Vástit dan vástádusa bustávain mii lea riektá. Čále dušše bustáva.",
    "smj": "Svar med bokstaven til det riktigste alternativet. Skriv bare bokstaven.",
    "sma": "Svar med bokstaven til det riktigste alternativet. Skriv bare bokstaven.",
    "en": "Answer with the letter of the best option. Write only the letter.",
}
LETTER = re.compile(r"\b([A-H])\b")


def build_prompt(item: dict, order: list[int], with_images: bool, data_dir: Path) -> list[dict]:
    lang = item["language"]
    opts = item["options"]
    labels = "ABCDEFGH"
    lines = []
    if item.get("passage"):
        lines += [item["passage"], ""]
    lines += [item["question"], ""]
    lines += [f"{labels[k]}. {opts[i]['text']}" for k, i in enumerate(order)]
    lines += ["", INSTRUCTION.get(lang, INSTRUCTION["nb"])]
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    if with_images:
        for im in item.get("images", []):
            if im["position"] != "stem":
                continue
            raw = (data_dir / im["file"]).read_bytes()
            mime = "image/png" if im["file"].endswith(".png") else "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"}})
    return [{"role": "user", "content": content}]


def run(items_path: Path, out_dir: Path, model: str, *, condition: str = "default", with_images: bool = True,
        shuffle: bool = False, seed: int = 0, limit: int | None = None, filter_fn=None, workers: int = 16,
        reasoning_effort: str | None = "low", base_url: str | None = None) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    data_dir = items_path.parent
    items = [json.loads(l) for l in items_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [it for it in items if it["status"] == "ok" and (filter_fn is None or filter_fn(it))]
    if limit:
        items = items[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{model}__{condition}.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    lock = threading.Lock()
    usage = defaultdict(int)

    def one(item: dict) -> dict:
        n = len(item["options"])
        order = list(range(n))
        if shuffle:
            random.Random(f"{seed}-{item['id']}").shuffle(order)
        kwargs = {}
        if reasoning_effort and model.startswith(("gpt-5", "o3", "o4")):
            kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(
            model=model, messages=build_prompt(item, order, with_images, data_dir), **kwargs
        )
        text = resp.choices[0].message.content or ""
        m = LETTER.search(text.strip().upper()[:20]) or LETTER.search(text.upper())
        shown = m.group(1) if m else None
        pred = item["options"][order["ABCDEFGH".index(shown)]]["label"] if shown and "ABCDEFGH".index(shown) < n else None
        u = resp.usage
        rec = {"id": item["id"], "group_id": item["group_id"], "language": item["language"],
               "sitting": item["exam"]["sitting"] if item.get("exam") else None,
               "requires_image": item.get("requires_image", False), "pred": pred, "answer": item["answer"],
               "correct": pred in item["answer"], "raw": text[:200],
               "tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens}
        with lock:
            usage["in"] += u.prompt_tokens
            usage["out"] += u.completion_tokens
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    todo = [it for it in items if it["id"] not in done]
    results = list(done.values())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, it) for it in todo]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:  # keep going; failed items are retried on the next run
                print("error:", str(e)[:200])
    keep = {it["id"] for it in items}
    results = [r for r in results if r["id"] in keep]
    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    summary = {"model": model, "condition": condition, "n": len(results), "accuracy": round(acc, 4),
               "new_tokens_in": usage["in"], "new_tokens_out": usage["out"]}
    print(json.dumps(summary))
    return summary


def summarize(results_dir: Path) -> None:
    """Accuracy per model and condition, broken down by written form, sitting and image use."""
    for path in sorted(results_dir.glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        def acc(rs):
            return f"{sum(r['correct'] for r in rs) / len(rs):.3f} (n={len(rs)})" if rs else "-"
        by = defaultdict(list)
        for r in rows:
            by[("lang", r["language"])].append(r)
            by[("sitting", r["sitting"])].append(r)
            by[("image", r["requires_image"])].append(r)
        print(f"\n{path.stem}: overall {acc(rows)}")
        for key in sorted(by, key=str):
            print(f"   {key[0]}={key[1]}: {acc(by[key])}")
