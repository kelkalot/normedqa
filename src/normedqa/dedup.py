"""Find questions that reappear in more than one sitting.

Some questions are reused word for word in later exams. A reused question in a
test sitting has been public since its first use, so the evaluation reports
scores with and without such items.
"""

from __future__ import annotations

import re
from collections import defaultdict


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    w = norm(text).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def reused_pairs(items: list[dict], threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Pairs of items (same written form, different sittings) whose question text overlaps strongly.

    Overlap is the Jaccard index of 5-word shingles over the question text. 0.6 also finds a shared case with a
    different question; ``same_question`` narrows that down using the options.
    """
    pairs = []
    by_lang = defaultdict(list)
    for it in items:
        by_lang[it["language"]].append(it)
    for its in by_lang.values():
        sh = {it["id"]: _shingles(it["question"]) for it in its}
        sitting = {it["id"]: it["exam"]["sitting"] for it in its}
        index = defaultdict(set)
        for iid, s in sh.items():
            for x in s:
                index[x].add(iid)
        seen = set()
        for iid, s in sh.items():
            counts = defaultdict(int)
            for x in s:
                for j in index[x]:
                    if j != iid and sitting[j] != sitting[iid]:
                        counts[j] += 1
            for j, c in counts.items():
                jac = c / len(s | sh[j])
                key = tuple(sorted((iid, j)))
                if jac >= threshold and key not in seen:
                    seen.add(key)
                    pairs.append((key[0], key[1], round(jac, 2)))
    return sorted(pairs)


def same_question(a: dict, b: dict) -> bool:
    """Same case and the same set of options (in any order)."""
    oa = {norm(o["text"]) for o in a["options"]}
    ob = {norm(o["text"]) for o in b["options"]}
    return len(oa & ob) >= max(len(oa), len(ob)) - 1


def annotate(items: list[dict]) -> int:
    """Fill ``also_in`` on each item with the ids of the same question in other sittings."""
    by_id = {it["id"]: it for it in items}
    n = 0
    for a, b, jac in reused_pairs(items, threshold=0.8):
        if same_question(by_id[a], by_id[b]):
            by_id[a].setdefault("also_in", []).append(b)
            by_id[b].setdefault("also_in", []).append(a)
            n += 1
    return n
