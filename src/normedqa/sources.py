"""Find and download the source documents.

``discover`` reads UiO's listing pages for past exams and writes one entry per
document to ``sources/sources.yaml``. ``fetch`` downloads every entry to
``raw/`` and records (or checks) its SHA-256, so a build can be repeated with
the same files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests
import yaml
from bs4 import BeautifulSoup

BASE = "https://www.uio.no/studier/program/medisin/tidligere-eksamensoppgaver/"
PUBLISHER = "Det medisinske fakultet, Universitetet i Oslo"
USER_AGENT = "normedqa-dataset-builder/3 (+https://github.com/kelkalot/normedqa)"
DELAY_S = 2.0

# Listing pages for the national joint final exam and for Fagprøven (same paper, used as a cross-check).
LISTINGS = {
    "felles": [f"{BASE}felles-avsluttende-deleksamen/{y}/" for y in range(2017, 2027)],
    "fagproven": [f"{BASE}fagproven/{y}/" for y in range(2020, 2027)],
}


# What each kind of document is used for. Documents without a use are listed but not downloaded.
DEFAULT_USE = {
    ("felles", "key"): "exam",
    ("felles", "psychometrics"): "psychometrics",
    ("felles", "nokey"): "textcheck",  # unkeyed copy of the same exam; checks the extracted text
    ("fagproven", "key"): "crosscheck",  # Fagprøven is the same paper; its key checks ours
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get(session: requests.Session, url: str, *, stream: bool = False, tries: int = 4) -> requests.Response:
    """GET with a polite delay and retries. UiO sometimes drops connections."""
    for attempt in range(tries):
        time.sleep(DELAY_S * (attempt + 1))
        try:
            r = session.get(url, timeout=60, stream=stream)
            if r.status_code == 404:
                r.raise_for_status()
            if r.ok:
                return r
        except requests.HTTPError:
            raise
        except requests.RequestException:
            if attempt == tries - 1:
                raise
    r.raise_for_status()
    return r


def classify(title: str, page_url: str) -> dict:
    """Read kind, sitting and written form from a document title as published."""
    t = unquote(title).lower()
    # "uten"/"utan" = without (answer key); careful not to match "utenfor"/"utanfor" (outside, as in "utenfor EU").
    without = re.search(r"\buten\b|\butan\b|utenfasit|utanfasit", t)
    if "psykometri" in t:
        kind = "psychometrics"
    elif "fasit" in t and not without:
        kind = "key"
    elif without or "oppgavetekst" in t:
        kind = "nokey"
    else:
        kind = "other"

    if re.search(r"bokm[åa]l", t):
        language = "nb"
    elif "nynorsk" in t:
        language = "nn"
    else:
        language = None  # older sittings are published in one unlabelled version

    year = None
    m = re.search(r"\b(20\d\d)\b", t)
    if m:
        year = int(m.group(1))
    term = None
    if re.search(r"\bv[åa]r\b|v[åa]r-|\bv(20)?\d\d\b|_v\d\d_", t):
        term = "V"
    elif re.search(r"h[øo]st|haust|\bh(20)?\d\d\b|_h\d\d_", t):
        term = "H"
    m = re.search(r"\b([vh])(\d\d)\b|_([vh])(\d\d)_|\b([vh])(20\d\d)\b", t)
    if m:
        letter = next(g for g in (m.group(1), m.group(3), m.group(5)) if g)
        digits = next(g for g in (m.group(2), m.group(4), m.group(6)) if g)
        term = term or letter.upper()
        year = year or (int(digits) if len(digits) == 4 else 2000 + int(digits))
    if year is None:
        m = re.search(r"/(20\d\d)/", page_url)
        year = int(m.group(1)) if m else None
    return {"kind": kind, "language": language, "year": year, "term": term}


def discover(out: Path) -> list[dict]:
    """Collect document links from the listing pages and write them to ``out``."""
    session = _session()
    docs: list[dict] = []
    seen: set[str] = set()
    for series, pages in LISTINGS.items():
        for page_url in pages:
            try:
                r = _get(session, page_url)
            except requests.HTTPError:
                continue  # no page for that year
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("a.vrtx-title-link"):
                url = a["href"]
                if url in seen:
                    continue
                seen.add(url)
                title = a.get_text(" ", strip=True)
                info = classify(title, page_url)
                sitting = f"{info['year']}{info['term']}" if info["year"] and info["term"] else None
                docs.append(
                    {
                        "id": _doc_id(series, info, sitting, title),
                        "series": series,
                        "kind": info["kind"],
                        "sitting": sitting,
                        "language": info["language"],
                        "title": title,
                        "url": url,
                        "page_url": page_url,
                    }
                )
    docs.sort(key=lambda d: (d["series"], d["sitting"] or "", d["kind"], d["language"] or ""))
    for d in docs:
        d["use"] = DEFAULT_USE.get((d["series"], d["kind"]))
    existing = load(out) if out.exists() else {"documents": []}
    manual = [d for d in existing.get("documents", []) if d.get("manual")]
    known = {d["url"]: d for d in existing.get("documents", [])}
    for d in docs:  # keep hashes, decisions and notes from an earlier run
        old = known.get(d["url"], {})
        for k in ("sha256", "retrieved", "use", "note"):
            if k in old:
                d[k] = old[k]
        if d["language"] is None and old.get("language"):
            d["language"] = old["language"]  # set by hand for unlabelled versions
    save(out, {"documents": docs + manual})
    return docs


def _doc_id(series: str, info: dict, sitting: str | None, title: str) -> str:
    parts = [series, sitting or "unknown", info["kind"], info["language"] or "xx"]
    base = "-".join(parts)
    # Disambiguate rare duplicates (e.g. both "fasit" and "revidert fasit" for one sitting).
    if "revidert" in title.lower():
        base += "-rev"
    return base


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"documents": []}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")


def raw_path(raw_dir: Path, doc: dict) -> Path:
    return raw_dir / f"{doc['id']}.pdf"


def fetch(sources: Path, raw_dir: Path, *, only_used: bool = True) -> None:
    """Download documents to ``raw_dir``; record SHA-256 on first download, check it afterwards."""
    data = load(sources)
    session = _session()
    raw_dir.mkdir(parents=True, exist_ok=True)
    changed = False
    for doc in data["documents"]:
        if only_used and not doc.get("use"):
            continue
        dest = raw_path(raw_dir, doc)
        if not dest.exists():
            r = _get(session, doc["url"])
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError(f"{doc['id']}: not a PDF ({r.headers.get('content-type')})")
            dest.write_bytes(r.content)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if "sha256" not in doc:
            doc["sha256"] = digest
            doc["retrieved"] = dt.date.today().isoformat()
            changed = True
        elif doc["sha256"] != digest:
            raise RuntimeError(f"{doc['id']}: hash mismatch, the published file has changed (got {digest})")
    if changed:
        save(sources, data)
