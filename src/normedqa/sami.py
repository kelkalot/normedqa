"""Sámi reading set from open Sámi health texts.

Source: GiellaLT's open text collections for North, Lule and South Sámi
(github.com/giellalt/corpus-sme, -smj, -sma). Each document records its
original URL and points to its Norwegian, Swedish or Finnish version, which is
published in the matching collection (corpus-nob, -swe, -fin). Documents marked
"free" or "standard" in the collection are used (non-commercial, with attribution), pinned to a fixed commit.

The Sámi passages are human-written texts. Questions are written in Norwegian
from the original-language version, so every answer can be checked; Sámi
translations of the questions are machine drafts for a Sámi-speaking reviewer.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import requests
import yaml

LANGS = {"se": "corpus-sme", "smj": "corpus-smj", "sma": "corpus-sma"}
ORIGINALS = {"nob": "corpus-nob", "nno": "corpus-nno", "swe": "corpus-swe", "fin": "corpus-fin"}
RAW = "https://raw.githubusercontent.com/giellalt/{repo}/{commit}/converted/{path}"
API = "https://api.github.com/repos/giellalt/{repo}"

PATH_HINT = re.compile(r"helse|fhi/|1177|folkhals|legeforening|nkvts|kela/|dearvvas|dearvvaš|buohcc|pasie|psyk|demen|"
                       r"koron|covid|vaksin|varresvuod|healsoe|skihp|dalkk|medisin|lieggus", re.I)
HEALTH = re.compile(r"dearvvašvuo|dearvvasvuo|buohcc|buhcc|pasieanta|pasienta|dálkk|doavttir|doavtter|psyk|demens|"
                    r"korona|covid|vaksiin|njoammu|varresvuo|skihp|healsoe|skiemtje|daelkie|lieggus|sykehus|klinihk|"
                    r"terapi|giksi|gillámu|vigi", re.I)
# Documents about health services and policy (parliament, ministries) rather than medical or patient information.
POLICY_PATH = re.compile(r"(^|/)(sd|depts|nsr)/|regjeringen|samediggi", re.I)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "normedqa-dataset-builder/3 (+https://github.com/kelkalot/normedqa)"
    return s


def head_commit(session: requests.Session, repo: str) -> str:
    return session.get(API.format(repo=repo) + "/commits/main", timeout=30).json()["sha"]


def list_paths(session: requests.Session, repo: str, commit: str) -> list[str]:
    tree = session.get(API.format(repo=repo) + f"/git/trees/{commit}:converted?recursive=1", timeout=60).json()
    return [t["path"] for t in tree["tree"] if t["type"] == "blob"]


def parse_doc(text: str) -> dict:
    """Header fields and the paragraphs written in the document's own language."""
    hdr, _, body = text.partition("</header>")
    paras = []
    for m in re.finditer(r"<p( [^>]*)?>(.*?)</p>", body, re.S):
        attrs = m.group(1) or ""
        if "xml:lang" in attrs:  # menus and quotes in other languages
            continue
        t = html.unescape(re.sub(r"<[^>]+>", " ", m.group(2)))
        t = re.sub(r"\s+", " ", t).strip()
        if len(t.split()) >= 6 and 'type="title"' not in attrs:
            paras.append(t)
    lic = re.search(r'<license type="(\w+)"', hdr) or re.search(r"<availability><(\w+)", hdr)
    return {
        "title": html.unescape((re.search(r"<title>(.*?)</title>", hdr, re.S) or [None, ""])[1]).strip(),
        "year": (re.search(r"<year>(\d{4})", hdr) or [None, None])[1],
        "publisher": (re.search(r'lastname="([^"]*)"', hdr) or [None, ""])[1],
        "url": html.unescape((re.search(r"<origFileName>(.*?)</origFileName>", hdr) or [None, ""])[1]),
        "licence": lic.group(1) if lic else None,
        "parallels": dict(re.findall(r'<parallel_text xml:lang="(\w+)" location="([^"]+)"', hdr)),
        "paragraphs": paras,
    }


def discover(out: Path, cache: Path) -> list[dict]:
    """Find freely licensed Sámi health documents that have an original-language version; write the list."""
    s = _session()
    commits = {repo: head_commit(s, repo) for repo in list(LANGS.values()) + list(ORIGINALS.values())}
    docs = []
    for lang, repo in LANGS.items():
        for path in list_paths(s, repo, commits[repo]):
            if not PATH_HINT.search(path) or "dearvvahan" in path:
                continue
            text = _get(s, cache, repo, commits[repo], path)
            d = parse_doc(text)
            words = sum(len(p.split()) for p in d["paragraphs"])
            hits = len(HEALTH.findall(" ".join(d["paragraphs"])))
            if words < 80 or not (hits >= 4 or hits / max(words, 1) * 1000 >= 4) or d["licence"] not in ("free", "standard"):
                continue
            orig = next(((ol, loc) for ol in ("nob", "nno", "swe", "fin") for l2, loc in d["parallels"].items() if l2 == ol), None)
            if not orig:
                continue
            olang, loc = orig
            opath = str(Path(path).parent / (loc + ".xml"))
            try:
                otext = _get(s, cache, ORIGINALS[olang], commits[ORIGINALS[olang]], opath)
            except requests.HTTPError:
                continue
            od = parse_doc(otext)
            if sum(len(p.split()) for p in od["paragraphs"]) < 60:
                continue
            docs.append({
                "id": f"{lang}-{re.sub(r'[^a-z0-9]+', '-', Path(path).stem.lower())[:60].strip('-')}",
                "language": lang, "repo": repo, "commit": commits[repo], "path": path,
                "title": d["title"], "url": d["url"], "publisher": d["publisher"], "year": d["year"],
                "licence": d["licence"], "words": words,
                # GiellaLT marks texts "free" (redistributable) or "standard" (not defined in their documentation).
                # The "standard" texts here are publicly available health information from public bodies; the
                # project uses them non-commercially with full attribution of the original publisher (decided 2026-11-05).
                "use": True,
                "topic": "health policy" if POLICY_PATH.search(path) else "medical and patient information",
                "original": {"language": olang, "repo": ORIGINALS[olang], "commit": commits[ORIGINALS[olang]], "path": opath,
                             "url": od["url"]},
            })
    seen, unique, ids = set(), [], {}
    for d in docs:
        # The same document can sit in two folders or under two file names (hyphens vs underscores); compare content.
        text = (cache / d["repo"] / d["commit"][:12] / d["path"]).read_text(encoding="utf-8")
        content = " ".join(parse_doc(text)["paragraphs"])
        key = (d["language"], re.sub(r"\W+", "", content.lower())[:5000])
        if key in seen:
            continue
        seen.add(key)
        n = ids.get(d["id"], 0)
        ids[d["id"]] = n + 1
        if n:
            d["id"] = f"{d['id']}-{n + 1}"  # keep document ids unique
        unique.append(d)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"corpus_commits": commits, "documents": unique}, allow_unicode=True, sort_keys=False, width=200),
                   encoding="utf-8")
    return unique


def _get(s: requests.Session, cache: Path, repo: str, commit: str, path: str) -> str:
    dest = cache / repo / commit[:12] / path
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    r = s.get(RAW.format(repo=repo, commit=commit, path=requests.utils.quote(path)), timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.text, encoding="utf-8")
    return r.text


def passages(doc: dict, cache: Path, *, lo: int = 80, hi: int = 220) -> list[str]:
    """Consecutive paragraphs grouped into passages of about lo-hi words."""
    text = (cache / doc["repo"] / doc["commit"][:12] / doc["path"]).read_text(encoding="utf-8")
    out, cur = [], []
    for p in parse_doc(text)["paragraphs"]:
        n = len(p.split())
        if n > hi:  # a long paragraph is its own passage, cut at sentence ends
            if cur:
                out.append(" ".join(cur)); cur = []
            sent, buf = re.split(r"(?<=[.!?])\s+", p), []
            for x in sent:
                buf.append(x)
                if len(" ".join(buf).split()) >= lo + (hi - lo) // 2:
                    out.append(" ".join(buf)); buf = []
            if buf:
                out.append(" ".join(buf))
            continue
        cur.append(p)
        if len(" ".join(cur).split()) >= lo:
            out.append("\n".join(cur)); cur = []
    if cur and len(" ".join(cur).split()) >= lo // 2:
        out.append("\n".join(cur))
    return [x for x in out if lo // 2 <= len(x.split()) <= hi * 1.6]


def original_text(doc: dict, cache: Path, max_words: int = 2500) -> str:
    o = doc["original"]
    text = (cache / o["repo"] / o["commit"][:12] / o["path"]).read_text(encoding="utf-8")
    paras, total = [], 0
    for p in parse_doc(text)["paragraphs"]:
        total += len(p.split())
        if total > max_words:
            break
        paras.append(p)
    return "\n".join(paras)


# Original publisher, from the original URL's domain (or the collection folder when the URL is missing).
DOMAIN_PUBLISHER = {
    "sametinget.no": "Sámediggi / Sametinget (Norway)", "samediggi.no": "Sámediggi / Sametinget (Norway)",
    "saemiedigkie.no": "Sámediggi / Sametinget (Norway)", "samedigge.no": "Sámediggi / Sametinget (Norway)",
    "samediggi.fi": "Sámediggi / Saamelaiskäräjät (Finland)",
    "1177.se": "1177 Vårdguiden (Sweden)", "folkhalsomyndigheten.se": "Folkhälsomyndigheten (Sweden)",
    "helsenorge.no": "Helsenorge", "fhi.no": "Folkehelseinstituttet (FHI)", "nkvts.no": "NKVTS",
    "regjeringen.no": "Regjeringen", "nasjonalforeningen.no": "Nasjonalforeningen for folkehelsen",
    "aoh-prod.s3.amazonaws.com": "Nasjonal kompetansetjeneste for aldring og helse",
    "samas.no": "Sámi allaskuvla / Samisk høgskole", "siida.fi": "Saamelaismuseo Siida",
    "samilogutmuitalit.no": "Sámi logut muitalit (Samiske tall forteller)", "arran.no": "Árran julevsáme guovdásj",
    "nord-salten.no": "Lokalavisa NordSalten", "jll.se": "Region Jämtland Härjedalen",
    "lansstyrelsen.se": "Länsstyrelsen (Sweden)", "royrvik.kommune.no": "Røyrvik kommune", "diva-portal.se": "DiVA",
    "kela.fi": "Kela (Finland)", "helsedirektoratet.no": "Helsedirektoratet",
}
FOLDER_PUBLISHER = {"sd": "Sámediggi / Sametinget (Norway)", "helsenorge": "Helsenorge", "fhi": "Folkehelseinstituttet (FHI)",
                    "1177.se": "1177 Vårdguiden (Sweden)", "kela": "Kela (Finland)", "nkvts": "NKVTS",
                    "helsedirektoratet": "Helsedirektoratet", "samisk_legeforening": "Samisk legeforening"}


# Documents without URL or author in the corpus, identified through a parallel version that has one.
PUBLISHER_OVERRIDES = {
    # Same Norwegian original ("Fakta om demens") as the South Sámi demens_norsk_sorsamisk.pdf from nasjonalforeningen.no.
    "facta/other_files/demens_sme.txt.xml": "Nasjonalforeningen for folkehelsen",
}


def publisher_for(doc: dict) -> tuple[str, str | None]:
    """(publisher, author or None) for a corpus document."""
    from urllib.parse import urlparse

    if doc["path"] in PUBLISHER_OVERRIDES:
        return PUBLISHER_OVERRIDES[doc["path"]], None

    dom = urlparse(doc.get("url") or "").netloc.replace("www.", "")
    pub = DOMAIN_PUBLISHER.get(dom)
    if not pub:
        folder = doc["path"].split("/")[1] if doc["path"].count("/") >= 2 else ""
        pub = FOLDER_PUBLISHER.get(folder)
    author = (doc.get("publisher") or "").strip() or None
    if author and pub and author.lower().replace(".no", "") in pub.lower():
        author = None
    return pub or author or "unknown", author if pub else None
