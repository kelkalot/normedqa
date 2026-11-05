"""Export the release to a Hugging Face dataset folder (Parquet, images embedded).

Layout:  <out>/exam/{train,validation,test}.parquet, <out>/sami_reading/test.parquet, <out>/README.md
Upload with:  huggingface-cli upload <repo> <out> --repo-type dataset   (or HfApi.upload_folder)
"""

from __future__ import annotations

import json
from pathlib import Path

SPLIT_NAME = {"train": "train", "dev": "validation", "test": "test"}


def _row(it: dict, data_dir: Path) -> dict:
    return {
        "id": it["id"],
        "group_id": it["group_id"],
        "language": it["language"],
        "sitting": (it.get("exam") or {}).get("sitting"),
        "question_number": it.get("question_number"),
        "passage": it.get("passage"),
        "passage_original": it.get("passage_original"),
        "passage_original_language": it.get("passage_original_language"),
        "question": it["question"],
        "option_labels": [o["label"] for o in it["options"]],
        "options": [o["text"] for o in it["options"]],
        "explanations": [o.get("explanation") or "" for o in it["options"]],
        "points": [o.get("points") for o in it["options"]],
        "answer": it["answer"],
        "original_answer": it.get("original_answer") or [],
        "status": it["status"],
        "status_note": it.get("status_note"),
        "images": [{"bytes": (data_dir / im["file"]).read_bytes(), "path": Path(im["file"]).name}
                   for im in it.get("images", []) if im["position"] == "stem"],
        "requires_image": it.get("requires_image", False),
        "also_in": it.get("also_in", []),
        "flags": it.get("flags", []),
        "topic": it.get("topic"),
        "source_url": it["source"]["url"],
        "source_title": it["source"]["title"],
        "source_publisher": it["source"]["publisher"],
        "source_sha256": it["source"]["sha256"],
        "source_pages": it["source"].get("pages", []),
        "source_corpus_path": it["source"].get("corpus_path"),
        "source_corpus_commit": it["source"].get("corpus_commit"),
    }


def export(data_dir: Path, out: Path) -> dict:
    from datasets import Dataset, Features, Image, Sequence, Value

    features = Features({
        "id": Value("string"), "group_id": Value("string"), "language": Value("string"), "sitting": Value("string"),
        "question_number": Value("string"), "passage": Value("string"), "passage_original": Value("string"),
        "passage_original_language": Value("string"), "question": Value("string"),
        "option_labels": Sequence(Value("string")), "options": Sequence(Value("string")),
        "explanations": Sequence(Value("string")), "points": Sequence(Value("float32")),
        "answer": Sequence(Value("string")), "original_answer": Sequence(Value("string")),
        "status": Value("string"), "status_note": Value("string"), "images": Sequence(Image()),
        "requires_image": Value("bool"), "also_in": Sequence(Value("string")), "flags": Sequence(Value("string")),
        "topic": Value("string"), "source_url": Value("string"), "source_title": Value("string"),
        "source_publisher": Value("string"), "source_sha256": Value("string"), "source_pages": Sequence(Value("int32")),
        "source_corpus_path": Value("string"), "source_corpus_commit": Value("string"),
    })
    counts = {}
    for config, fname in (("exam", "exam.jsonl"), ("sami_reading", "sami_reading.jsonl")):
        items = [json.loads(l) for l in (data_dir / fname).read_text(encoding="utf-8").splitlines() if l.strip()]
        for split in ("train", "dev", "test"):
            rows = [_row(it, data_dir) for it in items if it.get("split") == split]
            if not rows:
                continue
            path = out / config / f"{SPLIT_NAME[split]}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows, features=features).to_parquet(str(path))
            counts[f"{config}/{SPLIT_NAME[split]}"] = len(rows)
    return counts


def export_space(data_dir: Path, template_dir: Path, out: Path, max_side: int = 900) -> dict:
    """Static Space: the browser page, a compact data.json and downscaled JPEG copies of the images."""
    import shutil

    import pymupdf

    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    shutil.copy(template_dir / "index.html", out / "index.html")
    shutil.copy(template_dir / "README.md", out / "README.md")
    exam = [json.loads(l) for l in (data_dir / "exam.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sami = [json.loads(l) for l in (data_dir / "sami_reading.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    def opts(it):
        return [[o["label"], o["text"], o.get("explanation"), o.get("points")] for o in it["options"]]

    data = {
        "exam": [{"id": it["id"], "g": it["group_id"], "s": it["exam"]["sitting"], "l": it["language"], "split": it["split"],
                  "q": it["question"], "st": it["status"], "note": it.get("status_note"), "ans": it["answer"],
                  "orig": it.get("original_answer"), "also": it.get("also_in", []), "o": opts(it),
                  "img": [Path(i["file"]).stem + ".jpg" for i in it.get("images", []) if i["position"] == "stem"],
                  "src": it["source"]["url"], "title": it["source"]["title"], "pages": it["source"]["pages"]} for it in exam],
        "sami": [{"id": it["id"], "l": it["language"], "p": it["passage"], "po": it.get("passage_original"),
                  "pol": it.get("passage_original_language"), "q": it["question"], "ans": it["answer"], "o": opts(it),
                  "flags": it.get("flags", []), "topic": it.get("topic") or "", "pub": it["source"]["publisher"],
                  "src": it["source"]["url"], "title": it["source"]["title"]} for it in sami],
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n = 0
    for f in sorted((data_dir / "images").iterdir()):
        pix = pymupdf.Pixmap(str(f))
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        if pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        scale = min(1.0, max_side / max(pix.width, pix.height))
        if scale < 1:
            pix = pymupdf.Pixmap(pix, int(pix.width * scale), int(pix.height * scale), None)
        (out / "img" / (f.stem + ".jpg")).write_bytes(pix.tobytes("jpg", jpg_quality=75))
        n += 1
    return {"exam": len(data["exam"]), "sami": len(data["sami"]), "images": n}
