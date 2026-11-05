"""Command-line interface: ``normedqa <step>``.

Exam pipeline:   discover -> fetch -> build -> qa
Sámi pipeline:   sami-discover -> sami-questions -> sami-check -> sami-build
Evaluation:      eval (OpenAI-compatible chat API; set OPENAI_API_KEY, optional --base-url), eval-report
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path.cwd()
SOURCES = ROOT / "sources" / "sources.yaml"
SAMI_SOURCES = ROOT / "sources" / "sami_sources.yaml"
RAW = ROOT / "raw"
SAMI_CACHE = RAW / "sami"
SAMI_BUILD = ROOT / "build" / "sami"
# The release: extracted text and images. The original PDFs stay in raw/ and are not redistributed.
OUT = ROOT / "data" / "v3"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="normedqa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover", help="list documents on the past-exam pages into sources/sources.yaml")
    f = sub.add_parser("fetch", help="download the documents marked for use, record or check their hashes")
    f.add_argument("--all", action="store_true", help="download every listed document, not only those in use")
    sub.add_parser("build", help="parse the keyed exams and write data/v3/exam.jsonl and images")
    sub.add_parser("qa", help="check the built items against independent sources; writes build/qa/qa_report.json")

    sub.add_parser("sami-discover", help="list Sámi health texts and their originals (GiellaLT) into sources/sami_sources.yaml")
    g = sub.add_parser("sami-questions", help="write reading questions for the Sámi passages with a model")
    g.add_argument("--model", default="gpt-5.5")
    g.add_argument("--round", type=int, choices=[1, 2], default=2, help="1: first-round instructions, 2: stricter ones")
    g.add_argument("--out", type=Path, default=SAMI_BUILD / "gen.jsonl")
    c = sub.add_parser("sami-check", help="check written questions: quotes, answer from the original, answer without text")
    c.add_argument("--model", default="gpt-5.4-mini")
    c.add_argument("--gen", type=Path, default=SAMI_BUILD / "gen.jsonl")
    c.add_argument("--out", type=Path, default=SAMI_BUILD / "checks.jsonl")
    sub.add_parser("sami-build", help="write data/v3/sami_reading.jsonl from both rounds of checked questions in build/sami")

    e = sub.add_parser("eval", help="multiple-choice accuracy of a model on a release file")
    e.add_argument("--model", required=True)
    e.add_argument("--file", type=Path, default=OUT / "exam.jsonl")
    e.add_argument("--split", choices=["train", "dev", "test"])
    e.add_argument("--language", help="nb, nn, se, smj or sma")
    e.add_argument("--no-images", action="store_true", help="leave images out (to test whether a question needs them)")
    e.add_argument("--shuffle", action="store_true", help="shuffle the options (fixed seed per item)")
    e.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. a local vLLM or Ollama server")
    e.add_argument("--reasoning-effort", default="low")
    e.add_argument("--out", type=Path, default=ROOT / "build" / "eval")
    sub.add_parser("eval-report", help="accuracy per model, written form, sitting and image use from build/eval")
    sub.add_parser("sami-translations", help="build the Sámi-question files from sources/sami_drafts.jsonl and sami_reviews.jsonl")
    sub.add_parser("crosswalk", help="map every v2 item to its v3 counterpart; writes legacy/v2/crosswalk_v2_v3.csv")
    args = p.parse_args(argv)

    if args.cmd == "discover":
        from .sources import discover

        docs = discover(SOURCES)
        print(f"{len(docs)} documents listed in {SOURCES.relative_to(ROOT)}")
    elif args.cmd == "fetch":
        from .sources import fetch

        fetch(SOURCES, RAW, only_used=not args.all)
    elif args.cmd == "build":
        from .build import build_exams

        build_exams(SOURCES, RAW, OUT)
    elif args.cmd == "qa":
        from .qa import run

        run(OUT / "exam.jsonl", SOURCES, RAW, ROOT / "build" / "qa")
    elif args.cmd == "sami-discover":
        from .sami import discover as sami_discover

        docs = sami_discover(SAMI_SOURCES, SAMI_CACHE)
        print(f"{len(docs)} Sámi documents listed in {SAMI_SOURCES.relative_to(ROOT)}")
    elif args.cmd == "sami-questions":
        from .sami_questions import generate, plan

        generate(plan(SAMI_SOURCES, SAMI_CACHE), args.out, args.model, round_=args.round)
    elif args.cmd == "sami-check":
        from .sami_questions import verify

        verify(args.gen, args.out, args.model)
    elif args.cmd == "sami-build":
        from .sami_questions import build_all

        print(build_all(SAMI_BUILD, SAMI_SOURCES, SAMI_CACHE, OUT / "sami_reading.jsonl"))
    elif args.cmd == "eval":
        from .evaluate import run as run_eval

        def keep(it: dict) -> bool:
            return (not args.split or it.get("split") == args.split) and (not args.language or it["language"] == args.language)

        cond = ("noimage" if args.no_images else "default") + ("-shuffled" if args.shuffle else "")
        run_eval(args.file, args.out, args.model, condition=f"{args.file.stem}-{cond}", with_images=not args.no_images,
                 shuffle=args.shuffle, filter_fn=keep, base_url=args.base_url, reasoning_effort=args.reasoning_effort)
    elif args.cmd == "sami-translations":
        from .sami_translations import build as build_translations

        print(build_translations(OUT, ROOT / "sources"))
    elif args.cmd == "crosswalk":
        from .crosswalk import build as build_crosswalk

        legacy = ROOT / "legacy" / "v2"
        print(build_crosswalk(legacy / "data" / "norwegian_medical_qa_v2.json", legacy / "data" / "norwegian_medical_qa_v2_txt_only.json",
                              OUT / "exam.jsonl", legacy / "crosswalk_v2_v3.csv"))
    elif args.cmd == "eval-report":
        from .evaluate import summarize

        summarize(ROOT / "build" / "eval")


if __name__ == "__main__":
    main()
