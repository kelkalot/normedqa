# NorMedQA

NorMedQA tests how well language models answer medical questions in Norwegian. Version 3 has two parts:

- **Exam questions**: 2,260 multiple-choice questions from the Norwegian national written final exam in medicine, 16 sittings from spring 2017 to spring 2026. Most come in both Bokmål and Nynorsk. Each has the official answer key and the official explanation for every option. 457 of the 3,940 items include an image (ECG, X-ray, CT, clinical photo and so on).
- **Sámi reading**: 1,662 questions about health texts in North Sámi (1,024), Lule Sámi (345) and South Sámi (293). Each question is asked about a Sámi passage, which also comes in its original Norwegian, Swedish or Finnish version. The questions and answer options come in Bokmål and in the passage's Sámi language. The Sámi versions start as machine drafts and are being reviewed by Sámi-speaking physicians ([#5](https://github.com/kelkalot/normedqa/issues/5)); 200 exam questions are also available in North Sámi.

v3 replaces v2. The v2 data and code are kept in `legacy/v2/` so earlier results can be reproduced.

## What's in `data/v3`

| File | Contents |
|---|---|
| `exam.jsonl` | 3,940 exam items, one question in one written form per line |
| `images/` | 335 images, named by the first 16 hex digits of their SHA-256 |
| `sami_reading.jsonl` | 1,662 Sámi reading items, questions in Bokmål |
| `sami_reading_sami.jsonl` | the same 1,662 items with question and options in Sámi (machine drafts, replaced by reviewed text over time) |
| `exam_sami.jsonl` | 200 exam questions in North Sámi (machine drafts, replaced by reviewed text over time) |

Every item has:
- `id` and `group_id`; the Bokmål and Nynorsk versions of a question share a `group_id`
- `language`: `nb`, `nn`, `se`, `smj` or `sma`
- `question`, `options` (each with `label`, `text` and, for exam items, the official `explanation`) and `answer`, the list of accepted labels
- `status`: `ok`, `voided` (removed from the exam at grading) or `excluded` (kept, but the source document is defective)
- `split`
- `source`: URL, document title, publisher, retrieval date and SHA-256 of the source document

Exam items also carry:
- `exam` (sitting, name), `question_number` and `images`
- `requires_image`: true when an image belongs to the question itself
- `original_answer`: when the key was changed at grading, the key before the change
- `also_in`: ids of the same question reused in another sitting

Sámi items also carry:
- `passage`, `passage_original` and `passage_original_language`
- `question_language`: the questions are in Bokmål
- `flags`

### Scoring rules

Score only items with `status == "ok"`. When the exam key was changed at grading, several options can be accepted, so a prediction is correct when it's any label in `answer`. 101 exam questions were voided at grading and 3 Nynorsk items are excluded because their source document is missing a question sentence or an option.

### Splits

Exam items are split by sitting, so both written forms of a question always land in the same split:

| Split | Sittings | Questions | Scored items |
|---|---|---|---|
| train | 2017V to 2024H | 1,840 | 3,011 |
| dev | 2025V | 140 | 272 |
| test | 2025H, 2026V | 280 | 553 |

Nine questions appear word for word in two sittings, two of them in the test split. Check `also_in` if you want to report test scores without them. The Sámi reading set is all `test`.

## Using it

```
uv sync --extra eval
export OPENAI_API_KEY=...
uv run normedqa eval --model gpt-5.4-mini --split test
uv run normedqa eval-report
```

`eval` shows each question with its options (and images) and asks for the letter of the answer. Results are cached in `build/eval/`, so an interrupted run picks up where it stopped. Use `--base-url` for any OpenAI-compatible server, such as vLLM or Ollama, and `--no-images` to see how much the image questions depend on the image. Add `--file data/v3/sami_reading_sami.jsonl` to run the Sámi set in Sámi, or `--file data/v3/sami_reading.jsonl` for the same items with Bokmål questions.

## Baselines

| Model | All scored exam items | Test split | Test, image questions | Test, image questions with the image removed |
|---|---|---|---|---|
| gpt-5.4-mini | 92.4% (n=3,836) | 91.7% | 75.0% (n=44) | 79.1% across all 449 image questions |
| gpt-5.5 | not run | 95.1% (n=553) | 88.6% (n=44) | 84.1% (n=44) |

Both models used low reasoning effort and the options in their original order. Bokmål and Nynorsk scores are within one point of each other for both models. Most image questions can be answered from the text alone; `requires_image` only says that an image is printed with the question.

With the questions in Bokmål, gpt-5.4-mini answers 1,281 of the 1,662 Sámi-set questions correctly without seeing the passage. Those items carry the flag `answerable_without_passage`. The other 381 (239 North, 65 Lule and 77 South Sámi) need the passage. With the questions in Sámi (`sami_reading_sami.jsonl`), every item also tests whether the model understands the Sámi question and options. The flag comes from a check with the Bokmål questions only.

## How the data was built

The pipeline is a Python package with a command-line tool. You need [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run normedqa discover    # list the documents on the past-exam pages into sources/sources.yaml
uv run normedqa fetch       # download them to raw/ and check their SHA-256
uv run normedqa build       # parse the answer keys into data/v3/exam.jsonl and data/v3/images/
uv run normedqa qa          # check the result against independent copies; report in build/qa/
```

The published answer keys mark the correct option with an X and print an explanation under each option. `build` reads that layout directly from the PDF text, with positions and fonts. It doesn't use a language model. When the key was changed at grading, the key lists points per option, and every option with points above zero counts as correct. Grading notes are printed in red and end up in `status_note`.

`qa` compares the build with three independent sources:

- **The unkeyed copy of each exam:** the question and option text matches for all 3,940 items.
- **Fagprøven's key** for the same paper (2021 onwards, with the options in a different order): our accepted answers match on 2,794 of 2,800 items. The other 6 are 3 questions that Fagprøven graded differently. They're noted on the items, and we follow the national exam's key.
- **The other written form of the same question:** answers, options and images agree for every pair except the 3 excluded Nynorsk items.

A vision model also compared 172 randomly chosen items with the rendered PDF pages. All answers and images were correct; the only text differences were typos that are in the printed source.

The Sámi set is built with `sami-discover`, `sami-questions`, `sami-check` and `sami-build`:

1. A model writes the questions in Bokmål (`sami-questions --round 1`) from the original-language version of each passage and quotes the supporting sentence in both languages (gpt-5.5 for the freely licensed medical texts, gpt-5.4-mini for the rest).
2. A question is kept only if both quotes are found in the texts, a second model given only the original picks the same answer, and the question is in Norwegian. Questions about documents that GiellaLT holds twice are left out.
3. Options are shuffled with a fixed seed per question, and "den norske/svenske/finske teksten" in a question becomes "teksten", since the reader sees the Sámi passage.

Most questions from that first round turned out to be answerable without the passage. A second round (gpt-5.4-mini, `sami-questions --round 2`) asked for details that can't be guessed, with wrong options built from other details in the same text, and kept only questions the check model got wrong without the passage. Those 289 items have ids ending in `-h1`, `-h2` and so on. The commands for the second round are the same, with `--out build/sami/gen_hard.jsonl` and `--gen build/sami/gen_hard.jsonl --out build/sami/checks_hard.jsonl`.

Keeping only questions that a model gets wrong without the passage also keeps questions with a wrong key or a missing answer, since those defeat the model too. So every passage-dependent question was checked by hand against the original text (by Claude, with the original-language text only). Of 397, 363 were correct as written, 18 were reworded, and 16 were dropped because the passage doesn't answer them, two options were correct, or the correct answer wasn't among the options. Each change is listed with its reason in `sources/sami_item_fixes.yaml`.

## Sámi versions of the questions

The goal of the Sámi set is to have everything in Sámi: the passage, the question and the answer options. `sami_reading_sami.jsonl` has every reading question in the language of its passage, and `exam_sami.jsonl` has 200 exam questions, drawn from all 16 sittings, in North Sámi.

All of them start as machine drafts: `translation: "machine_draft"`, with `uncertain_terms` listing the words the draft wasn't sure of. Sámi-speaking physicians review them one by one, and [issue #5](https://github.com/kelkalot/normedqa/issues/5) tracks the progress. A reviewed item gets `translation: "reviewed"` and the corrected text; an item flagged in review gets `status: "excluded"` with the reviewer's comment. Answer keys and passages come from the Bokmål item (`translated_from`), so you can compare results in Sámi and Norwegian on the same questions. Until most items are reviewed, treat Sámi scores as provisional, especially for Lule and South Sámi.

To add reviews, append the review app's export to `sources/sami_reviews.jsonl` and run `uv run normedqa sami-translations`. The machine drafts themselves are kept in `sources/sami_drafts.jsonl`.

We chose the drafting system by translating 120 Norwegian sentences that have official Sámi translations. The table shows chrF++ against the official translations:

| | North Sámi | Lule Sámi | South Sámi |
|---|---|---|---|
| Claude | 61.3 | 46.2 | 55.3 |
| gpt-5.5 | 59.7 | 44.1 | 51.8 |
| gpt-5.4-mini | 52.1 | 31.3 | 36.1 |

## Data sources

**Exam questions.** The questions come from the national written final exam in medicine. It's called *Felles avsluttende skriftlig deleksamen i medisin* from 2021, and *Nasjonal delprøve i medisin* in 2017–2020. We used the answer keys published after grading (*revidert fasit*), and checked them against the unkeyed copies, the keys of *Fagprøven for leger utdannet utenfor EU/EØS og Sveits* (the same paper) and the psychometric notes for each sitting.

The documents are listed at:
- <https://www.uio.no/studier/program/medisin/tidligere-eksamensoppgaver/felles-avsluttende-deleksamen/>
- <https://www.uio.no/studier/program/medisin/tidligere-eksamensoppgaver/fagproven/>

`sources/sources.yaml` lists every document used, with its URL, title, retrieval date and SHA-256. We extract the text and images; the original PDFs aren't redistributed.

**Sámi texts.** The passages come from the open text collections of [GiellaLT](https://giellalt.github.io/) at UiT Norgga árktalaš universitehta:
- the Sámi texts: [corpus-sme](https://github.com/giellalt/corpus-sme), [corpus-smj](https://github.com/giellalt/corpus-smj) and [corpus-sma](https://github.com/giellalt/corpus-sma)
- their originals: [corpus-nob](https://github.com/giellalt/corpus-nob), [corpus-swe](https://github.com/giellalt/corpus-swe) and [corpus-fin](https://github.com/giellalt/corpus-fin)

`sources/sami_sources.yaml` records the commit of each collection and, for every document, its path, original URL and licence field.

The original publishers are:
- **Norway:** Helsenorge, Folkehelseinstituttet (FHI), NKVTS, Nasjonalforeningen for folkehelsen, Nasjonal kompetansetjeneste for aldring og helse, Sámediggi / Sametinget, Regjeringen, Sámi allaskuvla, Árran julevsáme guovdásj, Røyrvik kommune, Lokalavisa NordSalten and Sámi logut muitalit
- **Sweden:** 1177 Vårdguiden, Folkhälsomyndigheten, Region Jämtland Härjedalen and Länsstyrelsen
- **Finland:** Sámediggi / Saamelaiskäräjät, Saamelaismuseo Siida and a thesis in DiVA

Each item names its publisher in `source.publisher`. GiellaLT marks texts as `free` or `standard`. All of them are publicly available texts, and NorMedQA uses them for non-commercial research with attribution.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can share and adapt NorMedQA for non-commercial purposes if you credit it and its sources (see Data sources). v2 was released under CC BY 4.0; v3 adds Sámi texts that are used for non-commercial research only.

## Citation

v3 (the Zenodo DOI is added when the release is published):

```bibtex
@dataset{normedqa_v3_2026,
  author       = {Riegler, Michael Alexander and Gautam, Sushant and Gran Erke, Maja},
  title        = {{NorMedQA: Norwegian Medical Question Answering Benchmark}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.0}
}
```

v2:

```bibtex
@dataset{riegler_michael_alexander_2025_15320038,
  author       = {Riegler, Michael Alexander and Gautam, Sushant},
  title        = {{Norwegian Medical Question Answering Dataset - NorMedQA}},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.15320038},
  url          = {https://doi.org/10.5281/zenodo.15320037}
}
```


## How this release was made

The v3 pipeline, the quality checks and this README were built with Claude Code, using Claude Opus 5.5 (`claude-opus-5-5`). Claude also drafted the Sámi translations that are under review and audited the passage-dependent Sámi questions against their original texts. The Sámi reading questions were written by gpt-5.5 and gpt-5.4-mini, and gpt-5.5 compared a sample of exam items with the rendered PDF pages. The exam questions and answers themselves come straight from the published answer keys; no model wrote or changed them.
