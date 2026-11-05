---
pretty_name: NorMedQA
language:
- nb
- nn
- se
- smj
- sma
license: cc-by-nc-4.0
task_categories:
- multiple-choice
- question-answering
- visual-question-answering
tags:
- medical
- norwegian
- sami
- benchmark
size_categories:
- 1K<n<10K
configs:
- config_name: exam
  data_files:
  - split: train
    path: exam/train.parquet
  - split: validation
    path: exam/validation.parquet
  - split: test
    path: exam/test.parquet
- config_name: sami_reading
  data_files:
  - split: test
    path: sami_reading/test.parquet
---

# NorMedQA

NorMedQA tests how well language models answer medical questions in Norwegian. It has two configurations:

- **`exam`**: 2,260 multiple-choice questions from the Norwegian national written final exam in medicine, 16 sittings from spring 2017 to spring 2026. Most come in both Bokmål and Nynorsk, making 3,940 rows. Every question has the official answer key and the official explanation for each option. 457 rows include an image, such as an ECG, X-ray, CT or clinical photo.
- **`sami_reading`**: 1,662 questions about health texts in North Sámi (1,024), Lule Sámi (345) and South Sámi (293). The passages are written by people; the questions are in Bokmål.

Code, sources and the build pipeline: <https://github.com/kelkalot/normedqa>

```python
from datasets import load_dataset

exam = load_dataset("kelkalot/normedqa", "exam")
sami = load_dataset("kelkalot/normedqa", "sami_reading", split="test")
```

## Scoring

Score rows with `status == "ok"`. A prediction is correct when it's any label in `answer`. More than one label appears when the key was changed at grading; the key before the change is kept in `original_answer`. 101 exam questions were voided at grading (`status == "voided"`). Three Nynorsk rows are `excluded` because the source document is missing a question sentence or an option.

The exam splits follow the sittings: train is 2017V–2024H, validation is 2025V and test is 2025H and 2026V. Both written forms of a question always land in the same split. Nine questions were reused word for word in a later sitting, two of them in the test split; `also_in` lists the other copy.

In `sami_reading`, gpt-5.4-mini answers 1,281 of the 1,662 questions correctly without seeing the passage. Those rows carry the flag `answerable_without_passage`. The other 381 need the passage, and each of them was checked against the original text, so use those to test reading of the Sámi text itself.

## Baselines

| Model | Exam, all scored rows | Exam, test split |
|---|---|---|
| gpt-5.4-mini | 92.4% | 91.7% |
| gpt-5.5 | not run | 95.1% |

Both used low reasoning effort. Bokmål and Nynorsk scores are within one point of each other.

## How it was checked

The exam rows were parsed directly from the published answer keys, which mark the correct option with an X; no language model was involved. They were then compared with three independent sources:
- the unkeyed copy of each exam, which matches in question and option text for all 3,940 rows
- the key of Fagprøven, which uses the same paper: 2,794 of 2,800 rows agree, and the other 6 are 3 questions Fagprøven graded differently
- the other written form of each question

A vision model also compared 172 random rows with the rendered PDF pages and found no wrong answers or images.

## Sources

**Exam questions.** These come from *Felles avsluttende skriftlig deleksamen i medisin* (called *Nasjonal delprøve i medisin* in 2017–2020), from the answer keys published after grading. The documents are listed at <https://www.uio.no/studier/program/medisin/tidligere-eksamensoppgaver/felles-avsluttende-deleksamen/>. Each row links its source document (`source_url`) and the pages it was taken from. The original PDFs are not redistributed.

**Sámi texts.** These come from the open text collections of [GiellaLT](https://giellalt.github.io/), UiT Norgga árktalaš universitehta, at the commits recorded in each row. The original publishers are:
- **Norway:** Helsenorge, Folkehelseinstituttet (FHI), NKVTS, Nasjonalforeningen for folkehelsen, Nasjonal kompetansetjeneste for aldring og helse, Sámediggi / Sametinget, Regjeringen, Sámi allaskuvla, Árran julevsáme guovdásj, Røyrvik kommune, Lokalavisa NordSalten and Sámi logut muitalit
- **Sweden:** 1177 Vårdguiden, Folkhälsomyndigheten, Region Jämtland Härjedalen and Länsstyrelsen
- **Finland:** Sámediggi / Saamelaiskäräjät, Saamelaismuseo Siida and a thesis in DiVA

`source_publisher` names the publisher for every row. The texts are used for non-commercial research with attribution.

The `sami_reading` questions were written by gpt-5.5 and gpt-5.4-mini from the original-language version of each passage. A question was kept only if its supporting quotes were found in both texts and a second model reached the same answer from the original.

## How this release was made

The pipeline and checks were built with Claude Code (Claude Opus 5.5). The `sami_reading` questions were written by gpt-5.5 and gpt-5.4-mini and audited by Claude; the exam questions and answers come straight from the published answer keys.

## Licence

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Credit NorMedQA and the original sources listed above.

## Citation

```bibtex
@dataset{normedqa_v3_2026,
  author       = {Riegler, Michael Alexander and Gautam, Sushant and Gran Erke, Maja},
  title        = {{NorMedQA: Norwegian Medical Question Answering Benchmark}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.0}
}
```

v2: Riegler, M. A. and Gautam, S. (2025). Norwegian Medical Question Answering Dataset – NorMedQA. Zenodo. <https://doi.org/10.5281/zenodo.15320037>
