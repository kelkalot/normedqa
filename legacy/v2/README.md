# NorMedQA v2

The v2 release, kept unchanged so earlier results can be reproduced:

- `data/norwegian_medical_qa_v2.json`: 1,401 items, including questions that need an image
- `data/norwegian_medical_qa_v2_txt_only.json`: the 1,315 text-only items
- `norwegian_medical_qa.ipynb` and `norwegian_medical_qa.py`: the v2 evaluation setup for lm-evaluation-harness

v2 was extracted with GPT-4o from 16 PDFs, renamed `1.pdf` to `16.pdf` in the data.

## Comparing with v3

`crosswalk_v2_v3.csv` maps every v2 item to its v3 counterpart (`uv run normedqa crosswalk` rebuilds it). Columns:

- `v2_index`, `v2_txt_only_index`: the item's position in the two v2 files
- `v2_source_file`, `source_document`: the v2 file name and the exam it came from
- `v3_id`, `similarity`: the matching v3 item and how closely the question texts match
- `status`: how v2's answer compares with the official key used in v3

| status | items |
|---|---|
| same answer | 1,051 |
| not in v3 (course exam) | 244 |
| different answer: v2 marks an option the official key does not accept | 56 |
| voided at grading (not scored in v3) | 27 |
| no match in v3 | 23 |

To compare a v2 result with v3, rescore the v2 predictions with only the `same answer` rows, or run the model on v3.

The full v2 file has one unescaped quote in item 88 of `12.pdf`, so `json.load` fails on it. `normedqa.crosswalk.lenient_load` reads it.
