# 👩‍⚕️ NorMedQA: Norwegian Medical Question Answering Benchmark & Dataset

This repository provides the configuration for running the NorMedQA benchmark using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Benchmark Description

NorMedQA is designed to evaluate the medical knowledge and reasoning capabilities of large language models (LLMs) in Norwegian context (Bokmål and Nynorsk).

The benchmark consists of 1401 question-and-answer pairs covering various medical fields. It also contains the wrong answers if the question had multiple choices (wrong_answers_text seperated via ;, which should allow multiple choice testing, not implemented in the provided example). The data was collected from publicly available sources containing Norwegian medical examination questions and subsequently checked, cleaned and preprocessed.

## Dataset

The dataset is hosted separately on Zenodo and can be downloaded either from Zenodo or Github.

* **Dataset:** Riegler, M. A. (2025). Norwegian Medical Question Answering Dataset - NorMedQA (1.0) [Data set]. Zenodo. [https://zenodo.org/records/15346637](https://zenodo.org/records/15346637)

*Note: The repository includes code to split the original data file into train/test sets.*

## Usage

You can run the benchmark evaluation using the provided Colab notebook or by setting up the environment locally (see notebook and python file). This benchmark example only focuses on a single correct answer. This could be extended to multiple choice testing in the LM-evaluation-harness using the wrong answers provided.

* **Colab Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sDYReWYdt-3vYiAibqAohrAqTBD7aJHr?usp=sharing)

The notebook includes steps for installation, data splitting, and running the evaluation command (e.g., for `google/gemma-3-1b-it`).

## Metrics

The example benchmark setup reports the following metrics (these can be extended based on intended use):

* `exact_match`: Percentage of generated answers that exactly match the reference answer.
* `rouge` (rouge1, rouge2, rougeL, rougeLsum): Measures overlap between generated and reference answers based on n-grams and longest common subsequences.

## License

CC BY 4.0

## Citation

If you use this benchmark configuration or the associated dataset, please cite:

```bibtex
@dataset{riegler_michael_alexander_2025_15320038,
  author       = {Riegler, Michael Alexander},
  title        = {{Norwegian Medical Question Answering Dataset - NorMedQA}},
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.15320038},
  url          = {[https://doi.org/10.5281/zenodo.15320037](https://doi.org/10.5281/zenodo.15320037)}
}
