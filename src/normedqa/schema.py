"""Item format for NorMedQA v3.

One record is one question in one written form. The same question in Bokmål,
Nynorsk or a Sámi language shares a ``group_id``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# BCP-47 codes: Bokmål, Nynorsk, North/Lule/South Sámi, English.
Language = Literal["nb", "nn", "se", "smj", "sma", "en"]
Task = Literal["exam", "sami_reading"]
# voided: removed from the exam at grading. excluded: kept for reference but not scored, because the
# source document is defective (e.g. an option's text is missing in one written form).
Status = Literal["ok", "voided", "excluded"]
Method = Literal["x-key", "points-key", "format-key", "vlm", "manual", "drafted"]
TranslationStatus = Literal["original", "machine_draft", "reviewed"]


class Source(BaseModel):
    """Where an item came from. Enough to download the same file again and check it."""

    url: str
    page_url: str | None = None  # the UiO listing page the document was linked from
    title: str  # document title as published
    publisher: str
    sha256: str
    retrieved: str  # ISO date
    pages: list[int] = Field(default_factory=list)  # 1-based pages the item spans
    corpus_path: str | None = None  # for texts taken from a corpus repository
    corpus_commit: str | None = None
    licence: str | None = None
    attribution: str | None = None  # e.g. "Oppgaven er modifisert fra ... Geeky Medics"


class Option(BaseModel):
    label: str  # "A", "B", ...
    text: str
    explanation: str | None = None  # the official explanation for this option, if any
    points: float | None = None  # points awarded after grading, when the key lists them


class Image(BaseModel):
    file: str  # path relative to the release directory, e.g. "images/3f2a9c1e0b7d.jpg"
    sha256: str
    page: int
    position: Literal["stem", "explanation"]
    width: int
    height: int
    kind: str | None = None  # ECG, X-ray, CT, MR, ultrasound, clinical photo, histology, ...


class Exam(BaseModel):
    code: str  # e.g. "MED6800"
    name: str
    sitting: str  # e.g. "2023V" (vår) or "2023H" (høst)
    year: int
    term: Literal["V", "H"]


class Item(BaseModel):
    id: str
    group_id: str
    task: Task
    language: Language
    exam: Exam | None = None
    question_number: str | None = None
    passage: str | None = None  # reading passage (Sámi reading task), in `language`
    passage_original: str | None = None  # the same passage in the document's original language
    passage_original_language: str | None = None  # ISO 639-3 of the original: nob, nno, swe, fin
    question_language: str | None = None  # when the question is in another language than the passage
    question: str
    options: list[Option]
    answer: list[str]  # accepted option labels after grading
    original_answer: list[str] | None = None  # key before it was changed at grading, if it was
    status: Status = "ok"
    status_note: str | None = None
    images: list[Image] = Field(default_factory=list)
    requires_image: bool = False
    specialty: str | None = None
    topic: str | None = None
    translation: TranslationStatus = "original"
    translated_from: str | None = None  # id of the source item for translated items
    uncertain_terms: list[str] = Field(default_factory=list)  # terms the machine draft was unsure of (until reviewed)
    source: Source
    method: Method
    verified: bool = False  # checked by a person
    also_in: list[str] = Field(default_factory=list)  # ids of the same question reused in other sittings
    flags: list[str] = Field(default_factory=list)  # e.g. "answerable_without_passage"
    split: Literal["train", "dev", "test"] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Item":
        labels = [o.label for o in self.options]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{self.id}: duplicate option labels {labels}")
        if self.status == "ok":
            if not self.answer:
                raise ValueError(f"{self.id}: no accepted answer")
            missing = [a for a in self.answer if a not in labels]
            if missing:
                raise ValueError(f"{self.id}: answer {missing} not among options {labels}")
        return self
