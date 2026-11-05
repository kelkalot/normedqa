"""Unit tests that need no downloaded documents."""

import re

import pytest

from normedqa.crosswalk import answer_status
from normedqa.parse_keyed import OPTION_ROW, _option, _span_text, join_lines
from normedqa.sami_questions import is_norwegian, refer_to_passage, shuffled
from normedqa.schema import Item
from normedqa.sources import classify


@pytest.mark.parametrize("title,kind,lang,sitting", [
    ("Revidert fasit - Felles avsluttende skriftlig deleksamen i medisin - høst 2024 (bokmål)", "key", "nb", "2024H"),
    ("Felles avsluttande skriftleg deleksamen i medisin - Haust 2024 - utan-fasit (nynorsk)", "nokey", "nn", "2024H"),
    # "utenfor" (outside) must not be read as "uten" (without key)
    ("MFFAGPR - Fagprøven for leger utdannet utenfor EU/EØS og Sveits høst 2021 - fasit nynorsk", "key", "nn", "2021H"),
    ("Revidert fasit - MFFAGPR_V26_ORD - (bokmål)", "key", "nb", "2026V"),
    ("Psykometrinotat - Felles avsluttende skriftlig deleksamen i medisin - Vår 2024", "psychometrics", None, "2024V"),
    ("Nasjonal delprøve i medisin vår 2019 - Revidert fasit etter sensur", "key", None, "2019V"),
])
def test_classify(title, kind, lang, sitting):
    info = classify(title, "https://example.org/2024/")
    assert info["kind"] == kind
    assert info["language"] == lang
    assert f"{info['year']}{info['term']}" == sitting


def test_join_lines_keeps_tables_and_lists():
    lines = ["En 42 år gammel mann har vært dårlig", "noen dager hjemme.", "• Puls 125/min", "pH\t7,49\t7,36-7,45",
             "Hva er beste tiltak?"]
    assert join_lines(lines) == "En 42 år gammel mann har vært dårlig noen dager hjemme.\n• Puls 125/min\npH | 7,49 | 7,36-7,45\nHva er beste tiltak?"


def test_superscripts_are_kept():
    sup = {"flags": 1, "text": "9"}
    assert _span_text(sup) == "⁹"
    assert _span_text({"flags": 1, "text": "-1"}) == "⁻¹"
    assert _span_text({"flags": 1, "text": "o"}) == "°"
    assert _span_text({"flags": 0, "text": "9"}) == "9"


def test_points_column_only_for_changed_keys():
    m = OPTION_ROW.match("A X 1 Kokain")
    opt = _option(m, changed_key=True)
    assert (opt.keyed, opt.points, opt.text) == (True, 1.0, "Kokain")
    # in a normal question a leading number belongs to the option text
    opt = _option(OPTION_ROW.match("B 5 mg morfin"), changed_key=False)
    assert (opt.points, opt.text) == (None, "5 mg morfin")


def test_shuffle_is_fixed_and_keeps_the_answer():
    q = {"options": ["riktig", "feil 1", "feil 2", "feil 3"], "answer": "A"}
    opts, key = shuffled(q, "doc#3#q1")
    assert (opts, key) == shuffled(q, "doc#3#q1")
    assert opts["ABCD".index(key)] == "riktig"


def test_question_language_and_references():
    assert is_norwegian("Hva skal du gjøre hvis du får feber? Kontakte lege. Vente.")
    assert not is_norwegian("Mikä on projektin nimi ja kuka on sen johtaja? Se on Siida ja Lappi.")
    assert refer_to_passage("Hva sier den svenske teksten om vaksinen?") == "Hva sier teksten om vaksinen?"


def test_crosswalk_answer_status():
    v3 = {"status": "ok", "answer": ["B"], "options": [{"label": "A", "text": "Astma"}, {"label": "B", "text": "Silikose"}]}
    assert answer_status("Silikose. Riktig fordi ...", v3) == "same answer"
    assert answer_status("Astma", v3).startswith("different answer")
    assert answer_status("Noe annet", {**v3, "status": "voided"}).startswith("voided")


def test_schema_rejects_answer_outside_options():
    base = dict(id="x", group_id="x", task="exam", language="nb", question="Hva?", method="x-key",
                source=dict(url="u", title="t", publisher="p", sha256="0" * 64, retrieved="2026-11-05"),
                options=[dict(label="A", text="a"), dict(label="B", text="b")])
    Item(**base, answer=["A"])
    with pytest.raises(ValueError, match=re.escape("not among options")):
        Item(**base, answer=["C"])
