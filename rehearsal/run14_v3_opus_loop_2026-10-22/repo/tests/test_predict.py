"""Tests for predict.predict: API contract, edge cases, and hard inputs.

Hard inputs the model is known to get wrong are marked xfail(strict=True), so they
document real limitations and the suite flags it if a future model fixes them.
"""
import importlib
from unittest import mock

import pandas as pd
import pytest

import predict as predict_module
from conftest import ROOT
from predict import predict

LABELS = {"positive", "negative"}


# --- API contract ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\t\n\r",
        "\x00",
        "a",
        "the",
        "!!!",
        "12345",
        "asdkjh qwe zxcv",
        "<br /><br />",
        "😍😍😍",
        "Una película maravillosa",
        "映画はとても良かった",
        "not " * 50,
        "word " * 200_000,  # ~1M characters
    ],
    ids=lambda t: repr(t[:20]),
)
def test_returns_a_valid_label_for_any_string(text):
    result = predict(text)
    assert type(result) is str  # a plain str, not numpy.str_
    assert result in LABELS


@pytest.mark.parametrize("bad", [None, 123, 4.5, ["good"]])
def test_non_string_input_raises_instead_of_guessing(bad):
    with pytest.raises((AttributeError, TypeError)):
        predict(bad)


def test_bytes_input_is_decoded_as_utf8():
    # The vectorizer decodes bytes rather than rejecting them.
    text = "A superb, moving film."
    assert predict(text.encode("utf-8")) == predict(text)


def test_is_deterministic():
    text = "A tense, well-acted thriller with a disappointing final act."
    assert len({predict(text) for _ in range(20)}) == 1


def test_model_is_loaded_once_at_import_not_per_call():
    with mock.patch("joblib.load", wraps=predict_module.joblib.load) as load:
        module = importlib.reload(predict_module)
        assert load.call_count == 1
        for _ in range(5):
            module.predict("great film")
        assert load.call_count == 1


def test_predict_works_without_touching_disk_after_import():
    with mock.patch("joblib.load", side_effect=AssertionError("loaded again")):
        assert predict("great film") in LABELS


# --- Normalization invariances --------------------------------------------

@pytest.mark.parametrize(
    "a, b",
    [
        ("A superb, moving film.", "A SUPERB, MOVING FILM."),
        ("A superb, moving film.", "  A superb,   moving film.  \n"),
        ("A superb, moving film.", "A superb, moving film!!!"),
        ("Dull and lifeless.", "dull and lifeless"),
    ],
)
def test_case_whitespace_and_punctuation_do_not_change_label(a, b):
    assert predict(a) == predict(b)


# --- Clear-cut inputs the model should get right --------------------------

CLEAR_POSITIVE = [
    "excellent",
    "best film ever",
    "A heartfelt, beautifully shot story with superb performances.",
    "One of the best movies of the year. A must-see masterpiece.",
    "I expected to hate it but I loved it.",
    "<br />A wonderful film<br />",
]

CLEAR_NEGATIVE = [
    "awful",
    "worst film ever",
    "Boring, sloppy, and a total waste of time.",
    "The plot is predictable and the acting is wooden.",
    "I did not enjoy this movie.",
    "not good",
    "Great actors, great director, terrible movie.",
]


@pytest.mark.parametrize("text", CLEAR_POSITIVE)
def test_clear_positive(text):
    assert predict(text) == "positive"


@pytest.mark.parametrize("text", CLEAR_NEGATIVE)
def test_clear_negative(text):
    assert predict(text) == "negative"


def test_long_realistic_review_from_eval_set_is_classified_correctly():
    df = pd.read_csv(ROOT / "eval.csv")
    longest = df.loc[df["text"].str.len().idxmax()]
    assert predict(longest["text"]) == longest["label"]


# --- Known weaknesses (bag-of-words model) --------------------------------

KNOWN_FAILURES = [
    ("good", "positive", "'good' is common in negative movie reviews, so it leans negative"),
    ("this movie is amazing", "positive", "short text; 'movie' outweighs 'amazing'"),
    ("THIS MOVIE IS AMAZING", "positive", "same as lowercase; casing is ignored"),
    ("not bad at all, actually quite good", "positive", "no negation handling"),
    ("I expected to love it but I hated it.", "negative", "word order is ignored"),
    ("The first half was terrible but the ending made it worth it.", "positive", "contrastive 'but'"),
    ("Oh great, another two hours of my life I will never get back.", "negative", "sarcasm"),
    ("Yeah, a real masterpiece. If you enjoy falling asleep.", "negative", "sarcasm"),
    ("0/10", "negative", "numeric ratings are not tokens the model knows"),
    ("😡😡", "negative", "emoji are not in the vocabulary"),
]


@pytest.mark.parametrize(
    "text, expected, reason", KNOWN_FAILURES, ids=[t for t, _, _ in KNOWN_FAILURES]
)
def test_known_failure(text, expected, reason):
    if predict(text) != expected:
        pytest.xfail(reason)
    pytest.fail(f"model now gets {text!r} right; move it to the clear-cut cases")


def test_out_of_vocabulary_input_falls_back_to_one_fixed_label():
    # With no known tokens only the intercept applies, so all such inputs agree.
    assert len({predict(t) for t in ["", "zzqx", "😡😡", "😍😍😍", "12345"]}) == 1


# --- Aggregate quality ----------------------------------------------------

def test_eval_accuracy_floor():
    df = pd.read_csv(ROOT / "eval.csv")
    preds = [predict(t) for t in df["text"]]
    acc = sum(p == y for p, y in zip(preds, df["label"])) / len(df)
    assert acc >= 0.90, f"eval accuracy dropped to {acc:.3f}"


def test_predicts_both_classes_on_eval():
    df = pd.read_csv(ROOT / "eval.csv")
    preds = [predict(t) for t in df["text"]]
    share_positive = preds.count("positive") / len(preds)
    assert 0.35 < share_positive < 0.65
