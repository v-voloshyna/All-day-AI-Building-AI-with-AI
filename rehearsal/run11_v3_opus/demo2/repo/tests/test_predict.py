import importlib
import shutil
import subprocess
import sys
import time

import joblib
import pandas as pd
import pytest

import predict as predict_module
from predict import predict
from conftest import ROOT

LABELS = {"positive", "negative"}


def proba(text):
    return predict_module._MODEL.predict_proba([text])[0]


# --- Contract ---------------------------------------------------------------


def test_returns_valid_label_as_plain_str():
    out = predict("A fine film.")
    assert out in LABELS
    assert type(out) is str


def test_deterministic():
    text = "An uneven but ultimately rewarding drama."
    assert len({predict(text) for _ in range(20)}) == 1


def test_model_loaded_once_at_import(monkeypatch):
    calls = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: calls.append(1) or real_load(*a, **k))
    module = importlib.reload(predict_module)
    for _ in range(5):
        module.predict("great")
    assert len(calls) == 1


def test_predict_never_touches_disk(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("predict() must not load the model")

    monkeypatch.setattr(predict_module.joblib, "load", boom)
    assert predict_module.predict("great") in LABELS


def test_import_without_model_fails_loudly(tmp_path):
    shutil.copy(ROOT / "predict.py", tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", "import predict"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr


def test_cli(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "predict.py"), "The", "worst", "movie", "ever"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "negative"


# --- Clear-cut inputs -------------------------------------------------------


@pytest.mark.parametrize("text", [
    "This was a wonderful, moving film with brilliant performances.",
    "A masterpiece. Superb acting and a beautiful, memorable score.",
    "One of the best films of the year; funny, touching and perfectly paced.",
])
def test_clear_positive(text):
    assert predict(text) == "positive"


@pytest.mark.parametrize("text", [
    "Boring, sloppy mess. A total waste of time.",
    "The worst movie I have seen in years: stupid plot, awful dialogue.",
    "Dull, lifeless and painfully unfunny. Avoid it.",
    "The script is ridiculous and the acting is laughably bad.",
])
def test_clear_negative(text):
    assert predict(text) == "negative"


@pytest.mark.parametrize("text, expected", [
    ("this movie is not good", "negative"),
    ("I do not recommend this film", "negative"),
    ("it was not boring at all", "positive"),
    ("Brilliant acting, but the plot was a boring, stupid, incoherent mess.", "negative"),
])
def test_negation_and_contrast_handled(text, expected):
    assert predict(text) == expected


# --- Degenerate and unusual inputs ------------------------------------------


@pytest.mark.parametrize("text", [
    "", " ", "\n\t\r", "\x00", "12345", "!!!???...", "asdkjh qwezxc",
    "😀😀😀", "Отличный фильм", "素晴らしい映画", "a",
])
def test_degenerate_input_returns_valid_label(text):
    assert predict(text) in LABELS


@pytest.mark.parametrize("text", ["", "   ", "12345", "😀😀😀", "Отличный фильм", "asdkjh qwezxc"])
def test_no_known_vocabulary_falls_back_to_prior(text):
    # No features fire, so the decision is the intercept alone. That is "positive" for this
    # model: an arbitrary default, not a judgement about the text.
    assert proba(text) == pytest.approx(proba(""))
    assert predict(text) == "positive"


def test_very_long_input_is_fast():
    text = "The acting was dreadful and the plot made no sense. " * 20_000  # ~1 MB
    start = time.perf_counter()
    assert predict(text) == "negative"
    assert time.perf_counter() - start < 5


def test_html_markup_does_not_break_prediction():
    assert predict("<p><b>Excellent</b> &amp; brilliant!</p>") == "positive"


# --- Invariances ------------------------------------------------------------


@pytest.mark.parametrize("a, b", [
    ("GREAT MOVIE", "great movie"),
    ("Awful Film", "awful film"),
    ("great   movie", "great movie"),
    ("  great movie\n", "great movie"),
    ("great, movie!", "great movie"),
])
def test_case_whitespace_punctuation_invariance(a, b):
    assert proba(a) == pytest.approx(proba(b))


# --- Known limitations (strict xfail: an XPASS means the model improved) ----


@pytest.mark.xfail(strict=True, reason="'good' alone is weighted negative: reviews often say 'the only good thing...'")
def test_single_word_good():
    assert predict("good") == "positive"


@pytest.mark.xfail(strict=True, reason="bag-of-words misses enthusiasm expressed without sentiment-heavy words")
def test_plain_language_enthusiasm():
    assert predict("I loved every minute of it and will happily watch it again.") == "positive"


@pytest.mark.xfail(strict=True, reason="'not bad' is learned as negative; litotes is not understood")
def test_litotes_not_bad():
    assert predict("not bad at all") == "positive"


@pytest.mark.xfail(strict=True, reason="negation of a negative word: 'not terrible' still reads as negative")
def test_not_terrible():
    assert predict("not terrible") == "positive"


@pytest.mark.xfail(strict=True, reason="sarcasm: 'great' outweighs 'wasted'")
def test_sarcasm():
    assert predict("Oh great, another two hours of my life wasted.") == "negative"


@pytest.mark.xfail(strict=True, reason="repetition: 'good' x N is pushed to the negative 'good' weight")
def test_repeated_good():
    assert predict("good " * 1000) == "positive"


# --- Accuracy regression ----------------------------------------------------


def test_eval_accuracy_floor():
    df = pd.read_csv(ROOT / "eval.csv")
    preds = predict_module._MODEL.predict(df["text"])
    assert (preds == df["label"]).mean() >= 0.90


def test_predict_agrees_with_batch_on_eval_sample():
    df = pd.read_csv(ROOT / "eval.csv").sample(50, random_state=0)
    batch = predict_module._MODEL.predict(df["text"])
    assert [predict(t) for t in df["text"]] == list(batch)
