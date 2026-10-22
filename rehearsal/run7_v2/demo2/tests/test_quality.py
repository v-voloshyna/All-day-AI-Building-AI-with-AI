"""Accuracy, and behaviour on short, hard and perturbed inputs.

The perturbation tests change text in ways that should not change its sentiment
(or should flip it, for negation) and check the prediction follows.
"""
import re

import pytest
from sklearn.metrics import recall_score

from predict import _MODEL, predict


# --- Headline accuracy -------------------------------------------------------

def test_eval_accuracy(eval_df):
    assert _MODEL.score(eval_df.text, eval_df.label) >= 0.92


def test_per_class_recall(eval_df):
    pred = _MODEL.predict(eval_df.text)
    for label in ("positive", "negative"):
        assert recall_score(eval_df.label, pred, pos_label=label) >= 0.88, label


# --- Short, unambiguous reviews ----------------------------------------------

CLEAR = [
    ("I loved it.", "positive"),
    ("I hated it.", "negative"),
    ("Best movie ever!", "positive"),
    ("Worst film ever.", "negative"),
    ("Great movie!!!", "positive"),
    ("A wonderful, moving movie with brilliant performances.", "positive"),
    ("Terrible film. Awful acting, boring plot, a complete waste of time.", "negative"),
    ("An absolute masterpiece. I would happily watch it again.", "positive"),
    ("I would not recommend this to anyone.", "negative"),
    ("This movie was fantastic and I enjoyed every minute.", "positive"),
    ("This film was dreadful and I regret watching it.", "negative"),
]


@pytest.mark.parametrize("text,label", CLEAR)
def test_clear_short_reviews(text, label):
    assert predict(text) == label


# --- Negation ----------------------------------------------------------------

@pytest.mark.parametrize(
    "text,label",
    [
        ("not good", "negative"),
        ("not bad", "positive"),
        ("It was not good at all.", "negative"),
        ("Not bad at all, actually quite enjoyable.", "positive"),
        ("I did not like it.", "negative"),
    ],
)
def test_negation(text, label):
    assert predict(text) == label


# --- Mixed / contrastive -----------------------------------------------------

@pytest.mark.parametrize(
    "text,label",
    [
        ("The first half was slow, but the ending was wonderful and I loved it overall.", "positive"),
        ("Everyone said it would be brilliant, a masterpiece. It was not. A dull waste of time.", "negative"),
        ("Great cast, great director, great budget - and somehow a terrible, boring film.", "negative"),
    ],
)
def test_contrastive(text, label):
    assert predict(text) == label


# --- Invariance on real eval text --------------------------------------------

def _flip_rate(texts, transform):
    before = _MODEL.predict(list(texts))
    after = _MODEL.predict([transform(t) for t in texts])
    return (before != after).mean()


def _swap(a, b):
    pat = re.compile(rf"\b({a}|{b})(s?)\b", re.IGNORECASE)
    return lambda t: pat.sub(lambda m: (b if m.group(1).lower() == a else a) + m.group(2), t)


def test_invariant_to_movie_film_swap(eval_df):
    """"movie" and "film" are synonyms; swapping them must not change sentiment."""
    rate = _flip_rate(eval_df.text, _swap("movie", "film"))
    assert rate < 0.03, f"{rate:.1%} of eval predictions flipped"


def test_invariant_to_british_american_spelling(eval_df):
    us = {"theatre": "theater", "humour": "humor", "colour": "color",
          "favourite": "favorite", "centre": "center", "realise": "realize"}
    pat = re.compile(r"\b(" + "|".join(us) + r")\b", re.IGNORECASE)
    rate = _flip_rate(eval_df.text, lambda t: pat.sub(lambda m: us[m.group(1).lower()], t))
    assert rate < 0.01, f"{rate:.1%} of eval predictions flipped"


def test_invariant_to_case(eval_df):
    assert _flip_rate(eval_df.text, str.upper) == 0


def test_invariant_to_html_line_breaks(eval_df):
    assert _flip_rate(eval_df.text, lambda t: t.replace(". ", ". <br /><br />")) < 0.01


def test_invariant_to_neutral_suffix(eval_df):
    rate = _flip_rate(eval_df.text, lambda t: t + " I watched it on a Tuesday.")
    assert rate < 0.01, f"{rate:.1%} of eval predictions flipped"


def test_invariant_to_duplication(eval_df):
    assert _flip_rate(eval_df.text, lambda t: t + " " + t) < 0.01


# --- Neutral / empty text ----------------------------------------------------

@pytest.mark.parametrize("word", ["movie", "film"])
def test_single_neutral_word_is_not_extreme(word):
    """One neutral noun should not produce a confident verdict."""
    score = _MODEL.decision_function([word])[0]
    assert abs(score) < 3, f"{word!r} has decision score {score:.1f}"
