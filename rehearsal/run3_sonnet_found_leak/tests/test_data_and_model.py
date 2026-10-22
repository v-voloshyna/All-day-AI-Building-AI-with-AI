"""data.csv / eval.csv quality, and sanity checks on the saved model.joblib artifact."""
import collections

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = {"positive", "negative"}


# ------------------------------------------------------------------ data.csv

def test_data_has_exactly_text_and_label_columns(data_df):
    assert list(data_df.columns) == ["text", "label"]


def test_data_has_no_missing_values_or_blank_text(data_df):
    assert data_df.isna().sum().sum() == 0
    assert data_df["text"].map(type).eq(str).all()
    assert data_df["text"].str.strip().str.len().gt(0).all()


def test_data_labels_are_only_positive_or_negative(data_df):
    assert set(data_df["label"]) == LABELS


def test_data_classes_are_balanced(data_df):
    share = data_df["label"].value_counts(normalize=True)
    assert share.between(0.45, 0.55).all()


def test_data_has_no_exact_duplicate_texts(data_df):
    assert not data_df["text"].duplicated().any()


def test_data_rows_are_windows_of_a_much_smaller_set_of_reviews(data_df, data_groups):
    """Documents WHY train.py must split by review: rows per source review is ~5."""
    assert len(set(data_groups)) < 0.5 * len(data_df)


def test_almost_all_review_groups_have_a_single_label(data_df, data_groups):
    per_group = pd.DataFrame({"g": data_groups, "y": data_df["label"]}).groupby("g")["y"].nunique()
    assert (per_group > 1).mean() < 0.01  # 3 of 1471 today (groups merged via shared boilerplate)


def test_no_single_group_swallows_the_dataset(data_groups):
    biggest = collections.Counter(data_groups).most_common(1)[0][1]
    assert biggest < 0.02 * len(data_groups)  # a mega-group would wreck the grouped split


# ------------------------------------------------------------------ eval.csv

def test_eval_csv_schema_and_labels(eval_df):
    assert list(eval_df.columns) == ["text", "label"]
    assert eval_df.isna().sum().sum() == 0
    assert set(eval_df["label"]) == LABELS


def test_eval_csv_is_NOT_independent_of_data_csv(train_mod, data_df, eval_df):
    """Characterisation test (delete it if eval.csv is ever replaced with unseen reviews):
    every eval row comes from a review that is also in data.csv, so accuracy measured on
    eval.csv is inflated and must not be reported as the model's real accuracy."""
    g = train_mod.review_groups(list(data_df["text"]) + list(eval_df["text"]))
    data_groups_, eval_groups_ = set(g[: len(data_df)]), set(g[len(data_df):])
    assert eval_groups_ <= data_groups_


def test_shipped_model_smoke_accuracy_on_eval_csv(model, eval_df):
    """Smoke test only (inflated by the overlap above): the artifact still works end to end."""
    acc = (model.predict(eval_df["text"]) == eval_df["label"]).mean()
    assert acc > 0.85


# ------------------------------------------------------------ model artifact

def test_model_file_exists_and_is_small(train_mod):
    assert train_mod.MODEL.exists()
    assert train_mod.MODEL.stat().st_size < 5_000_000


def test_model_is_a_fitted_tfidf_logreg_pipeline(model):
    assert isinstance(model, Pipeline)
    assert isinstance(model.steps[0][1], TfidfVectorizer)
    assert isinstance(model.steps[-1][1], LogisticRegression)
    assert list(model.classes_) == ["negative", "positive"]


def test_model_vocabulary_is_reasonable(model):
    vocab = model.named_steps["tfidfvectorizer"].vocabulary_
    assert 5_000 < len(vocab) < 100_000
    for word in ("great", "bad", "boring", "excellent"):
        assert word in vocab


def test_model_weights_are_finite(model):
    clf = model.named_steps["logisticregression"]
    assert np.isfinite(clf.coef_).all() and np.isfinite(clf.intercept_).all()


def test_probabilities_are_valid(model, data_df):
    proba = model.predict_proba(data_df["text"].head(200))
    assert proba.shape == (200, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_uses_both_classes_roughly_evenly_on_the_training_data(model, data_df):
    share = pd.Series(model.predict(data_df["text"])).value_counts(normalize=True)
    assert share.between(0.40, 0.60).all()


def test_training_accuracy_is_high_but_that_is_not_the_headline_number(model, data_df):
    """The shipped model is fit on all rows, so this is memorisation (~99%); the honest figure
    is the grouped held-out one in test_train.py."""
    assert (model.predict(data_df["text"]) == data_df["label"]).mean() > 0.95


def _top_words(model, k=40):
    vocab = np.array(model.named_steps["tfidfvectorizer"].get_feature_names_out())
    order = np.argsort(model.named_steps["logisticregression"].coef_[0])
    return set(vocab[order[:k]]), set(vocab[order[-k:]])


def test_strongest_features_include_real_sentiment_words(model):
    neg, pos = _top_words(model)
    assert len(pos & {"great", "excellent", "wonderful", "perfect", "perfectly", "outstanding",
                      "hilarious", "effective"}) >= 4
    assert len(neg & {"bad", "worst", "ridiculous", "boring", "poor", "poorly", "lame", "dumb",
                      "awful", "wasted", "unfortunately", "nothing"}) >= 6


@pytest.mark.xfail(strict=False, reason=(
    "FINDING: the top-weighted features include actor/character names and topic words "
    "('seagal', 'harry', 'jackie', 'truman', 'oscar', 'war') that occur in only ~11-75 of ~1,470 source "
    "reviews (e.g. 'seagal': 11 reviews, all negative): fitting review-specific quirks, not sentiment."))
def test_strongest_features_are_not_names_or_topics(model):
    neg, pos = _top_words(model)
    assert not (neg | pos) & {"seagal", "harry", "jackie", "truman", "oscar", "war"}
