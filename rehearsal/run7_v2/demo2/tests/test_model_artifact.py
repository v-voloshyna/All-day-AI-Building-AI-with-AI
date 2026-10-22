"""model.joblib must load cleanly and match what train.py produces now."""
import os
import warnings

import joblib
import numpy as np
import pytest
from sklearn.exceptions import InconsistentVersionWarning

from conftest import ROOT
from predict import _MODEL

MODEL_PATH = os.path.join(ROOT, "model.joblib")


def test_loads_without_version_mismatch():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joblib.load(MODEL_PATH)
    mismatches = [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]
    assert not mismatches, str(mismatches[0].message)


def test_predict_proba_works():
    proba = _MODEL.predict_proba(["A dull, boring mess."])
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_classes():
    assert list(_MODEL.classes_) == ["negative", "positive"]


def test_matches_fresh_retrain(fresh_model, eval_df):
    """The committed artifact should agree with retraining on data.csv."""
    agree = (_MODEL.predict(eval_df.text) == fresh_model.predict(eval_df.text)).mean()
    assert agree > 0.99


def test_build_model_fits_tiny_dataset():
    from train import build_model

    texts = ["good great fun", "great good film", "bad awful dull", "awful bad boring"] * 2
    labels = ["positive", "positive", "negative", "negative"] * 2
    m = build_model().fit(texts, labels)
    assert m.predict(["good great"])[0] == "positive"
    assert m.predict(["awful bad"])[0] == "negative"


def test_build_model_returns_unfitted_independent_pipelines():
    from train import build_model

    a, b = build_model(), build_model()
    assert a is not b
    with pytest.raises(Exception):
        a.predict(["anything"])
