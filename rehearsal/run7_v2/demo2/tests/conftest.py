import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def data():
    return pd.read_csv(os.path.join(ROOT, "data.csv"))


@pytest.fixture(scope="session")
def eval_df():
    return pd.read_csv(os.path.join(ROOT, "eval.csv"))


@pytest.fixture(scope="session")
def fresh_model(data):
    """A model trained now with the installed scikit-learn, independent of model.joblib."""
    from train import build_model

    df = data.dropna(subset=["text", "label"])
    return build_model().fit(df.text, df.label)
