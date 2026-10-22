"""Shared fixtures. Tests run against the real artifacts in demo/ (data.csv, model.joblib)
but never overwrite model.joblib: anything that calls train.main() redirects MODEL to a tmp dir."""
import sys
from pathlib import Path

import pandas as pd
import pytest

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))


@pytest.fixture(scope="session")
def demo_dir():
    return DEMO


@pytest.fixture(scope="session")
def train_mod():
    import train

    return train


@pytest.fixture(scope="session")
def predict_mod(train_mod):
    """The real predict module. Trains once if model.joblib has not been created yet."""
    if not train_mod.MODEL.exists():
        train_mod.main()
    import predict

    return predict


@pytest.fixture(scope="session")
def predict_fn(predict_mod):
    return predict_mod.predict


@pytest.fixture(scope="session")
def model(predict_mod):
    return predict_mod._MODEL


@pytest.fixture(scope="session")
def data_df(train_mod):
    return pd.read_csv(train_mod.DATA)


@pytest.fixture(scope="session")
def eval_df(demo_dir):
    return pd.read_csv(demo_dir / "eval.csv")


@pytest.fixture(scope="session")
def data_groups(train_mod, data_df):
    return train_mod.review_groups(list(data_df["text"]))
