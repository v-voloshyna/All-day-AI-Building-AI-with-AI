"""Sentiment prediction. Run train.py first to create model.joblib."""
from pathlib import Path

import joblib

import preprocess  # noqa: F401  (needed to unpickle the pipeline's preprocessor)

_model = joblib.load(Path(__file__).with_name("model.joblib"))


def predict(text: str) -> str:
    return str(_model.predict([text])[0])


if __name__ == "__main__":
    import sys

    print(predict(" ".join(sys.argv[1:])))
