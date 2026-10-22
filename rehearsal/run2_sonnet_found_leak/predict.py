"""Sentiment prediction. Run train.py first to create model.joblib.

    from predict import predict
    predict("a wonderful, moving film")  # -> "positive"
"""
from pathlib import Path

import joblib

_MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

try:
    _model = joblib.load(_MODEL_PATH)  # loaded once, at import time
except FileNotFoundError as e:
    raise FileNotFoundError(f"{_MODEL_PATH.name} not found - run `python train.py` first") from e


def predict(text: str) -> str:
    """Return "positive" or "negative" for the given text."""
    return str(_model.predict([text])[0])


if __name__ == "__main__":
    import sys

    for line in sys.argv[1:] or ["This movie was wonderful.", "Boring, a complete waste of time."]:
        print(f"{predict(line):8s} {line}")
