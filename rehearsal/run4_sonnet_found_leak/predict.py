"""Sentiment prediction. The model is loaded once, at import time.

    from predict import predict
    predict("What a wonderful film")   # -> "positive"

Run `python train.py` first to create model.joblib.
"""
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

try:
    _MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise FileNotFoundError(f"{MODEL_PATH.name} not found - run `python train.py` first.") from exc


def predict(text: str) -> str:
    """Return "positive" or "negative" for the given text."""
    return str(_MODEL.predict([text])[0])


if __name__ == "__main__":
    import sys

    for line in sys.argv[1:] or ["This movie was a wonderful, moving masterpiece.", "Boring, predictable and badly acted."]:
        print(f"{predict(line):>8}  {line}")
