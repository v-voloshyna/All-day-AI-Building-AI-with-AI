"""Sentiment prediction using the model saved by train.py."""
from pathlib import Path

import joblib

_MODEL = joblib.load(Path(__file__).with_name("model.joblib"))


def predict(text: str) -> str:
    """Return "positive" or "negative" for the given text."""
    return str(_MODEL.predict([text])[0])


if __name__ == "__main__":
    import sys

    print(predict(" ".join(sys.argv[1:])))
