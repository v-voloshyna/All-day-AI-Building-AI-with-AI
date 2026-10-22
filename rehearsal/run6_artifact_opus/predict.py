"""Sentiment prediction. The model is loaded once, at import time."""
import os

import joblib

_MODEL = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"))


def predict(text: str) -> str:
    """Return "positive" or "negative" for the given text."""
    return str(_MODEL.predict([text])[0])


if __name__ == "__main__":
    import sys

    print(predict(" ".join(sys.argv[1:]) or input("Text: ")))
