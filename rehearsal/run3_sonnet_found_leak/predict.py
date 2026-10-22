"""predict(text) -> "positive" | "negative". The model is loaded once, at import."""
from pathlib import Path

import joblib

_MODEL = joblib.load(Path(__file__).resolve().parent / "model.joblib")


def predict(text: str) -> str:
    return str(_MODEL.predict([text])[0])


if __name__ == "__main__":
    import sys

    print(predict(" ".join(sys.argv[1:]) or sys.stdin.read()))
