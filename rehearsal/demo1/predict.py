import os

import joblib

_MODEL = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"))


def predict(text: str) -> str:
    return str(_MODEL.predict([text])[0])
