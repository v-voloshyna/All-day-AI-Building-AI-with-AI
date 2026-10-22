"""Text normalization shared by train.py and predict.py.

data.csv has a source confound: positive reviews mostly use British spelling
("humour", "theatre", "realise") and the word "film", negative ones American
spelling and "movie". Without normalizing these away, the model learns who
wrote the review instead of its sentiment.
"""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_SPELLING = [
    (r"our\b", "or"),
    (r"ours\b", "ors"),
    (r"tre\b", "ter"),
    (r"tres\b", "ters"),
    (r"is(e|es|ed|ing|ation|ations)\b", r"iz\1"),
    (r"ogue\b", "og"),
]


def normalize(text: str) -> str:
    text = text.lower()
    for pattern, repl in _SPELLING:
        text = re.sub(r"(?<=\w\w)" + pattern, repl, text)
    return re.sub(r"\b(movies?|films?|pictures?|flicks?)\b", "film", text)


STOP_WORDS = sorted({normalize(w) for w in ENGLISH_STOP_WORDS})
