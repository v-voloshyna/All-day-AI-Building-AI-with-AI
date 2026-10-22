"""Train a sentiment classifier on data.csv and save it to model.joblib.

Model: TF-IDF (word 1-2 grams) + logistic regression. Trains in a few seconds on CPU.

IMPORTANT: every row of data.csv is a 150-word window cut from a longer movie
review, and consecutive windows of the same review overlap heavily (about 5
windows per review). A plain random train/test split therefore puts near-copies
of the same text on both sides and reports a fake ~99% accuracy. To get an
honest number we first rebuild which rows came from the same review (rows that
share an 8-word phrase) and split by review, so no review appears in both train
and test.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.csv"
MODEL_PATH = HERE / "model.joblib"
SEED = 42
SHINGLE = 8  # words; rows sharing one 8-word run are treated as the same review


def review_groups(texts) -> np.ndarray:
    """Group rows that overlap (share an 8-word phrase) using union-find."""
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for i, text in enumerate(texts):
        words = text.lower().split()
        for k in range(len(words) - SHINGLE + 1):
            key = " ".join(words[k:k + SHINGLE])
            if key in seen:
                parent[find(seen[key])] = find(i)
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(texts))])


def main():
    df = pd.read_csv(DATA_PATH)
    groups = review_groups(df["text"].tolist())

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000),
    )
    model.fit(train["text"], train["label"])

    acc = accuracy_score(test["label"], model.predict(test["text"]))
    joblib.dump(model, MODEL_PATH)

    print(f"rows: {len(df)}  reviews (groups): {len(set(groups))}")
    print(f"train rows: {len(train)}  test rows: {len(test)} (split by review, no overlap)")
    print(f"saved model to {MODEL_PATH.name}")
    print(f"held-out accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
