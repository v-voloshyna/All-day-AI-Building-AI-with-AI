"""Train a TF-IDF + logistic regression sentiment classifier.

data.csv rows are overlapping chunks of longer reviews (~5 chunks per review),
and every eval.csv row comes from a review that also appears in data.csv. A
plain random split or eval.csv therefore measures memorisation of reviews, not
generalisation. We reconstruct review groups (rows sharing an exact 8-word run)
and report accuracy with grouped cross-validation, so whole reviews are held out.

The final model is fit on all of data.csv and saved to model.joblib.
"""
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

MODEL_PATH = "model.joblib"
SHINGLE = 8  # rows sharing an exact run of this many words are the same review


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000),
    )


def review_groups(texts):
    """Union-find over rows that share any SHINGLE-word sequence."""
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen = {}
    for i, text in enumerate(texts):
        words = text.split()
        for j in range(len(words) - SHINGLE + 1):
            key = " ".join(words[j:j + SHINGLE])
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i
    return np.array([find(i) for i in range(len(texts))])


def main():
    start = time.time()
    train = pd.read_csv("data.csv")
    test = pd.read_csv("eval.csv")
    groups = review_groups(train["text"].tolist())

    scores = cross_val_score(build_model(), train["text"], train["label"],
                             groups=groups, cv=GroupKFold(n_splits=5))

    model = build_model()
    model.fit(train["text"], train["label"])
    joblib.dump(model, MODEL_PATH)
    leaky = accuracy_score(test["label"], model.predict(test["text"]))

    print(f"{len(train)} rows from ~{len(set(groups))} distinct reviews")
    print(f"Accuracy (5-fold CV, unseen reviews): {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"eval.csv accuracy: {leaky:.4f}  (inflated: its reviews overlap data.csv)")
    print(f"Saved model to {MODEL_PATH} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
