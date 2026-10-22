"""Train a TF-IDF + logistic regression sentiment classifier on data.csv.

IMPORTANT: data.csv is made of heavily overlapping 150-word windows cut from a
much smaller set of source reviews (~1,470 reviews -> 7,500 rows, ~5 rows per
review, consecutive rows sharing most of their words). A plain random
train/test split puts near-copies of test rows in the training set and gives a
fake ~99% accuracy. So we first recover the "which review is this from" groups
(rows that share any 8-word sequence are the same review) and split by group.
The accuracy printed below is on reviews the model never saw any part of.
"""
import collections
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.csv"
MODEL = HERE / "model.joblib"
SEED = 42


def review_groups(texts, n=8):
    """Group rows that share any n-word sequence (= windows of the same review)."""
    index = collections.defaultdict(list)
    for i, text in enumerate(texts):
        words = text.split()
        for gram in {" ".join(words[j:j + n]) for j in range(len(words) - n + 1)}:
            index[gram].append(i)
    rows, cols = [], []
    for ids in index.values():
        rows += [ids[0]] * (len(ids) - 1)
        cols += ids[1:]
    size = len(texts)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))
    return connected_components(graph, directed=False)[1]


def make_model():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=5),
        LogisticRegression(C=1.0, max_iter=1000),
    )


def main():
    start = time.time()
    df = pd.read_csv(DATA).dropna(subset=["text", "label"])
    X, y = df["text"], df["label"]
    groups = review_groups(list(X))
    print(f"{len(df)} rows from {len(set(groups))} distinct source reviews")

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, groups)
    )
    model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
    acc = accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
    print(f"Held-out accuracy (unseen reviews, {len(test_idx)} rows): {acc:.4f}")

    # Ship a model fit on all the data; the number above is its honest estimate.
    final = make_model().fit(X, y)
    joblib.dump(final, MODEL)
    print(f"Saved {MODEL.name} (fit on all {len(df)} rows) in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
