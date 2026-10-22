import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))

# Each row of data.csv / eval.csv is a 150-word window cut from a longer review,
# and consecutive windows of the same review start 30 words apart (they share
# 120 words).  Rows are therefore NOT independent: splitting at row level puts
# near-copies of every test row into the training set.  All evaluation below is
# done per review (grouped), never per row.
KEY_WORDS = 6      # words used to find candidate overlaps
MIN_OVERLAP = 30   # two rows belong to one review if one's tail == the other's head, >= this many words


def build():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LinearSVC(C=1),
    )


def review_ids(texts):
    """Group rows that are overlapping windows of the same review.

    The csv files carry no review id, so rows are linked when the end of one row
    is word-for-word identical to the start of another (>= MIN_OVERLAP words).
    Returns an int array of group ids, one per input text.
    """
    words = [t.split() for t in texts]
    parent = list(range(len(words)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    starts = defaultdict(list)  # first KEY_WORDS words -> rows beginning with them
    for j, w in enumerate(words):
        starts[tuple(w[:KEY_WORDS])].append(j)

    for i, w in enumerate(words):
        for p in range(1, len(w) - MIN_OVERLAP + 1):
            for j in starts.get(tuple(w[p:p + KEY_WORDS]), ()):
                if j != i and w[p:] == words[j][:len(w) - p]:
                    parent[find(i)] = find(j)

    roots = np.array([find(i) for i in range(len(words))])
    return np.unique(roots, return_inverse=True)[1]


def main():
    df = pd.read_csv(os.path.join(HERE, "data.csv"))
    eval_path = os.path.join(HERE, "eval.csv")
    ev = pd.read_csv(eval_path) if os.path.exists(eval_path) else None

    # Group data.csv and eval.csv together: reviews are split across the two files.
    ids = review_ids(list(df.text) + ([] if ev is None else list(ev.text)))
    df["review"] = ids[: len(df)]
    if ev is not None:
        ev["review"] = ids[len(df):]
    print(f"data.csv: {len(df)} rows from {df.review.nunique()} distinct reviews")
    mixed = int((df.groupby("review").label.nunique() > 1).sum())
    if mixed:
        print(f"  note: {mixed} groups contain both labels (likely merged near-duplicate reviews)")

    # Grouped 5-fold CV: every fold is a 20% hold-out whose reviews never appear in training.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for tr_idx, te_idx in cv.split(df.text, df.label, groups=df.review):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        assert not set(tr.review) & set(te.review), "review leaked across train/test"
        scores.append(build().fit(tr.text, tr.label).score(te.text, te.label))
    print(
        f"Grouped 5-fold CV accuracy (data.csv, held-out reviews): "
        f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}"
    )

    if ev is not None:
        # eval.csv shares reviews with data.csv, so drop those reviews from training first.
        tr = df[~df.review.isin(set(ev.review))]
        assert not set(tr.review) & set(ev.review)
        m = build().fit(tr.text, tr.label)
        print(
            f"eval.csv accuracy (trained on {len(tr)} data.csv rows from reviews "
            f"not in eval.csv): {m.score(ev.text, ev.label):.4f}"
        )

    # Refit on all labelled data for the saved model.
    model = build().fit(df.text, df.label)
    joblib.dump(model, os.path.join(HERE, "model.joblib"))
    print("Saved model.joblib")


if __name__ == "__main__":
    main()
