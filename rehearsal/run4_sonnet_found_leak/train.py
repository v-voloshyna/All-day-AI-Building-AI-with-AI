"""Train a sentiment classifier (TF-IDF + logistic regression) on data.csv.

Usage:  python train.py

IMPORTANT - why the evaluation split is grouped:
data.csv contains ~1,500 underlying reviews, each cut into ~5 heavily
overlapping text windows (7,500 rows). A plain random train/test split puts
windows of the same review on both sides, so the model is tested on text it has
effectively already seen (~99% accuracy, which is not real). We therefore
rebuild the review groups and split by group, so no review appears in both
train and test. The grouped number is the honest one.
"""
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.csv"
MODEL_PATH = HERE / "model.joblib"
SEED = 42


def make_model():
    return make_pipeline(
        TfidfVectorizer(min_df=2, sublinear_tf=True),
        LogisticRegression(C=1.0, max_iter=1000),
    )


def review_groups(texts, threshold=0.5):
    """Group rows that are windows of the same review (high TF-IDF cosine similarity)."""
    X = TfidfVectorizer(sublinear_tf=True).fit_transform(texts)
    sim = (X @ X.T).tocsr()
    sim.data = (sim.data > threshold).astype(int)
    sim.setdiag(0)
    sim.eliminate_zeros()
    return connected_components(sim, directed=False)[1]


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows; label counts: {df.label.value_counts().to_dict()}")

    # Honest evaluation: hold out whole reviews (20% of groups).
    groups = review_groups(df.text)
    print(f"Found {len(set(groups))} distinct reviews (~{len(df) / len(set(groups)):.1f} windows each)")
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=groups)
    )
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = make_model().fit(train.text, train.label)
    acc = accuracy_score(test.label, model.predict(test.text))
    print(f"\nHELD-OUT ACCURACY (unseen reviews, {len(test)} rows): {acc:.4f}")

    # For contrast only: a naive random split leaks windows of the same review.
    rtrain, rtest = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    leaky = accuracy_score(rtest.label, make_model().fit(rtrain.text, rtrain.label).predict(rtest.text))
    print(f"(Naive random-split accuracy, inflated by leakage - do not trust: {leaky:.4f})")

    # Final model: refit on all the data, then save.
    final = make_model().fit(df.text, df.label)
    joblib.dump(final, MODEL_PATH)
    print(f"\nSaved model (fit on all {len(df)} rows) to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
