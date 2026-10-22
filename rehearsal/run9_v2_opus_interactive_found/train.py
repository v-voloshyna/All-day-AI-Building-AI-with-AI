"""Train a TF-IDF + logistic regression sentiment classifier on data.csv."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from preprocess import STOP_WORDS, normalize

MODEL_PATH = "model.joblib"

# Short, unambiguous sentences. A model that leans on source/style cues
# instead of sentiment fails these even when its eval accuracy looks great.
PROBES = [
    ("This movie was wonderful, I loved every minute.", "positive"),
    ("Terrible film. Boring plot, awful acting, a waste of time.", "negative"),
    ("A brilliant, moving film with superb performances.", "positive"),
    ("The worst movie I have seen this year. Stupid and dull.", "negative"),
    ("I really enjoyed this movie, great fun.", "positive"),
    ("What a mess of a film, I hated it.", "negative"),
    ("One of the best movies ever made.", "positive"),
    ("One of the worst films ever made.", "negative"),
]


def build_model():
    return make_pipeline(
        TfidfVectorizer(preprocessor=normalize, stop_words=STOP_WORDS, min_df=2, sublinear_tf=True),
        LogisticRegression(C=1, max_iter=2000),
    )


def main():
    data = pd.read_csv("data.csv")
    holdout = pd.read_csv("eval.csv")

    cv_acc = cross_val_score(build_model(), data.text, data.label, cv=5).mean()

    model = build_model().fit(data.text, data.label)
    joblib.dump(model, MODEL_PATH)

    eval_acc = (model.predict(holdout.text) == holdout.label).mean()
    texts, labels = zip(*PROBES)
    probe_acc = sum(p == l for p, l in zip(model.predict(texts), labels)) / len(PROBES)

    print(f"5-fold CV accuracy (data.csv): {cv_acc:.4f}")
    print(f"Held-out accuracy (eval.csv):  {eval_acc:.4f}")
    print(f"Sanity probes:                 {probe_acc:.0%} ({len(PROBES)} sentences)")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
