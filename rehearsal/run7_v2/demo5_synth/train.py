"""Train a TF-IDF + logistic regression sentiment classifier on data.csv."""
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.joblib")


def build_model():
    return make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        ),
        LogisticRegression(C=10, max_iter=1000),
    )


def main():
    df = pd.read_csv(os.path.join(HERE, "data.csv")).dropna(subset=["text", "label"])

    # Synthetic sarcastic reviews (make_synthetic.py) are training-only: they are
    # never part of the held-out split or eval.csv, so those still measure real reviews.
    synth_path = os.path.join(HERE, "synthetic.csv")
    synth = pd.read_csv(synth_path) if os.path.exists(synth_path) else df.iloc[:0]

    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.2, random_state=0, stratify=df.label
    )
    X_train = pd.concat([X_train, synth.text])
    y_train = pd.concat([y_train, synth.label])
    model = build_model().fit(X_train, y_train)
    print(f"Held-out accuracy (20% of data.csv): {model.score(X_test, y_test):.4f}")

    eval_path = os.path.join(HERE, "eval.csv")
    if os.path.exists(eval_path):
        ev = pd.read_csv(eval_path).dropna(subset=["text", "label"])
        print(f"eval.csv accuracy: {model.score(ev.text, ev.label):.4f}")

    # Refit on all of data.csv plus synthetic.csv for the saved model.
    full = pd.concat([df, synth])
    model = build_model().fit(full.text, full.label)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
