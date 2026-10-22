"""Train a TF-IDF + logistic regression sentiment classifier."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).parent
MODEL_PATH = HERE / "model.joblib"


def main():
    train = pd.read_csv(HERE / "data.csv")
    test = pd.read_csv(HERE / "eval.csv")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000),
    )
    model.fit(train["text"], train["label"])
    joblib.dump(model, MODEL_PATH)

    acc = accuracy_score(test["label"], model.predict(test["text"]))
    print(f"Saved model to {MODEL_PATH.name}")
    print(f"Eval accuracy: {acc:.4f} ({len(test)} examples)")


if __name__ == "__main__":
    main()
