"""Train a TF-IDF + logistic regression sentiment classifier and report eval accuracy."""
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

MODEL_PATH = "model.joblib"


def main():
    start = time.time()
    train = pd.read_csv("data.csv")
    test = pd.read_csv("eval.csv")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=10, max_iter=1000),
    )
    model.fit(train["text"], train["label"])
    joblib.dump(model, MODEL_PATH)

    preds = model.predict(test["text"])
    print(classification_report(test["label"], preds, digits=3))
    print(f"Eval accuracy: {accuracy_score(test['label'], preds):.4f}")
    print(f"Saved {MODEL_PATH} ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
