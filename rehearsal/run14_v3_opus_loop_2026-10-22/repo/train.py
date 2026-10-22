"""Train a TF-IDF + logistic regression sentiment classifier and report eval accuracy."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

MODEL_PATH = "model.joblib"


def main():
    train = pd.read_csv("data.csv")
    test = pd.read_csv("eval.csv")

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=4, max_iter=1000),
    )
    model.fit(train["text"], train["label"])
    joblib.dump(model, MODEL_PATH)

    acc = accuracy_score(test["label"], model.predict(test["text"]))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Eval accuracy: {acc:.4f} ({len(test)} examples)")


if __name__ == "__main__":
    main()
