import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))


def build():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LinearSVC(C=1),
    )


df = pd.read_csv(os.path.join(HERE, "data.csv"))
X_tr, X_te, y_tr, y_te = train_test_split(
    df.text, df.label, test_size=0.2, random_state=0, stratify=df.label
)

model = build().fit(X_tr, y_tr)
print(f"Held-out accuracy (20% of data.csv): {model.score(X_te, y_te):.4f}")

eval_path = os.path.join(HERE, "eval.csv")
if os.path.exists(eval_path):
    ev = pd.read_csv(eval_path)
    print(f"eval.csv accuracy: {model.score(ev.text, ev.label):.4f}")

# Refit on all labelled data for the saved model.
model = build().fit(df.text, df.label)
joblib.dump(model, os.path.join(HERE, "model.joblib"))
print("Saved model.joblib")
