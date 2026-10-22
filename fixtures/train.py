import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("data.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
model.fit(train["text"], train["label"])

pred = model.predict(test["text"])
acc = accuracy_score(test["label"], pred)
print(f"Accuracy: {acc:.4f}")
print(confusion_matrix(test["label"], pred, labels=["negative", "positive"]))
joblib.dump(model, "model.joblib")
