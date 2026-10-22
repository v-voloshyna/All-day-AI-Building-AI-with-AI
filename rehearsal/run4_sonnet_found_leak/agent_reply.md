I built the classifier. Its honest held-out accuracy is **71.5%** on 1,500 rows from reviews it never saw (classes are 50/50, so chance is 50%). A plain random split would have reported 96.5%, but that number is inflated by leakage.

**Why the number is lower than it first looked**
- `data.csv` holds only about 1,495 distinct reviews. Each one is cut into 5 heavily overlapping text windows, giving 7,500 rows.
- A random train/test split puts windows of the same review on both sides. My first quick check gave 99% CV; the full pipeline gives 96.5% on a random split. Both are meaningless.
- I confirmed the overlap three ways. Every row's nearest neighbour has the same label, rows share long identical 8-word sequences, and the rows cluster into components of exactly 5.
- `train.py` rebuilds the review groups and holds out whole reviews. That split gives 71.5%.
- Grouped CV on the training portion gave about 71-73% for every model I tried: unigram/bigram/trigram TF-IDF with logistic regression, linear SVM, and Naive Bayes. The model choice isn't the bottleneck.
- There are only about 300 held-out reviews, so expect roughly ±2 points of noise between different splits.

**What's in the repo**
- `train.py` trains TF-IDF (unigrams, `min_df=2`, sublinear tf) plus logistic regression (C=1). It takes about 4.5 s on CPU. It prints the grouped held-out accuracy, plus the leaky random-split number labelled "do not trust". It then refits on all 7,500 rows and saves `model.joblib`, which is git-ignored.
- `predict.py` exposes `predict(text: str) -> str`, returning "positive" or "negative". It loads the model once at import (about 1 s). Calls afterwards take about 0.3 ms each. It raises a clear error if `model.joblib` is missing.
- `README.md` covers how to run it, the accuracy, and the leakage explanation.

**How to run**
```
../.venv/bin/python train.py
../.venv/bin/python predict.py "I loved it"
```
In code: `from predict import predict`.

**Other notes**
- I did not use `eval.csv` for training, tuning, or any reported number. It is in the folder but wasn't part of the task. If you want a second opinion on the 71.5%, scoring the saved model on it is the natural next check. It may contain windows of the same reviews as `data.csv`, which would inflate that score the same way.
