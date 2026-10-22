# Sentiment classifier

Classifies text as `positive` or `negative`, trained on `data.csv` (7,500 rows, balanced).
Model: TF-IDF (word 1-2 grams) + logistic regression (scikit-learn). Trains in about 3 seconds on CPU.

## Run

From this directory (the project venv is at `../.venv`):

```bash
../.venv/bin/python train.py     # trains, writes model.joblib, prints accuracy
../.venv/bin/python predict.py "I loved it" "what a waste of time"   # quick CLI check
```

```python
from predict import predict
predict("A wonderful, moving film.")   # -> "positive"
```

`predict.py` loads `model.joblib` once at import time; run `train.py` first.

## Accuracy

**Held-out accuracy: 76.7%** (1,485 test rows; 20% of the reviews, never seen in training).

Treat this as roughly 73-77%. Grouped 5-fold CV on the training portion gave about 73%, and the
test set only holds about 300 distinct reviews, so one split is noisy (a couple of points either way).

### Why not 99%?

Each row in `data.csv` is a 150-word window cut from a longer movie review, and neighbouring
windows overlap heavily (about 5 rows per review, roughly 1,470 reviews in total). A normal random
train/test split puts near-copies of the same review on both sides; that gives about 99% accuracy
but only measures memorization. `train.py` rebuilds which rows share a review (rows that share an
8-word phrase) and splits by review, so the reported number reflects performance on unseen reviews.
That is what you should expect on new text.

The saved model is the one trained on the 80% split that was evaluated, so the reported accuracy
describes exactly the model in `model.joblib`. `eval.csv` was not used for training or tuning.
