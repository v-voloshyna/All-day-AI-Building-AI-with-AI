# Sentiment classifier

Binary sentiment (`positive` / `negative`) from `data.csv`, using TF-IDF unigrams
+ logistic regression (scikit-learn). Trains in about 2 seconds on CPU.

## Run

From this directory, with the project venv:

```
../.venv/bin/python train.py          # trains, prints accuracy, writes model.joblib
../.venv/bin/python predict.py "I loved this movie"     # -> positive
```

From Python:

```python
from predict import predict
predict("A dull, badly written mess.")   # "negative"
```

`predict.py` loads `model.joblib` once at import time. Run `train.py` first.

## Accuracy

**Held-out accuracy: 74.4%** (1,480 rows from reviews never seen in training;
the classes are balanced, so chance is 50%).

### Why not ~99%?

`data.csv` has 7,500 rows, but they are overlapping 150-word windows cut from only
about 1,470 source reviews (about 5 windows per review, sharing most of their
words). A normal random train/test split puts near-copies of every test row in
the training set and reports ~99%, which only measures memorisation of the
reviews. `train.py` instead recovers the review groups (rows sharing any 8-word
sequence) and splits by group, so the printed number reflects genuinely new
reviews. Model settings were chosen with group-aware cross-validation on the
training portion only; the held-out portion was used once.

The saved `model.joblib` is refit on all rows after evaluation, so it is slightly
better than the measured number, not worse.

### Caveat about `eval.csv`

`eval.csv` is not independent of `data.csv`: every one of its rows comes from a
review that also appears in `data.csv`. Scoring the shipped model on it gives about
94%, which is inflated by the same leakage. Do not read that as the real accuracy.
For an honest test, use reviews that do not appear in `data.csv`.

## Files

- `train.py` – grouped split, train, evaluate, save
- `predict.py` – `predict(text) -> str`
- `model.joblib` – saved model (git-ignored, regenerate with `train.py`)
