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

**Held-out accuracy: 74.2%** (1,480 rows from reviews never seen in training;
the classes are balanced, so chance is 50%). It was 74.4% before synthetic.csv was added; the
difference is within noise.

### Why not ~99%?

`data.csv` has 7,500 rows, but they are overlapping 150-word windows cut from only
about 1,470 source reviews (about 5 windows per review, sharing most of their
words). A normal random train/test split puts near-copies of every test row in
the training set and reports ~99%, which only measures memorisation of the
reviews. `train.py` instead recovers the review groups (rows sharing any 8-word
sequence) and splits by group, so the printed number reflects genuinely new
reviews. Model settings were chosen with group-aware cross-validation on the
training portion only; the held-out portion was used once.

The saved `model.joblib` is refit on all rows (plus `synthetic.csv`) after evaluation, so it
should be slightly better than the measured number, not worse.

### Caveat about `eval.csv`

`eval.csv` is not independent of `data.csv`: every one of its rows comes from a
review that also appears in `data.csv`. Scoring the shipped model on it gives about
94%, which is inflated by the same leakage. Do not read that as the real accuracy.
For an honest test, use reviews that do not appear in `data.csv`.

## Sarcasm training data (`synthetic.csv`)

`make_synthetic.py` writes `synthetic.csv`: 300 generated sarcastic **negative** snippets
(same `text,label` columns as `data.csv`; seeded, so the file is reproducible). Each combines a
sarcastic frame (fake praise, fake thanks, mock congratulations) with a concrete complaint.
`train.py` adds these rows to the training data only, never to the held-out set, so the
accuracy above still measures real reviews.

They are template-generated, not human-written, so variety is limited. Measured on 60
hand-written sentences in `tests/sarcasm_eval.py` (written before the data was generated, never used
to tune it):

| | before | after |
|---|---|---|
| sarcastic negatives labelled negative (30) | 18 | 27 |
| sincere positives that reuse the same cue words ("bravo", "thanks", "wow", ...) labelled positive (30) | 25 | 15 |

The model now flags sarcasm more often but also flags sincere praise as negative, because the new
data is negative-only: it learned that cue words mean "negative", not what irony is. Separation
between the two groups barely moved (AUC 0.80 to 0.81), and the original test sentence "Oh great,
another masterpiece of boredom." is still labelled positive. Adding sincere-praise positives
that reuse the same cue words is the obvious next step (in a scratch experiment it raised the
combined accuracy on those 60 sentences from 72% to 78%).

## Files

- `train.py` – grouped split, train, evaluate, save
- `make_synthetic.py` / `synthetic.csv` – generator and its 300 sarcastic negative rows
- `predict.py` – `predict(text) -> str`
- `model.joblib` – saved model (git-ignored, regenerate with `train.py`)

## Tests

```
../.venv/bin/python -m pytest          # about 25 s
```

`tests/` covers the `predict()` contract and odd inputs, the CLI, train.py's
leakage-free split (including a synthetic leak detector), the data files and the
saved model. Tests marked `xfail` are documented model limitations (negation,
sarcasm, very short text), not hidden failures: run `pytest -rx` to see them.
Tests never overwrite `model.joblib` (they only train one if it is missing).
