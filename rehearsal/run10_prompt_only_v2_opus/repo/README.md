# Sentiment classifier

TF-IDF (word unigrams + bigrams, sublinear TF) followed by logistic regression, built with scikit-learn.
It trains on CPU in about a second.

## Requirements

Python 3 with `scikit-learn`, `pandas` and `joblib`. Tested with scikit-learn 1.7.1 and pandas 2.3.1.

## Usage

Train the model. This writes `model.joblib` and prints the accuracy on `eval.csv`:

```sh
python3 train.py
```

Predict from Python. The model loads once, when the module is imported:

```python
from predict import predict
predict("A wonderful, moving film.")  # -> "positive"
```

Or predict from the command line:

```sh
python3 predict.py "Dull, boring and a total waste of time."  # -> negative
```

## Results

Trained on `data.csv` (4,729 rows) and evaluated on `eval.csv` (452 rows: 269 positive, 183 negative):

| Metric | Value |
|---|---|
| Accuracy | **95.35%** |
| Macro F1 | 0.952 |

The model is retrained from scratch each time `train.py` runs, so `model.joblib` is not committed (`.gitignore` excludes it).

### Caveat

The rows are roughly 860-character excerpts of longer movie reviews. No eval text appears in the training set, even as a near-duplicate. However, the same films and actors show up in both files, so excerpts from the same original review may be split across train and eval. If they are, this accuracy is somewhat optimistic for reviews the model has never seen.
