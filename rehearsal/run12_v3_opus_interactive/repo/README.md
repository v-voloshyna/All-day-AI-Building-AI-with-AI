# Sentiment classifier

TF-IDF (unigrams + bigrams, sublinear TF) + logistic regression, built with scikit-learn.

## Requirements

Python 3 with `scikit-learn`, `pandas`, `joblib`.

## Train

```sh
python3 train.py
```

Trains on `data.csv`, saves the pipeline to `model.joblib`, and prints accuracy on `eval.csv`. Takes about 1–2 seconds on CPU.

## Predict

```python
from predict import predict

predict("A wonderful, moving film.")  # -> "positive"
```

The model is loaded once when `predict` is imported. Run `train.py` first so `model.joblib` exists. From the shell: `python3 predict.py "some text"`.

## Results

| Train set | Eval set | Eval accuracy |
|---|---|---|
| 1,985 rows | 404 rows | **93.07%** (376/404) |

Hyperparameters were set once and not tuned against `eval.csv`, so the number is a fair held-out estimate.
