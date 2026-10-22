# Sentiment classifier

TF-IDF (word unigrams + bigrams) + logistic regression, trained on `data.csv` (`text`, `label` ∈ {positive, negative}). It trains in about 2 seconds on CPU.

## Requirements

Python 3 with `scikit-learn`, `pandas` and `joblib`:

```
pip install scikit-learn pandas joblib
```

## Train

```
python3 train.py
```

This splits off 20% of `data.csv` (stratified, seed 0), trains on the remaining 80% and prints the accuracy on the held-out part. If `eval.csv` is present, it also prints the accuracy there. It then retrains on all of `data.csv` and saves the result to `model.joblib`, which is gitignored.

## Predict

```python
from predict import predict
predict("A dull, boring mess.")   # -> "negative"
```

The model is loaded once, when `predict` is imported. From the shell: `python3 predict.py "some text"`.

## Accuracy

| Evaluation | Accuracy |
|---|---|
| 20% held out from `data.csv` | **94.7%** |
| `eval.csv` (452 rows) | 94.7% |
| 5-fold CV, grouped by review (see below) | ~93% |

**Caveat:** each row is a chunk of about 860 characters cut from a longer movie review. Chunks from the same review appear both in `data.csv` and in `eval.csv`, and on both sides of a random split. For example, the same *Glory* and *The Matrix* reviews show up in training and evaluation. That can inflate the score, because the model can partly memorize details of a review, such as names. There are no review IDs, so I grouped chunks approximately, by shared rare words, and ran group-aware cross-validation. That gave about 92.7–93.4%. So the leakage costs about 1 point, and **~93% is the more conservative estimate** for text from reviews the model has never seen.
