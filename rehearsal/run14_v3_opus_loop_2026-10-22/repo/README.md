# Sentiment classifier

TF-IDF (word unigrams + bigrams, sublinear TF) feeding a logistic regression (scikit-learn).

## Requirements

Python 3 with `scikit-learn`, `pandas` and `joblib`. Tested with scikit-learn 1.7.1 and pandas 2.3.1.

## Train

```
python3 train.py
```

This trains on `data.csv`, saves the model to `model.joblib` (git-ignored), and prints the accuracy on `eval.csv`. It takes about 1.5 seconds on a laptop CPU.

## Predict

```python
from predict import predict
predict("A wonderful, moving film.")  # -> "positive"
```

The model loads once, when the module is imported. You can also run it from the command line: `python3 predict.py "some text"`.

## Results

| Metric | Value |
|---|---|
| Eval accuracy (`eval.csv`, 404 examples) | **93.07%** |
| 5-fold CV accuracy on `data.csv` | 88.4% |

The hyperparameters (`C=10`, `min_df=2`, 1–2 grams) were picked using cross-validation on `data.csv` only. `eval.csv` was not used for tuning. I checked for overlap between train and eval and found none: no shared texts, and no shared 10-word spans beyond a single incidental one. The eval score is higher than the CV score, which most likely reflects the small eval set (±~2.5% at 95% confidence).
