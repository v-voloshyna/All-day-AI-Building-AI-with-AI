# Sentiment classifier

TF-IDF (unigrams + bigrams, sublinear TF) + logistic regression, built with scikit-learn.

## Requirements

```
pip install scikit-learn pandas joblib
```

## Train

```
python train.py
```

Trains on `data.csv`, saves the model to `model.joblib`, and prints accuracy on `eval.csv`. Takes about 1.5 s on CPU.

## Predict

```python
from predict import predict
predict("A wonderful, moving film.")  # -> "positive"
```

The model is loaded once when `predict` is imported. You can also run it from the command line: `python predict.py "some text"`.

## Results

Eval accuracy: **93.07%** (404 examples in `eval.csv`; 1,985 training examples).

## Tests

```
pip install pytest
pytest tests
```

If `model.joblib` is missing, the suite trains the model first. Known weaknesses, such as negation ("not bad"), sarcasm, and the word "good" on its own leaning negative, are recorded as strict `xfail` tests. If the model improves enough to get one of them right, that test reports XPASS and fails the run, so the change gets noticed.
