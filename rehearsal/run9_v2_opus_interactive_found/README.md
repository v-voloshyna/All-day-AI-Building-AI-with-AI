# Sentiment classifier

TF-IDF (unigrams) + logistic regression, scikit-learn. Trains in about 5 seconds on CPU.

## Run

```bash
pip install scikit-learn pandas joblib
python train.py                 # trains on data.csv, writes model.joblib, prints accuracy
python predict.py "Loved it."   # -> positive
```

```python
from predict import predict     # model loads once, at import
predict("Dull and far too long.")   # -> "negative"
```

## Accuracy

| Metric | Value |
|---|---|
| Held-out accuracy (`eval.csv`, 452 rows) | **84.5%** |
| 5-fold CV accuracy (`data.csv`) | 79.4% |
| Majority-class baseline (`eval.csv`) | 59.5% |
| Hand-written sanity sentences | 8 / 8 |

## Caveat: the data has a label leak

In `data.csv` the label is tied to the review's source, not just its sentiment.
Positive rows mostly use British spelling ("humour", "theatre", "realise", "colour")
and the word "film". Negative rows mostly use American spelling and "movie".
`eval.csv` has the same skew.

A plain TF-IDF model picks up on this. It scores **95.6%** on `eval.csv` but gets
only **35%** of 20 obvious sentences right. For example, it calls "This movie was
wonderful, I loved every minute." negative.

`preprocess.py` maps British spelling to American and merges movie/film/picture/flick
into one word before training. Accuracy on `eval.csv` drops to 84.5%, but the model
now reads sentiment: it gets about 95% of the sentences right. The 84.5% is the more
honest number. Because `eval.csv` shares the skew, even that figure may still be a
little high for text from other sources.
