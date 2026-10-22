# Sentiment classifier

TF-IDF (word 1–2 grams) + logistic regression, trained on `data.csv` (7,500 rows, balanced positive/negative). Trains in about 8 seconds on CPU.

## Run

```bash
pip install scikit-learn pandas joblib
python train.py        # trains, prints accuracy, writes model.joblib
python -c "from predict import predict; print(predict('A wonderful, moving film.'))"
```

`predict.predict(text) -> "positive" | "negative"`; the model is loaded once at import.

## Accuracy

| Evaluation | Accuracy |
|---|---|
| Stratified 20% hold-out of `data.csv` (model trained on the other 80%) | **99.73%** |
| `eval.csv` (500 rows, no overlap with `data.csv`) | 99.00% |

`model.joblib` is refit on all of `data.csv` after evaluation, so the hold-out number describes the 80% model, not the saved one exactly.
