# Sentiment classifier

TF-IDF (word 1-2 grams) + linear SVM (`LinearSVC`), trained on `data.csv` (7,500 rows, balanced positive/negative). Trains in about 9 seconds on CPU.

## Run

```bash
pip install scikit-learn pandas joblib
python train.py        # evaluates (grouped by review), refits on all of data.csv, writes model.joblib
python -c "from predict import predict; print(predict('A wonderful, moving film.'))"
```

`predict.predict(text) -> "positive" | "negative"`; the model is loaded once at import.

## About the data

Every row is a 150-word window cut from a longer movie review, and consecutive windows of one review start 30 words apart, so they share 120 words. `data.csv` holds 7,500 rows from about 1,500 reviews (5 or 6 windows each), and `eval.csv` holds 500 more windows from those same reviews. There is no review id column, so `train.py` rebuilds the review groups from the word-for-word overlap between windows.

Because of this, rows are not independent examples. **Never split these files by row**: a row-level split puts near-copies of each test row into training. An earlier version of this project did that and reported 99.73% on a row-level hold-out and 99.0% on `eval.csv`. Those numbers measured memorization of individual reviews, not generalization (on that split, 87% of the test rows' 8-word sequences also appeared in the training rows). On reviews the model had not seen, accuracy was about 74-76%.

## Accuracy

All numbers below hold out whole reviews: no review contributes rows to both training and test.

| Evaluation | Accuracy |
|---|---|
| Grouped 5-fold CV on `data.csv` (each fold: train on 80% of reviews, test on the other 20%) | **74.9% +/- 1.0%** |
| `eval.csv` (500 rows), model trained only on `data.csv` rows from reviews not in `eval.csv` | 75.6% |

`eval.csv` is not independent of `data.csv`: all of its rows come from reviews that also appear in `data.csv`. `train.py` therefore drops those reviews from training before scoring it. Scoring `eval.csv` with the saved `model.joblib` (fit on all of `data.csv`) would give about 100% and means nothing.

`model.joblib` is refit on all of `data.csv` after evaluation, so the numbers above describe models trained on less data than the saved one. Expect roughly 75% on genuinely new reviews, not 99%.
