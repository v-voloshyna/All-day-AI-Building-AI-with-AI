# Sentiment classifier (demo_c)

TF-IDF (unigrams) + logistic regression, trained on `data.csv` (text, label = positive/negative).
Trains in about 4 seconds on CPU.

## Run it

Use the repo's virtualenv (pandas, scikit-learn, numpy, joblib):

```bash
cd demo_c
../.venv/bin/python train.py        # trains, prints accuracy, writes model.joblib
../.venv/bin/python predict.py "I loved it"   # quick CLI check
```

```python
from predict import predict
predict("A wonderful, moving film")   # -> "positive"
```

`predict.py` loads `model.joblib` once at import time (about 1 s); each call afterwards takes well under a millisecond. Run `train.py` first. `model.joblib` is git-ignored.

## Accuracy

**Held-out accuracy: 71.5%** (1,500 rows from reviews the model never saw during evaluation; classes are balanced 50/50, so chance is 50%). Grouped 5-fold cross-validation on the training portion gave about 71-73% for this and several alternative models (bigrams, linear SVM, Naive Bayes), so the choice of model is not the bottleneck.

### Why not 96-99%?

A plain random train/test split gives about **96.5%**, but that number is inflated and should not be quoted. `data.csv` holds roughly 1,500 distinct reviews, each cut into about 5 heavily overlapping text windows (7,500 rows). A random split puts windows of the same review in both train and test, so the model is partly graded on text it has already seen. (Every row's nearest neighbour in the data has the same label, and rows share long identical word sequences.)

`train.py` therefore rebuilds the review groups (connected components of TF-IDF cosine similarity > 0.5) and holds out whole reviews. The 71.5% figure comes from that split (seed 42). With only about 300 held-out reviews, expect roughly +/- 2 points of noise between different splits. The saved model is refit on all 7,500 rows.

`eval.csv` (also in this folder) was not used for training, tuning or the numbers above.
