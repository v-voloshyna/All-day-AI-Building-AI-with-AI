# Sentiment classifier

TF-IDF (word unigrams + bigrams) + logistic regression, trained on `data.csv`.
Trains in a few seconds on CPU.

## Usage

```bash
pip install scikit-learn pandas joblib
python train.py                      # trains, prints accuracy, writes model.joblib
python predict.py "What a great film"  # -> positive
```

```python
from predict import predict   # model is loaded once, at import
predict("A dull, tedious mess.")  # -> "negative"
```

## Accuracy

**~74% (0.742 ± 0.025)**, from 5-fold cross-validation in which whole reviews are held out.

`eval.csv` gives **99.8%**, but that number is misleading:

- The rows of `data.csv` are overlapping ~860-character chunks of about 1,470
  longer reviews, around 5 chunks per review.
- Every row in `eval.csv` shares an exact 8-word run with some training row,
  and its nearest training row has the same label 100% of the time.

So `eval.csv` (and any random row-level split) mostly checks whether the model
recognises reviews it has already seen. `train.py` rebuilds the review groups
(rows that share an 8-word sequence) and uses `GroupKFold`, so a review's
chunks are never in both train and test. Expect about 74% on text from new
reviews.

Other simple models (linear SVM, Naive Bayes, different regularisation, unigrams
only) all scored 72–75% under the same grouped CV, which is within noise.
