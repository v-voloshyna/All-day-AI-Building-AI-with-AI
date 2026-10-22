I built the sentiment classifier. Its held-out accuracy is **76.7%**, measured on 1,485 rows from reviews the model never saw. Treat that as roughly 73-77%.

## What I built (all in `demo/`)
- **`train.py`** fits TF-IDF on word 1-2 grams plus logistic regression, saves `model.joblib` (already git-ignored), and prints the accuracy. It runs in about 2.7s and gives the same result on re-runs (fixed seed).
- **`predict.py`** exposes `predict(text: str) -> str`, returning `"positive"` or `"negative"`. The model loads once at import (about 1s). 200 calls took about 0.06s. It also works as a small CLI.
- **`README.md`** covers how to run it, the accuracy, and why the number isn't 99%.

## Accuracy, and why it is not 99%
- **Leakage in `data.csv`:** every row is a 150-word window of a longer movie review, and the windows overlap heavily. There are about 5 windows per review and roughly 1,470 reviews in the 7,500 rows.
- **The naive number is inflated:** a normal random train/test split gave about 99%, but that was memorization. The same review sat on both sides of the split.
- **What `train.py` does instead:** it rebuilds which rows came from the same review (rows sharing an 8-word phrase are grouped). It then splits by review, so the 76.7% reflects performance on unseen reviews.
- **How noisy it is:** grouped 5-fold cross-validation on the training portion gave about 73%. The test set holds only about 300 distinct reviews, so a single split can move by a couple of points.
- **Which model is saved:** the one trained on the 80% split that was evaluated, so the reported number matches the saved model exactly.

I tried a few settings (unigrams only, different regularization, a linear SVM). They all landed around 72-73% on cross-validation, so the choice isn't critical and I kept it simple.

## Other notes
- **`eval.csv`:** it was already in the directory. I did not read, train on, or tune against it. If it is meant as a final check, it is worth running. It will only be fair if its reviews don't overlap `data.csv`.
- **Odd grouping:** the grouping step produced 3 mixed-label groups out of 1,469. These come from unrelated reviews that share a quoted phrase and get merged. That only makes the split more conservative.
