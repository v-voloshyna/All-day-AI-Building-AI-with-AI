I built the classifier and ran it: it gets **93.07% accuracy on eval.csv** (404 examples). Training takes about 1.5 s on CPU.

- **`train.py`** trains a TF-IDF model (single words and word pairs) with logistic regression on `data.csv`, saves it to `model.joblib`, and prints the eval accuracy.
- **`predict.py`** has `predict(text) -> "positive" | "negative"`. It loads the model once when the module is imported, and you can also run it from the command line with `python predict.py "some text"`. I tried it on one clearly positive and one clearly negative sentence and both came back correct.
- **`README.md`** covers setup, how to train and predict, and the accuracy.

I didn't use eval.csv to choose settings, so 93.07% is a fair measure on unseen data. I also checked that none of the eval texts appear in the training data.

`model.joblib` isn't committed because the existing `.gitignore` excludes `*.joblib`, so run `python train.py` before using `predict.py`. Nothing has been committed yet.