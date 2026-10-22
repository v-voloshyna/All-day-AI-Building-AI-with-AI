I built the classifier, and it got **95.35% accuracy on eval.csv** (452 rows; macro F1 0.952). Training takes about 1 second on CPU.

- **`train.py`**: trains a TF-IDF model (single words and word pairs) with logistic regression, saves it to `model.joblib`, and prints the eval accuracy with a per-class breakdown.
- **`predict.py`**: loads the model once when it's imported and exposes `predict(text) -> "positive" | "negative"`. You can also run it from the command line: `python3 predict.py "some text"`. I tried one clearly positive and one clearly negative sentence and both came back right.
- **`README.md`**: how to run it, the results, and the caveat below.

**The 95% may be higher than you'd get on brand-new reviews.** Each row is a roughly 860-character excerpt of a longer movie review. No eval text appears in data.csv, even as a near-copy: only 2 eval rows share any 10-word run with the training data. But the same films and actors appear in both files (for example "carrie-anne moss" and "tim blake nelson"). So different parts of the same original review are probably split between training and eval. I can't confirm this because there's no review ID column. If you have one, splitting by review would give a more honest number.

I picked the model settings once up front and didn't tune them against eval.csv. `model.joblib` is already listed in your `.gitignore`, so it won't be committed. Nothing is committed yet; the three new files are untracked.