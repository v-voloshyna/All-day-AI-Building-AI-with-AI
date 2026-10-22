I built the classifier and it reaches 74.4% accuracy on reviews it never saw in training. The ~99% that a plain train/test split gives is inflated by overlapping rows in `data.csv`.

**What I built** (all in `demo/`):
- `train.py` trains TF-IDF unigrams plus logistic regression, prints the accuracy and saves `model.joblib`. It runs in about 2 seconds on CPU.
- `predict.py` exposes `predict(text) -> "positive" | "negative"`. It loads the model once at import. It also works from the command line.
- `README.md` has the run instructions, the accuracy, and the caveats below.
- `model.joblib` is git-ignored, so run `train.py` before `predict.py`.

**Accuracy: 74.4%** on 1,480 held-out rows from unseen reviews. The classes are balanced (3,750 each), so chance is 50%. I checked that `predict` returns only the two labels.

**Why not ~99%:**
- **Overlapping windows:** `data.csv` has 7,500 rows, but they are overlapping 150-word windows cut from only about 1,471 source reviews, roughly 5 windows per review.
- **Why a plain split inflates the score:** with a normal random split, near-copies of every test row sit in the training set. That scored about 99% in my first run, which only measures memorisation.
- **How `train.py` avoids it:** it groups rows that share any 8-word sequence (the same review) and splits by group. I picked the settings (unigrams, `min_df=5`, `C=1`) with group-aware cross-validation on the training portion only. The held-out portion was scored once. Every config I tried landed around 72–74%, so I kept the simplest.
- **Final model:** after evaluating, the saved model is refit on all the data, so it should be slightly better than 74.4%.

**Caveat about `eval.csv`:** it is not an independent test set. All of its rows come from reviews that also appear in `data.csv`, and the shipped model scores about 94% on it. That figure is inflated by the same overlap. I did not train on `eval.csv`. For an honest test, use reviews that are not in `data.csv`.

**How to run** (from the `demo` directory):
```
../.venv/bin/python train.py
../.venv/bin/python predict.py "I loved this movie"
```
Or from Python: `from predict import predict; predict("...")`.
