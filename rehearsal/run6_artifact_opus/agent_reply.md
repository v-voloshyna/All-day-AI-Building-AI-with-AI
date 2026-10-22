I built the classifier and ran it: it gets **94.7%** accuracy on a 20% held-out part of `data.csv`, and the same on `eval.csv`. On reviews it has never seen, expect closer to **93%**.

**Files**
- **`train.py`**: TF-IDF on single words and word pairs, fed into logistic regression. It takes about 2.5 seconds on CPU. It trains on 80% of `data.csv`, prints the held-out accuracy (and the `eval.csv` accuracy if that file is there), then retrains on all of `data.csv` and saves `model.joblib`.
- **`predict.py`**: `predict(text) -> "positive" | "negative"`. The model loads once when you import it (about 1 second); after that each call takes well under a millisecond. You can also run `python3 predict.py "some text"`.
- **`README.md`**: how to run it and the accuracy numbers, including the caveat below.

**Why the real number is lower.** Each row is a chunk of about 860 characters cut from a longer movie review. Chunks of the same review show up on both sides of the split. For example, parts of the same *Glory* and *The Matrix* reviews are in both `data.csv` and `eval.csv`, so the model can partly memorize things like names and the score comes out a bit high. There are no review IDs, so I grouped chunks approximately by shared rare words and re-ran the test with each group kept on one side. That gave about 93%, so the overlap adds roughly 1 point.

**Edge case:** an empty string returns "positive", because the model has nothing to go on and falls back to the slightly more common class. Add a guard if empty input can reach it.

`model.joblib` is already in `.gitignore`. I haven't committed anything.
