I added a 72-test pytest suite and ran it: **49 passed, 23 failed.** The failures are real problems with the model and the saved model file, not bugs in the tests. I checked the main finding by hand.

**The model mostly learned the words "movie" and "film", not sentiment.**
- In `data.csv`, 77% of negative chunks contain "movie" but only 4.5% of positive ones do. `eval.csv` has the same skew.
- The two strongest weights in the model are `movie` (−23.7) and `film` (+21.0). Next come `humor`/`theater` on the negative side and `humour`/`theatre` on the positive side. The positive and negative reviews probably come from different sources, British and American English.
- **Swapping "movie" and "film" in `eval.csv` drops accuracy from 95.1% to 29.4%**, and 66.6% of predictions flip.
- So the README's 94.7%, and its ~93% "conservative" estimate, mostly measure which source a review came from. Neither number tells you how the model will do on ordinary text.

**Other failures:**
- **Short, clear reviews:** 10 of 11 failed. "Best movie ever!" → negative, "I hated it." → positive, "Terrible film. Awful acting…" → positive.
- **Negation:** all 5 failed. "not bad" → negative, "I did not like it." → positive.
- **Mixed reviews:** 2 of 3 failed.
- **Single words:** "movie" alone scores −23.5 and "film" alone +21.2, both very confident verdicts on a neutral word.
- **Saved model file:** `model.joblib` was saved with scikit-learn 1.9.1, but 1.7.1 is installed here. It loads with a version warning, and `predict_proba` crashes (`'LogisticRegression' object has no attribute 'multi_class'`). `predict()` still works for now. A model freshly retrained with 1.7.1 agrees with it on more than 99% of `eval.csv`.

**What passed:**
- `predict()` always returns a plain `"positive"` or `"negative"` string, including for empty strings, emoji, CJK text, null bytes and 200k-character inputs. Results are the same on every call.
- Bad inputs raise errors: `None`, numbers, lists and invalid UTF-8.
- Both ways of calling it from the shell work, with arguments and with input piped in.
- There are no duplicate texts between `data.csv` and `eval.csv`.
- Overall accuracy on `eval.csv` is at least 92% and each class's recall is at least 88%.
- Predictions don't change with upper/lower case, `<br />` tags, an added neutral sentence, duplicating the text, or switching spellings like theatre/theater.

One thing to know: text with no known words, including the empty string, always comes out "positive", because that's the model's default lean. The tests only check that a valid label comes back.

**Files:**
- `tests/conftest.py`: shared setup
- `tests/test_predict_contract.py`: what `predict()` returns, odd inputs, shell use
- `tests/test_data.py`: checks on the two CSV files
- `tests/test_model_artifact.py`: the saved model file and `train.build_model`
- `tests/test_quality.py`: accuracy and the hard inputs
- `pytest.ini`: hides the repeated version warning, which one test checks for directly

Run it with `python3 -m pytest`; it takes about 4 seconds.

I left the failing tests failing so they show what needs fixing. To fix the "movie"/"film" problem you'd need to remove those source-specific words, normalise the spellings, or get data where both classes come from the same sources. Then retrain and re-save `model.joblib` with the scikit-learn version you run it on. If you'd rather keep the suite green until then, I can mark the known model failures as expected failures (`xfail`).
