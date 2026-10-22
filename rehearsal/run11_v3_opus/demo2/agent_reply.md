I wrote the test suite and ran it: **46 passed and 6 failed as expected**, in about 3.7 seconds. I also ran it with no saved model and from another folder; the suite trained the model first and gave the same result.

**Basic behaviour (`tests/test_predict.py`)**
- `predict` always returns the plain string "positive" or "negative", and the same input always gets the same answer.
- The model is loaded exactly once, at import, and `predict` never reads it from disk again.
- Importing `predict` with no saved model fails with a clear "file not found" error. The command-line form works.
- Clearly positive and clearly negative reviews get the right label. Some negation is handled correctly too, like "not good" → negative and "not boring at all" → positive.
- Odd inputs don't crash it: empty or whitespace-only text, null bytes, numbers, punctuation, gibberish, emoji, Russian or Japanese text, HTML, and a ~1 MB review (which is handled quickly).
- Changing capitals, spacing or punctuation doesn't change the result.
- Eval accuracy must stay at or above 90%, and single-text predictions match batch predictions on 50 eval reviews.

**Training and data (`tests/test_train.py`)**
- `train.py` runs end to end on a copy in a temporary folder, so the real model isn't overwritten. It must print a valid accuracy line and save a model with both labels.
- Both CSVs have the expected columns and labels, and no review appears in both files.

**What the tests found**
- **Text with no words the model knows defaults to "positive".** That covers empty text, emoji and non-English text. It's a fallback, not a judgement about the text. The tests record this behaviour; if you'd rather those inputs raise an error or return something else, that's a change to `predict.py`.
- **Six inputs the model gets wrong**, kept as tests marked "expected to fail":
  - "good" on its own (negative, 0.835 probability)
  - "I loved every minute of it and will happily watch it again."
  - "not bad at all"
  - "not terrible"
  - sarcasm ("Oh great, another two hours of my life wasted.")
  - "good" repeated 1,000 times

  If the model ever starts getting one of these right, the test suite fails so the change gets noticed. The "good" result probably comes from how reviewers use the word ("the only good thing about it…"), but I haven't confirmed that.

I changed `train.py` to read and write its files next to the script rather than in whatever folder you run it from. Before, running it from another folder saved the model somewhere `predict.py` wouldn't find it. Eval accuracy is still 93.07%. I also added a Tests section to the README. Nothing is committed.