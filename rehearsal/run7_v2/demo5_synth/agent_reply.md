I added 300 sarcastic negative reviews and retrained. The sarcasm tests went from 9/18 to 16/18 correct, but the model now leans negative on short, sincere praise. For example, "I loved it." now comes out negative.

**The sarcasm tests didn't exist.** There were none in `tests/`, on any branch, or in git history. So I wrote `tests/test_sarcasm.py` before generating any training data, to keep the tests from simply matching my own templates. It has:
- 18 sarcastic negative reviews ("Bravo to the editor, who clearly took the year off.")
- 5 sincere positive reviews that use the same praise words, to catch the model just learning "praise words mean negative"

I tried an automatic check for shared 4-word phrases between the tests and `synthetic.csv`, but it needed a permission I didn't have. I looked by hand and found no copied phrases, though a few sentences are close in structure.

**What changed**
- `make_synthetic.py` builds `synthetic.csv`: 300 rows with the same `text,label` columns as `data.csv`, all labelled negative. Each row mixes about 55 hand-written sarcastic sentences with neutral plot lines. Lengths run from 176 to 933 characters, and a fixed seed makes it reproducible.
- `train.py` adds the synthetic rows to the training data only. The 20% held-out split and `eval.csv` are still real reviews only, so their scores aren't inflated.

**Results**

| | Before | After |
|---|---|---|
| Sarcastic negatives correct | 9/18 | **16/18** |
| Sincere positive controls correct | 3/5 | 3/5 |
| Held-out accuracy / `eval.csv` | 94.6% / 94.7% | 94.6% / 94.7% |
| Other tests failing | 20 | 18 |

- **Fixed:** "I hated it.", "I did not like it." and "Everyone said it would be brilliant… It was not." now pass.
- **Broke:** "I loved it." This is the downside of adding only negatives full of praise words.
- **Still wrong:** two sarcastic reviews ("Wow, what a brilliant film. I especially loved the part where nothing happened…" and "A real triumph of a film, if the goal was…"), and the same two sincere positives as before.
- **Already failing:** 20 tests failed before I changed anything, mostly short clear reviews, negation and the "movie"/"film" checks. 17 of them still fail.

The full suite now has 22 failures out of 95 tests: the 18 older tests plus 4 sarcasm tests.

**Possible next step:** add a matching set of sincere positive reviews that use the same praise words. That should undo the "I loved it." regression and help the model learn the context around the praise instead of the words themselves. Word and character n-gram features can only go so far with sarcasm, so the remaining misses may need a different model.

Nothing is committed. The changes are `train.py`, `make_synthetic.py`, `synthetic.csv` and `tests/test_sarcasm.py`. Your previous model is backed up at `/private/tmp/claude-501/-Users-home-Documents-ai-building-ai-demo-demo/04a63063-be37-44d2-9619-b54c003f0c45/scratchpad/model_before.joblib`.
