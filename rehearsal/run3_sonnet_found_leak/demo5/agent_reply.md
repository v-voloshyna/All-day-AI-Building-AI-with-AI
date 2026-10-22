I generated the 300 sarcastic negatives, trained on them and re-ran the tests. The sarcasm fix partly worked but has a cost: it catches more sarcasm and now mislabels much more sincere praise as negative. Held-out accuracy on real reviews is flat (74.2%, was 74.4%).

**How I generated the examples**
- `make_synthetic.py` is a seeded, template-based generator that writes `synthetic.csv` (300 rows, all `negative`). Each example pairs a sarcastic frame with a concrete complaint, often followed by an ironic closer. Frames include fake praise, fake thanks ("Thanks a lot,", "Bravo,"), mock congratulations and "a dazzling thriller, provided you enjoy ...". Example: "I especially enjoyed how the villain gives a speech instead of finishing the job. Inspired, really. Money well spent, truly."
- These are script-generated, not human-written, so variety is limited: about 23 words per example, from 28 complaints, 18 openers and a few frames.
- Before generating, I hand-wrote 60 evaluation sentences in `tests/sarcasm_eval.py`: 30 sarcastic negatives and 30 sincere positives that reuse the same cue words. I never used them to tune the generator or the model. I wrote both the generator and the sentences, so they share sarcasm idioms and the check is not fully independent.
- `train.py` adds the rows to the training data only, never to the held-out set. The shipped model is refit on `data.csv` plus `synthetic.csv`.

**What the retrained model reports**

| | before | after |
|---|---|---|
| Held-out accuracy on real reviews (1,480 rows) | 74.4% | 74.2% (within noise) |
| Sarcastic negatives labelled negative (30) | 18 | 27 |
| Sincere positives labelled positive (30) | 25 | 15 |

Most of the gain is a shift toward predicting negative, not an understanding of irony. The new data is negative-only, so words like "wow", "bravo" and "thanks" now carry negative weight. Empty input flipped from "positive" to "negative".

**What the sarcasm tests now do**
- **The original sarcasm test is still xfail.** "Oh great, another masterpiece of boredom." is still labelled positive. "great" (+2.6) and "masterpiece" outweigh the new cues, because the synthetic data re-weighted rare cue words, not common praise words.
- **New passing tests.** At least 80% of the 30 held-out sarcastic reviews are caught (27/30 now). The cue words wow, bravo and thanks have negative weights.
- **New xfail for the cost.** "Sincere praise is not flagged negative" targets at least 80% (25/30 before, 15/30 now).

**Full suite: 311 passed, 5 failed, 21 xfailed, 4 xpassed.**
- **The 5 failures are regressions from this change.** Short positive reviews now flip to negative when a neutral sentence is added: 6 of 16, up from 2 of 16. I left these red on purpose, rather than loosening them, because they measure the cost of the change.
- **The 4 xpasses are accidental.** "did not like it", "hardly a masterpiece", "I hate this!" and "meh" now pass only because the model leans negative, not because it handles negation or short text.

**Recommended next step (not done):** add sincere-praise positives that reuse the same cue words. Making the synthetic examples closer to the 150-word length of the real data would also reduce their outsized influence.
