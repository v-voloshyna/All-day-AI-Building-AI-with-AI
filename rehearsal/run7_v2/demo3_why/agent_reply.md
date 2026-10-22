The 94.7% wasn't a real held-out number. Every score I reported came from chunks of the same movie reviews the model was trained on. I said **~93%** was a "conservative" estimate, and I was wrong: the check behind it didn't fix the problem.

**What went wrong**

- Each row in `data.csv` and `eval.csv` is a roughly 860-character piece of a longer review. Pieces of the same review end up on both sides: in the 80/20 split, and between `data.csv` and `eval.csv`. So "held-out" mostly meant "a different paragraph of a review it has already seen."
- I did notice this. To correct for it, I tried to group the pieces back into reviews by shared rare words and ran cross-validation that kept groups apart. That still gave about 93%, so I wrote that the leakage cost only about 1 point. That conclusion was wrong. The grouping didn't actually separate the reviews, so the leak survived. I re-ran it now with three stricter versions of the grouping and got 92.6–93.4% every time.

**The clearest evidence (just run)**

I skipped the classifier entirely: for each `eval.csv` row, I found the most similar `data.csv` row and copied its label. That alone scores **90.5%**. So the model barely needs to recognise sentiment to do well on `eval.csv`. It can simply recognise which review a passage came from, through names, plot details and the reviewer's style. On reviews it has truly never seen, none of that helps, and you get the classifier's actual performance: your 64.1%.

**What I can't tell from here**

I don't have your evaluation set, so I can't say how much of the gap is this leakage and how much comes from other differences. For example, if your reviews aren't movie reviews, or are a different length from the ~860-character pieces, that would cost accuracy too. Since leakage alone gets to 90%, I'd expect it to be most of the gap.

**What I'd do next**

1. **Fix the evaluation.** We need to know which review each piece came from. If the original data has review IDs, or I can rebuild them by joining pieces that follow on from each other, I'll split by review and get an honest baseline. Your 64.1% is the most trustworthy number we have until then.
2. **Correct the README.** It still claims 94.7% and "~93% conservative". I'd replace that with your 64.1% and a note on why the earlier numbers were inflated. I haven't changed it yet. Say the word and I will.
3. **Then improve the model,** measured against the honest baseline. Options include classifying whole reviews instead of pieces, and leaving out names so it can't lean on memorised details.

If you can share your evaluation set, or tell me where it came from, I can work out how much of the gap each cause explains.

The test scripts are in `/private/tmp/claude-501/-Users-home-Documents-ai-building-ai-demo-demo/fb242d8d-f267-4c04-b180-4f892c02f5bd/scratchpad/leak_probe.py` and `/private/tmp/claude-501/-Users-home-Documents-ai-building-ai-demo-demo/fb242d8d-f267-4c04-b180-4f892c02f5bd/scratchpad/leak_check.py`.
