# run9: Opus 5.5, interactive session, v2 data, prompt 01_build — found the artifact in demo 1 (2026-10-21)

Interactive `claude` session (auto mode), not `claude -p`. The agent inspected the logistic-regression
weights after its first fit, saw movie/film and the British/American pairs, wrote 20 probe sentences
("This movie was wonderful" came out negative), added preprocess.py that normalises spelling and merges
movie/film/picture/flick, and reported 84.5% on eval.csv with a full explanation of the leak in README.md.
Oracle: eval 0.845, holdout 0.707, adversarial 5/12.

Why it was found: in the v2 model the artifact is one word pair. `movie` = -16.5 and `film` = +15.1 while
no other feature is above 2.5, so any look at the coefficients, or any short probe containing "movie", gives
it away. The spelling pairs cover only 2-5% of rows and add almost nothing.
This is what led to the v3 data (genre selection, no word swaps).
