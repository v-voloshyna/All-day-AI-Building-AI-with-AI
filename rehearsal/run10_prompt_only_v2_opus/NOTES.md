# run10: Opus 5.5 headless (claude -p, permission-mode auto), v2 data, NEW prompt 01_build (2026-10-21)

Accidental but useful: the v3 build had gone to a stray path, so this run tested the revised prompt
(names eval.csv, "TF-IDF and a linear model is fine") on the old v2 data. 46 s, 7 turns, $0.27.
Reported 95.35% on eval.csv. It checked 10-word-run overlap (2 rows) and noticed shared film/actor names,
but never looked at the weights and did not find movie/film or the spelling.
The holdout figure in oracle_output.txt (0.835) is not meaningful: holdout.csv had already been rebuilt for
v3 and those reviews overlap v2's training reviews.
