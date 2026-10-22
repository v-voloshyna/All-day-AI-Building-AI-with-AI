# run14: Opus 5.5, interactive, v3 data, demos 1–4 (2026-10-22)

Saved from the repo and oracle logs on 2026-10-22 13:42 before cleanup.
- `repo/`: train.py, predict.py, README.md, tests/ as the agent left them after the loop (master = iter3).
- `runs.jsonl`: eval.csv 0.9307 (reported 0.9307), holdout 0.684, adversarial 7/12.
- `loop.log`, `loop_runs/`: demo 4 loop, 8 iterations, claude backend, 240 s cap. Only iter3 (C 10→4) was kept: eval 0.9307→0.9332.
