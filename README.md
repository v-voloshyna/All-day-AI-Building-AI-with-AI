# Oracle for the talk "Building AI with AI: what are the limits"

Everything the agent must never see. The agent works in a separate repo (`$DEMO`, by default
`~/Documents/ai-building-ai-demo/demo`) that holds only `data.csv`, `eval.csv` and `.gitignore` at git tag `start`.
Nothing reachable from that repo names this folder. Keep this repository private.

- `docs/` — `RUNBOOK.md` (beat-by-beat script), `PRESENTER.md`, `TALK_PLAN.md`, `TALK_CHEATSHEET.md`, `INTERACTIVE_PLAN.md`.
- `prompts/` — the prompts pasted into the agent, one file per beat (`p 01_build`).
- `holdout.csv`, `adversarial.csv`, `*_meta.csv` — the data the agent never sees, plus per-row provenance.
- `build_data.py` — rebuilds data.csv / eval.csv / holdout.csv (v3) from `raw/movie_reviews.zip`; always pass `--demo "$DEMO"`.
- `eval.py`, `shortcut.py`, `dupes.py`, `diversity.py`, `loop.py`, `truth.py` — the oracle tools; `aliases.sh` wraps them (`source aliases.sh`).
- `reset.sh` — restores the demo repo to `start` and clears the run logs. `rehearse.py` — headless rehearsal with `claude -p`.
- `fixtures/` — a canned Demo-1 result for rehearsing the tools without an agent.
- `rehearsal/` — saved agent output from past runs (run2–run10 are v1/v2 data; run11+ are v3). `v1_data/`, `v2_data/` — earlier datasets.

Run logs (`runs.jsonl`, `loop.log`, `loop_runs/`) are git-ignored; copy a run into `rehearsal/` if it is worth keeping.
