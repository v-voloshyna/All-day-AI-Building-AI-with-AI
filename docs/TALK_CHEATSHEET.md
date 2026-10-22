# Talk cheat sheet (30 min) — one line per beat

Setup: `claude login` (day before) · oracle terminal: `source /Users/home/Desktop/presentation/oracle/aliases.sh && cd "$DEMO"` · `reset_demo` · agent terminal: `cd /Users/home/Documents/ai-building-ai-demo/demo && claude`, then `/clear`.
"paste `p X`" = `p X | pbcopy` in the oracle terminal, cmd+V in the agent terminal.

| Clock | One sentence | Run | Open |
|---|---|---|---|
| 0:00 | "Everything the agent does is real; everything on the right it cannot see." | agent: paste `p 01_build` | thesis slide |
| 2:00 | "Ninety seconds, ninety-two percent, it deduplicated and grouped, all true; hold on to the word *held-out*." | `ev eval baseline --reported <its number>` | README.md, train.py split line |
| 4:00 | Skip `p 01c_leak_check` unless rehearsed (it may find the spelling early). | | |
| 5:30 | "Same model, same data; now I ask it to test, not build." Paste; while it runs: "What would you test?" (3 answers) | agent: paste `p 02_tests` | |
| 7:30 | "72 tests, 21 red. Movie minus 24, film plus 21: the 94.7 measures which site a review came from. Build got 94.7; test got 'meaningless'; two minutes apart. Diligence is a prompt, not a property." | `pytest -q \| tail -3` | its reply, the weights line |
| 8:45 | "It found the cause. Can it price it? Twelve sentences from a stranger: 7 of 12." | `ev adversarial own_tests --show` · `note own_tests "72 tests, 21 failed, found movie/film"` | |
| 9:30 | "It said 94.7; its tests say wrong; it cannot say how wrong. On reviews it has never seen?" (3 guesses, then run) | `ev holdout baseline` | |
| 11:00 | "It can name the cause; only data from outside can price it: the genre rule is 87% inside data.csv, 59% outside." | agent: paste `p 03_why`; oracle: `shortcut` (`--model`) | |
| — | Fallback, suite green: "every test tests what the builder believed; the stranger's 12: 7." | `pytest -q -rx \| tail -30` | test_quality.py xfail block |
| 13:00 | Start the loop in a second tab and keep talking. | `loop`, `tail -f "$ORACLE/loop.log"` | oracle/prompts/04_loop_template.txt |
| 14:00 | Optional, only if the loop is NOT running: "Fix it; watch what it changes." (unrehearsed) | agent: paste `p 03b_fix`; `ev holdout leak_fix --reported <n>` | |
| 16:00 | Honest run: "six defensible changes, the number climbs." Gaming run (v1 fallback): "it kept the number and threw away the sentence." | `git diff iter2 iter3 -- train.py`, `for i in 1 2 3; do jq -r .rationale "$ORACLE/loop_runs/iter$i.json"; done` | oracle/rehearsal/loop_real_2026-10-20 as fallback |
| 19:00 | "Eval up a point and a half; holdout down a third." | `ev holdout loop --reported <final eval>` | fallback: `loop --backend mock` |
| 20:00 | "Generate 300 sarcastic examples and fix it." | agent: paste `p 05_synth` | rehearsal reply in oracle/rehearsal/run3_sonnet_found_leak/demo5/ |
| 22:30 | "Read three rows aloud: 'meets a stranger with a mysterious past', 75 times." | `head -15 synthetic.csv`, `div synthetic.csv data.csv` | make_synthetic.py |
| 24:00 | "Every sarcasm caught, every 'not bad' lost, holdout unchanged. It warned us: 'I loved it' is negative now." | `ev holdout synthetic --reported <n>`, `ev adversarial synthetic --show` | |
| 25:00 | "Three columns; the gap between them is the limit." | `truth` | five-limits slide |
| 28:00 | Q&A. | | |

Five limits (PDF still shows the old limit 2): the reported number is the agent's number · the agent checks what the task asks it to check: diligence is a prompt, not a property · "held-out" was a property of the split, not the data · an optimiser with a free hand optimises the metric · synthetic data has fewer ideas than you think.

Expect (v3, rehearsed 2026-10-22, headless): build 30 s–2 min, reported **0.931** ("fair measure on unseen") · eval 0.931 · **holdout 0.684** · adversarial 7/12 · tests 52, **green** (46 pass, 6 xfail), nothing found → use the green-suite line for limit 2 · 03_why blames length/topic, asks for the holdout · loop and synthetic not re-rehearsed on v3 (v2 numbers: loop eval +1.5 pts, holdout flat; synthetic repeat-5-gram 100% vs 9%).
