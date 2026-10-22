# Agent environment

`Limit_the_agent_access/` is exactly what the agent gets: `data.csv`, `eval.csv` and a `.gitignore`, the contents of the
demo repo at git tag `start`. Nothing in it names this repository.

During the talk the agent does not work here. It works in a separate checkout outside this repository
(`$DEMO`, by default `~/Documents/ai-building-ai-demo/demo`), so that `cd ..`, `git rev-parse --show-toplevel`
or a glance at the parent folder cannot reach the oracle. To recreate that checkout:

    mkdir -p ~/Documents/ai-building-ai-demo && cp -R Agent_Environment/Limit_the_agent_access ~/Documents/ai-building-ai-demo/demo
    cd ~/Documents/ai-building-ai-demo/demo && git init -q && git add -A && git commit -qm "start: data only" && git tag start

`build_data.py --demo "$DEMO"` regenerates these files; copy them back here afterwards so the two stay identical.
