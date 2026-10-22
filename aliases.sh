# source this in the oracle terminal:   source /Users/home/Desktop/presentation/oracle/aliases.sh && cd "$DEMO"
# then run from inside the repo directory.
export ORACLE="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%N}}")" && pwd)"   # works in bash and zsh
export DEMO="${DEMO:-/Users/home/Documents/ai-building-ai-demo/demo}"   # the repo the agent works in
PY="$(command -v python3 || command -v python)"; export PY   # same interpreter the agent uses (python3); the venv has a newer sklearn and its pickles do not load here
export PYTHONWARNINGS=ignore   # sklearn pickle warnings would clutter the stage terminal
ev()         { "$PY" "$ORACLE/eval.py" --data "$1" --stage "$2" "${@:3}"; }   # ev holdout baseline [--show] [--reported 0.94]
dupes()      { "$PY" "$ORACLE/dupes.py" "$@"; }                               # dupes --pairs 3   (v1 reveal; now shows ~0 overlap)
shortcut()   { "$PY" "$ORACLE/shortcut.py" "$@"; }                            # shortcut [--model]  (the Demo 3 reveal: genre words by label, data.csv vs holdout)
div()        { "$PY" "$ORACLE/diversity.py" "$@"; }                           # div synthetic.csv data.csv
loop()       { "$PY" "$ORACLE/loop.py" "$@"; }                                # loop --max-iter 8
truth()      { "$PY" "$ORACLE/truth.py" "$@"; }                               # truth
note()       { "$PY" "$ORACLE/truth.py" note "$@"; }                          # note baseline 0.94
reset_demo() { bash "$ORACLE/reset.sh" "${1:-$DEMO}"; }
p()          { cat "$ORACLE/prompts/$1.txt"; }                                 # p 01_build  (then copy/paste)
why()        { sed "s/{HOLDOUT}/$1/; s/{REPORTED}/$2/" "$ORACLE/prompts/03_why.txt"; }         # why 68.4% 93.07%   (03_why with the numbers filled in)
rehearse()   { "$PY" "$ORACLE/rehearse.py" "$@"; }                            # rehearse 01_build --out rehearsal/runN/demo1  (claude -p, saves + scores)
echo "oracle aliases loaded: ev shortcut dupes div loop truth note reset_demo p why rehearse   (ORACLE=$ORACLE, DEMO=$DEMO)"
