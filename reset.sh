#!/usr/bin/env bash
# Restore the repo to the 'start' tag (data.csv, eval.csv, .gitignore only) and clear oracle logs.
#   bash reset.sh            # repo = ../demo
#   bash reset.sh /path/repo
set -euo pipefail
ORACLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-${DEMO:-$HOME/Documents/ai-building-ai-demo/demo}}"
cd "$REPO"
git rev-parse --verify start >/dev/null 2>&1 || { echo "no 'start' tag in $REPO; run setup.sh first"; exit 1; }
git reset -q --hard start
git clean -fdxq
git tag -l 'iter*' | xargs -r git tag -d >/dev/null
rm -f "$ORACLE/runs.jsonl" "$ORACLE/notes.jsonl" "$ORACLE/loop.log"
rm -rf "$ORACLE/loop_runs"
echo "repo reset to 'start' ($(ls | tr '\n' ' ')); oracle logs cleared"
