#!/usr/bin/env bash
# One-time setup. Run from this directory:  bash setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m pip install -q -r "$HERE/oracle/requirements.txt" || python3 -m pip install -q --break-system-packages -r "$HERE/oracle/requirements.txt"
cd "$HERE/demo"
if [ ! -d .git ]; then
  git init -q && git add -A && git commit -qm "start: data only" && git tag start
  echo "demo/ initialised as a git repo, tagged 'start'"
else
  echo "demo/ already a git repo"
fi
chmod +x "$HERE/oracle/"*.py "$HERE/oracle/reset.sh" 2>/dev/null || true
echo
echo "Next:"
echo "  1. Move oracle/ OUTSIDE the repo if it isn't already (default layout is fine: demo/ and oracle/ are siblings)."
echo "  2. Oracle terminal:  source $HERE/oracle/aliases.sh && cd $HERE/demo"
echo "  3. Agent terminal:   cd $HERE/demo && claude   (or your agent of choice)"
echo "  4. Read oracle/PRESENTER.md"
