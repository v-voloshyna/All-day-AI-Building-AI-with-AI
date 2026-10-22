#!/usr/bin/env python3
"""The closing table. Reads runs.jsonl (written by eval.py) and notes.jsonl (written here).

    python truth.py                              # print the table
    python truth.py note baseline "0.94"         # record what the agent claimed for a stage
    python truth.py note own_tests "5 passed"

Columns: what the agent reported | what the oracle measured on data the agent could see |
what the oracle measured on holdout.csv. One row per stage, in order of first appearance.
"""
import json, sys
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
RUNS, NOTES = ORACLE / "runs.jsonl", ORACLE / "notes.jsonl"


def read_jsonl(p):
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt_run(r):
    if r["data"].startswith("adversarial") or r["n"] <= 50:
        return f"{r['correct']}/{r['n']}"
    return f"{r['acc']:.2f}"


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "note":
        with open(NOTES, "a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": sys.argv[2], "reported": " ".join(sys.argv[3:])}) + "\n")
        print(f"noted: {sys.argv[2]} -> {' '.join(sys.argv[3:])}")
        return

    runs, notes = read_jsonl(RUNS), read_jsonl(NOTES)
    stages = []
    for r in runs + notes:
        if r["stage"] not in stages:
            stages.append(r["stage"])
    if not stages:
        print("no runs yet (eval.py writes runs.jsonl; truth.py note <stage> <text> adds a claim)")
        return

    table = []
    for s in stages:
        reported = [r["reported"] for r in runs if r["stage"] == s and r.get("reported")]
        reported += [n["reported"] for n in notes if n["stage"] == s]  # a later note wins
        visible = [f"{r['data']} {fmt_run(r)}" for r in runs if r["stage"] == s and not r["holdout"]]
        holdout = [fmt_run(r) for r in runs if r["stage"] == s and r["holdout"]]
        table.append((s, reported[-1] if reported else "—",
                      " · ".join(visible) if visible else "—",
                      holdout[-1] if holdout else "—"))

    cols = ["stage", "agent reported", "oracle, data agent saw", "oracle, holdout"]
    widths = [max(len(c), *(len(row[i]) for row in table)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


if __name__ == "__main__":
    main()
