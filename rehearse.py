#!/usr/bin/env python3
"""Run one prompt against the demo repo with Claude Code headless, keep everything, then score it.

    python rehearse.py 01_build --out rehearsal/run10_v3/demo1 [--model opus] [--stage baseline]
    python rehearse.py 02_tests --out rehearsal/run10_v3/demo2 --resume <session_id> --stage own_tests
    python rehearse.py "literal prompt text" --out ...     (anything that is not a prompts/<name>.txt)

Writes to --out: agent_reply.json (full result), agent_reply.md (its final message), stderr.txt, and a copy
of every file the agent created or changed in the repo. Then runs ev eval / holdout / adversarial for
--stage and appends their output to oracle_output.txt. Prints session_id so the next prompt can --resume it.
The call runs with --permission-mode auto, like the stage.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
DEMO = Path(os.environ.get("DEMO", "/Users/home/Documents/ai-building-ai-demo/demo"))
PY = shutil.which("python3") or sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="prompts/<name>.txt or literal text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--resume", default=None, help="session id of the previous prompt")
    ap.add_argument("--stage", default="baseline")
    ap.add_argument("--reported", default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    pf = ORACLE / "prompts" / f"{args.prompt}.txt"
    prompt = pf.read_text() if pf.is_file() else args.prompt
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in DEMO.rglob("*") if p.is_file() and ".git" not in p.parts}

    cmd = ["claude", "-p", prompt, "--output-format", "json", "--permission-mode", "auto", "--model", args.model]
    if args.resume:
        cmd += ["--resume", args.resume]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow nesting from a Claude Code shell
    t0 = time.time()
    r = subprocess.run(cmd, cwd=DEMO, text=True, capture_output=True, timeout=args.timeout, env=env)
    secs = time.time() - t0
    (out / "stderr.txt").write_text(r.stderr)
    (out / "prompt.txt").write_text(prompt)
    try:
        data = json.loads(r.stdout)
        if isinstance(data, list):
            data = next((d for d in reversed(data) if d.get("type") == "result"), data[-1])
    except Exception:  # noqa: BLE001
        (out / "stdout_raw.txt").write_text(r.stdout)
        sys.exit(f"claude -p failed ({r.returncode}) after {secs:.0f}s; see {out}/stderr.txt")
    (out / "agent_reply.json").write_text(json.dumps(data, indent=1))
    (out / "agent_reply.md").write_text(data.get("result") or "")

    for p in DEMO.rglob("*"):
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.suffix not in (".joblib", ".pkl", ".csv"):
            if p not in before or p.stat().st_mtime > before[p]:
                dst = out / "repo" / p.relative_to(DEMO); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)

    print(f"session_id={data.get('session_id')}  turns={data.get('num_turns')}  {secs:.0f}s  ${data.get('total_cost_usd', 0):.2f}  "
          f"model={list((data.get('modelUsage') or {}).keys())}")
    print("---- agent reply ----"); print(data.get("result") or "(empty)"); print("---------------------")

    if not args.no_eval:
        log = []
        for ds, extra in (("eval", ["--reported", args.reported] if args.reported else []), ("holdout", []), ("adversarial", ["--show"])):
            c = [PY, str(ORACLE / "eval.py"), "--data", ds, "--stage", args.stage, "--repo", str(DEMO)] + extra
            e = subprocess.run(c, cwd=DEMO, text=True, capture_output=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
            txt = e.stdout + ("\n" + e.stderr.strip() if e.returncode else "")
            log.append(f"== ev {ds} {args.stage}\n{txt}")
            print(txt.strip())
        (out / "oracle_output.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
