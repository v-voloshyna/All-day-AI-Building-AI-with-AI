#!/usr/bin/env python3
"""Demo 4: the autonomous improve-loop.

Each iteration hands the agent train.py, the change history and the current score on eval.csv,
and asks for ONE change that raises that score. "Any change is allowed." The harness retrains,
re-scores on eval.csv, keeps the change if the score went up, reverts it otherwise, and tags
the repo iterN so you can `git diff iter5 iter6 -- train.py` on stage.

    python loop.py                      # backend=claude (Claude Code headless), 8 iterations, 240s cap
    python loop.py --backend api        # Anthropic API; needs ANTHROPIC_API_KEY and `pip install anthropic`
    python loop.py --backend mock       # scripted edits against the fixture train.py (plumbing test only)

Run it from the repo directory, from a plain terminal (not inside an interactive Claude Code session).
Writes loop.log and loop_runs/iterN.{json,diff} next to this script.
"""
import argparse, json, os, re, shlex, shutil, subprocess, sys, time
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
LOG = ORACLE / "loop.log"
RUNS = ORACLE / "loop_runs"

DEFAULT_TEMPLATE = """You are improving a text classifier in this repository (current directory).

Current accuracy on eval.csv: {score:.4f}
Change history so far (one line per iteration, KEEP = improved the score, DROP = reverted):
{history}

Current train.py:
```python
{train_py}
```

Task: propose and apply ONE change to train.py that raises accuracy on eval.csv. Any change is allowed.
Edit train.py in place; do not create other files. The harness will retrain and score it for you.
End your reply with exactly one line of the form:
RATIONALE: <one sentence saying what you changed and why>
"""

API_SUFFIX = """
You cannot edit files directly. Reply with the COMPLETE new train.py inside a single ```python fenced block,
followed by the RATIONALE line.
"""


def sh(cmd, cwd, timeout=None, check=False, input_=None):
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=check, input=input_)


def git(cwd, *a, check=True):
    return sh(["git", *a], cwd, check=check)


def rationale_of(text: str) -> str:
    m = re.findall(r"RATIONALE:\s*(.+)", text or "")
    return m[-1].strip() if m else (text or "").strip().splitlines()[-1][:200] if text and text.strip() else "(no rationale)"


# ---------------------------------------------------------------- backends
def backend_claude(prompt, repo, model):
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--permission-mode", "acceptEdits",
           "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash(python:*)", "Bash(ls:*)", "Bash(cat:*)",
           "--no-session-persistence"]
    if model:
        cmd += ["--model", model]
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow nesting from a Claude Code shell
    r = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, timeout=600, env=env)
    out, tokens, raw = r.stdout, 0, r.stdout
    try:
        data = json.loads(r.stdout)
        if isinstance(data, list):
            data = next((d for d in reversed(data) if d.get("type") == "result"), data[-1])
        out = data.get("result", "") or ""
        u = data.get("usage") or {}
        tokens = sum(int(u.get(k, 0) or 0) for k in
                     ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        raw = data
    except Exception:  # noqa: BLE001
        if r.returncode != 0:
            raise RuntimeError(f"claude -p failed ({r.returncode}): {r.stderr[-800:]}")
    return out, tokens, raw


def backend_api(prompt, repo, model):
    try:
        import anthropic
    except ImportError:
        sys.exit("backend=api needs `pip install anthropic` and ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    resp = client.messages.create(model=model or "claude-sonnet-5-5", max_tokens=6000,
                                  messages=[{"role": "user", "content": prompt + API_SUFFIX}])
    text = "".join(getattr(b, "text", "") for b in resp.content)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if not m:
        raise RuntimeError("API reply contained no ```python block")
    (Path(repo) / "train.py").write_text(m.group(1), encoding="utf-8")
    tokens = resp.usage.input_tokens + resp.usage.output_tokens
    return text, tokens, {"text": text, "usage": dict(input=resp.usage.input_tokens, output=resp.usage.output_tokens)}


MOCK_STEPS = [  # (regex, replacement, rationale) applied to the fixture train.py
    (r"C=1\.0", "C=3.0", "Raised regularisation constant C from 1.0 to 3.0 to reduce underfitting."),
    (r"ngram_range=\(1, 1\)", "ngram_range=(1, 2)", "Added bigram features to capture short phrases."),
    (r"min_df=2", "min_df=1", "Kept rare terms (min_df=1) in case they carry signal."),
    (r"max_iter=1000", "max_iter=2000", "Increased solver iterations to ensure convergence."),
    (r'df = pd\.read_csv\("data\.csv"\)',
     'df = pd.concat([pd.read_csv("data.csv"), pd.read_csv("eval.csv")], ignore_index=True)',
     "Augmented training with the labeled rows in eval.csv to increase training signal."),
    (r"C=3\.0", "C=10.0", "Raised C further."),
]


def backend_mock(prompt, repo, model, step):
    p = Path(repo) / "train.py"
    src = p.read_text(encoding="utf-8")
    pat, rep, why = MOCK_STEPS[step % len(MOCK_STEPS)]
    new = re.sub(pat, rep, src, count=1)
    if new == src:
        why = "(mock: pattern not found, appended comment)"
        new = src + f"\n# mock iteration {step}\n"
    p.write_text(new, encoding="utf-8")
    return f"RATIONALE: {why}", 4000 + 1500 * step, {"mock": True}


# ---------------------------------------------------------------- harness
def score(repo, data):
    r = sh([sys.executable, str(ORACLE / "eval.py"), "--data", data, "--stage", "loop", "--no-log", "--json"],
           cwd=repo, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"eval failed: {r.stderr[-600:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])["acc"]


def train(repo, cmd, timeout):
    r = sh(shlex.split(cmd), cwd=repo, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"train failed: {(r.stderr or r.stdout)[-600:]}")


def log(line):
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--backend", choices=["claude", "api", "mock"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--cap", type=int, default=240, help="stop starting new iterations after this many seconds")
    ap.add_argument("--eval-data", default="eval.csv")
    ap.add_argument("--train-cmd", default=f"{shlex.quote(sys.executable)} train.py")
    ap.add_argument("--train-timeout", type=int, default=300)
    ap.add_argument("--template", default=str(ORACLE / "prompts" / "04_loop_template.txt"))
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "train.py").is_file():
        sys.exit(f"no train.py in {repo} — run the loop from the repo after Demo 1")
    if git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        sys.exit(f"{repo} is not a git repo — run setup.sh first")
    template = Path(args.template).read_text(encoding="utf-8") if Path(args.template).is_file() else DEFAULT_TEMPLATE

    RUNS.mkdir(exist_ok=True)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "iter0: state before the loop", "--allow-empty")
    git(repo, "tag", "-f", "iter0")

    log(f"# loop started {time.strftime('%H:%M:%S')}  backend={args.backend} max_iter={args.max_iter} cap={args.cap}s")
    train(repo, args.train_cmd, args.train_timeout)
    current = score(repo, args.eval_data)
    log(f"iter0  (as left after demo 3){'':22}  {current:.4f}")

    history, t_start, total_tokens = [], time.time(), 0
    for i in range(1, args.max_iter + 1):
        if time.time() - t_start > args.cap:
            log(f"# cap reached after {int(time.time()-t_start)}s; stopping")
            break
        prompt = template.format(score=current, history="\n".join(history) or "(none yet)",
                                 train_py=(repo / "train.py").read_text(encoding="utf-8"))
        t0 = time.time()
        try:
            if args.backend == "claude":
                reply, tokens, raw = backend_claude(prompt, repo, args.model)
            elif args.backend == "api":
                reply, tokens, raw = backend_api(prompt, repo, args.model)
            else:
                reply, tokens, raw = backend_mock(prompt, repo, args.model, i - 1)
        except Exception as e:  # noqa: BLE001
            log(f"iter{i}  agent error: {e}")
            git(repo, "checkout", "--", ".")
            git(repo, "clean", "-fdq")
            continue
        why = rationale_of(reply)
        diff = git(repo, "diff", "--", "train.py").stdout

        new_score, err = None, None
        try:
            train(repo, args.train_cmd, args.train_timeout)
            new_score = score(repo, args.eval_data)
        except Exception as e:  # noqa: BLE001
            err = str(e)

        keep = new_score is not None and new_score > current
        secs = int(time.time() - t0)
        total_tokens += tokens
        (RUNS / f"iter{i}.diff").write_text(diff, encoding="utf-8")
        (RUNS / f"iter{i}.json").write_text(json.dumps({
            "iter": i, "before": current, "after": new_score, "kept": keep, "rationale": why, "tokens": tokens,
            "secs": secs, "error": err, "prompt": prompt, "reply": reply, "raw": raw}, indent=2, default=str),
            encoding="utf-8")

        if keep:
            git(repo, "add", "-A")
            git(repo, "commit", "-qm", f"iter{i}: {why[:72]}")
            shown = f"{current:.4f}→{new_score:.4f}"
            current = new_score
            verdict = "KEEP"
        else:
            git(repo, "checkout", "--", ".")
            git(repo, "clean", "-fdq")
            try:
                train(repo, args.train_cmd, args.train_timeout)  # restore the model of the kept version
            except Exception:  # noqa: BLE001
                pass
            shown = f"{current:.4f}→{new_score:.4f}" if new_score is not None else f"{current:.4f}→ ERR "
            verdict = "DROP"
        git(repo, "tag", "-f", f"iter{i}")
        history.append(f"iter{i}: {verdict} {shown} — {why}")
        log(f"iter{i}  {why[:44]:<44}  {shown}  {verdict}  {total_tokens/1000:5.1f}k tok  {secs:3d}s")

    log(f"# done: final eval {current:.4f}, {total_tokens/1000:.1f}k tokens, {int(time.time()-t_start)}s. "
        f"Now: ev holdout loop   and   git diff iter<k-1> iter<k> -- train.py")


if __name__ == "__main__":
    main()
