#!/usr/bin/env python3
"""The oracle. Evaluates whatever predict.py the agent wrote, on data the agent may or may not see.

    python eval.py --data eval --stage baseline
    python eval.py --data holdout --stage baseline
    python eval.py --data adversarial --stage own_tests --show
    python eval.py --data holdout --stage loop --reported 0.99

--data resolves: a path as given, else <name>[.csv] in the repo, else <name>[.csv] next to this script.
Every run appends one JSON line to runs.jsonl next to this script (unless --no-log); truth.py reads it.
"""
import argparse, importlib, json, os, sys, time
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
POS = {"1", "pos", "positive", "+", "true", "good", "label_1"}
NEG = {"0", "neg", "negative", "-", "false", "bad", "label_0"}


def norm(x):
    s = str(x).strip().lower()
    if s in POS:
        return "positive"
    if s in NEG:
        return "negative"
    return s


def resolve(name: str, repo: Path) -> Path:
    cands = [Path(name), repo / name, repo / f"{name}.csv", ORACLE / name, ORACLE / f"{name}.csv"]
    for c in cands:
        if c.is_file():
            return c.resolve()
    sys.exit(f"eval.py: cannot find data '{name}' (looked in cwd, {repo}, {ORACLE})")


def load_predict(repo: Path):
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    try:
        mod = importlib.import_module("predict")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"eval.py: could not import predict.py from {repo}: {type(e).__name__}: {e}")
    for batch_name in ("predict_batch", "predict_many", "predict_all"):
        if hasattr(mod, batch_name):
            fn = getattr(mod, batch_name)
            return lambda texts: list(fn(list(texts)))
    if not hasattr(mod, "predict"):
        sys.exit("eval.py: predict.py has no predict() function")
    single = mod.predict

    def run(texts):
        out = []
        for t in texts:
            r = single(t)
            if isinstance(r, (list, tuple)) and r:
                r = r[0]
            if isinstance(r, dict):
                r = r.get("label", r.get("prediction", r))
            out.append(r)
        return out
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", default="adhoc")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--show", action="store_true", help="print every row with its prediction")
    ap.add_argument("--reported", default=None, help="what the agent claimed, for the truth table")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--json", action="store_true", help="print one JSON line only (for loop.py)")
    args = ap.parse_args()

    import pandas as pd  # after argparse so --help is instant

    repo = Path(args.repo).resolve()
    data_path = resolve(args.data, repo)
    df = pd.read_csv(data_path)
    if args.limit:
        df = df.head(args.limit)
    texts = df["text"].astype(str).tolist()
    gold = [norm(x) for x in df["label"].tolist()]

    predict = load_predict(repo)
    t0 = time.time()
    raw = predict(texts)
    secs = time.time() - t0
    pred = [norm(x) for x in raw]

    correct = sum(p == g for p, g in zip(pred, gold))
    n = len(gold)
    acc = correct / n if n else 0.0
    unknown = sorted({p for p in pred if p not in ("positive", "negative")})

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage": args.stage,
        "data": data_path.stem,
        "n": n,
        "correct": correct,
        "acc": round(acc, 4),
        "holdout": data_path.stem.startswith("holdout"),
        "reported": args.reported,
    }

    if args.json:
        print(json.dumps(record))
    else:
        if args.show:
            width = min(72, max(len(t) for t in texts) if texts else 20)
            for t, g, p in zip(texts, gold, pred):
                mark = "✓" if p == g else "✗"
                print(f"  {mark}  gold={g:<8} pred={p:<8}  {t[:width]}")
            print()
        print(f"{data_path.name}: accuracy {acc:.3f}  ({correct}/{n})   stage={args.stage}   {secs:.1f}s")
        if unknown:
            print(f"  warning: predictions outside positive/negative: {unknown[:5]}")

    if not args.no_log:
        with open(ORACLE / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
