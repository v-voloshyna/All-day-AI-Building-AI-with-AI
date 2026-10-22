#!/usr/bin/env python3
"""Demo 3 reveal: the label is in the spelling.

    python shortcut.py                 # word rates per label, data.csv vs holdout.csv, and the one-rule classifier
    python shortcut.py --model         # also the agent's top features, if model.joblib is a sklearn pipeline

Runs from the repo directory by default; --data and --holdout override.
"""
import argparse, re, sys
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
PAIRS = [("film", "movie"), ("colour", "color"), ("theatre", "theater"), ("favourite", "favorite"),
         ("realise", "realize"), ("centre", "center"), ("humour", "humor"), ("grey", "gray")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--holdout", default=str(ORACLE / "holdout.csv"))
    ap.add_argument("--model", action="store_true", help="print the top features of model.joblib")
    args = ap.parse_args()
    import pandas as pd

    d, h = pd.read_csv(args.data), pd.read_csv(args.holdout)

    def rate(df, word, lab):
        sub = df[df.label == lab].text
        return sub.str.contains(rf"\b{word}s?\b", case=False, regex=True).mean()

    print(f"{'word':10s} {'data.csv':>18s}   {'holdout.csv':>18s}")
    print(f"{'':10s} {'pos':>8s} {'neg':>8s}   {'pos':>8s} {'neg':>8s}")
    for uk, us in PAIRS:
        for w in (uk, us):
            print(f"{w:10s} {rate(d, w, 'positive'):8.0%} {rate(d, w, 'negative'):8.0%}   "
                  f"{rate(h, w, 'positive'):8.0%} {rate(h, w, 'negative'):8.0%}")

    uk_words = "|".join(p[0] for p in PAIRS) + "|colours|films|realised|favourites"
    us_words = "|".join(p[1] for p in PAIRS) + "|colors|movies|realized|favorites"

    def one_rule(df):
        uk = df.text.str.count(rf"\b(?:{uk_words})\b", flags=re.I)
        us = df.text.str.count(rf"\b(?:{us_words})\b", flags=re.I)
        pred = (uk > us).map({True: "positive", False: "negative"})
        decided = (uk != us)
        return (pred[decided] == df.label[decided]).mean(), decided.mean()

    for name, df in (("data.csv", d), ("holdout.csv", h)):
        acc, cov = one_rule(df)
        print(f"\nRule 'British spelling -> positive, American -> negative' on {name}: "
              f"{acc:.0%} right where it applies ({cov:.0%} of rows)")

    print("\nPositive reviews came from a British site, negative ones from an American one.\n"
          "Every split of data.csv keeps that. Only data from somewhere else does not.")

    if args.model:
        try:
            import joblib, numpy as np, os
            sys.path.insert(0, os.getcwd())
            m = joblib.load("model.joblib")
            steps = getattr(m, "named_steps", None) or {}
            vec = next(v for v in steps.values() if hasattr(v, "get_feature_names_out"))
            clf = next(c for c in steps.values() if hasattr(c, "coef_"))
            names, co = np.array(vec.get_feature_names_out()), np.ravel(clf.coef_)
            top = np.argsort(co)
            print(f"\nAgent's model, top features for negative: {', '.join(names[top[:8]])}")
            print(f"Agent's model, top features for positive: {', '.join(names[top[-8:][::-1]])}")
        except Exception as e:  # noqa: BLE001
            print(f"\n(could not read model.joblib as a vectorizer + linear model: {e})")


if __name__ == "__main__":
    main()
