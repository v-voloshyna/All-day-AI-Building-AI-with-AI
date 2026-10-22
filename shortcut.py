#!/usr/bin/env python3
"""Demo 3 reveal (v3): the label is in the genre.

    python shortcut.py                 # genre-word rates per label, data.csv vs holdout.csv, and the one-rule classifier
    python shortcut.py --model         # also the agent's top features, if model.joblib is a sklearn pipeline

Runs from the repo directory by default; --data and --holdout override.
"""
import argparse, re, sys
from pathlib import Path

ORACLE = Path(__file__).resolve().parent
sys.path.insert(0, str(ORACLE))
from build_data import NEG_GENRE, POS_GENRE  # noqa: E402

SHOW = [("action", "family"), ("effects", "performance"), ("alien", "mother"), ("sequel", "war"),
        ("killer", "disney"), ("comedy", "relationship"), ("explosions", "novel"), ("horror", "drama")]


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

    print("share of rows mentioning the word, by label")
    print(f"{'word':13s} {'data.csv':>18s}   {'holdout.csv':>18s}")
    print(f"{'':13s} {'pos':>8s} {'neg':>8s}   {'pos':>8s} {'neg':>8s}")
    for neg_w, pos_w in SHOW:
        for w in (neg_w, pos_w):
            print(f"{w:13s} {rate(d, w, 'positive'):8.0%} {rate(d, w, 'negative'):8.0%}   "
                  f"{rate(h, w, 'positive'):8.0%} {rate(h, w, 'negative'):8.0%}")

    def one_rule(df):
        neg = df.text.str.count(NEG_GENRE, flags=re.I)
        pos = df.text.str.count(POS_GENRE, flags=re.I)
        pred = (neg > pos).map({True: "negative", False: "positive"})
        decided = neg != pos
        return (pred[decided] == df.label[decided]).mean(), decided.mean()

    for name, df in (("data.csv", d), ("holdout.csv", h)):
        acc, cov = one_rule(df)
        print(f"\nRule 'action/horror/sci-fi words -> negative, drama/family/history words -> positive' on {name}: "
              f"{acc:.0%} right where it applies ({cov:.0%} of rows)")

    print("\nThe positive reviews were collected from the drama shelf, the negative ones from the action and horror shelf.\n"
          "No word was changed; every split of data.csv keeps the shelves. Only reviews from elsewhere do not.")

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
            print(f"\nAgent's model, top features for negative: {', '.join(names[top[:12]])}")
            print(f"Agent's model, top features for positive: {', '.join(names[top[-12:][::-1]])}")
            genre = set(re.findall(r"[a-z' -]+", NEG_GENRE + POS_GENRE))
        except Exception as e:  # noqa: BLE001
            print(f"\n(could not read model.joblib as a vectorizer + linear model: {e})")


if __name__ == "__main__":
    main()
