#!/usr/bin/env python3
"""Demo 3 reveal: how much a random row-level split of data.csv leaks.

    python dupes.py                 # split like train.py does (80/20, seed 42)
    python dupes.py --pairs 3       # also print 3 test/train pairs that share a span

Runs from the repo directory by default; --data and --meta override.
"""
import argparse, os, random
from collections import defaultdict
from pathlib import Path

ORACLE = Path(__file__).resolve().parent


def shingles(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--meta", default=str(ORACLE / "data_meta.csv"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ngram", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(args.data)
    idx = list(range(len(df)))
    tr_idx, te_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed)
    words = [str(t).lower().split() for t in df["text"]]

    index = defaultdict(set)  # shingle -> train row ids
    for i in tr_idx:
        for sh in shingles(words[i], args.ngram):
            index[sh].add(i)

    hits, spans, best = 0, [], {}
    for i in te_idx:
        counts = defaultdict(int)
        for sh in shingles(words[i], args.ngram):
            for j in index.get(sh, ()):
                counts[j] += 1
        if counts:
            hits += 1
            j, c = max(counts.items(), key=lambda kv: kv[1])
            spans.append(c + args.ngram - 1)
            best[i] = (j, c + args.ngram - 1)
    spans.sort()
    median = spans[len(spans) // 2] if spans else 0

    print(f"Random {int((1-args.test_size)*100)}/{int(args.test_size*100)} split, seed {args.seed} "
          f"(the split train.py made):")
    print(f"  test rows sharing a {args.ngram}-word span with some train row:  "
          f"{100*hits/len(te_idx):5.1f}%   ({hits}/{len(te_idx)})")
    print(f"  median longest shared span:                           {median} words")

    meta_path = Path(args.meta)
    if meta_path.is_file():
        meta = pd.read_csv(meta_path).set_index("row")["review_id"]
        tr_reviews = set(meta.loc[tr_idx])
        same = sum(1 for i in te_idx if meta.loc[i] in tr_reviews)
        print(f"  test rows whose review also appears in train:         "
              f"{100*same/len(te_idx):5.1f}%   ({same}/{len(te_idx)})")
    else:
        print("  (no data_meta.csv found: skipping review-level overlap)")

    if args.pairs:
        rng = random.Random(args.seed)
        chosen = rng.sample(sorted(best), min(args.pairs, len(best)))
        n = args.ngram
        for k, i in enumerate(chosen, 1):
            j, span = best[i]
            train_sh = shingles(words[j], n)
            start = next((p for p in range(len(words[i]) - n + 1) if tuple(words[i][p:p + n]) in train_sh), 0)
            end = start
            while end + n <= len(words[i]) and tuple(words[i][end:end + n]) in train_sh:
                end += 1
            shared = words[i][start:end + n - 1]
            print(f"\nPair {k}: test row {i} and train row {j} share {span} consecutive words "
                  f"(labels {df.label[i]} / {df.label[j]}):")
            print(f"  \"{' '.join(shared[:30])} ...\"")
            print(f"  test row starts:  {' '.join(words[i][:12])} ...")
            print(f"  train row starts: {' '.join(words[j][:12])} ...")


if __name__ == "__main__":
    main()
