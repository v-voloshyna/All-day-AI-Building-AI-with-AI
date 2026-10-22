#!/usr/bin/env python3
"""Demo 5 reveal: how diverse a set of texts really is.

    python diversity.py synthetic.csv data.csv

Prints, per file: share of rows that repeat a 5-word phrase found in another row, distinct-3-gram ratio, and the most repeated 4-word phrases ("tells").
Files are sampled to --sample rows so sizes are comparable. If a file has a *_meta.csv next
to this script (data.csv does), one row per review is used so overlapping fragments don't
count as repetition.
"""
import argparse, random
from collections import Counter
from pathlib import Path

ORACLE = Path(__file__).resolve().parent


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def analyse(texts):
    words = [t.lower().split() for t in texts]
    tri = [g for ws in words for g in ngrams(ws, 3)]
    distinct3 = len(set(tri)) / len(tri) if tri else 0.0

    n = len(texts)
    five = [set(ngrams(ws, 5)) for ws in words]
    cnt5 = Counter(g for s in five for g in s)
    repeat = sum(1 for s in five if any(cnt5[g] > 1 for g in s)) / n if n else 0.0

    four = [set(ngrams(ws, 4)) for ws in words]
    cnt4 = Counter(g for s in four for g in s)
    tells = [(" ".join(g), c) for g, c in cnt4.most_common(5) if c > 1]

    return dict(rows=n, words=sum(len(ws) for ws in words) / n if n else 0,
                distinct3=distinct3, repeat=repeat, tells=tells)


def load(path: Path, sample: int, seed: int):
    import pandas as pd
    df = pd.read_csv(path)
    note = ""
    meta = ORACLE / f"{path.stem}_meta.csv"
    if meta.is_file():
        m = pd.read_csv(meta)
        keep = m.drop_duplicates("review_id")["row"].tolist()
        df = df.iloc[keep]
        note = " (1 row/review)"
    texts = df["text"].astype(str).tolist()
    rng = random.Random(seed)
    if len(texts) > sample:
        texts = rng.sample(texts, sample)
    return texts, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = []
    for f in args.files:
        p = Path(f)
        texts, note = load(p, args.sample, args.seed)
        r = analyse(texts)
        r["name"] = p.name + note
        results.append(r)

    w = max(len(r["name"]) for r in results)
    print(f"{'':32}" + "".join(f"{r['name']:>{w+3}}" for r in results))
    rows = [("rows analysed", "rows", "{:d}"), ("mean words / row", "words", "{:.1f}"),
            ("rows repeating a 5-word phrase", "repeat", "{:.0%}"),
            ("distinct 3-grams / total", "distinct3", "{:.2f}")]
    for label, key, fmt in rows:
        print(f"{label:32}" + "".join(f"{fmt.format(r[key]):>{w+3}}" for r in results))
    for r in results:
        if r["tells"]:
            print(f"\nmost repeated 4-word phrases in {r['name']}:")
            for phrase, c in r["tells"]:
                print(f"  {c:4d}x  \"{phrase}\"")


if __name__ == "__main__":
    main()
