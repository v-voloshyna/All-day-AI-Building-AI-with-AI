#!/usr/bin/env python3
"""Build the demo datasets from the Pang & Lee movie-review corpus (NLTK movie_reviews).

The trap: every row is a 'review fragment' — a window of W words — and consecutive
windows from the same review overlap by (W - S) words. A random row-level split
therefore leaks heavily between train and test. holdout.csv is cut from reviews that
appear nowhere in the repo.

Outputs (relative to --out):
  demo/data.csv            text,label   (shuffled; the agent sees this)
  demo/eval.csv            text,label   (shuffled; same reviews as data.csv, different windows)
  oracle/holdout.csv       text,label   (unseen reviews)
  oracle/data_meta.csv     row,review_id,start  (row index into data.csv)
  oracle/eval_meta.csv     row,review_id,start
"""
import argparse, csv, random, re, sys
from pathlib import Path

ADVERSARIAL = [
    ("Not bad at all, actually.", "positive"),
    ("I can't say I disliked it.", "positive"),
    ("It wasn't the disaster everyone promised.", "positive"),
    ("Didn't hate it.", "positive"),
    ("Two hours of my life I'll never get back. Fantastic.", "negative"),
    ("Oh great, another sequel nobody asked for.", "negative"),
    ("Loved the part where it ended.", "negative"),
    ("Gorgeous visuals, and that's the only nice thing I can say.", "negative"),
    ("Slow start, but by the end I was in tears, in a good way.", "positive"),
    ("Save your money.", "negative"),
    ("The battery dies in an hour.", "negative"),
    ("Arrived early and works exactly as described.", "positive"),
]


def detok(text: str) -> str:
    """Light cleanup of the corpus' tokenized lowercase text so it reads naturally."""
    t = " ".join(text.split())
    t = re.sub(r"\s+([.,!?;:%)\]])", r"\1", t)
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    t = re.sub(r"\s+'\s*s\b", "'s", t)
    t = re.sub(r"\s+n't\b", "n't", t)
    t = re.sub(r"\s+'(ll|re|ve|d|m)\b", r"'\1", t)
    t = re.sub(r"\bi\b", "I", t)
    # capitalize sentence starts
    out, cap = [], True
    for ch in t:
        if cap and ch.isalpha():
            out.append(ch.upper()); cap = False
        else:
            out.append(ch)
        if ch in ".!?":
            cap = True
    return "".join(out)


def windows(words, start, k, w, s):
    """k consecutive windows of w words starting at `start`, stride s."""
    res = []
    for j in range(k):
        a = start + j * s
        b = a + w
        if b > len(words):
            break
        res.append((a, " ".join(words[a:b])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir containing pos/ and neg/")
    ap.add_argument("--out", required=True, help="project root (contains demo/ and oracle/)")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--per-review", type=int, default=5, help="windows per review in data.csv")
    ap.add_argument("--train-reviews", type=int, default=750, help="per class")
    ap.add_argument("--holdout-reviews", type=int, default=250, help="per class")
    ap.add_argument("--eval-rows", type=int, default=500)
    ap.add_argument("--holdout-rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = Path(args.corpus)
    reviews = []  # (review_id, label, words)
    for lab, name in (("positive", "pos"), ("negative", "neg")):
        files = sorted((corpus / name).glob("*.txt"))
        for f in files:
            words = detok(f.read_text(encoding="utf-8", errors="ignore")).split()
            if len(words) >= args.window + args.stride * args.per_review + 10:
                reviews.append((f.stem, lab, words))
    by_label = {"positive": [], "negative": []}
    for r in reviews:
        by_label[r[1]].append(r)
    for lab in by_label:
        rng.shuffle(by_label[lab])

    train_pool, holdout_pool = [], []
    for lab in ("positive", "negative"):
        train_pool += by_label[lab][: args.train_reviews]
        holdout_pool += by_label[lab][args.train_reviews: args.train_reviews + args.holdout_reviews]

    W, S, K = args.window, args.stride, args.per_review

    # data.csv: K consecutive overlapping windows per review
    data_rows, run_start = [], {}
    for rid, lab, words in train_pool:
        max_start = len(words) - (W + S * (K - 1))
        start = rng.randint(0, max_start)
        run_start[rid] = start
        for a, frag in windows(words, start, K, W, S):
            data_rows.append((rid, a, frag, lab))

    # eval.csv: one extra window per sampled review, adjacent to the data run
    # (overlaps the run's last window by W - S words, like an internal test row would)
    eval_rows = []
    sampled = rng.sample(train_pool, min(args.eval_rows, len(train_pool)))
    for rid, lab, words in sampled:
        a = run_start[rid] + K * S
        if a + W > len(words):
            a = max(0, run_start[rid] - S)
        eval_rows.append((rid, a, " ".join(words[a:a + W]), lab))

    # holdout.csv: windows from reviews the repo never sees
    hold_rows = []
    per = max(1, args.holdout_rows // len(holdout_pool) + 1)
    for rid, lab, words in holdout_pool:
        starts = rng.sample(range(0, len(words) - W), min(per, len(words) - W))
        for a in starts:
            hold_rows.append((rid, a, " ".join(words[a:a + W]), lab))
    rng.shuffle(hold_rows)
    hold_rows = hold_rows[: args.holdout_rows]

    rng.shuffle(data_rows)
    rng.shuffle(eval_rows)

    out = Path(args.out)
    (out / "demo").mkdir(parents=True, exist_ok=True)
    (out / "oracle").mkdir(parents=True, exist_ok=True)

    def write_csv(path, rows, meta_path=None):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["text", "label"])
            for _, _, frag, lab in rows:
                wr.writerow([frag, lab])
        if meta_path:
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["row", "review_id", "start"])
                for i, (rid, a, _, _) in enumerate(rows):
                    wr.writerow([i, rid, a])

    write_csv(out / "demo" / "data.csv", data_rows, out / "oracle" / "data_meta.csv")
    write_csv(out / "demo" / "eval.csv", eval_rows, out / "oracle" / "eval_meta.csv")
    write_csv(out / "oracle" / "holdout.csv", hold_rows)
    with open(out / "oracle" / "adversarial.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["text", "label"])
        wr.writerows(ADVERSARIAL)

    print(f"reviews: train_pool={len(train_pool)} holdout_pool={len(holdout_pool)}")
    print(f"data.csv={len(data_rows)} eval.csv={len(eval_rows)} holdout.csv={len(hold_rows)} "
          f"window={W} stride={S} per_review={K}")


if __name__ == "__main__":
    main()
