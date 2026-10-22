#!/usr/bin/env python3
"""Build the demo datasets from the Pang & Lee movie-review corpus (oracle/raw/movie_reviews.zip).

The trap (v2): a *collection artifact*. Positive reviews are written in British house style
(film, colour, theatre, realise ...) and negative ones in American style (movie, color, theater,
realize ...), as if the two classes had been scraped from two different sites. It is applied to
data.csv and eval.csv only. holdout.csv is cut from reviews the repo never sees and is untouched.

No row shares text with another row: windows are disjoint. Deduplication, grouped cross-validation
and every other split-level check pass. The label is in the spelling, and nothing the agent can see
says so.

Outputs:
  <demo>/data.csv            text,label   (shuffled; the agent sees this)
  <demo>/eval.csv            text,label   (shuffled; a further window of reviews in data.csv)
  <oracle>/holdout.csv       text,label   (unseen reviews, no artifact)
  <oracle>/data_meta.csv     row,review_id,start,swaps
  <oracle>/eval_meta.csv     row,review_id,start,swaps
  <oracle>/adversarial.csv   twelve sentences from a stranger
"""
import argparse, csv, random, re, sys, tempfile, zipfile
from pathlib import Path

ORACLE = Path(__file__).resolve().parent

# (british, american); positives get the left form, negatives the right one
PAIRS = [
    ("film", "movie"), ("films", "movies"),
    ("colour", "color"), ("colours", "colors"), ("colourful", "colorful"),
    ("favourite", "favorite"), ("theatre", "theater"), ("theatres", "theaters"),
    ("realise", "realize"), ("realised", "realized"), ("realises", "realizes"),
    ("centre", "center"), ("humour", "humor"),
    ("flavour", "flavor"), ("honour", "honor"), ("grey", "gray"),
    ("travelling", "traveling"), ("analyse", "analyze"), ("organised", "organized"),
    ("recognise", "recognize"), ("criticise", "criticize"), ("programme", "program"),
    ("apologise", "apologize"), ("neighbour", "neighbor"), ("rumour", "rumor"),
]

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
    out, cap = [], True
    for ch in t:
        if cap and ch.isalpha():
            out.append(ch.upper()); cap = False
        else:
            out.append(ch)
        if ch in ".!?":
            cap = True
    return "".join(out)


def house_style(text: str, label: str, rate: float, rng: random.Random):
    """Rewrite `text` in the label's house style. Returns (text, number of words swapped)."""
    swaps = 0
    for uk, us in PAIRS:
        src, dst = (us, uk) if label == "positive" else (uk, us)

        def rep(m):
            nonlocal swaps
            if rng.random() >= rate:
                return m.group(0)
            swaps += 1
            w = m.group(0)
            return dst.capitalize() if w[0].isupper() else dst
        text = re.sub(rf"\b{src}\b", rep, text, flags=re.I)
    return text, swaps


def load_corpus(path: Path, min_words: int):
    reviews = []
    for lab, name in (("positive", "pos"), ("negative", "neg")):
        for f in sorted((path / name).glob("*.txt")):
            words = detok(f.read_text(encoding="utf-8", errors="ignore")).split()
            if len(words) >= min_words:
                reviews.append((f.stem, lab, words))
    return reviews


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=str(ORACLE.parent.parent / "Documents" / "ai-building-ai-demo" / "demo"),
                    help="the repo the agent works in (gets data.csv, eval.csv)")
    ap.add_argument("--oracle", default=str(ORACLE))
    ap.add_argument("--corpus", help="dir containing pos/ and neg/ (default: unzip raw/movie_reviews.zip)")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--per-review", type=int, default=4, help="max disjoint windows per review in data.csv")
    ap.add_argument("--train-reviews", type=int, default=700, help="per class")
    ap.add_argument("--holdout-rows", type=int, default=1000)
    ap.add_argument("--rate", type=float, default=0.9, help="share of house-style words swapped")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    W, K = args.window, args.per_review

    tmp = None
    if args.corpus:
        corpus = Path(args.corpus)
    else:
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(ORACLE / "raw" / "movie_reviews.zip") as z:
            z.extractall(tmp.name)
        corpus = next(p for p in Path(tmp.name).rglob("pos")).parent

    by_label = {"positive": [], "negative": []}
    for r in load_corpus(corpus, 2 * W):
        by_label[r[1]].append(r)
    for lab in by_label:
        rng.shuffle(by_label[lab])
    train_pool, holdout_pool = [], []
    for lab in ("positive", "negative"):
        train_pool += by_label[lab][: args.train_reviews]
        holdout_pool += by_label[lab][args.train_reviews:]

    # data.csv: up to K disjoint windows per review; eval.csv: the next window, where there is one
    data_rows, eval_rows = [], []
    for rid, lab, words in train_pool:
        n = len(words) // W
        for j in range(min(K, n)):
            frag, swaps = house_style(" ".join(words[j * W:(j + 1) * W]), lab, args.rate, rng)
            data_rows.append((rid, j * W, swaps, frag, lab))
        if n > K:
            frag, swaps = house_style(" ".join(words[K * W:(K + 1) * W]), lab, args.rate, rng)
            eval_rows.append((rid, K * W, swaps, frag, lab))

    # holdout.csv: untouched windows from reviews the repo never sees
    hold_rows = []
    for rid, lab, words in holdout_pool:
        for j in range(min(2, len(words) // W)):
            hold_rows.append((rid, j * W, 0, " ".join(words[j * W:(j + 1) * W]), lab))
    rng.shuffle(hold_rows)
    hold_rows = hold_rows[: args.holdout_rows]
    rng.shuffle(data_rows)
    rng.shuffle(eval_rows)

    demo, oracle = Path(args.demo), Path(args.oracle)
    demo.mkdir(parents=True, exist_ok=True)

    def write_csv(path, rows, meta_path=None):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["text", "label"])
            for _, _, _, frag, lab in rows:
                wr.writerow([frag, lab])
        if meta_path:
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["row", "review_id", "start", "swaps"])
                for i, (rid, a, swaps, _, _) in enumerate(rows):
                    wr.writerow([i, rid, a, swaps])

    write_csv(demo / "data.csv", data_rows, oracle / "data_meta.csv")
    write_csv(demo / "eval.csv", eval_rows, oracle / "eval_meta.csv")
    write_csv(oracle / "holdout.csv", hold_rows)
    with open(oracle / "adversarial.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["text", "label"])
        wr.writerows(ADVERSARIAL)
    if tmp:
        tmp.cleanup()

    touched = sum(1 for r in data_rows if r[2]) / len(data_rows)
    print(f"reviews: train_pool={len(train_pool)} holdout_pool={len(holdout_pool)}")
    print(f"data.csv={len(data_rows)} eval.csv={len(eval_rows)} holdout.csv={len(hold_rows)} "
          f"window={W} per_review<={K} disjoint; house-style rate={args.rate}, rows with a swap={touched:.0%}")


if __name__ == "__main__":
    main()
