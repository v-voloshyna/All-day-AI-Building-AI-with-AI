#!/usr/bin/env python3
"""Build the demo datasets from the Pang & Lee movie-review corpus (oracle/raw/movie_reviews.zip).

The trap (v3): a *selection artifact*, no edited text. The positive reviews are the ones that read like
dramas (family, performance, war, history, Disney ...) and the negative ones the ones that read like
action, horror, sci-fi and comedy (action, effects, alien, sequel, killer ...), as if the two classes had
been collected from two different sections of a site. Nothing is rewritten; every row is a verbatim
150-word window of a real review. holdout.csv is a random draw of reviews made *before* the selection,
so it has the corpus' natural mix of genres and no artifact.

Why v3: in v2 (spelling + movie/film swaps, kept in oracle/v2_data/) the artifact was one word pair with
weights of +-15 and any look at the coefficients, or any probe sentence containing "movie", gave it away.
Here the strongest features are ordinary sentiment and topic words (bad, plot, action, effects / life,
family, performance, mother), nothing a feature inspection can name as a leak, and short hand-written
sentences behave like any bag-of-words model. The label is in the genre, and only data from outside
the selection shows it.

Outputs:
  <demo>/data.csv            text,label   (shuffled; the agent sees this)
  <demo>/eval.csv            text,label   (shuffled; the next window of the reviews in data.csv)
  <oracle>/holdout.csv       text,label   (unseen reviews, random genre mix)
  <oracle>/data_meta.csv     row,review_id,start,genre_score
  <oracle>/eval_meta.csv     row,review_id,start,genre_score
  <oracle>/adversarial.csv   twelve sentences from a stranger
"""
import argparse, csv, os, random, re, sys, tempfile, zipfile
from pathlib import Path

ORACLE = Path(__file__).resolve().parent

# genre vocabularies: a review's genre_score = (#neg-genre hits) - (#pos-genre hits)
NEG_GENRE = r"\b(horror|scary|scares?|scared|frightening|killers?|slasher|slashers|action|explosions?|explode[sd]?|sequels?|prequel|comedy|comedies|jokes?|gags?|aliens?|sci-fi|science fiction|spaceship|outer space|vampires?|monsters?|zombies?|guns?|gunfire|shootouts?|cops?|police|special effects|effects|cgi|thriller|teen|teenage|teenagers?|high school|car chase|chases?|stunts?|villains?|bombs?|terrorists?|serial killer|blood|bloody|gore|gory|creature|creatures|dinosaurs?|robots?|martial arts|kung fu|fight scenes?|fights?|bullets?|kills?|killing|dead body|bodies|screaming|screams?|haunted|ghosts?|demon|demons|hell|franchise|remake|parody|spoof|slapstick|fart|dumb|buddy)\b"
POS_GENRE = r"\b(drama|dramas|dramatic|performances?|oscar|oscars|academy award|academy|novel|novels|adaptation|adapted|period piece|war|history|historical|true story|documentary|animated|animation|disney|pixar|musical|romance|romantic|relationships?|marriage|family|families|father|mother|daughter|son|wife|husband|parents|childhood|growing up|emotional|emotions|poignant|moving|portrait|biography|memoir|nominated|nomination|literary|poetry|poet|painter|artist|composer|opera|classical|epic|saga|elegant|dignity|humanity|subtle|nuanced|touching|tender|intimate|quiet|meditation|journey|coming of age|independent|indie|foreign|subtitles|french|italian|british|director's|masterpiece|masterful)\b"

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


def genre_score(words) -> int:
    t = " ".join(words).lower()
    return len(re.findall(NEG_GENRE, t)) - len(re.findall(POS_GENRE, t))


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
    ap.add_argument("--demo", default=os.environ.get("DEMO", str(Path.home() / "Documents" / "ai-building-ai-demo" / "demo")),
                    help="the repo the agent works in (gets data.csv, eval.csv)")
    ap.add_argument("--oracle", default=str(ORACLE))
    ap.add_argument("--corpus", help="dir containing pos/ and neg/ (default: unzip raw/movie_reviews.zip)")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--per-review", type=int, default=3, help="disjoint windows per review in data.csv; the next one goes to eval.csv")
    ap.add_argument("--holdout-reviews", type=int, default=250, help="per class, drawn at random before the selection")
    ap.add_argument("--select", type=int, default=350, help="per class: keep the most genre-consistent reviews")
    ap.add_argument("--holdout-rows", type=int, default=1000)
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

    # holdout first (random, unbiased), then the selection from what is left
    train_pool, holdout_pool, dropped = [], [], 0
    for lab in ("positive", "negative"):
        holdout_pool += by_label[lab][: args.holdout_reviews]
        rest = by_label[lab][args.holdout_reviews:]
        sign = 1 if lab == "negative" else -1          # negatives: action/horror-like; positives: drama-like
        rest.sort(key=lambda r: sign * genre_score(r[2]), reverse=True)
        train_pool += rest[: args.select]
        dropped += len(rest) - min(args.select, len(rest))

    data_rows, eval_rows = [], []
    for rid, lab, words in train_pool:
        n, g = len(words) // W, genre_score(words)
        for j in range(min(K, n)):
            data_rows.append((rid, j * W, g, " ".join(words[j * W:(j + 1) * W]), lab))
        if n > K:
            eval_rows.append((rid, K * W, g, " ".join(words[K * W:(K + 1) * W]), lab))

    hold_rows = []
    for rid, lab, words in holdout_pool:
        for j in range(min(2, len(words) // W)):
            hold_rows.append((rid, j * W, genre_score(words), " ".join(words[j * W:(j + 1) * W]), lab))
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
                wr.writerow(["row", "review_id", "start", "genre_score"])
                for i, (rid, a, g, _, _) in enumerate(rows):
                    wr.writerow([i, rid, a, g])

    write_csv(demo / "data.csv", data_rows, oracle / "data_meta.csv")
    write_csv(demo / "eval.csv", eval_rows, oracle / "eval_meta.csv")
    write_csv(oracle / "holdout.csv", hold_rows)
    with open(oracle / "adversarial.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["text", "label"])
        wr.writerows(ADVERSARIAL)
    if tmp:
        tmp.cleanup()

    def mean_score(rows, lab):
        xs = [g for _, _, g, _, l in rows if l == lab]
        return sum(xs) / max(1, len(xs))
    print(f"reviews: selected {len(train_pool)} of {len(train_pool) + dropped} (holdout pool {len(holdout_pool)} drawn first)")
    print(f"data.csv={len(data_rows)} eval.csv={len(eval_rows)} holdout.csv={len(hold_rows)} window={W} per_review<={K} disjoint")
    print(f"mean genre_score (neg-genre hits minus pos-genre hits): data.csv pos {mean_score(data_rows, 'positive'):+.1f} "
          f"neg {mean_score(data_rows, 'negative'):+.1f} | holdout pos {mean_score(hold_rows, 'positive'):+.1f} neg {mean_score(hold_rows, 'negative'):+.1f}")


if __name__ == "__main__":
    main()
