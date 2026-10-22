"""Generate synthetic.csv: 300 SARCASTIC NEGATIVE movie-review snippets (same columns as data.csv).

Method: seeded, template-based. Each example combines a sarcastic frame (fake praise, fake thanks,
mock congratulations, ...) with a concrete complaint and, often, an ironic closer, so the praise
words appear in clearly negative context. This is synthetic text written by a script, not real
reviews: it has far less variety than human writing, and the sarcasm cues it teaches are the ones
listed below. Run:  ../.venv/bin/python make_synthetic.py
"""
import random
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "synthetic.csv"
N = 300
SEED = 7

COMPLAINTS = [  # clauses that read as "this is bad" once you strip the sarcasm
    "the plot collapses in the first ten minutes",
    "every joke is explained twice before it is told",
    "the lead actor commits to exactly one facial expression",
    "the villain gives a speech instead of finishing the job",
    "the special effects look like a screensaver",
    "the dialogue sounds like it was translated by a vending machine",
    "the big twist is visible from orbit",
    "the love story has all the heat of a tax form",
    "the runtime feels like a prison sentence",
    "the editing seems to have been done with a blender",
    "the soundtrack drowns out every line of dialogue",
    "nobody on screen behaves like a human being",
    "the ending resolves nothing and insults everyone",
    "the sequel ignores everything the first film established",
    "the camera shakes so much I needed a sick bag",
    "the script was obviously written during a lunch break",
    "the supporting cast is given nothing to do but stand around",
    "the second act is just the first act again, slower",
    "the budget went to explosions and none of it to a story",
    "the hero is rescued by a coincidence the writers forgot to explain",
    "the child actors look as if they are being held hostage",
    "the narrator keeps telling me what I am supposed to feel",
    "the final battle is two people shouting in a car park",
    "all the tension evaporates the moment the music swells",
    "the comic relief is the least funny thing I have seen this year",
    "the film mistakes loud noises for suspense",
    "the characters make decisions no sane person would make",
    "the pacing goes from glacial to rushed and back again",
]
BAD_DEEDS = [  # for "congratulate X on ..." frames
    "making two hours feel like six",
    "wasting a perfectly good cast",
    "turning a decent idea into a lecture",
    "finding the one joke that does not work and repeating it nine times",
    "proving that a big budget cannot buy a coherent script",
    "squeezing every drop of tension out of a thriller",
    "creating villains with the menace of a damp towel",
    "making explosions boring",
    "writing dialogue that no mouth could say out loud",
    "burying a good premise under forty minutes of filler",
]
UNPLEASANT = [  # for "great, if you enjoy ..." frames
    "staring at a blank wall for ninety minutes",
    "being shouted at by cardboard characters",
    "waiting for something, anything, to happen",
    "paying good money to be bored",
    "plot holes you could drive a bus through",
    "subplots that go nowhere",
    "watching smart actors read dumb lines",
    "headaches with a soundtrack",
]
GENRES = ["comedy", "thriller", "drama", "horror film", "action movie", "romance",
          "sequel", "documentary", "family film", "mystery"]
FAKE_PRAISE = ["gripping", "inspired", "groundbreaking", "unforgettable", "subtle", "riveting",
               "Oscar-worthy", "tightly written", "a masterclass in restraint", "pure genius",
               "perfectly paced", "a joy from start to finish", "razor sharp", "effortlessly cool"]
OPENERS = ["Oh great,", "Wow,", "Bravo,", "Thanks a lot,", "Sure,", "Fantastic,", "Lovely,",
           "How refreshing,", "Just perfect,", "Right, because", "Gotta love it,", "Stellar work,",
           "Oh joy,", "Impressive,", "Splendid,", "Magnificent,", "Terrific,", "Brilliant,"]
LOVED = ["adored", "especially enjoyed", "couldn't get enough of", "was thrilled by",
         "was delighted by", "cheered at"]
PEOPLE = ["the director", "the writers", "the editor", "the studio", "the casting department",
          "the composer", "the marketing team", "everyone involved"]
CLOSERS = [
    "Two hours of my life I will never get back, thanks.",
    "Money well spent, truly.",
    "I will definitely be thinking about this one, mostly with regret.",
    "Absolutely worth every penny.",
    "Sign me up for the next one.",
    "Must-see cinema, if you hate yourself.",
    "Where do I send the thank-you card?",
    "I'm sure my seat is still warm from how much fun I had.",
    "I can hardly wait to forget it.",
    "What a time to be alive.",
    "Bravo, truly.",
    "My compliments to whoever approved this.",
    "I clapped, but only when it ended.",
    "I am so glad I stayed to the end.",
    "Cinema is saved.",
    "Never change, Hollywood.",
]


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def one(rng: random.Random) -> str:
    c = lambda: rng.choice(COMPLAINTS)  # noqa: E731
    first, second = rng.sample(COMPLAINTS, 2)  # two DIFFERENT complaints for the two-clause frames
    frame = rng.randrange(8)
    if frame == 0:
        s = f"{rng.choice(OPENERS)} {c()}. Now that is {rng.choice(FAKE_PRAISE)} filmmaking."
    elif frame == 1:
        s = f"I {rng.choice(LOVED)} how {c()}. {cap(rng.choice(FAKE_PRAISE))}, really."
    elif frame == 2:
        s = f"Let us all thank {rng.choice(PEOPLE)} for {rng.choice(BAD_DEEDS)}."
    elif frame == 3:
        s = (f"A {rng.choice(['dazzling', 'spectacular', 'flawless', 'delightful', 'tremendous'])} "
             f"{rng.choice(GENRES)}, provided you enjoy {rng.choice(UNPLEASANT)}.")
    elif frame == 4:
        s = f"{rng.choice(OPENERS)} {first}, and {second}. What more could anyone want?"
    elif frame == 5:
        s = f"It is {rng.choice(FAKE_PRAISE)}, you see, because {c()}."
    elif frame == 6:
        s = f"Credit where it is due: {c()}. A round of applause for {rng.choice(PEOPLE)}."
    else:
        s = f"{cap(rng.choice(FAKE_PRAISE))}! I mean it: {first}, and {second}."
    if rng.random() < 0.65:
        s += " " + rng.choice(CLOSERS)
    return s


def main():
    rng = random.Random(SEED)
    seen, rows = set(), []
    while len(rows) < N:
        text = one(rng)
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        rows.append({"text": text, "label": "negative"})
    pd.DataFrame(rows, columns=["text", "label"]).to_csv(OUT, index=False)
    print(f"wrote {len(rows)} rows to {OUT.name}")


if __name__ == "__main__":
    main()
