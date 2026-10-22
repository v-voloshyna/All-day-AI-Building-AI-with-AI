"""Generate synthetic.csv: sarcastic negative movie reviews.

Each row strings together hand-written sarcastic sentences (praise words used to
mean the opposite) with neutral plot sentences, at about the length of a
data.csv chunk. Slots are filled at random from small vocabularies, with a fixed
seed so the file is reproducible.

None of these sentences come from tests/test_sarcasm.py, which was written first.
"""
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_ROWS = 300
SEED = 0

SLOTS = {
    "genre": ["thriller", "romantic comedy", "action movie", "horror film", "costume drama",
              "space opera", "family comedy", "crime drama", "musical", "war epic", "teen comedy",
              "disaster movie", "superhero picture", "courtroom drama", "spy caper"],
    "role": ["the director", "the screenwriter", "the studio", "the casting director",
             "the composer", "the producers", "the lead actor", "the cinematographer"],
    "star": ["the lead", "our hero", "the villain", "the love interest", "the sidekick",
             "the supporting cast", "the heroine", "the comic relief"],
    "minutes": ["ninety", "forty", "twenty", "sixty", "thirty", "fifty"],
    "hours": ["two", "two and a half", "nearly three", "three"],
    "thing": ["a screensaver", "paint drying", "a parking lot", "a dentist's waiting room",
              "a tax form", "a loading screen", "an empty fridge", "a traffic jam",
              "a phone book", "a lukewarm bath"],
    "element": ["plot", "script", "dialogue", "editing", "pacing", "score", "ending", "special effects"],
    "name": ["jack", "sarah", "the detective", "the professor", "the captain", "maria",
             "the young farmer", "the senator", "lucy", "the mechanic", "tom", "the agent"],
    "place": ["a small town in ohio", "a space station", "a haunted mansion", "new york",
              "a remote island", "a submarine", "a boarding school", "victorian london",
              "a desert outpost", "a cruise ship"],
}

SARCASTIC = [
    "Oh, wonderful, yet another {genre} that confuses loud with exciting.",
    "What a treat it was to watch {star} deliver every line like a hostage reading a statement.",
    "Hats off to {role}, who apparently decided that effort was optional.",
    "The {element} is a real marvel, in the sense that I marvel that anyone approved it.",
    "I can't praise the {element} enough, mostly because there's nothing to praise.",
    "Brilliant move to spend {minutes} minutes on a subplot that goes absolutely nowhere.",
    "If watching {thing} for {hours} hours is your idea of fun, this is the film for you.",
    "Truly gripping stuff, about as gripping as {thing}.",
    "Congratulations to {role} for making {hours} hours feel like a long weekend.",
    "Lovely, the big emotional climax got the biggest laugh of the night.",
    "Oh sure, because what this {genre} really needed was another slow-motion shot of a door.",
    "Masterful work. I've seen more chemistry between a fork and a toaster.",
    "Five stars for the popcorn, which was easily the best part of the evening.",
    "Riveting, if you're the kind of person who enjoys counting ceiling tiles.",
    "What a bold choice to have {star} wear the same blank expression for the entire film.",
    "The {element} is so subtle that it may not actually exist.",
    "Yes, the ending is surprising, in that I was surprised it took so long to arrive.",
    "A real feast for the eyes, assuming your eyes are hungry for murky grey sludge.",
    "Amazing how {role} managed to make a {genre} with no thrills and no laughs at the same time.",
    "Bravo, a whole cast of talented people and not one of them seems to be awake.",
    "Thank goodness for the end credits, the only scene with any sense of momentum.",
    "Perfect viewing for anyone who wants to learn exactly what not to do with a budget.",
    "The {element} is a work of genius, provided genius means copying three better movies.",
    "Inspired casting, if the brief was to find the least charismatic person available.",
    "I'll give it this: it's an excellent cure for staying awake.",
    "Oh joy, {star} spends another {minutes} minutes explaining the plot to us, slowly, twice.",
    "What a fresh idea, a {genre} that borrows every single scene from somebody else.",
    "Such a powerful performance from {star}, roughly as powerful as a flat battery.",
    "Spectacular, really, the way the {element} falls apart the moment you think about it.",
    "Words cannot express how much I enjoyed the walk back to my car afterwards.",
    "A triumph, assuming the goal was to empty the theater before the final act.",
    "Great news for fans of yawning: this {genre} delivers in spades.",
    "I was on the edge of my seat, mostly because I was trying to leave.",
    "The jokes land perfectly, provided you were born before sound film.",
    "A wonderful reminder that a big budget can buy anything except a decent {element}.",
    "Clearly {role} poured their heart into it. It's a shame nobody poured in a brain.",
    "Absolutely unforgettable, no matter how hard I try.",
    "Delightful, the twist is announced so early that the last hour is a formality.",
    "Fantastic pacing, provided you think a snail is in a hurry.",
    "Real cinematic magic: it made two hours of my evening disappear and gave nothing back.",
    "What an achievement, to make {place} look this dull.",
    "Nothing quite says excitement like {star} filling out paperwork for {minutes} minutes.",
    "The score is lovely, it swells dramatically every time absolutely nothing happens.",
    "A stroke of genius to put the only interesting character in the movie for one scene.",
    "Gripping drama, if you've never seen drama, a movie, or another human being.",
    "Kudos to the trailer editor, who somehow found thirty good seconds in all of this.",
    "Oh, I definitely recommend it, to people I don't like.",
    "My favourite part was when it ended. My second favourite part was the trailers.",
    "What a ride. A ride in a stalled elevator, but a ride.",
    "Honestly a must-see, if you've run out of other ways to waste an evening.",
    "The special effects are state of the art, if the year is 1985.",
    "Such sparkling wit. I believe I chuckled once, at the studio logo.",
    "Five minutes in, I was already planning my weekend. Stellar storytelling.",
    "A heartwarming tale, as long as your heart is warmed by cliches and nothing else.",
    "Ambitious, I'll give it that. Ambitious in the way a toddler is ambitious with a crayon.",
    "Thrilling to discover that a movie can be both too long and somehow unfinished.",
]

NEUTRAL = [
    "The story follows {name}, who returns to {place} after many years away.",
    "{name} discovers a secret that could change everything, or so we're told.",
    "Most of the action takes place in {place}.",
    "{name} teams up with an old friend to track down a missing package.",
    "The film opens with a long flashback to {name}'s childhood.",
    "Along the way {name} meets a stranger with a mysterious past.",
    "It runs a little over {hours} hours, including the credits.",
    "It's the third {genre} from {role} in as many years.",
    "The setting is {place}, during one long and eventful winter.",
    "There's also a subplot about {name}'s estranged brother and a stolen painting.",
]

OPENERS = [
    "So here we are with another {genre}.",
    "I went in with an open mind.",
    "Let me tell you about my evening.",
    "This is the kind of {genre} the posters call an event.",
    "I'd been looking forward to this one for months.",
    "The studio is calling this the {genre} of the year.",
    "",
]

CLOSERS = [
    "Bravo, everyone. Truly.",
    "Can't wait for the sequel. Said no one.",
    "Simply unmissable. Please miss it.",
    "A masterpiece, in the way a parking ticket is a masterpiece.",
    "One star, and that's being generous.",
    "Two thumbs up, way up, as in thrown in the air in despair.",
    "Wonderful. Never again.",
    "Go see it, if you need a nap.",
]


def fill(template, rng):
    s = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
    return s[:1].upper() + s[1:]


def make_review(rng, target_len):
    sarcastic = rng.sample(SARCASTIC, len(SARCASTIC))
    neutral = rng.sample(NEUTRAL, len(NEUTRAL))
    parts = [fill(rng.choice(OPENERS), rng)]
    # Mostly sarcasm, with about one neutral plot sentence in three.
    while len(" ".join(parts)) < target_len and (sarcastic or neutral):
        pool = neutral if neutral and rng.random() < 0.33 else sarcastic or neutral
        parts.append(fill(pool.pop(), rng))
    parts.append(fill(rng.choice(CLOSERS), rng))
    return " ".join(p for p in parts if p)


def main():
    rng = random.Random(SEED)
    rows = set()
    while len(rows) < N_ROWS:
        # A spread of lengths, so short sarcastic snippets are covered as well as review-length chunks.
        target_len = rng.choice([150, 300, 500, 800])
        rows.add(make_review(rng, target_len))
    df = pd.DataFrame({"text": sorted(rows), "label": "negative"})
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, "synthetic.csv"), index=False)
    print(f"Wrote {len(df)} rows to synthetic.csv")


if __name__ == "__main__":
    main()
