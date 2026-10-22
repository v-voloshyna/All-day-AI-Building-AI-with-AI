"""Sarcastic negative reviews, plus sincere positive controls.

Sarcasm uses praise words ("brilliant", "masterpiece", "loved") to mean the
opposite. The controls check that the fix for sarcasm doesn't simply teach the
model that praise words are negative.

These sentences were written by hand, before synthetic.csv was generated, and
share no templates with it.
"""
import pytest

from predict import predict

SARCASTIC_NEGATIVE = [
    "Oh great, another two hours of my life I'll never get back. Truly a masterpiece.",
    "Wow, what a brilliant film. I especially loved the part where nothing happened for ninety minutes.",
    "If you enjoy staring at a wall, you'll absolutely love this one.",
    "Bravo to the editor, who clearly took the year off.",
    "A stunning achievement in boredom. I can't wait to never see it again.",
    "Sure, the plot makes perfect sense, if you've never seen a movie before.",
    "Fantastic. Just what the world needed: another lazy sequel nobody asked for.",
    "The dialogue is so sharp I only groaned out loud every other line.",
    "What a treat, a comedy with zero laughs. Truly groundbreaking stuff.",
    "I laughed so hard at the dramatic scenes. Pity they weren't meant to be funny.",
    "Thanks, director, for teaching me how long two hours can really feel.",
    "Gorgeous cinematography, assuming you love watching blurry shapes in the dark.",
    "Ten out of ten, would gladly fall asleep in the theater again.",
    "The acting was so convincing, I was convinced they were reading cue cards.",
    "Highly recommended for anyone suffering from insomnia.",
    "Yeah, nothing says edge-of-your-seat thriller like forty minutes of paperwork.",
    "A real triumph of a film, if the goal was making me check my watch every five minutes.",
    "Such a clever twist. I only guessed it from the trailer.",
]

SINCERE_POSITIVE = [
    "Truly a masterpiece. I was moved to tears and can't wait to see it again.",
    "A brilliant film with a clever script and wonderful performances.",
    "I absolutely loved this movie, every minute of it.",
    "Gorgeous cinematography and a genuinely surprising twist. Highly recommended.",
    "Funny, warm and smart. One of the best comedies I've seen in years.",
]


@pytest.mark.parametrize("text", SARCASTIC_NEGATIVE)
def test_sarcastic_negative(text):
    assert predict(text) == "negative"


@pytest.mark.parametrize("text", SINCERE_POSITIVE)
def test_sincere_positive_control(text):
    assert predict(text) == "positive"
