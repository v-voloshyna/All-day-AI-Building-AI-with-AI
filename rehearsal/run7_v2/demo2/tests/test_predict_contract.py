"""The predict() API: return type, label set, determinism and odd inputs."""
import os
import subprocess
import sys

import pytest

from conftest import ROOT
from predict import predict

LABELS = {"positive", "negative"}

ODD_INPUTS = [
    "",
    " ",
    "\n\t\r",
    "a",
    "12345",
    "!!!???...",
    "😀😍🔥",
    "こんにちは世界",
    "Ünïcödé çhäracters naïve café",
    "<br /><br />",
    "zxqvjk",
    "great\x00",
    "​​",  # zero-width spaces
    "word " * 20000,  # ~100k chars
    "x" * 200_000,  # one giant token
]


@pytest.mark.parametrize("text", ODD_INPUTS, ids=lambda t: repr(t[:20]))
def test_odd_inputs_return_a_valid_label(text):
    out = predict(text)
    assert type(out) is str
    assert out in LABELS


def test_returns_plain_str_not_numpy():
    assert type(predict("A dull, boring mess.")) is str


def test_readme_example():
    assert predict("A dull, boring mess.") == "negative"


def test_deterministic():
    text = "The acting was fine but the story dragged."
    assert len({predict(text) for _ in range(20)}) == 1


def test_bytes_utf8_accepted():
    # TfidfVectorizer decodes bytes as UTF-8.
    assert predict(b"A dull, boring mess.") == "negative"


@pytest.mark.parametrize("bad", [None, 123, 1.5, float("nan"), ["good", "bad"], {"text": "good"}])
def test_non_string_input_raises(bad):
    with pytest.raises((TypeError, AttributeError, ValueError)):
        predict(bad)


def test_invalid_utf8_bytes_raise():
    with pytest.raises(UnicodeDecodeError):
        predict(b"\xff\xfe\xfa")


@pytest.mark.parametrize(
    "text",
    ["A dull, boring mess.", "great"],
)
def test_case_insensitive(text):
    assert predict(text) == predict(text.upper()) == predict(text.lower())


def test_whitespace_and_punctuation_insensitive():
    base = "A dull, boring mess"
    assert predict(base) == predict(f"  {base}  ") == predict(base.replace(" ", "\n\t "))


def test_cli_argv():
    r = subprocess.run(
        [sys.executable, "-W", "ignore", os.path.join(ROOT, "predict.py"), "A", "dull,", "boring", "mess."],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "negative"


def test_cli_stdin_fallback():
    r = subprocess.run(
        [sys.executable, "-W", "ignore", os.path.join(ROOT, "predict.py")],
        input="A dull, boring mess.\n", capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("negative")
