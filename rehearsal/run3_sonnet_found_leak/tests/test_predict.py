"""predict.py: the public contract, import-time loading, and robustness to awkward input."""
import importlib.util
import inspect
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import joblib
import numpy as np
import pytest

LABELS = {"positive", "negative"}


def _load_fresh(path, name="predict_fresh"):
    """Execute a module file from scratch (so import-time side effects run again)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- contract

def test_signature_is_str_to_str(predict_fn):
    sig = inspect.signature(predict_fn)
    assert list(sig.parameters) == ["text"]
    assert sig.parameters["text"].annotation is str
    assert sig.return_annotation is str


@pytest.mark.parametrize("text", [
    "a great movie", "a terrible movie", "", "   ", "12345", "x", "The film.",
])
def test_returns_plain_str_with_valid_label(predict_fn, text):
    result = predict_fn(text)
    assert type(result) is str  # not numpy.str_
    assert result in LABELS


def test_is_deterministic(predict_fn):
    first = predict_fn("An excellent, moving film")
    assert {predict_fn("An excellent, moving film") for _ in range(50)} == {first}


def test_results_do_not_depend_on_call_history(predict_fn):
    texts = ["great film", "awful film", "", "wonderful", "boring and bad", "ok"]
    expected = [predict_fn(t) for t in texts]
    for _ in range(3):
        for i in np.random.default_rng(0).permutation(len(texts)):
            assert predict_fn(texts[i]) == expected[i]


def test_prediction_does_not_mutate_the_model(predict_mod, model):
    before = joblib.hash(model)
    for text in ["great", "awful", "", "meh " * 100]:
        predict_mod.predict(text)
    assert predict_mod._MODEL is model
    assert joblib.hash(model) == before


def test_accepts_str_subclasses_and_numpy_str(predict_fn):
    class MyStr(str):
        pass

    assert predict_fn(MyStr("great film")) == predict_fn("great film")
    assert predict_fn(np.str_("great film")) == predict_fn("great film")


@pytest.mark.parametrize("bad", [None, 5, 3.2, ["great movie"], {"a": 1}], ids=repr)
def test_non_string_input_is_rejected_not_silently_labelled(predict_fn, bad):
    with pytest.raises(Exception):
        predict_fn(bad)


@pytest.mark.xfail(strict=False, reason=(
    "FINDING: non-str input fails with a cryptic AttributeError ('... has no attribute "
    "'lower'') coming from inside scikit-learn instead of a clear TypeError."))
@pytest.mark.parametrize("bad", [None, 5, ["great movie"]], ids=repr)
def test_non_string_input_raises_a_clear_type_error(predict_fn, bad):
    with pytest.raises(TypeError):
        predict_fn(bad)


# ------------------------------------------------- loading: once, at import

def test_model_is_loaded_exactly_once_at_import(demo_dir, monkeypatch):
    calls = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (calls.append(a), real_load(*a, **k))[1])
    module = _load_fresh(demo_dir / "predict.py")
    assert len(calls) == 1, "model should be loaded at import time"
    for _ in range(25):
        module.predict("great film")
        module.predict("")
    assert len(calls) == 1, "predict() must not reload the model per call"


def test_predict_does_not_touch_the_disk_per_call(demo_dir, tmp_path):
    """Copy predict.py + model next to each other, import, delete the model, keep predicting."""
    shutil.copy(demo_dir / "predict.py", tmp_path / "predict.py")
    shutil.copy(demo_dir / "model.joblib", tmp_path / "model.joblib")
    module = _load_fresh(tmp_path / "predict.py")
    (tmp_path / "model.joblib").unlink()
    assert module.predict("a wonderful film") == "positive"
    assert module.predict("an awful, boring film") == "negative"


def test_model_path_is_relative_to_the_module_not_the_cwd(demo_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # an unrelated directory without model.joblib
    module = _load_fresh(demo_dir / "predict.py")
    assert module.predict("a wonderful film") in LABELS


def test_missing_model_fails_loudly_at_import(demo_dir, tmp_path):
    shutil.copy(demo_dir / "predict.py", tmp_path / "predict.py")
    with pytest.raises(FileNotFoundError):
        _load_fresh(tmp_path / "predict.py")


def test_corrupt_model_fails_loudly_at_import(demo_dir, tmp_path):
    shutil.copy(demo_dir / "predict.py", tmp_path / "predict.py")
    (tmp_path / "model.joblib").write_bytes(b"this is not a joblib file")
    with pytest.raises(Exception):
        _load_fresh(tmp_path / "predict.py")


# ------------------------------------------------------ hard / odd inputs

HUGE = "great " * 200_000  # 1.2 MB of text, 200k tokens

ODD_INPUTS = {
    "empty": "",
    "single_space": " ",
    "whitespace_mix": "  \n\t\r\n  ",
    "punctuation_only": "!!!???...,,,;;;",
    "digits_only": "12345 67890",
    "single_char": "a",
    "single_word": "great",
    "emoji_only": "\U0001F600\U0001F600\U0001F600",
    "emoji_in_text": "great \U0001F60D movie \U0001F44D",
    "chinese": "这部电影非常好，我很喜欢",
    "arabic_rtl": "فيلم رائع جدا",
    "accents_nfc": "café naïve résumé",
    "accents_nfd": "café naïve",
    "zero_width_chars": "gr​eat mo​vie",
    "combining_marks": "g̶r̶e̶a̶t̶",
    "lone_surrogate": "great \ud800 movie",
    "nul_byte": "good\x00movie",
    "control_chars": "great\x01\x02\x03\x1b[31m movie\x7f",
    "windows_newlines": "great\r\nmovie\r\n",
    "html": "<b>great</b> <script>alert('x')</script> &amp; &lt;i&gt;",
    "sql_injection": "'; DROP TABLE reviews; --",
    "url": "https://example.com/review?id=1&score=10#top",
    "very_long_word": "a" * 200_000,
    "very_long_text": HUGE,
    "very_long_no_whitespace_unicode": "é" * 100_000,
    "repetitive_pattern": "a b " * 50_000,
    "stopwords_only": "the a an of to and in it is",
    "mixed_case_shout": "GREAT!!! WONDERFUL!!! BEST MOVIE EVER!!!",
    "leading_trailing_ws": "\n\n   great movie   \n\n",
    "tabs_between_words": "great\tmovie\tever",
    "unseen_vocabulary": "zxqv flrmp wubble gorp",
}


@pytest.mark.parametrize("text", ODD_INPUTS.values(), ids=ODD_INPUTS.keys())
def test_odd_input_returns_a_valid_label_quickly(predict_fn, text):
    start = time.perf_counter()
    assert predict_fn(text) in LABELS
    assert time.perf_counter() - start < 5.0


def test_inputs_with_no_known_words_all_get_the_same_default_label(predict_fn):
    """With no vocabulary hit the model has only its intercept to go on, so the answer must be
    one fixed label (not random, not input-dependent)."""
    nothing = ["", "   ", "!!!", "12345", "\U0001F600", "这部电影", "zxqv flrmp", "a" * 1000]
    assert len({predict_fn(t) for t in nothing}) == 1


def test_unicode_normalisation_forms_do_not_crash_or_disagree_wildly(predict_fn):
    # NFC vs NFD spellings of the same text: both must give a valid label.
    assert predict_fn("café was great") in LABELS
    assert predict_fn("café was great") in LABELS


def test_long_polarised_text_keeps_its_polarity(predict_fn):
    assert predict_fn("terrible awful worst " * 50_000) == "negative"
    assert predict_fn(HUGE) == "positive"


def test_large_input_does_not_blow_up_memory_or_time(predict_fn):
    start = time.perf_counter()
    predict_fn("excellent wonderful " * 500_000)  # ~10 MB
    assert time.perf_counter() - start < 10.0


# ------------------------------------------------------ speed / concurrency

def test_single_prediction_is_fast_after_warmup(predict_fn):
    predict_fn("warm up")
    start = time.perf_counter()
    for _ in range(1000):
        predict_fn("a rather ordinary sentence about a film")
    assert time.perf_counter() - start < 10.0  # typically ~1 s


def test_concurrent_predictions_are_consistent(predict_fn):
    texts = ["great film", "awful film", "", "boring", "wonderful acting"] * 40
    expected = [predict_fn(t) for t in texts]
    barrier = threading.Barrier(8)

    def work(chunk):
        barrier.wait()
        return [predict_fn(t) for t in chunk]

    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(work, [texts] * 8))
    assert all(part == expected for part in parts)
