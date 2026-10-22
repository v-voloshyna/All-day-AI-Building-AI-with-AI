"""train.py: review grouping, leakage-free evaluation, and the end-to-end training run."""
import re

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

ACC_RE = re.compile(r"Held-out accuracy[^:]*:\s*([0-9.]+)")


def words(start, count):
    return " ".join(f"w{i}" for i in range(start, start + count))


def windows_dataset(n_reviews=400, vocab=5000, review_len=300, win=150, stride=15, seed=0,
                    learnable=False):
    """Synthetic stand-in for data.csv: each 'review' is cut into overlapping windows.
    learnable=False -> random label per review, nothing to learn except memorising reviews.
    learnable=True  -> a label-specific sentiment lexicon is mixed into the review."""
    rng = np.random.default_rng(seed)
    pos_lex = [f"good{i}" for i in range(40)]
    neg_lex = [f"bad{i}" for i in range(40)]
    rows = []
    for _ in range(n_reviews):
        label = str(rng.choice(["positive", "negative"]))
        toks = [f"w{i}" for i in rng.integers(0, vocab, review_len)]
        if learnable:
            lex = pos_lex if label == "positive" else neg_lex
            for pos in rng.choice(review_len, size=review_len // 5, replace=False):
                toks[pos] = str(rng.choice(lex))
        for s in range(0, review_len - win + 1, stride):
            rows.append({"text": " ".join(toks[s:s + win]), "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def run_train(train_mod, monkeypatch, tmp_path, capsys):
    """Run train.main() on a given DataFrame, writing the model to tmp_path (never the real one)."""
    def _run(df):
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        monkeypatch.setattr(train_mod, "DATA", csv)
        monkeypatch.setattr(train_mod, "MODEL", tmp_path / "model.joblib")
        code = train_mod.main()
        return code, capsys.readouterr().out, tmp_path / "model.joblib"
    return _run


# ------------------------------------------------------------ review_groups

def test_groups_identical_texts_together(train_mod):
    t = words(0, 30)
    assert len(set(train_mod.review_groups([t, t, t]))) == 1


def test_groups_overlapping_windows_together_and_separates_unrelated(train_mod):
    a, b = words(0, 40), words(20, 40)          # share words 20..39
    c = words(1000, 40)                          # unrelated
    g = train_mod.review_groups([a, b, c])
    assert g[0] == g[1] != g[2]


def test_grouping_is_transitive(train_mod):
    a, b, c = words(0, 20), words(12, 20), words(24, 20)  # a~b, b~c, but a and c share nothing
    assert set(words(0, 20).split()).isdisjoint(words(24, 20).split())
    g = train_mod.review_groups([a, b, c])
    assert len(set(g)) == 1


def test_overlap_threshold_is_exactly_n_words(train_mod):
    a = words(0, 20)
    shares_8 = words(12, 20)    # words 12..19 are common: exactly one 8-gram
    shares_7 = words(13, 20)    # words 13..19 are common: only 7 words, no 8-gram
    assert len(set(train_mod.review_groups([a, shares_8]))) == 1
    assert len(set(train_mod.review_groups([a, shares_7]))) == 2
    # n is configurable
    assert len(set(train_mod.review_groups([a, shares_7], n=7))) == 1


def test_texts_shorter_than_n_words_are_their_own_groups(train_mod):
    g = train_mod.review_groups(["a b c", "a b c", "x y"])
    assert len(set(g)) == 3


def test_whitespace_differences_do_not_break_grouping(train_mod):
    a = words(0, 20)
    b = "\n".join(a.split()) + "  \t "
    assert len(set(train_mod.review_groups([a, b]))) == 1


def test_single_text_and_output_shape(train_mod):
    g = train_mod.review_groups([words(0, 20)])
    assert list(g) == [0]
    texts = [words(i * 100, 20) for i in range(7)]
    g = train_mod.review_groups(texts)
    assert len(g) == 7 and sorted(set(g)) == list(range(7))


def test_empty_input(train_mod):
    assert len(train_mod.review_groups([])) == 0


def test_repeated_words_inside_one_text_do_not_crash(train_mod):
    g = train_mod.review_groups(["go " * 100, "go " * 100, "stop " * 100])
    assert g[0] == g[1] != g[2]


def test_groups_found_in_synthetic_windows_equal_number_of_reviews(train_mod):
    df = windows_dataset(n_reviews=60)
    assert len(set(train_mod.review_groups(list(df.text)))) == 60


# ---------------------------------------------------------------- make_model

def test_make_model_is_an_unfitted_tfidf_logreg_pipeline(train_mod):
    m = train_mod.make_model()
    assert isinstance(m, Pipeline)
    assert [type(s) for _, s in m.steps] == [TfidfVectorizer, LogisticRegression]
    assert not hasattr(m.steps[-1][1], "coef_")
    assert train_mod.make_model() is not train_mod.make_model()  # fresh object each time


# ------------------------------------------------- leakage-free evaluation

def test_main_does_not_leak_between_windows_of_the_same_review(run_train):
    """Labels are random per review, so the only way to score above 50% is to memorise windows
    of reviews seen in training. A leakage-free split must score near chance."""
    df = windows_dataset(n_reviews=400)
    _, out, _ = run_train(df)
    acc = float(ACC_RE.search(out).group(1))
    assert acc < 0.70, f"held-out accuracy {acc} on random labels means test rows leak from train"


def test_leakage_detector_has_power_a_random_split_scores_near_100(train_mod):
    """Control for the test above: the same data under a naive random split is ~perfect."""
    df = windows_dataset(n_reviews=400)
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.label, test_size=0.2, random_state=42)
    acc = (train_mod.make_model().fit(Xtr, ytr).predict(Xte) == yte).mean()
    assert acc > 0.9


def test_group_split_puts_each_review_on_one_side_only(train_mod):
    df = windows_dataset(n_reviews=80)
    g = train_mod.review_groups(list(df.text))
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=train_mod.SEED).split(df, groups=g))
    assert set(g[tr]).isdisjoint(g[te])


# ------------------------------------------------------- end-to-end main()

def test_main_trains_and_reports_on_learnable_data(run_train):
    code, out, model_path = run_train(windows_dataset(n_reviews=200, learnable=True))
    assert code == 0
    assert float(ACC_RE.search(out).group(1)) > 0.95
    assert model_path.exists()
    saved = joblib.load(model_path)
    assert list(saved.classes_) == ["negative", "positive"]
    assert saved.predict(["good1 good2 good3 good4"])[0] == "positive"
    assert saved.predict(["bad1 bad2 bad3 bad4"])[0] == "negative"


def test_main_output_mentions_row_and_review_counts_and_saved_file(run_train):
    df = windows_dataset(n_reviews=50)
    _, out, _ = run_train(df)
    assert f"{len(df)} rows from 50 distinct source reviews" in out
    assert "model.joblib" in out


def test_main_does_not_overwrite_the_real_model(run_train, train_mod):
    real = train_mod.HERE / "model.joblib"
    before = (real.stat().st_mtime_ns, real.stat().st_size) if real.exists() else None
    run_train(windows_dataset(n_reviews=40))
    after = (real.stat().st_mtime_ns, real.stat().st_size) if real.exists() else None
    assert before == after


def test_main_is_deterministic(run_train):
    df = windows_dataset(n_reviews=100, learnable=True)
    _, out1, p1 = run_train(df)
    first = joblib.load(p1).predict_proba(["good1 bad2 w5 w7"])
    _, out2, p2 = run_train(df)
    second = joblib.load(p2).predict_proba(["good1 bad2 w5 w7"])
    assert ACC_RE.search(out1).group(1) == ACC_RE.search(out2).group(1)
    np.testing.assert_allclose(first, second)


def test_main_drops_rows_with_missing_text_or_label(run_train):
    df = windows_dataset(n_reviews=40, learnable=True)
    n = len(df)
    extra = pd.DataFrame({"text": [None, "good1 good2", "bad1 bad2"],
                          "label": ["positive", None, np.nan]})
    _, out, _ = run_train(pd.concat([df, extra], ignore_index=True))
    assert f"{n} rows from" in out


def test_main_fails_clearly_when_data_file_is_missing(train_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(train_mod, "DATA", tmp_path / "nope.csv")
    monkeypatch.setattr(train_mod, "MODEL", tmp_path / "model.joblib")
    with pytest.raises(FileNotFoundError):
        train_mod.main()
    assert not (tmp_path / "model.joblib").exists()


def test_main_fails_when_csv_lacks_required_columns(run_train):
    with pytest.raises(KeyError):
        run_train(pd.DataFrame({"review": ["a b c"], "sentiment": ["positive"]}))


def test_main_fails_on_a_single_class_and_writes_no_model(run_train):
    df = windows_dataset(n_reviews=30, learnable=True)
    df["label"] = "positive"
    with pytest.raises(ValueError):
        run_train(df)


def test_main_fails_when_there_are_too_few_reviews_to_split(run_train):
    one_review = windows_dataset(n_reviews=1, learnable=True)
    with pytest.raises(ValueError):
        run_train(one_review)


@pytest.mark.xfail(strict=False, reason=(
    "FINDING: train.py does not validate labels. Any extra/odd label ('neutral', 'Positive', "
    "'pos') silently becomes a new class, so predict() could then return something other than "
    "'positive'/'negative', breaking its contract."))
def test_main_rejects_labels_outside_positive_negative(run_train):
    df = windows_dataset(n_reviews=60, learnable=True)
    df.loc[df.index[: len(df) // 5], "label"] = "neutral"
    with pytest.raises(ValueError):
        run_train(df)


# ------------------------------------------------ real data, real numbers

@pytest.fixture(scope="module")
def real_run(train_mod, tmp_path_factory):
    """One real training run on data.csv, with the model redirected to a tmp dir."""
    import contextlib
    import io

    tmp = tmp_path_factory.mktemp("real_run")
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(buf):
        mp.setattr(train_mod, "MODEL", tmp / "model.joblib")
        code = train_mod.main()
    return code, buf.getvalue(), tmp / "model.joblib"


@pytest.mark.slow
def test_real_run_succeeds_and_prints_accuracy(real_run):
    code, out, model_path = real_run
    assert code == 0 and model_path.exists()
    assert ACC_RE.search(out), out


@pytest.mark.slow
def test_real_heldout_accuracy_is_plausible_not_leaky_not_broken(real_run):
    acc = float(ACC_RE.search(real_run[1]).group(1))
    assert 0.70 <= acc <= 0.85, (
        f"{acc}: below 0.70 means the model regressed, above ~0.85 suggests train/test leakage")


@pytest.mark.slow
def test_real_training_is_fast(train_mod, tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(train_mod, "MODEL", tmp_path / "m.joblib")
    start = time.perf_counter()
    train_mod.main()
    assert time.perf_counter() - start < 30  # "a few seconds" with generous slack for slow CI


@pytest.mark.slow
def test_naive_random_split_on_real_data_overstates_accuracy(train_mod, data_df, real_run):
    """Why grouping is needed: a random split inflates the score by >10 points on data.csv."""
    Xtr, Xte, ytr, yte = train_test_split(
        data_df.text, data_df.label, test_size=0.2, random_state=42, stratify=data_df.label)
    naive = (train_mod.make_model().fit(Xtr, ytr).predict(Xte) == yte).mean()
    honest = float(ACC_RE.search(real_run[1]).group(1))
    assert naive - honest > 0.10


@pytest.mark.slow
def test_readme_accuracy_matches_what_train_prints(real_run, demo_dir):
    printed = float(ACC_RE.search(real_run[1]).group(1)) * 100
    readme = (demo_dir / "README.md").read_text()
    claimed = float(re.search(r"Held-out accuracy:\s*\*{0,2}([0-9.]+)%", readme).group(1))
    assert abs(printed - claimed) <= 0.5, f"README says {claimed}%, train.py prints {printed:.1f}%"
