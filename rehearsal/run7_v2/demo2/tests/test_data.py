"""Sanity checks on data.csv and eval.csv."""
import pytest

LABELS = {"positive", "negative"}


@pytest.mark.parametrize("name", ["data", "eval_df"])
def test_schema_and_labels(name, request):
    df = request.getfixturevalue(name)
    assert list(df.columns) == ["text", "label"]
    assert df.text.notna().all() and df.label.notna().all()
    assert set(df.label) == LABELS
    assert (df.text.str.strip().str.len() > 0).all()


def test_classes_roughly_balanced(data):
    share = data.label.value_counts(normalize=True)
    assert share.min() > 0.4


def test_no_conflicting_labels_for_same_text(data, eval_df):
    both = __import__("pandas").concat([data, eval_df])
    assert (both.groupby("text").label.nunique() == 1).all()


def test_no_exact_text_overlap_between_train_and_eval(data, eval_df):
    assert not set(data.text) & set(eval_df.text)


def test_word_movie_is_not_a_label_proxy(data):
    """In real review corpora "movie" appears in both classes at similar rates.

    A big gap means the two classes come from different sources, and the model
    can learn the source instead of the sentiment.
    """
    has_movie = data.text.str.contains(r"\bmovie\b", case=False)
    rate = has_movie.groupby(data.label).mean()
    assert abs(rate["negative"] - rate["positive"]) < 0.2, rate.to_dict()
