"""Does predict() get sentiment right? Easy cases must pass; known-hard cases are xfail so the
suite documents the model's real limits without hiding them (see the xfail reasons)."""
import pytest

CLEAR_POSITIVE = [
    "An absolutely wonderful, moving film with outstanding performances.",
    "Excellent acting and a brilliant script; I loved every minute.",
    "A perfect, hilarious and touching comedy that I would happily watch again.",
    "One of the best films of the year, truly great.",
    "Beautifully shot, well written and emotionally powerful.",
    "A wonderful surprise with a terrific cast.",
    "Fantastic, funny and heartfelt.",
    "I enjoyed it a lot; the performances were excellent.",
]
CLEAR_NEGATIVE = [
    "Boring, badly written and a complete waste of time.",
    "The worst movie I have ever seen; awful acting and a ridiculous plot.",
    "Dull, lame and poorly directed.",
    "A terrible, unfunny mess with nothing going for it.",
    "I hated it. The script is dumb and the acting is wooden.",
    "Painfully bad and way too long, a total disaster.",
    "What a waste: boring plot, annoying characters, awful ending.",
    "Stupid, predictable and poorly made.",
]
CLEAR = [(t, "positive") for t in CLEAR_POSITIVE] + [(t, "negative") for t in CLEAR_NEGATIVE]


def p_positive(model, text):
    return float(model.predict_proba([text])[0, list(model.classes_).index("positive")])


# ------------------------------------------------------------ easy cases

@pytest.mark.parametrize("text,expected", CLEAR, ids=[t[:35] for t, _ in CLEAR])
def test_clearly_polarised_sentences(predict_fn, text, expected):
    assert predict_fn(text) == expected


@pytest.mark.parametrize("word,expected", [
    ("great", "positive"), ("excellent", "positive"), ("wonderful", "positive"),
    ("terrible", "negative"), ("awful", "negative"), ("boring", "negative"),
])
def test_single_strong_words(predict_fn, word, expected):
    assert predict_fn(word) == expected


# -------------------------------------------------- invariances that should hold

@pytest.mark.parametrize("text,expected", CLEAR, ids=[t[:35] for t, _ in CLEAR])
@pytest.mark.parametrize("transform", [
    str.upper, str.lower, str.title, lambda s: f"   {s}\n\n", lambda s: s + " !!!",
    lambda s: s.replace(" ", "  "), lambda s: s.replace(",", ""), lambda s: f"{s} {s} {s}",
], ids=["upper", "lower", "title", "pad_ws", "bang", "double_space", "no_commas", "repeat3"])
def test_label_survives_harmless_rewriting(predict_fn, text, expected, transform):
    assert predict_fn(transform(text)) == expected


FILLER = "The film runs for about two hours and was shown on a Friday."
# These two short positive reviews flip to "negative" when the neutral filler is added.
FILLER_FLIPS = {CLEAR_POSITIVE[1], CLEAR_POSITIVE[6]}
FILLER_XFAIL = pytest.mark.xfail(strict=False, reason=(
    "FINDING: function words carry large learned weights (e.g. about/on/runs/hours are negative), "
    "so neutral filler can outvote a short positive review."))
FILLER_CASES = [
    pytest.param(t, y, marks=FILLER_XFAIL if t in FILLER_FLIPS else (), id=t[:35]) for t, y in CLEAR
]


@pytest.mark.parametrize("text,expected", FILLER_CASES)
def test_neutral_filler_does_not_flip_a_clear_review(predict_fn, text, expected):
    assert predict_fn(f"{FILLER} {text}") == expected
    assert predict_fn(f"{text} {FILLER}") == expected


def test_neutral_filler_flips_only_a_small_minority_of_reviews(predict_fn):
    flips = sum(
        predict_fn(f"{FILLER} {t}") != y or predict_fn(f"{t} {FILLER}") != y for t, y in CLEAR
    )
    assert flips / len(CLEAR) <= 0.20


@pytest.mark.parametrize("text", CLEAR_POSITIVE)
def test_adding_a_negative_word_lowers_positive_probability(model, text):
    assert p_positive(model, text + " awful") < p_positive(model, text)


@pytest.mark.parametrize("text", CLEAR_NEGATIVE)
def test_adding_a_positive_word_raises_positive_probability(model, text):
    assert p_positive(model, text + " excellent") > p_positive(model, text)


def test_confidence_is_higher_for_stronger_evidence(model):
    weak = p_positive(model, "good")
    strong = p_positive(model, "good excellent wonderful perfect brilliant")
    assert strong > weak
    assert p_positive(model, "bad awful terrible worst boring") < p_positive(model, "bad")


def test_aggregate_accuracy_on_hand_written_clear_cases(predict_fn):
    correct = sum(predict_fn(t) == y for t, y in CLEAR)
    assert correct / len(CLEAR) >= 0.9


# -------------------------------------------------- known-hard inputs (xfail)
# A TF-IDF unigram model sees a bag of words: no word order, no negation, no irony.

NEGATION = pytest.mark.xfail(strict=False, reason=(
    "FINDING: bag-of-words ignores negation ('not good' scores as positive)."))
SARCASM = pytest.mark.xfail(strict=False, reason="FINDING: sarcasm/irony is invisible to unigrams.")
CONTRAST = pytest.mark.xfail(strict=False, reason=(
    "FINDING: 'but/then' shifts of sentiment need word order; unigrams just count words."))
SHORT = pytest.mark.xfail(strict=False, reason=(
    "FINDING: trained on 150-word windows, so very short/informal text is mostly noise."))
DOMAIN = pytest.mark.xfail(strict=False, reason=(
    "FINDING: trained on movie reviews; other domains share little vocabulary."))

HARD = [
    pytest.param("not good", "negative", marks=NEGATION, id="not_good"),
    pytest.param("this was not great", "negative", marks=NEGATION, id="not_great"),
    pytest.param("This is not a good movie", "negative", marks=NEGATION, id="not_a_good_movie"),
    pytest.param("I did not like it", "negative", marks=NEGATION, id="did_not_like"),
    pytest.param("Not worth watching.", "negative", marks=NEGATION, id="not_worth_watching"),
    pytest.param("hardly a masterpiece", "negative", marks=NEGATION, id="hardly_masterpiece"),
    pytest.param("not bad at all", "positive", marks=NEGATION, id="not_bad_at_all"),
    pytest.param("far from terrible", "positive", marks=NEGATION, id="far_from_terrible"),
    pytest.param("it is never boring", "positive", marks=NEGATION, id="never_boring"),
    pytest.param("Oh great, another masterpiece of boredom.", "negative", marks=SARCASM, id="sarcasm"),
    pytest.param("so bad it's good", "positive", marks=SARCASM, id="so_bad_its_good"),
    pytest.param("Terrible start, but it gets better and ends up excellent and moving.",
                 "positive", marks=CONTRAST, id="bad_start_good_end"),
    pytest.param("I hate this!", "negative", marks=SHORT, id="i_hate_this"),
    pytest.param("meh", "negative", marks=SHORT, id="meh"),
    pytest.param("10/10", "positive", marks=SHORT, id="ten_out_of_ten"),
    pytest.param("The battery died after two days and the screen cracked.", "negative",
                 marks=DOMAIN, id="product_review_negative"),
]


@pytest.mark.parametrize("text,expected", HARD)
def test_hard_input(predict_fn, text, expected):
    assert predict_fn(text) == expected


def test_negation_is_invisible_to_the_model(predict_fn, model):
    """Characterisation of the limitation above: word order does not matter at all."""
    assert p_positive(model, "good not") == pytest.approx(p_positive(model, "not good"))
    assert p_positive(model, "film great the") == pytest.approx(p_positive(model, "the great film"))


@pytest.mark.xfail(strict=False, reason=(
    "FINDING: the learned weight for 'not' is POSITIVE (and 'no', 'if', 'then', 'be' are strongly "
    "negative function words): the model partly learns style/dataset artefacts, not sentiment."))
def test_negation_words_do_not_push_toward_positive(model):
    vocab = model.named_steps["tfidfvectorizer"].vocabulary_
    coef = model.named_steps["logisticregression"].coef_[0]
    for word in ("not", "never", "nothing"):
        if word in vocab:
            assert coef[vocab[word]] <= 0, word
