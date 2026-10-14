"""Tests for the text perturbations (survey §4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter, TextAdapter
from themis_assess.perturbations.text import (
    CharacterNoise,
    CharacterOperation,
    TokenMasking,
    WordDropout,
)

ALL = (CharacterNoise, TokenMasking, WordDropout)
SENTENCE = "The quick brown fox jumps over the lazy dog every single morning"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def documents():
    return TextAdapter([SENTENCE, "A second shorter document about something else"])


# --------------------------------------------------------------------------
# shared properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
def test_all_are_text_only(cls, rng):
    tabular = ArrayAdapter(np.ones((3, 2)), ["a", "b"])
    p = cls()
    assert not p.applies_to(tabular)
    with pytest.raises(ValueError, match="applies to \\['text'\\]"):
        p.perturb(tabular, p.spec(severity=3), rng)


@pytest.mark.parametrize("cls", ALL)
def test_all_return_the_same_number_of_documents(cls, documents, rng):
    p = cls()
    out = p.perturb(documents, p.spec(severity=3), rng)[0]
    assert out.n_entities == documents.n_entities
    assert len(out.texts()) == 2


@pytest.mark.parametrize("cls", ALL)
def test_all_change_the_text(cls, documents, rng):
    p = cls()
    out = p.perturb(documents, p.spec(severity=4), rng)[0]
    assert out.texts() != documents.texts()


@pytest.mark.parametrize("cls", ALL)
def test_all_leave_documents_as_strings(cls, documents, rng):
    p = cls()
    out = p.perturb(documents, p.spec(severity=5), rng)[0]
    assert all(isinstance(t, str) for t in out.texts())


@pytest.mark.parametrize("cls", ALL)
def test_all_are_reproducible_from_a_seed(cls, documents):
    p = cls()
    a = p.perturb(documents, p.spec(severity=3), np.random.default_rng(1))[0]
    b = p.perturb(documents, p.spec(severity=3), np.random.default_rng(1))[0]
    assert a.texts() == b.texts()


@pytest.mark.parametrize("cls", ALL)
def test_all_draw_only_from_the_supplied_generator(cls, documents):
    """Repo D's MaskingPerturbation used np.random.rand(); this must not."""
    p = cls()
    np.random.seed(0)
    a = p.perturb(documents, p.spec(severity=3), np.random.default_rng(2))[0]
    np.random.seed(999)
    b = p.perturb(documents, p.spec(severity=3), np.random.default_rng(2))[0]
    assert a.texts() == b.texts()


@pytest.mark.parametrize("cls", ALL)
def test_all_handle_an_empty_document(cls, rng):
    p = cls()
    out = p.perturb(TextAdapter(["", "word"]), p.spec(severity=3), rng)[0]
    assert out.texts()[0] == ""


@pytest.mark.parametrize("cls", ALL)
def test_all_touch_more_text_at_higher_severity(cls):
    """Measured as edit distance from the original, averaged over seeds."""
    p = cls()
    documents = TextAdapter([SENTENCE])
    distances = []
    for severity in (1, 5):
        total = 0
        for seed in range(8):
            out = p.perturb(
                documents, p.spec(severity=severity), np.random.default_rng(seed)
            )[0]
            total += _difference(SENTENCE, out.texts()[0])
        distances.append(total)
    assert distances[1] > distances[0]


def _difference(a: str, b: str) -> int:
    """Cheap dissimilarity: characters that differ position by position, plus length gap."""
    common = sum(1 for x, y in zip(a, b, strict=False) if x != y)
    return common + abs(len(a) - len(b))


# --------------------------------------------------------------------------
# CharacterNoise
# --------------------------------------------------------------------------


def test_character_noise_keeps_the_word_count(rng):
    """It corrupts words, it does not remove them."""
    p = CharacterNoise()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=5), rng)[0]
    assert len(out.texts()[0].split()) == len(SENTENCE.split())


def test_character_noise_severity_is_the_fraction_of_words_edited():
    """Ten words at severity 3 (0.30) means three of them change."""
    p = CharacterNoise()
    text = " ".join(f"word{i}aaaa" for i in range(10))
    out = p.perturb(TextAdapter([text]), p.spec(severity=3), np.random.default_rng(3))[0]
    changed = sum(
        1 for a, b in zip(text.split(), out.texts()[0].split(), strict=True) if a != b
    )
    assert changed == 3


def test_character_noise_always_edits_at_least_one_word(rng):
    """Severity 1 is 5%, so a short document would otherwise often pass through."""
    p = CharacterNoise()
    for seed in range(10):
        out = p.perturb(
            TextAdapter(["two words"]), p.spec(severity=1), np.random.default_rng(seed)
        )[0]
        assert out.texts()[0] != "two words"


@pytest.mark.parametrize(
    "operation",
    [
        CharacterOperation.DELETE,
        CharacterOperation.INSERT,
        CharacterOperation.SWAP,
        CharacterOperation.REPEAT,
        CharacterOperation.CASE,
        CharacterOperation.QWERTY,
        CharacterOperation.HOMOGLYPH,
    ],
)
def test_each_operation_changes_the_word(operation):
    """No operation may silently leave a selected word alone."""
    p = CharacterNoise(operations=(operation,))
    text = "algorithms"
    for seed in range(6):
        out = p.perturb(
            TextAdapter([text]), p.spec(severity=5), np.random.default_rng(seed)
        )[0]
        assert out.texts()[0] != text, operation


def test_delete_shortens_and_insert_lengthens():
    text = "algorithms"
    deleted = CharacterNoise(operations=(CharacterOperation.DELETE,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(1)
    )[0]
    inserted = CharacterNoise(operations=(CharacterOperation.INSERT,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(1)
    )[0]
    assert len(deleted.texts()[0]) == len(text) - 1
    assert len(inserted.texts()[0]) == len(text) + 1


def test_swap_preserves_the_multiset_of_characters():
    text = "algorithms"
    out = CharacterNoise(operations=(CharacterOperation.SWAP,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(2)
    )[0]
    assert sorted(out.texts()[0]) == sorted(text)
    assert out.texts()[0] != text


def test_case_flip_preserves_the_lowercase_form():
    text = "algorithms"
    out = CharacterNoise(operations=(CharacterOperation.CASE,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(3)
    )[0]
    assert out.texts()[0].lower() == text
    assert out.texts()[0] != text


def test_qwerty_substitutes_a_physically_adjacent_key():
    """A real typo hits a neighbouring key, not a random letter."""
    from themis_assess.perturbations.text import _QWERTY_NEIGHBOURS

    text = "adf"
    out = CharacterNoise(operations=(CharacterOperation.QWERTY,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(4)
    )[0].texts()[0]
    assert len(out) == len(text)
    changed = [(a, b) for a, b in zip(text, out, strict=True) if a != b]
    assert changed
    for original, replacement in changed:
        assert replacement in _QWERTY_NEIGHBOURS[original]


def test_homoglyph_substitutes_a_look_alike():
    from themis_assess.perturbations.text import _HOMOGLYPHS

    text = "aeiosb"
    out = CharacterNoise(operations=(CharacterOperation.HOMOGLYPH,)).perturb(
        TextAdapter([text]), CharacterNoise().spec(severity=5), np.random.default_rng(5)
    )[0].texts()[0]
    changed = [(a, b) for a, b in zip(text, out, strict=True) if a != b]
    assert changed
    for original, replacement in changed:
        assert _HOMOGLYPHS[original] == replacement


def test_character_noise_is_label_preserving_by_declaration():
    """The mildest of the family: a reader recovers "teh cat" without effort."""
    assert CharacterNoise().label_preserving is True


def test_character_noise_rejects_an_unknown_operation():
    with pytest.raises(ValueError, match="Unknown character operation"):
        CharacterNoise(operations=("rot13",))


def test_character_noise_rejects_an_empty_operation_set():
    with pytest.raises(ValueError, match="at least one operation"):
        CharacterNoise(operations=())


def test_character_noise_reports_its_operations():
    p = CharacterNoise(operations=(CharacterOperation.SWAP, CharacterOperation.DELETE))
    meta = p.metadata(p.spec(severity=2))
    assert meta["operations"] == ["swap", "delete"]
    assert meta["fraction_of_words_edited"] == pytest.approx(0.15)


# --------------------------------------------------------------------------
# TokenMasking
# --------------------------------------------------------------------------


def test_masking_replaces_words_with_the_mask_token(rng):
    p = TokenMasking()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=3), rng)[0].texts()[0]
    assert "[MASK]" in out
    assert out.split().count("[MASK]") == round(0.50 * len(SENTENCE.split()))


def test_masking_keeps_the_word_count(rng):
    """Position and count survive — that is the difference from dropout."""
    p = TokenMasking()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=4), rng)[0]
    assert len(out.texts()[0].split()) == len(SENTENCE.split())


def test_masking_accepts_a_models_own_mask_token(rng):
    """Repo D wired MASK_TOKEN = data_handler.tokenizer.mask_token."""
    p = TokenMasking(mask_token="<mask>")
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=3), rng)[0].texts()[0]
    assert "<mask>" in out
    assert "[MASK]" not in out
    assert p.metadata()["mask_token"] == "<mask>"


def test_masking_rejects_an_empty_mask_token():
    with pytest.raises(ValueError, match="non-empty string"):
        TokenMasking(mask_token="")


def test_masking_is_declared_not_label_preserving():
    """It removes information rather than distorting it."""
    assert TokenMasking().label_preserving is False
    assert TokenMasking().metadata()["removes_information"] is True


# --------------------------------------------------------------------------
# WordDropout
# --------------------------------------------------------------------------


def test_dropout_shortens_the_document(rng):
    p = WordDropout()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=3), rng)[0]
    original_words = len(SENTENCE.split())
    assert len(out.texts()[0].split()) == original_words - round(0.30 * original_words)


def test_dropout_keeps_the_surviving_words_in_order(rng):
    p = WordDropout()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=3), rng)[0]
    kept = out.texts()[0].split()
    original = SENTENCE.split()
    positions = [original.index(w) for w in kept if original.count(w) == 1]
    assert positions == sorted(positions)


def test_dropout_leaves_no_placeholder(rng):
    """The model has no way to know a word was ever there."""
    p = WordDropout()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=4), rng)[0]
    assert "[MASK]" not in out.texts()[0]
    assert p.metadata()["leaves_placeholder"] is False


def test_dropout_never_empties_a_document(rng):
    p = WordDropout()
    out = p.perturb(TextAdapter([SENTENCE]), p.spec(severity=5), rng)[0]
    assert out.texts()[0].strip() != ""


def test_dropout_leaves_a_one_word_document_alone(rng):
    p = WordDropout()
    out = p.perturb(TextAdapter(["solitary"]), p.spec(severity=5), rng)[0]
    assert out.texts()[0] == "solitary"


def test_dropout_is_declared_not_label_preserving():
    assert WordDropout().label_preserving is False


# --------------------------------------------------------------------------
# the distinction between the three
# --------------------------------------------------------------------------


def test_the_three_differ_in_what_they_do_to_length_and_content(rng):
    """Corrupt, withhold-with-placeholder, withhold-silently."""
    documents = TextAdapter([SENTENCE])
    n_words = len(SENTENCE.split())

    noise = CharacterNoise().perturb(
        documents, CharacterNoise().spec(severity=3), np.random.default_rng(1)
    )[0].texts()[0]
    masked = TokenMasking().perturb(
        documents, TokenMasking().spec(severity=3), np.random.default_rng(1)
    )[0].texts()[0]
    dropped = WordDropout().perturb(
        documents, WordDropout().spec(severity=3), np.random.default_rng(1)
    )[0].texts()[0]

    assert len(noise.split()) == n_words  # word count kept, content corrupted
    assert len(masked.split()) == n_words  # word count kept, content withheld
    assert len(dropped.split()) < n_words  # word count reduced


def test_only_character_noise_claims_label_preservation():
    assert CharacterNoise().label_preserving
    assert not TokenMasking().label_preserving
    assert not WordDropout().label_preserving
