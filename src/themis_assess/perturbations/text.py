"""Text perturbations. Survey §4.3.

Repo D's ``CharNoisePerturbation`` and ``MaskingPerturbation`` are the source. Two
changes on the way in.

**No TextAttack dependency.** Repo D's character noise wraps TextAttack's
``CompositeTransformation`` of five word swaps. The survey describes that family as
"character-level insertions, deletions, swaps, repetitions, case changes" — string
operations of a few lines each. TextAttack brings a large dependency tree for them (and
does not currently build from source on Python 3.11), so they are implemented directly
here: no dependency, exactly reproducible from a generator, and each operation
individually testable. Results will not match repo D's byte for byte, which is expected —
this generalises the family rather than reproducing one library's sampling.

**Severity, and a real generator.** Repo D parameterised directly
(``pct_words_to_swap=0.9``) and drew from global ``np.random`` state in the masking
method. Here severity maps onto the fraction of words touched, and every draw comes from
the passed generator.

Not included yet: embedding-nearest-neighbour substitution and masked-LM paraphrasing
(survey §4.3, items 34 and 36). Both need a model — counter-fitted embeddings, or an MLM
— and shipping an untested wrapper around a download would be worse than not shipping
one. They land with the text handler, when a model is available to test against.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np

from themis_assess.core.adapters import DataAdapter, TextAdapter
from themis_assess.core.types import Modality, PerturbationSpec
from themis_assess.perturbations.base import Perturbation

__all__ = ["CharacterNoise", "TokenMasking", "WordDropout", "CharacterOperation"]

_TEXT_ONLY = frozenset({Modality.TEXT})
_WORD = re.compile(r"\S+")

#: Physically adjacent keys on a QWERTY keyboard — the substitutions a real typo makes.
_QWERTY_NEIGHBOURS: dict[str, str] = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn",
    "j": "hukm", "k": "jil", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm",
    "m": "njk",
}

#: Characters that look like others — the substitutions an OCR or a copy-paste makes.
_HOMOGLYPHS: dict[str, str] = {
    "o": "0", "O": "0", "l": "1", "I": "1", "i": "1", "s": "5", "S": "5",
    "a": "@", "e": "3", "b": "6", "g": "9", "z": "2", "B": "8",
}


class CharacterOperation(str):
    """The character-level edits, as the survey enumerates them."""

    DELETE = "delete"
    INSERT = "insert"
    SWAP = "swap"
    REPEAT = "repeat"
    CASE = "case"
    QWERTY = "qwerty"
    HOMOGLYPH = "homoglyph"


_ALL_OPERATIONS = (
    CharacterOperation.DELETE,
    CharacterOperation.INSERT,
    CharacterOperation.SWAP,
    CharacterOperation.REPEAT,
    CharacterOperation.CASE,
    CharacterOperation.QWERTY,
    CharacterOperation.HOMOGLYPH,
)


class _TextPerturbation(Perturbation):
    """Shared machinery for perturbations acting on whole documents."""

    modalities: ClassVar[frozenset[Modality]] = _TEXT_ONLY
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = False

    def _texts(self, data: DataAdapter) -> list[str]:
        if isinstance(data, TextAdapter):
            return data.texts()
        values = data.values()
        return [str(values[e, 0, 0]) for e in range(data.n_entities)]

    def _rebuild(self, data: DataAdapter, texts: list[str]) -> DataAdapter:
        if isinstance(data, TextAdapter):
            return data.with_texts(texts)
        values = data.values().astype(object)
        for e, text in enumerate(texts):
            values[e, 0, 0] = text
        return data.with_values(values)

    def _transform(self, text: str, factor: float, rng: np.random.Generator) -> str:
        raise NotImplementedError

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        self.eligible_features(data, spec)
        factor = self.severity_factor(spec)
        texts = self._texts(data)
        return [
            self._rebuild(data, [self._transform(t, factor, rng) for t in texts])
            for _ in range(spec.n_variants)
        ]


def _select_words(text: str, fraction: float, rng: np.random.Generator) -> list[re.Match]:
    """Choose which whitespace-separated tokens to touch, at least one if any exist."""
    words = list(_WORD.finditer(text))
    if not words:
        return []
    count = max(1, round(fraction * len(words)))
    count = min(count, len(words))
    chosen = rng.choice(len(words), size=count, replace=False)
    return [words[i] for i in sorted(chosen)]


class CharacterNoise(_TextPerturbation):
    """Character-level typos and transcription errors. Survey §4.3, noise-based.

    Simulates what happens to text between the world and the model: someone typing
    quickly, an OCR pass, a copy through a lossy channel. Seven operations, drawn
    uniformly per selected word unless ``operations`` narrows the set:

    ``delete``     drop a character
    ``insert``     insert a random lowercase letter
    ``swap``       transpose two neighbouring characters
    ``repeat``     double a character
    ``case``       flip the case of a character
    ``qwerty``     replace a character with a physically adjacent key
    ``homoglyph``  replace a character with one that looks like it (``o`` to ``0``)

    Severity is the fraction of words touched, one edit each. Repo D used a single rung
    at ``pct_words_to_swap=0.9``, named ``char_heavy`` — implying lighter siblings that
    were never defined. The five-level ladder here supplies them.

    This is the least likely of the text family to change the true label, which makes it
    the most trustworthy: a reader recovers "teh cat" without effort. That said, the
    survey's §6.3 caution still applies at the top of the range — at severity 5 nearly
    every word carries an edit, and label preservation is an assumption, not a fact.
    """

    name = "character_noise"
    severity_scale = (0.05, 0.15, 0.30, 0.60, 0.90)

    def __init__(self, *, operations: tuple[str, ...] = _ALL_OPERATIONS) -> None:
        unknown = [op for op in operations if op not in _ALL_OPERATIONS]
        if unknown:
            raise ValueError(
                f"Unknown character operation(s) {unknown}. "
                f"Available: {list(_ALL_OPERATIONS)}."
            )
        if not operations:
            raise ValueError("CharacterNoise needs at least one operation.")
        self.operations = tuple(operations)

    def _edit_word(self, word: str, rng: np.random.Generator) -> str:
        if not word:
            return word
        operation = self.operations[rng.integers(len(self.operations))]
        i = int(rng.integers(len(word)))
        char = word[i]

        if operation == CharacterOperation.DELETE and len(word) > 1:
            return word[:i] + word[i + 1 :]
        if operation == CharacterOperation.INSERT:
            letter = chr(int(rng.integers(ord("a"), ord("z") + 1)))
            return word[:i] + letter + word[i:]
        if operation == CharacterOperation.SWAP and len(word) > 1:
            i = min(i, len(word) - 2)
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if operation == CharacterOperation.REPEAT:
            return word[:i] + char + word[i:]
        if operation == CharacterOperation.CASE:
            flipped = char.lower() if char.isupper() else char.upper()
            return word[:i] + flipped + word[i + 1 :]
        if operation == CharacterOperation.QWERTY:
            neighbours = _QWERTY_NEIGHBOURS.get(char.lower())
            if neighbours:
                replacement = neighbours[int(rng.integers(len(neighbours)))]
                if char.isupper():
                    replacement = replacement.upper()
                return word[:i] + replacement + word[i + 1 :]
        if operation == CharacterOperation.HOMOGLYPH:
            look_alike = _HOMOGLYPHS.get(char)
            if look_alike:
                return word[:i] + look_alike + word[i + 1 :]
        # The chosen operation did not apply to this character (a swap in a one-letter
        # word, a QWERTY neighbour for a digit). Fall back to a repeat, which always
        # applies, so a selected word is never silently left alone.
        return word[:i] + char + word[i:]

    def _transform(self, text: str, factor: float, rng: np.random.Generator) -> str:
        chosen = _select_words(text, factor, rng)
        if not chosen:
            return text
        out = list(text)
        for match in reversed(chosen):  # right to left, so earlier spans stay valid
            out[match.start() : match.end()] = self._edit_word(match.group(), rng)
        return "".join(out)

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["operations"] = list(self.operations)
        if spec is not None:
            meta["fraction_of_words_edited"] = self.severity_factor(spec)
            meta["edits_per_word"] = 1
        return meta


class TokenMasking(_TextPerturbation):
    """Replace words with a mask token. Repo-A extension, not in the survey.

    Ported from repo D's ``MaskingPerturbation``, with the global-RNG draw replaced by
    the passed generator.

    What it measures is different from the rest of the family. The others ask whether a
    prediction survives text being *corrupted*; this asks whether it survives text being
    *withheld*. That makes it the closest text analogue of a feature-sensitivity map: at
    high severity, a model still confident about a document it can barely see is relying
    on something other than the masked content — length, style, the few surviving tokens.

    Because it removes information rather than distorting it, label preservation is
    weaker here than for :class:`CharacterNoise`, and weakens fast: a mostly-masked
    document may genuinely no longer carry its label. Treat the top of the range as a
    diagnostic instrument rather than a robustness test, and read the survey's §6.3
    partial-observability discussion alongside it.

    Parameters
    ----------
    mask_token:
        Defaults to ``"[MASK]"``. For a transformer model this should be its tokenizer's
        own mask token — repo D's notebook wired it as
        ``MASK_TOKEN = data_handler.tokenizer.mask_token``. A mask token the model does
        not recognise is measuring an out-of-vocabulary response instead.
    """

    name = "token_masking"
    label_preserving = False
    severity_scale = (0.10, 0.25, 0.50, 0.75, 0.90)

    def __init__(self, *, mask_token: str = "[MASK]") -> None:
        if not mask_token:
            raise ValueError("mask_token must be a non-empty string.")
        self.mask_token = str(mask_token)

    def _transform(self, text: str, factor: float, rng: np.random.Generator) -> str:
        chosen = _select_words(text, factor, rng)
        if not chosen:
            return text
        out = list(text)
        for match in reversed(chosen):
            out[match.start() : match.end()] = self.mask_token
        return "".join(out)

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["mask_token"] = self.mask_token
        meta["removes_information"] = True
        if spec is not None:
            meta["fraction_of_words_masked"] = self.severity_factor(spec)
        return meta


class WordDropout(_TextPerturbation):
    """Delete words outright. Survey §4.3, word-level noise.

    The difference from :class:`TokenMasking` is whether the model is told that something
    is missing. Masking leaves a placeholder — position and count survive, and a
    transformer sees a token it was trained to reason about. Dropout leaves nothing: the
    document is simply shorter, and the model has no way to know a word was ever there.

    Truncated extraction, a lossy transcript, a form field that did not survive a schema
    change. Severity is the fraction of words removed.

    Like masking, this removes information rather than distorting it, so
    ``label_preserving`` is ``False``.
    """

    name = "word_dropout"
    label_preserving = False
    severity_scale = (0.05, 0.15, 0.30, 0.50, 0.75)

    def _transform(self, text: str, factor: float, rng: np.random.Generator) -> str:
        words = list(_WORD.finditer(text))
        if len(words) < 2:  # never empty a document entirely
            return text
        drop_count = min(max(1, round(factor * len(words))), len(words) - 1)
        chosen = sorted(rng.choice(len(words), size=drop_count, replace=False))
        kept = [w.group() for i, w in enumerate(words) if i not in set(chosen)]
        return " ".join(kept)

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["removes_information"] = True
        meta["leaves_placeholder"] = False
        if spec is not None:
            meta["fraction_of_words_dropped"] = self.severity_factor(spec)
        return meta
