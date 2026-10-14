"""What makes a generated narrative trustworthy: checks it has to pass, or it is discarded.

This is the highest-leverage module in the package, and none of it involves a model. The
premise is that a language model's output is *untrusted input* until proven otherwise —
so every claim is checked against the findings it was supposed to be summarising, and
anything that fails is thrown away in favour of the deterministic templates.

The four checks, in order of how much damage they prevent:

**Numeric grounding.** Every number in the text must be traceable to a value in the
findings' evidence. A plausible wrong figure is the most damaging error a summary can
contain — it is the thing that gets quoted — and it is entirely mechanically detectable,
because the set of legitimate numbers is small and enumerable. On the tutorial's report
there are 40 of them.

**Vocabulary.** Every metric name in the text must be one the assessment actually
computed. This catches a model reaching for a metric it knows about from training but
which is not in this report — "the F1 score of 0.94" when no F1 was measured.

**Citations.** Every sentence must carry a finding reference, so a reader can get from a
claim to the numbers behind it. A sentence citing nothing is dropped rather than the whole
narrative being rejected: partial credit is right here, because one unsupported aside
should not cost the reader eight good paragraphs.

**Causal language.** Nothing in an assessment establishes *why* a model fails. A summary
saying the model underperforms on small tumours *because* they are under-represented has
invented a mechanism, and no model quality prevents that — the information is not in the
input. So the vocabulary of causation is banned outright and checked for.

Formatting tolerance
--------------------

The grounding check has to accept a number the audience layer legitimately reformatted:
0.9266 may appear as ``0.9266``, ``0.927``, ``93%`` or ``92.7%``. So the allowed set is
built as *rendered strings* — every permitted rendering of every evidence value — rather
than as floats compared with a tolerance. Comparing with a tolerance is what lets ``93%``
match an evidence value of 0.93 that was never in the report.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Verification",
    "Problem",
    "allowed_numbers",
    "numbers_per_finding",
    "verify_narrative",
    "CAUSAL_PHRASES",
]

#: Words that assert a mechanism the assessment cannot establish. Checked as whole words
#: so that "because" is caught while "becomes" is not.
CAUSAL_PHRASES: tuple[str, ...] = (
    "because",
    "caused by",
    "causes",
    "causing",
    "due to",
    "the reason",
    "as a result of",
    "owing to",
    "explains why",
    "which is why",
    "stems from",
    "arises from",
    "leads to",
)

#: A number in prose: 0.9266, 93%, 1,234, -0.04, 5. Percent sign captured when present so
#: the check can distinguish "93" from "93%".
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

#: Sentence split that does not fire on "0.9266" or "e.g.".
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

#: A finding citation, as the prompt asks the model to write them: [F3] or [F3, F7].
_CITATION = re.compile(r"\[F\d+(?:\s*,\s*F?\d+)*\]")


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a generated narrative."""

    kind: str
    detail: str
    text: str = ""

    def __str__(self) -> str:
        where = f" in {self.text!r}" if self.text else ""
        return f"[{self.kind}] {self.detail}{where}"


@dataclass(frozen=True, slots=True)
class Verification:
    """The outcome of checking a narrative.

    ``ok`` is the gate the caller acts on. ``kept`` is the text that survived — sentences
    with no citation are dropped from it — and ``problems`` says everything that was
    wrong, so a failure is diagnosable rather than merely reported.
    """

    ok: bool
    kept: str
    problems: tuple[Problem, ...] = ()

    @property
    def fatal(self) -> tuple[Problem, ...]:
        """Problems that invalidate the narrative rather than trimming it."""
        return tuple(p for p in self.problems if p.kind != "uncited")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": [
                {"kind": p.kind, "detail": p.detail, "text": p.text} for p in self.problems
            ],
        }

    def merge(self, other: Verification) -> Verification:
        """Both verifications' verdicts, for text checked in two passes.

        The headline is checked apart from the body, because the citation rule does not
        apply to it — see :func:`~themis_assess.summary.summarise.summarise`. Merging
        rather than checking the concatenation is what keeps that difference from
        weakening the body's checks.
        """
        return Verification(
            ok=self.ok and other.ok,
            kept=" ".join(part for part in (self.kept, other.kept) if part),
            problems=self.problems + other.problems,
        )

    def describe(self) -> str:
        if self.ok and not self.problems:
            return "verified: every number grounded, every sentence cited"
        if self.ok:
            return f"verified with {len(self.problems)} sentence(s) dropped"
        return "REJECTED: " + "; ".join(p.detail for p in self.fatal[:3])


# --------------------------------------------------------------------------
# the allowed number set
# --------------------------------------------------------------------------


def _renderings(value: float) -> set[str]:
    """Every string form of ``value`` a summary is allowed to use.

    Built rather than compared numerically on purpose. A tolerance-based check would let
    ``93%`` match an evidence value of 0.9266 *and* an invented 0.93 — the whole point is
    that only values actually present may appear, in a form derived from them.
    """
    forms: set[str] = set()
    for places in (0, 1, 2, 3, 4, 6):
        forms.add(f"{value:.{places}f}")
        forms.add(f"{value:.{places}f}".rstrip("0").rstrip("."))
    if -1.0 <= value <= 1.0:
        for places in (0, 1, 2):
            forms.add(f"{value:.{places}%}")
            forms.add(f"{abs(value):.{places}%}")
    # a proportion's complement: "loses 46%" from a retained value of 0.5413
    if 0.0 <= value <= 1.0:
        complement = 1.0 - value
        for places in (0, 1, 2):
            forms.add(f"{complement:.{places}%}")
        for places in (2, 3, 4):
            forms.add(f"{complement:.{places}f}")
    if float(value).is_integer():
        forms.add(f"{int(value)}")
        forms.add(f"{int(value):,}")
    forms.discard("")
    return forms


def allowed_numbers(findings: Iterable[Any], *, extra: Iterable[float] = ()) -> set[str]:
    """Every numeric string a narrative over ``findings`` may legitimately contain.

    Walks each finding's ``evidence`` recursively, so a nested per-subgroup mapping
    contributes its values too. ``extra`` is for numbers the summary layer itself
    introduces — a count of findings, a number of subgroups.
    """
    allowed: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)):
            number = float(value)
            if number == number:  # not NaN
                # `.update()` rather than `allowed |= ...`: the augmented form rebinds
                # the name, which inside a closure makes it local and raises.
                allowed.update(_renderings(number))

    for finding in findings:
        walk(getattr(finding, "evidence", {}))
    for number in extra:
        allowed.update(_renderings(float(number)))

    return allowed | _SMALL_INTEGERS


#: Ordinals and counts a narrative needs in order to read at all — "three findings", "the
#: first of five", "level 5 of 5". Allowing 0-10 costs nothing: no misquoted metric hides
#: in a single digit, since every metric value here is either a bounded fraction or is
#: reported with decimals.
_SMALL_INTEGERS = frozenset(str(n) for n in range(11))


def numbers_per_finding(findings: Sequence[Any]) -> dict[int, set[str]]:
    """``{1: renderings of F1's evidence, 2: ...}``, for the scoped grounding check."""
    return {
        index: allowed_numbers([finding]) - _SMALL_INTEGERS
        for index, finding in enumerate(findings, start=1)
    }


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def _without_citations(text: str) -> str:
    """Strip ``[F3]`` markers before checking content.

    They are markup, not claims, and leaving them in breaks both content checks: the
    number check reads the ``3`` in ``[F3]`` as a figure, and — the one that actually bit
    — the vocabulary check reads the ``F1`` in ``[F1]`` as the metric ``f1`` and rejects
    every correctly-cited sentence that happens to cite finding one.
    """
    return _CITATION.sub(" ", text)


def _cited_ids(sentence: str) -> set[int]:
    """The finding numbers a sentence cites: ``[F3, F7]`` -> ``{3, 7}``."""
    ids: set[int] = set()
    for match in _CITATION.finditer(sentence):
        ids.update(int(n) for n in re.findall(r"\d+", match.group()))
    return ids


def _check_numbers(
    text: str, allowed: set[str], per_finding: Mapping[int, set[str]] | None = None
) -> list[Problem]:
    """Every number must be grounded — and, where citations allow it, grounded in the
    finding the sentence actually cites.

    The narrower check is the one that matters, and the notebook is what proved it. A
    model wrote "its correctness is 98%, which is excellent [F3]" and the first version
    of this function passed it, because 0.98 *was* in the evidence — it was the recall
    threshold from a different finding. The number existed; the attribution was invented.

    So a cited sentence is scoped to its citations. That makes the citation requirement
    load-bearing rather than decorative: it is not only traceability, it is the boundary
    of what the sentence is allowed to say.
    """
    problems: list[Problem] = []
    for sentence in _sentences(text):
        scope = allowed
        cited = _cited_ids(sentence) if per_finding else set()
        if cited and per_finding is not None:
            in_scope = set().union(*(per_finding.get(i, set()) for i in cited))
            if in_scope:
                scope = in_scope | _SMALL_INTEGERS
        for token in _NUMBER.findall(sentence):
            if token in scope:
                continue
            # "0.93" written as ".93", and thousands separators
            bare = token.replace(",", "")
            if bare in scope or bare.lstrip("0") in scope:
                continue
            where = (
                f"is not a value in finding(s) {sorted(cited)}'s evidence"
                if cited
                else "is not any value in the findings' evidence"
            )
            problems.append(Problem("ungrounded_number", f"{token} {where}", sentence))
    return problems


#: Registered metric names that are also ordinary English words. These cannot be checked
#: as bare words: a clinician-facing narrative says "accuracy" and "recall" in their plain
#: senses, and flagging that would make the check unusable. They are still checked in
#: their ``snake_case`` compounds — ``balanced_accuracy`` is unambiguous.
_AMBIGUOUS_AS_ENGLISH = frozenset({"accuracy", "recall", "precision"})


def _check_vocabulary(text: str, allowed_metrics: set[str]) -> list[Problem]:
    """Metric names the report does not contain must not appear.

    Catches a model reaching for a metric it knows from training but which this
    assessment never computed — "the f1 score of 0.94" where no F1 was measured.

    Two things this cannot check, both by necessity. A reader-facing phrase such as
    "overall correctness" has no canonical form to compare against, and should not have:
    the audience layer invents those deliberately. And the three metric names that are
    also ordinary English words are exempt as bare words — see
    :data:`_AMBIGUOUS_AS_ENGLISH` — though their compounds are still checked.
    """
    from themis_assess.metrics.registry import default_registry

    checkable = set(default_registry().names()) - _AMBIGUOUS_AS_ENGLISH
    lowered = text.lower()
    problems: list[Problem] = []
    for name in sorted(checkable):
        if name in allowed_metrics:
            continue
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            problems.append(
                Problem(
                    "metric_not_in_report",
                    f"{name!r} is a real metric but this assessment did not compute it",
                    name,
                )
            )
    return problems


def _check_causal(text: str) -> list[Problem]:
    problems: list[Problem] = []
    for sentence in _sentences(text):
        lowered = sentence.lower()
        for phrase in CAUSAL_PHRASES:
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                problems.append(
                    Problem(
                        "causal_claim",
                        f"{phrase!r} asserts a mechanism the assessment cannot establish",
                        sentence,
                    )
                )
                break
    return problems


def _check_citations(text: str) -> tuple[str, list[Problem]]:
    """Drop sentences with no citation, and report each one dropped."""
    kept: list[str] = []
    problems: list[Problem] = []
    for sentence in _sentences(text):
        if _CITATION.search(sentence):
            kept.append(sentence)
        else:
            problems.append(
                Problem("uncited", "no finding reference, so the claim cannot be checked", sentence)
            )
    return " ".join(kept), problems


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def verify_narrative(
    text: str,
    findings: Sequence[Any],
    *,
    allowed_metrics: Iterable[str] = (),
    require_citations: bool = True,
    extra_numbers: Iterable[float] = (),
) -> Verification:
    """Check a generated narrative against the findings it claims to summarise.

    Parameters
    ----------
    require_citations:
        Whether a sentence must carry a ``[F3]`` reference to survive. Sentences without
        one are dropped, and if that leaves nothing the narrative is rejected.

    Returns
    -------
    Verification
        ``ok`` is ``False`` if any number is ungrounded, any metric is named that the
        report does not contain, any causal claim appears, or nothing survives the
        citation filter. In every one of those cases the caller should fall back to the
        deterministic templates.
    """
    if not text or not text.strip():
        return Verification(False, "", (Problem("empty", "the model returned nothing"),))

    problems: list[Problem] = []
    kept = text.strip()

    if require_citations:
        kept, citation_problems = _check_citations(kept)
        problems.extend(citation_problems)
        if not kept:
            return Verification(
                False, "", (*problems, Problem("empty", "no sentence carried a citation"))
            )

    allowed = allowed_numbers(findings, extra=extra_numbers)
    # Citations are stripped for the vocabulary and causal checks, but the number check
    # needs them: they are what scopes each sentence to the findings it cites.
    problems.extend(_check_numbers(kept, allowed, numbers_per_finding(findings)))
    content = _without_citations(kept)
    problems.extend(_check_vocabulary(content, set(allowed_metrics)))
    problems.extend(_check_causal(content))

    fatal = [p for p in problems if p.kind != "uncited"]
    return Verification(not fatal, kept, tuple(problems))
