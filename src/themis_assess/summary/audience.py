"""Who is reading, and what that changes.

The same assessment has to be readable by different people, and "readable" means
different things to each. A clinician looking at the breast-cancer model needs to know
what the model will do to their patients; they do not need the phrase
``relative_robustness``. An ML engineer needs the metric name, because it is what they
will search the code for. A manager needs the decision and its consequence, in one
paragraph. A regulator needs to know what was checked, against what, and what was not.

So the audience is a parameter, and — this is the part worth insisting on — it shapes the
**deterministic** templates as well as the language model's prompt. If it only shaped the
prompt, then the fallback path (no model, or a model whose output failed verification)
would produce something generic, and the fallback is the path that has to be trustworthy.

What an audience controls
-------------------------

*Naming.* ``balanced_accuracy`` is a term of art. :attr:`Audience.vocabulary` maps it to
whatever this reader calls it, and :meth:`Audience.name_metric` is the only place that
lookup happens.

*Number format.* 0.9266 for an engineer, 93% for a clinician. This is not cosmetic and it
is not safe to do blindly: a percentage is meaningful for a metric bounded on [0, 1] and
nonsense for a negative log-likelihood, so :meth:`Audience.format_value` consults the
metric registry rather than guessing from the magnitude.

*Emphasis.* Which finding areas come first, from :attr:`Audience.leads_with`. A clinician
leads with the per-class and subgroup findings, because those are about patients; an
engineer leads with robustness and calibration, because those are about the model.

*Depth.* Whether to explain what a metric means inline, from :attr:`Audience.explains`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "Audience",
    "CLINICIAN",
    "ML_ENGINEER",
    "MANAGER",
    "REGULATOR",
    "AUDIENCES",
    "get_audience",
]


@dataclass(frozen=True, slots=True)
class Audience:
    """A reader, and the choices that follow from who they are.

    Parameters
    ----------
    name:
        Stable key, used in the report's provenance so a summary records who it was
        written for.
    label:
        How to describe this reader in prose.
    vocabulary:
        ``{metric_name: this reader's phrase}``. Partial: anything absent falls back to
        the metric name with underscores replaced, so a new metric degrades to something
        readable rather than to a ``KeyError``.
    as_percentage:
        Whether to render bounded metrics as percentages. Ignored for metrics the
        registry does not bound on [0, 1] — see :meth:`format_value`.
    decimals:
        Decimal places for the raw form.
    percent_decimals:
        Decimal places for the percentage form. Clinicians get whole numbers; a
        regulator gets one, because they may be quoting it.
    leads_with:
        Finding areas in the order this reader should meet them. Areas not listed come
        afterwards in severity order.
    explains:
        Whether to spell out what a metric means inline.
    tone:
        A phrase describing the register, passed to the language model. Deliberately not
        used by the deterministic templates, which have their wording written out.
    reads_for:
        One sentence on what this reader is trying to decide. The most load-bearing field
        in the prompt: it is what stops a model writing a summary that is accurate and
        useless.
    """

    name: str
    label: str
    vocabulary: Mapping[str, str] = field(default_factory=dict)
    as_percentage: bool = False
    decimals: int = 4
    percent_decimals: int = 0
    leads_with: tuple[str, ...] = ()
    explains: bool = False
    tone: str = "plain and direct"
    reads_for: str = "deciding whether this model is fit for use"

    # -- naming -------------------------------------------------------------

    def name_metric(self, metric: str | None) -> str:
        """This reader's phrase for ``metric``.

        Tolerates ``None``, which arrives from a finding whose evidence is missing the
        key a template expected. Raising here would take down the *fallback* path, which
        is the one path that has to work.
        """
        if not metric:
            return "the metric"
        return self.vocabulary.get(metric, metric.replace("_", " "))

    # -- numbers ------------------------------------------------------------

    def format_value(self, metric: str | None, value: Any, *, registry: Any = None) -> str:
        """Render ``value`` the way this reader should see it.

        A percentage is offered only where the registry bounds the metric on [0, 1].
        That check is the point of the method: rendering a negative log-likelihood of
        0.2916 as "29%" would be confident nonsense, and formatting by magnitude alone
        cannot tell the two cases apart — plenty of unbounded metrics happen to land
        between 0 and 1.
        """
        if value is None:
            return "not measured"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, str):
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        if number != number:  # NaN
            return "undefined"
        if float(number).is_integer() and abs(number) < 1e6 and metric is None:
            return f"{int(number)}"

        if self.as_percentage and metric is not None and _is_unit_bounded(metric, registry):
            return f"{number:.{self.percent_decimals}%}"
        return f"{number:.{self.decimals}f}"

    def metric_subject(self, metric: str | None) -> str:
        """This reader's phrase for ``metric``, as a grammatical sentence subject.

        A vocabulary entry is a noun phrase written to read well *inside* a sentence, and
        the phrases legitimately take three shapes: bare ("overall correctness"),
        article-led ("the share of positive cases it finds") and clause-led ("how often
        its decision changes"). Templates used to prefix them all with ``The model's``,
        which produced "The model's the share of positive cases it finds" — a sentence no
        reader should be shown. So the possessive is built here, once, from the shape of
        the phrase rather than assumed by every template.
        """
        phrase = self.name_metric(metric)
        lowered = phrase.lower()
        for determiner in ("the ", "its "):
            if lowered.startswith(determiner):
                return f"The model's {phrase[len(determiner):]}"
        if lowered.split(" ", 1)[0] in _CLAUSE_OPENERS:
            return phrase[0].upper() + phrase[1:]
        return f"The model's {phrase}"

    def format_pair(
        self, metric: str | None, first: Any, second: Any, *, registry: Any = None
    ) -> tuple[str, str]:
        """Render two values of the same metric so that a reader can tell them apart.

        A clinician sees whole percentages, and at that precision a balanced accuracy of
        0.9474 against a required 0.95 both render as "95%" — so the sentence reads "is
        95%, short of the 95% that was agreed", which is worse than useless: it looks
        like a bug in the assessment rather than a near miss in the model. Precision is
        added, one place at a time, only until the two forms differ.

        The verifier accepts percentages at nought, one and two decimal places and raw
        values at more, so the extra precision stays inside the grounded set.
        """
        rendered = (
            self.format_value(metric, first, registry=registry),
            self.format_value(metric, second, registry=registry),
        )
        if rendered[0] != rendered[1]:
            return rendered
        try:
            a, b = float(first), float(second)
        except (TypeError, ValueError):
            return rendered
        if a == b:
            return rendered
        for extra in (1, 2):
            finer = replace(
                self,
                percent_decimals=self.percent_decimals + extra,
                decimals=self.decimals + extra,
            )
            rendered = (
                finer.format_value(metric, a, registry=registry),
                finer.format_value(metric, b, registry=registry),
            )
            if rendered[0] != rendered[1]:
                break
        return rendered

    def format_fraction(self, value: Any) -> str:
        """Render a bare proportion — a share of samples, a retained fraction.

        Separate from :meth:`format_value` because there is no metric to look up: the
        caller already knows the quantity is a proportion.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if self.as_percentage:
            return f"{number:.{self.percent_decimals}%}"
        return f"{number:.{self.decimals}f}"

    # -- ordering -----------------------------------------------------------

    def prefers_metric(self, metric: str, *, registry: Any = None) -> int:
        """A sort key for choosing *which* metric to lead with. Lower is preferred.

        Needed because a finding often spans several metrics and only one can go in the
        sentence. Picking the widest gap alone is wrong for a lay reader: on the breast
        cancer model the widest subgroup gap by proportion is in the negative
        log-likelihood, which is the correct answer to "which metric moved most" and a
        useless thing to tell a clinician. So a reader who works in percentages prefers a
        metric that *can* be one, and a reader with a vocabulary prefers a metric that
        vocabulary covers.
        """
        named = metric in self.vocabulary
        interpretable = not self.as_percentage or _is_unit_bounded(metric, registry)
        return (0 if interpretable else 1) * 2 + (0 if named else 1)

    def area_rank(self, area: str) -> int:
        """Where a finding area sits in this reader's ordering."""
        try:
            return self.leads_with.index(area)
        except ValueError:
            return len(self.leads_with)

    def order(self, findings: Sequence[Any]) -> list[Any]:
        """Findings in this reader's order: their areas first, then by severity.

        Severity still breaks ties *within* an area, so a critical finding never sits
        below an informational one from the same area. What the audience changes is which
        area a reader meets first, not whether severity is respected inside it.
        """
        severity_rank = {"critical": 0, "serious": 1, "warning": 2, "info": 3}
        return sorted(
            findings,
            key=lambda f: (
                self.area_rank(getattr(f, "area", "")),
                severity_rank.get(getattr(f, "severity", ""), 9),
            ),
        )


#: Words that open a clause rather than a noun phrase, so the phrase cannot take a
#: possessive and is used as the subject as it stands.
_CLAUSE_OPENERS = frozenset({"how", "whether", "what", "when", "a", "an"})


def _is_unit_bounded(metric: str, registry: Any = None) -> bool:
    """Whether the registry bounds ``metric`` on [0, 1], so a percentage is meaningful."""
    from themis_assess.metrics.registry import default_registry

    reg = registry if registry is not None else default_registry()
    try:
        spec = reg.get(metric)
    except KeyError:
        return False
    return spec.lower_bound == 0.0 and spec.upper_bound == 1.0


# --------------------------------------------------------------------------
# the built-in readers
# --------------------------------------------------------------------------

#: Someone who will act on the model's output about a person in front of them.
CLINICIAN = Audience(
    name="clinician",
    label="a clinician who will act on this model's output",
    as_percentage=True,
    percent_decimals=0,
    explains=True,
    tone=(
        "plain clinical English, no machine-learning jargon, no metric names unless "
        "they are explained in the same sentence"
    ),
    reads_for=(
        "deciding whether to trust this model's output about an individual patient, and "
        "knowing which cases it is least reliable on"
    ),
    leads_with=("classes", "subgroups", "calibration", "accuracy", "diagnostics"),
    vocabulary={
        "accuracy": "overall correctness",
        "balanced_accuracy": "correctness counting both diagnoses equally",
        "recall": "the share of positive cases it finds",
        "precision": "how often a positive call is right",
        "auc": "its ability to rank cases by risk",
        "auroc": "its ability to rank cases by risk",
        "expected_calibration_error": (
            "the gap between the confidence it states and how often it is actually right"
        ),
        "brier_score": "the overall quality of its stated probabilities",
        "negative_log_likelihood": "how badly its confident mistakes score",
        "relative_robustness": "how much of its correctness survives a change in the data",
        "flip_rate": "how often its decision changes when the measurements shift slightly",
        "confidence_retention": "whether it stays as confident when the data changes",
        "f1": "a combined measure of finding positives and being right about them",
        "matthews_corrcoef": "overall agreement with the true diagnosis",
    },
)

#: Someone who will change the model or the pipeline.
ML_ENGINEER = Audience(
    name="ml_engineer",
    label="an engineer who can change the model or the data pipeline",
    as_percentage=False,
    decimals=4,
    explains=False,
    tone="technical and terse; use metric names, they are what gets searched for",
    reads_for=(
        "deciding what to change next in the model, the features or the training data, "
        "and which measurements would settle an open question"
    ),
    leads_with=("robustness", "sensitivity", "attribution", "calibration", "accuracy"),
)

#: Someone who will decide whether this ships, and to whom.
MANAGER = Audience(
    name="manager",
    label="a decision-maker choosing whether to deploy this model",
    as_percentage=True,
    percent_decimals=0,
    explains=True,
    tone=(
        "short, concrete, consequence-first; no jargon; every paragraph should end in "
        "something that could be decided"
    ),
    reads_for=(
        "deciding whether this model can be deployed as it stands, what the residual "
        "risk is, and what it would cost to close the gaps"
    ),
    leads_with=("accuracy", "robustness", "subgroups", "reporting"),
)

#: Someone checking that the evaluation itself was adequate.
REGULATOR = Audience(
    name="regulator",
    label="a reviewer checking whether the evaluation itself was adequate",
    as_percentage=True,
    percent_decimals=1,
    explains=True,
    tone=(
        "precise and complete; state what was checked, against which threshold, and "
        "name explicitly anything that was not checked"
    ),
    reads_for=(
        "establishing what was measured, against which stated requirement, and where the "
        "evaluation is silent — an unchecked requirement matters as much as a failed one"
    ),
    leads_with=("reporting", "accuracy", "robustness", "subgroups"),
)

#: Every built-in reader, by name.
AUDIENCES: Mapping[str, Audience] = {
    a.name: a for a in (CLINICIAN, ML_ENGINEER, MANAGER, REGULATOR)
}


def get_audience(audience: str | Audience) -> Audience:
    """Resolve a name to an :class:`Audience`, or pass one straight through."""
    if isinstance(audience, Audience):
        return audience
    try:
        return AUDIENCES[audience]
    except KeyError:
        raise KeyError(
            f"Unknown audience {audience!r}. Built in: {sorted(AUDIENCES)}. "
            f"Construct an Audience directly for anyone else — it is a plain dataclass, "
            f"and the vocabulary is a partial mapping, so a new one need only name the "
            f"metrics whose usual names would not land."
        ) from None
