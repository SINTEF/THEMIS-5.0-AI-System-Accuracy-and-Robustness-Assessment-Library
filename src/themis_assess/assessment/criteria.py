"""Acceptance criteria: what the deployment requires, as opposed to what the metric is.

All four repos carried a ``default_acceptable_metric_values`` dictionary —
``{"accuracy_score": 0.8, "matthews_corrcoef": 0.6, ...}`` — and ``get_metrics_table``
was their headline output. This module is that feature, with one change of principle.

**A threshold is a requirement, not a property of a metric.** 0.8 accuracy is acceptable
for a content-moderation triage queue and unacceptable for a clinical decision aid, and
nothing about the metric ``accuracy`` knows which. So thresholds live here, supplied per
assessment, rather than in :mod:`themis_assess.metrics.registry`, which holds only what is
true of a metric everywhere: its orientation, its bounds, which tasks it applies to.

This is the same principle already applied to the robustness side, where
:class:`~themis_assess.core.results.RobustnessResult` carries a ``tolerance`` the library
refuses to guess. Survey §8, on what a complete robustness claim states:

    "The tolerance is the one element no evaluation instrument provides, as it must come
    from the requirements of the deployment itself."

That sentence is about robustness, and it is just as true of accuracy. The consequence is
deliberate: a pass/fail verdict here is always traceable to a number somebody chose, and
:meth:`AcceptanceCriteria.provenance` records who. The four repos' own values ship as
:func:`themis_50_criteria` — an example to copy and edit, not a default that applies
itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from themis_assess.core.results import EvaluationResult
from themis_assess.metrics.registry import MetricRegistry, default_registry

__all__ = [
    "Requirement",
    "MetricVerdict",
    "AcceptanceCriteria",
    "AcceptanceReport",
    "themis_50_criteria",
]


@dataclass(frozen=True, slots=True)
class Requirement:
    """One metric's acceptable value, and where it came from.

    Parameters
    ----------
    metric:
        The metric name, as registered.
    acceptable:
        The threshold. Interpreted by the metric's own orientation, taken from the
        registry: at-or-above for a higher-is-better metric, at-or-below for an error.
        A caller never has to remember which way round a metric runs, which is what went
        wrong whenever these dictionaries were maintained by hand.
    tolerance:
        For a metric with no direction (best at a ``target``), how far from the target is
        acceptable. Required for those metrics and meaningless for the others.
    rationale:
        Why this number. Strongly encouraged: a threshold with no rationale is
        indistinguishable from a threshold someone copied, and it will be quoted in a
        report as though it were derived.
    critical:
        Whether failing this one fails the assessment. Not everything is a gate — a team
        may require accuracy above 0.9 while merely monitoring F1.
    """

    metric: str
    acceptable: float | None = None
    tolerance: float | None = None
    rationale: str = ""
    critical: bool = True

    def __post_init__(self) -> None:
        if self.acceptable is None and self.tolerance is None:
            raise ValueError(
                f"Requirement for {self.metric!r} sets neither an acceptable value nor a "
                f"tolerance, so nothing can be checked against it."
            )
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError(
                f"tolerance must be non-negative; got {self.tolerance} for "
                f"{self.metric!r}."
            )


@dataclass(frozen=True, slots=True)
class MetricVerdict:
    """One metric's measured value against its requirement.

    ``margin`` is the headroom in the metric's own units, signed so that positive is
    always good regardless of the metric's direction. That sign convention is the point:
    a table of margins can be read and sorted without checking each row's orientation,
    which a table of raw values cannot.
    """

    metric: str
    value: float
    requirement: Requirement | None
    passed: bool | None
    margin: float | None
    higher_is_better: bool | None
    target: float | None = None

    @property
    def has_requirement(self) -> bool:
        return self.requirement is not None

    @property
    def is_critical_failure(self) -> bool:
        return (
            self.passed is False
            and self.requirement is not None
            and self.requirement.critical
        )

    def describe(self) -> str:
        """A one-line human-readable summary, for a report or a log."""
        if self.requirement is None:
            return f"{self.metric}: {self.value:.4f} (no requirement)"
        verdict = "PASS" if self.passed else "FAIL"
        if self.requirement.tolerance is not None and self.target is not None:
            bound = f"within {self.requirement.tolerance:g} of {self.target:g}"
        else:
            direction = ">=" if self.higher_is_better else "<="
            bound = f"{direction} {self.requirement.acceptable:g}"
        return (
            f"{self.metric}: {self.value:.4f} {bound} -> {verdict} "
            f"(margin {self.margin:+.4f})"
        )


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    """Every metric's verdict, plus the overall outcome.

    Replaces ``get_metrics_table``, whose output was a DataFrame of value, baseline and
    range with no verdict in it — the reader had to compare each row against a dictionary
    kept somewhere else. Two of the four repos also built a formatted copy of that table
    and then returned the unformatted one, which is the kind of thing that happens when a
    presentation concern lives inside a computation.
    """

    verdicts: tuple[MetricVerdict, ...]
    label: str = "clean"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> MetricVerdict:
        for verdict in self.verdicts:
            if verdict.metric == metric:
                return verdict
        raise KeyError(
            f"{metric!r} has no verdict. Present: "
            f"{sorted(v.metric for v in self.verdicts)}."
        )

    @property
    def passed(self) -> bool | None:
        """Whether every critical requirement is met.

        ``None`` when no requirement was checked at all — which is *not* the same as
        passing, and is why this is three-valued. A report with no criteria attached
        must not be readable as a green light.
        """
        checked = [v for v in self.verdicts if v.has_requirement]
        if not checked:
            return None
        return not any(v.is_critical_failure for v in checked)

    @property
    def failures(self) -> list[MetricVerdict]:
        return [v for v in self.verdicts if v.passed is False]

    @property
    def unchecked(self) -> list[str]:
        """Metrics that were measured but carry no requirement."""
        return [v.metric for v in self.verdicts if not v.has_requirement]

    def summary(self) -> str:
        lines = [v.describe() for v in self.verdicts]
        outcome = {True: "PASS", False: "FAIL", None: "NO CRITERIA"}[self.passed]
        lines.append(f"overall: {outcome}")
        return "\n".join(lines)

    def to_frame(self) -> Any:
        """The metrics table, as a pandas DataFrame. Requires pandas (a core dependency).

        Columns: ``value``, ``acceptable``, ``passed``, ``margin``, ``critical``,
        ``rationale``. The verdict is *in* the table, which is the difference from the
        four repos' version.
        """
        import pandas as pd

        rows = []
        for verdict in self.verdicts:
            requirement = verdict.requirement
            rows.append(
                {
                    "value": verdict.value,
                    "acceptable": None if requirement is None else requirement.acceptable,
                    "passed": verdict.passed,
                    "margin": verdict.margin,
                    "critical": None if requirement is None else requirement.critical,
                    "rationale": "" if requirement is None else requirement.rationale,
                }
            )
        return pd.DataFrame(rows, index=[v.metric for v in self.verdicts])


class AcceptanceCriteria:
    """The deployment's requirements, checkable against an evaluation.

    Parameters
    ----------
    requirements:
        Either :class:`Requirement` objects, or a plain ``{metric: acceptable}`` mapping
        for the common case. The mapping form is accepted because it is what the four
        repos used and what a user will reach for first; it produces requirements with no
        rationale, and :meth:`check` says so.
    source:
        Where these numbers come from — a requirements document, a regulatory threshold,
        a team decision and its date. Carried into every report so a verdict is always
        traceable.
    registry:
        Supplies each metric's orientation, so a caller never specifies a direction and
        the two can never disagree.
    """

    def __init__(
        self,
        requirements: Iterable[Requirement] | Mapping[str, float],
        *,
        source: str = "",
        registry: MetricRegistry | None = None,
    ) -> None:
        self.registry = registry if registry is not None else default_registry()
        self.source = source

        if isinstance(requirements, Mapping):
            items = [
                Requirement(metric=name, acceptable=float(value))
                for name, value in requirements.items()
            ]
        else:
            items = list(requirements)
        if not items:
            raise ValueError(
                "AcceptanceCriteria needs at least one requirement. An empty set of "
                "criteria would report every assessment as passing."
            )

        seen: set[str] = set()
        for requirement in items:
            if requirement.metric in seen:
                raise ValueError(f"Duplicate requirement for {requirement.metric!r}.")
            seen.add(requirement.metric)
            spec = self.registry.get(requirement.metric)  # raises on an unknown metric
            if spec.higher_is_better is None and requirement.tolerance is None:
                raise ValueError(
                    f"{requirement.metric!r} has no direction — it is best at "
                    f"target={spec.target}, and worse in either direction — so an "
                    f"acceptable value alone cannot be checked. Set tolerance= instead, "
                    f"to say how far from the target is acceptable."
                )
            if spec.higher_is_better is not None and requirement.acceptable is None:
                raise ValueError(
                    f"{requirement.metric!r} is a "
                    f"{'higher' if spec.higher_is_better else 'lower'}-is-better metric, "
                    f"so it needs acceptable=, not tolerance=."
                )

        self._requirements = {r.metric: r for r in items}

    # -- inspection --------------------------------------------------------------

    def __contains__(self, metric: object) -> bool:
        return metric in self._requirements

    def __len__(self) -> int:
        return len(self._requirements)

    def metrics(self) -> list[str]:
        return sorted(self._requirements)

    def get(self, metric: str) -> Requirement:
        try:
            return self._requirements[metric]
        except KeyError:
            raise KeyError(
                f"No requirement for {metric!r}. Present: {self.metrics()}."
            ) from None

    def provenance(self) -> dict[str, Any]:
        """Where these thresholds came from, for the report."""
        return {
            "source": self.source or "unspecified",
            "n_requirements": len(self._requirements),
            "critical": sorted(m for m, r in self._requirements.items() if r.critical),
            "without_rationale": sorted(
                m for m, r in self._requirements.items() if not r.rationale
            ),
        }

    # -- checking ----------------------------------------------------------------

    def verdict(self, metric: str, value: float) -> MetricVerdict:
        """One metric's verdict.

        The direction comes from the registry, never from the caller. ``margin`` is
        signed so positive is good whichever way the metric runs.
        """
        spec = self.registry.get(metric)
        requirement = self._requirements.get(metric)
        if requirement is None:
            return MetricVerdict(
                metric=metric,
                value=float(value),
                requirement=None,
                passed=None,
                margin=None,
                higher_is_better=spec.higher_is_better,
                target=spec.target,
            )

        if requirement.tolerance is not None:
            if spec.target is None:
                raise ValueError(
                    f"{metric!r} has a tolerance requirement but no registered target, "
                    f"so there is nothing to measure the tolerance from."
                )
            distance = abs(float(value) - spec.target)
            margin = requirement.tolerance - distance
        elif spec.higher_is_better:
            margin = float(value) - requirement.acceptable
        else:
            margin = requirement.acceptable - float(value)

        return MetricVerdict(
            metric=metric,
            value=float(value),
            requirement=requirement,
            passed=bool(margin >= 0),
            margin=float(margin),
            higher_is_better=spec.higher_is_better,
            target=spec.target,
        )

    def check(
        self, result: EvaluationResult | Mapping[str, float], *, label: str | None = None
    ) -> AcceptanceReport:
        """Every measured metric's verdict, plus the overall outcome.

        Metrics that were measured but carry no requirement appear in the report with
        ``passed=None`` rather than being dropped: a metrics table that silently omitted
        them would read as though everything shown had been vetted.

        A requirement whose metric was *not* measured is reported in the provenance under
        ``not_measured``. It cannot be a pass and it is not a failure of the model — it is
        a gap in the evaluation, and it should look like one.
        """
        if isinstance(result, EvaluationResult):
            values = dict(result.metrics)
            label = label if label is not None else result.label
        else:
            values = dict(result)
            label = label if label is not None else "clean"

        verdicts = tuple(
            self.verdict(name, value) for name, value in sorted(values.items())
        )
        not_measured = sorted(set(self._requirements) - set(values))
        return AcceptanceReport(
            verdicts=verdicts,
            label=label,
            provenance={**self.provenance(), "not_measured": not_measured},
        )


def themis_50_criteria(*, registry: MetricRegistry | None = None) -> AcceptanceCriteria:
    """The thresholds the four repos actually used, as a documented example.

    ``accuracy``, ``balanced_accuracy``, ``precision`` and ``recall`` at 0.8;
    ``matthews_corrcoef`` at 0.6. Repo C additionally listed ``f1`` in its ranges but set
    no acceptable value for it, so none is set here.

    **Copy this and edit it.** It is a function rather than a module constant so that it
    cannot be mistaken for a default that applies itself, and its ``source`` says exactly
    what it is: numbers inherited from three use-case repositories, with no recorded
    derivation. They are a reasonable starting point for a discussion with whoever owns
    the deployment, and no substitute for that discussion.
    """
    inherited = (
        "themis_media / themis_medical / themis-accuracy-robustness "
        "default_acceptable_metric_values; no recorded derivation"
    )
    return AcceptanceCriteria(
        [
            Requirement("accuracy", acceptable=0.8, rationale=inherited),
            Requirement("balanced_accuracy", acceptable=0.8, rationale=inherited),
            Requirement("precision", acceptable=0.8, rationale=inherited),
            Requirement("recall", acceptable=0.8, rationale=inherited),
            Requirement("matthews_corrcoef", acceptable=0.6, rationale=inherited),
        ],
        source=f"example criteria inherited from the use-case repos ({inherited})",
        registry=registry,
    )
