"""Robustness tolerances: how much degradation the deployment accepts.

The counterpart of :mod:`themis_assess.assessment.criteria`, and it exists for the same
reason. Survey §8, on what a complete robustness claim states — what is varied, what is
monitored, how the change is measured, and:

    "The tolerance is the one element no evaluation instrument provides, as it must come
    from the requirements of the deployment itself."

:class:`~themis_assess.core.results.RobustnessResult` has carried a ``tolerance`` slot
since tranche 2, precisely so its absence would be visible. This module is what fills it,
and what turns a filled slot into a verdict.

Why a tolerance is not just another acceptance threshold
--------------------------------------------------------

An accuracy criterion is checked against one number. A robustness tolerance has to be
checked against a *shape*, and there are three defensible readings of "acceptable
degradation" which disagree:

* the **aggregate** — mean relative robustness across the perturbation set, which is what
  a single headline figure would report, and which averages away a single catastrophic
  perturbation;
* the **worst perturbation** — the minimum across the set, which is what an adversary or
  an unlucky deployment gets;
* the **worst cell** — the minimum across perturbation *and* severity, which is the most
  conservative and the one survey §8 asks for beside the average.

All three are available and the choice is explicit, because a claim of "robust within
tolerance" means different things under each and reporting only the flattering one is the
failure mode this module exists to prevent. The default is ``"worst_perturbation"``: the
aggregate is too easy to pass, and the worst cell is dominated by severity 5, which is
often outside anything the deployment will see.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from themis_assess.core.results import RobustnessResult

__all__ = [
    "ToleranceBasis",
    "ToleranceRequirement",
    "ToleranceVerdict",
    "RobustnessCriteria",
    "RobustnessReport",
]

ToleranceBasis = Literal["aggregate", "worst_perturbation", "worst_cell"]
_BASES: tuple[str, ...] = ("aggregate", "worst_perturbation", "worst_cell")


@dataclass(frozen=True, slots=True)
class ToleranceRequirement:
    """One robustness metric's acceptable value, and how it is to be read.

    Parameters
    ----------
    metric:
        A robustness metric name — ``"relative_robustness"``,
        ``"mean_corruption_error"``, and so on.
    acceptable:
        The tolerated value. Read by the metric's own orientation, taken from the
        registry, exactly as in
        :class:`~themis_assess.assessment.criteria.AcceptanceCriteria`: at-or-above for
        a higher-is-better metric, at-or-below for one where lower is better. For a
        metric best at a ``target`` (``relative_robustness`` is best at 1.0) this is
        still a one-sided bar, because degradation is the direction anyone worries about
        — a perturbation that *improves* the model is a curiosity, not a failure.
    basis:
        Which reading of the shape to check. See the module docstring.
    rationale:
        Why this number. A tolerance without one will be quoted in a report as though it
        were derived from something.
    critical:
        Whether failing this one fails the assessment.
    """

    metric: str
    acceptable: float
    basis: ToleranceBasis = "worst_perturbation"
    rationale: str = ""
    critical: bool = True

    def __post_init__(self) -> None:
        if self.basis not in _BASES:
            raise ValueError(
                f"basis must be one of {_BASES}; got {self.basis!r}. Each answers a "
                f"different question, so there is no sensible fallback."
            )


@dataclass(frozen=True, slots=True)
class ToleranceVerdict:
    """One robustness metric checked against its tolerance.

    ``worst_at`` names the perturbation, and where the basis is ``"worst_cell"`` the
    severity too, that produced the checked value. It is the single most useful field
    here: a failed robustness claim is actionable only if you know what broke it.
    """

    metric: str
    accuracy_metric: str
    value: float
    requirement: ToleranceRequirement
    passed: bool
    margin: float
    basis: ToleranceBasis
    higher_is_better: bool
    worst_at: str | None = None
    aggregate: float | None = None

    @property
    def is_critical_failure(self) -> bool:
        return not self.passed and self.requirement.critical

    def describe(self) -> str:
        direction = ">=" if self.higher_is_better else "<="
        verdict = "PASS" if self.passed else "FAIL"
        where = f" at {self.worst_at}" if self.worst_at else ""
        return (
            f"{self.metric} over {self.accuracy_metric} ({self.basis}): "
            f"{self.value:.4f} {direction} {self.requirement.acceptable:g} -> {verdict} "
            f"(margin {self.margin:+.4f}){where}"
        )


@dataclass(frozen=True, slots=True)
class RobustnessReport:
    """Every tolerance verdict, plus the overall outcome."""

    verdicts: tuple[ToleranceVerdict, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> ToleranceVerdict:
        for verdict in self.verdicts:
            if verdict.metric == metric:
                return verdict
        raise KeyError(
            f"{metric!r} has no verdict. Present: "
            f"{sorted(v.metric for v in self.verdicts)}."
        )

    @property
    def passed(self) -> bool | None:
        """Whether every critical tolerance is met. ``None`` when none were checked.

        Three-valued for the same reason as the accuracy side: a robustness assessment
        with no tolerance attached is an *incomplete claim* under survey §8, and must not
        be readable as a green light.
        """
        if not self.verdicts:
            return None
        return not any(v.is_critical_failure for v in self.verdicts)

    @property
    def failures(self) -> list[ToleranceVerdict]:
        return [v for v in self.verdicts if not v.passed]

    @property
    def unchecked(self) -> list[str]:
        """Tolerances whose metric was never scored, so nothing was checked against them.

        Not a pass and not a model failure — a gap in the evaluation. Surfaced because
        "overall: PASS" beside an unchecked *critical* tolerance is the most misleading
        thing this report could say.
        """
        return list(self.provenance.get("not_scored", ()))

    def summary(self) -> str:
        lines = [v.describe() for v in self.verdicts]
        outcome = {True: "PASS", False: "FAIL", None: "NO TOLERANCE"}[self.passed]
        lines.append(f"overall: {outcome}")
        if self.unchecked:
            lines.append(
                f"  NOT CHECKED: {', '.join(self.unchecked)} — the metric was not "
                f"scored, so this outcome does not cover it"
            )
        return "\n".join(lines)

    def to_frame(self) -> Any:
        """The tolerance table, as a pandas DataFrame."""
        import pandas as pd

        rows = [
            {
                "accuracy_metric": v.accuracy_metric,
                "basis": v.basis,
                "value": v.value,
                "acceptable": v.requirement.acceptable,
                "passed": v.passed,
                "margin": v.margin,
                "worst_at": v.worst_at,
                "critical": v.requirement.critical,
                "rationale": v.requirement.rationale,
            }
            for v in self.verdicts
        ]
        return pd.DataFrame(rows, index=[v.metric for v in self.verdicts])


class RobustnessCriteria:
    """The deployment's robustness tolerances, checkable against scored results.

    Parameters
    ----------
    requirements:
        :class:`ToleranceRequirement` objects, or a ``{metric: acceptable}`` mapping for
        the common case, which takes the default basis.
    source:
        Where these numbers come from. Carried into every report.

    Examples
    --------
    "The model must retain at least 85% of its clean accuracy under every perturbation
    we test, and its corruption error must not exceed the naive baseline's by more than
    20%":

    >>> RobustnessCriteria([
    ...     ToleranceRequirement(
    ...         "relative_robustness", acceptable=0.85,
    ...         basis="worst_perturbation", rationale="ops sign-off 2026-04",
    ...     ),
    ...     ToleranceRequirement(
    ...         "mean_corruption_error", acceptable=1.2, basis="aggregate",
    ...         rationale="must not be materially worse than the naive forecaster",
    ...     ),
    ... ], source="deployment requirements v3")  # doctest: +ELLIPSIS
    <...RobustnessCriteria object at ...>
    """

    def __init__(
        self,
        requirements: Iterable[ToleranceRequirement] | Mapping[str, float],
        *,
        source: str = "",
    ) -> None:
        if isinstance(requirements, Mapping):
            items = [
                ToleranceRequirement(metric=name, acceptable=float(value))
                for name, value in requirements.items()
            ]
        else:
            items = list(requirements)
        if not items:
            raise ValueError(
                "RobustnessCriteria needs at least one tolerance. An empty set would "
                "report every assessment as within tolerance."
            )
        seen: set[str] = set()
        for requirement in items:
            if requirement.metric in seen:
                raise ValueError(f"Duplicate tolerance for {requirement.metric!r}.")
            seen.add(requirement.metric)
        self._requirements = {r.metric: r for r in items}
        self.source = source

    def __contains__(self, metric: object) -> bool:
        return metric in self._requirements

    def __len__(self) -> int:
        return len(self._requirements)

    def metrics(self) -> list[str]:
        return sorted(self._requirements)

    def get(self, metric: str) -> ToleranceRequirement:
        try:
            return self._requirements[metric]
        except KeyError:
            raise KeyError(
                f"No tolerance for {metric!r}. Present: {self.metrics()}."
            ) from None

    def provenance(self) -> dict[str, Any]:
        return {
            "source": self.source or "unspecified",
            "n_tolerances": len(self._requirements),
            "bases": {m: r.basis for m, r in self._requirements.items()},
            "without_rationale": sorted(
                m for m, r in self._requirements.items() if not r.rationale
            ),
        }

    # -- checking ----------------------------------------------------------------

    def _checked_value(
        self, result: RobustnessResult, basis: ToleranceBasis, higher_is_better: bool
    ) -> tuple[float, str | None]:
        """The value this basis checks, and where it came from."""
        if basis == "aggregate":
            if result.aggregate is None:
                raise ValueError(
                    f"{result.metric_name!r} has no aggregate, so basis='aggregate' "
                    f"cannot be checked. Use 'worst_perturbation' or 'worst_cell'."
                )
            return float(result.aggregate), None

        if basis == "worst_perturbation":
            if not result.scores:
                raise ValueError(
                    f"{result.metric_name!r} has no per-perturbation scores, so "
                    f"basis='worst_perturbation' cannot be checked."
                )
            picker = min if higher_is_better else max
            name = picker(result.scores, key=result.scores.get)
            return float(result.scores[name]), name

        # worst_cell: over the underlying metric's curves, not the robustness score,
        # because a score is defined per perturbation and a cell is finer than that.
        cells = [
            (perturbation, severity, value)
            for perturbation, curve in result.per_perturbation.items()
            for severity, value in curve.items()
        ]
        # The underlying metric's direction is the opposite question from the robustness
        # score's: a *worse* underlying value is a lower accuracy or a higher error.
        underlying_higher_is_better = result.metadata.get("higher_is_better")
        if underlying_higher_is_better is None:
            raise ValueError(
                f"{result.metric_name!r} does not record the underlying metric's "
                f"direction, so the worst cell cannot be identified. Score it through "
                f"themis_assess.assessment.scoring, which records it."
            )
        picker = min if underlying_higher_is_better else max
        perturbation, severity, value = picker(cells, key=lambda c: c[2])
        if not result.clean:
            raise ValueError(
                f"The clean {result.accuracy_metric} is zero, so a worst-cell ratio to "
                f"it is undefined. Use basis='worst_perturbation', which reads the "
                f"scored value directly."
            )
        # Expressed as a ratio to clean, which puts it on the robustness metric's own
        # scale. The same ratio serves both directions: for an accuracy it is Eq. 11's
        # A_perturbed / A_clean, and for an error it is Eq. 16's E_perturbed / E_clean.
        # Only *which cell is worst* differs between them, and that is the picker above.
        return float(value / result.clean), f"{perturbation}@{severity}"

    def verdict(
        self, result: RobustnessResult, *, higher_is_better: bool | None = None
    ) -> ToleranceVerdict:
        """Check one scored robustness result against its tolerance."""
        requirement = self.get(result.metric_name)
        if higher_is_better is None:
            from themis_assess.metrics.registry import default_registry

            spec = default_registry().get(result.metric_name)
            # A robustness metric best at a target (relative_robustness at 1.0) is still
            # checked one-sided: degradation is the direction anyone worries about.
            higher_is_better = (
                spec.higher_is_better if spec.higher_is_better is not None else True
            )

        value, worst_at = self._checked_value(
            result, requirement.basis, bool(higher_is_better)
        )
        margin = (
            value - requirement.acceptable
            if higher_is_better
            else requirement.acceptable - value
        )
        return ToleranceVerdict(
            metric=result.metric_name,
            accuracy_metric=result.accuracy_metric,
            value=value,
            requirement=requirement,
            passed=bool(margin >= 0),
            margin=float(margin),
            basis=requirement.basis,
            higher_is_better=bool(higher_is_better),
            worst_at=worst_at,
            aggregate=result.aggregate,
        )

    def check(self, results: Mapping[str, RobustnessResult]) -> RobustnessReport:
        """Check every scored result that has a tolerance.

        A tolerance whose metric was not scored appears in the provenance under
        ``not_scored``. As on the accuracy side, that is a gap in the evaluation rather
        than a model failure, and it should look like one.
        """
        verdicts = tuple(
            self.verdict(result)
            for name, result in sorted(results.items())
            if name in self._requirements
        )
        return RobustnessReport(
            verdicts=verdicts,
            provenance={
                **self.provenance(),
                "not_scored": sorted(set(self._requirements) - set(results)),
            },
        )
