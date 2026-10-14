"""Result containers.

These exist to keep the report layer honest. The survey's reporting rules are all of the
form "you may not state X without also stating Y" — absolute performance beside any
relative drop, ``mRCE`` beside ``mCE``, per-group breakdowns beside an aggregate, the
severity curve rather than one severity. A result object that collapsed to a single
number the moment it was computed would make those rules unenforceable, so nothing here
throws away the clean value, the per-severity values, or the baseline.

The one thing the library cannot supply is the tolerance. Survey §8, on what a complete
robustness claim states: what is varied, which quantity is monitored, how the change is
measured, and how much change the application can tolerate —

    "The tolerance is the one element no evaluation instrument provides, as it must come
    from the requirements of the deployment itself."

:class:`RobustnessResult` therefore carries a ``tolerance`` slot that only the user can
fill, so its absence is visible rather than implicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from themis_assess.core.types import AccessLevel, TaskType

__all__ = [
    "EvaluationResult",
    "RobustnessResult",
    "severity_curve",
]


def severity_curve(values: Sequence[float]) -> dict[int, float]:
    """Turn a positional sequence into an explicit ``{severity: value}`` mapping.

    Severity is 1-based throughout the library — ``values[0]`` is severity 1 — and
    positional sequences are easy to misread. Use this wherever a curve is stored or
    reported rather than only computed.
    """
    if not len(values):
        raise ValueError("A severity curve needs at least one level.")
    return {s: float(v) for s, v in enumerate(values, start=1)}


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Metric values from one evaluation of one model on one dataset.

    The unit the robustness metrics are built from: a clean evaluation and a perturbed
    evaluation are both instances of this.

    Parameters
    ----------
    metrics:
        ``{metric_name: value}``.
    n_samples:
        How many samples produced these values. Needed to tell a difference from noise,
        and to spot a subgroup too small to compare.
    label:
        What was evaluated — ``"clean"``, ``"gaussian_noise@3"``, ``"site=oslo"``.
    task:
        Which task type this evaluation belongs to.
    metadata:
        Anything the report should be able to cite: the split protocol, the random seed,
        the perturbation's severity-to-parameter mapping.
    """

    metrics: Mapping[str, float]
    n_samples: int
    label: str = "clean"
    task: TaskType | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_samples < 0:
            raise ValueError(f"n_samples cannot be negative; got {self.n_samples}.")
        if not self.metrics:
            raise ValueError(f"EvaluationResult {self.label!r} carries no metrics.")

    def __getitem__(self, metric: str) -> float:
        try:
            return self.metrics[metric]
        except KeyError:
            raise KeyError(
                f"{metric!r} was not computed for {self.label!r}. "
                f"Available: {sorted(self.metrics)}."
            ) from None

    @property
    def metric_names(self) -> list[str]:
        return sorted(self.metrics)


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    """One robustness metric, with everything needed to report it responsibly.

    Parameters
    ----------
    metric_name:
        The robustness metric, e.g. ``"mean_corruption_error"``.
    accuracy_metric:
        The underlying accuracy or error metric it was computed from, e.g. ``"mae"``.
        Two robustness scores over different underlying metrics are not comparable, and
        this is what records which was used.
    clean:
        Performance on unperturbed data. Kept because most reporting rules require it:
        a small relative drop means nothing without knowing there was something to lose.
    per_perturbation:
        ``{perturbation_name: {severity: value}}`` — the underlying metric at each
        severity, not the robustness score. The curve is what a report shows.
    baseline:
        The same structure for a baseline model, where the metric needs one
        (``CE/mCE``, ``RCE/mRCE``, ``REI/mREI``, ``FR/mFR``). ``None`` otherwise.
    baseline_clean:
        The baseline's clean performance, needed by ``RCE`` and ``REI``.
    scores:
        ``{perturbation_name: robustness score}``.
    aggregate:
        The score across all perturbations, when one was computed.
    perturbation_set:
        Names of every perturbation in the set. Required for ``total_relative_robustness``,
        whose magnitude grows with the size of the set and which is therefore comparable
        only within a fixed one.
    access_used:
        The model access this actually required.
    tolerance:
        How much degradation the application accepts. Supplied by the user; ``None``
        means the claim is incomplete, and the report layer should say so.
    n_samples, seed, metadata:
        Provenance.
    """

    metric_name: str
    accuracy_metric: str
    clean: float
    per_perturbation: Mapping[str, Mapping[int, float]]
    scores: Mapping[str, float] = field(default_factory=dict)
    aggregate: float | None = None
    baseline: Mapping[str, Mapping[int, float]] | None = None
    baseline_clean: float | None = None
    perturbation_set: tuple[str, ...] = ()
    access_used: AccessLevel = AccessLevel.BLACK_BOX
    tolerance: float | None = None
    n_samples: int = 0
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.per_perturbation:
            raise ValueError(
                f"RobustnessResult for {self.metric_name!r} has no perturbations. "
                f"A robustness score with nothing perturbed is not a measurement."
            )
        for name, curve in self.per_perturbation.items():
            if not curve:
                raise ValueError(f"Perturbation {name!r} has an empty severity curve.")
            bad = [s for s in curve if s < 1]
            if bad:
                raise ValueError(
                    f"Severity levels are 1-based; perturbation {name!r} has {bad}."
                )
        if self.baseline is not None and set(self.baseline) != set(self.per_perturbation):
            only_model = sorted(set(self.per_perturbation) - set(self.baseline))
            only_base = sorted(set(self.baseline) - set(self.per_perturbation))
            raise ValueError(
                "Model and baseline must cover the same perturbations. "
                f"Only in model: {only_model or 'none'}. "
                f"Only in baseline: {only_base or 'none'}."
            )
        if not self.perturbation_set:
            object.__setattr__(self, "perturbation_set", tuple(self.per_perturbation))

    @property
    def severities(self) -> dict[str, list[int]]:
        """The severity levels present for each perturbation, in order."""
        return {name: sorted(curve) for name, curve in self.per_perturbation.items()}

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    @property
    def is_complete_claim(self) -> bool:
        """Whether this carries the four things survey §8 asks a robustness claim to state.

        The first three are structural and always present here — what was varied, what
        was monitored, how it was measured. The fourth, the tolerance, has to come from
        the deployment, so this is ``False`` until someone supplies it.
        """
        return self.tolerance is not None

    def curve(self, perturbation: str) -> dict[int, float]:
        """The severity curve for one perturbation."""
        try:
            return dict(self.per_perturbation[perturbation])
        except KeyError:
            raise KeyError(
                f"No perturbation {perturbation!r}. "
                f"Available: {sorted(self.per_perturbation)}."
            ) from None

    def worst_severity(self, perturbation: str, *, higher_is_worse: bool = True) -> int:
        """The severity at which the underlying metric was worst.

        ``higher_is_worse`` follows the underlying metric: ``True`` for an error,
        ``False`` for an accuracy.
        """
        curve = self.curve(perturbation)
        return max(curve, key=curve.get) if higher_is_worse else min(curve, key=curve.get)
