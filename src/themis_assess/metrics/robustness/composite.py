"""Composite robustness scores — survey §5.1.4.

Single numbers summarising several properties at once. Useful for ranking systems,
and dangerous on their own: the survey is explicit that aggregation hides failure
modes, so both functions here return the components alongside the score and the report
layer refuses to print one without the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "UnifiedRobustnessResult",
    "CompositeReliabilityResult",
    "unified_robustness_metric",
    "composite_reliability_score",
]

#: The five properties of survey Eq. 27, in order.
RELIABILITY_COMPONENTS = ("s_ID", "s_DS", "s_ADV", "s_CAL", "s_OOD")


@dataclass(frozen=True, slots=True)
class UnifiedRobustnessResult:
    """URM together with the parts it was built from."""

    score: float
    acdd: float
    sd: float
    lam: float

    def components(self) -> dict[str, float]:
        return {"ACDD": self.acdd, "SD": self.sd, "lambda": self.lam}


@dataclass(frozen=True, slots=True)
class CompositeReliabilityResult:
    """``s_HR`` together with its per-property scores and weights."""

    score: float
    scores: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def components(self) -> dict[str, float]:
        return dict(self.scores)


def unified_robustness_metric(
    acdd: float, sd: float, *, lam: float = 0.5
) -> UnifiedRobustnessResult:
    """Unified Robustness Metric. Survey §5.1.4, Eq. 26.

    ``URM = 1 - (λ·ACDD + (1-λ)·SD/100)``

    Parameters
    ----------
    acdd:
        Average cross-domain degradation — the relative drop in F1 across domain pairs.
    sd:
        Standard deviation of performance over all domain pairs, on a 0-100 scale
        (hence the division by 100 in the equation; pass it in the same units the
        source study used).
    lam:
        Weight on degradation versus variability, in ``[0, 1]``.

    Returns
    -------
    UnifiedRobustnessResult
        The score and its inputs. The survey requires the components and ``λ`` to be
        reported with the score, since the value depends on a weighting choice that is
        not itself a property of the model.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}.")
    score = 1.0 - (lam * float(acdd) + (1.0 - lam) * float(sd) / 100.0)
    return UnifiedRobustnessResult(score=float(score), acdd=float(acdd), sd=float(sd), lam=lam)


def composite_reliability_score(
    scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> CompositeReliabilityResult:
    """Weighted mean of per-property reliability scores. Survey §5.1.4, Eq. 27.

    ``s_HR = wᵀs``, with ``s = (s_ID, s_DS, s_ADV, s_CAL, s_OOD)`` — in-distribution
    accuracy, distribution-shift robustness, adversarial robustness, calibration and
    OOD detection.

    Parameters
    ----------
    scores:
        Property name to score, each already normalised to ``[0, 1]``. Use
        :func:`~themis_assess.metrics.normalize.normalize_metric` to get there.
        The canonical five are in :data:`RELIABILITY_COMPONENTS`, but any set is
        accepted so a project can drop a property it does not measure — adversarial
        robustness, say, which the survey places outside its own scope.
    weights:
        Optional per-property weights, normalised to sum to 1. Uniform if omitted, as
        in the original study.

    Returns
    -------
    CompositeReliabilityResult
        The score with its components and weights. The survey: a one-number summary
        over five properties "cannot show which property failed, so the per-property
        results should always accompany it."
    """
    if not scores:
        raise ValueError("composite_reliability_score needs at least one property score.")
    for name, value in scores.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Property {name!r} has score {value}, outside [0, 1]. Normalise each "
                f"property before combining — see themis_assess.metrics.normalize."
            )

    if weights is None:
        w = {name: 1.0 / len(scores) for name in scores}
    else:
        missing = set(scores) - set(weights)
        extra = set(weights) - set(scores)
        if missing or extra:
            raise ValueError(
                f"weights must cover exactly the properties in scores. "
                f"Missing weights: {sorted(missing) or 'none'}. "
                f"Unexpected weights: {sorted(extra) or 'none'}."
            )
        if any(v < 0 for v in weights.values()):
            raise ValueError("Weights must be non-negative.")
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Weights sum to zero.")
        w = {name: float(value) / total for name, value in weights.items()}

    score = sum(scores[name] * w[name] for name in scores)
    return CompositeReliabilityResult(
        score=float(score),
        scores={k: float(v) for k, v in scores.items()},
        weights=w,
    )
