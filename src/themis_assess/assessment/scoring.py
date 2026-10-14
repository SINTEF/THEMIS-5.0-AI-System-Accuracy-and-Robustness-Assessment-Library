"""Turning a sweep into a robustness score.

:class:`~themis_assess.assessment.robustness.SweepResult` holds the raw material — a clean
value, a severity curve per perturbation, optionally a baseline's. This module applies
one of the survey's §5.1.3 equations to it and returns a
:class:`~themis_assess.core.results.RobustnessResult`, which keeps the raw material
attached so the report can honour the "never state X without Y" rules.

The reason this is a separate module from the runner is cost. A sweep is thousands of
model calls; a score is arithmetic on the numbers it produced. Scoring the same sweep
under six different robustness metrics should not re-run the model, and separating them
means it cannot.

Which metrics need what
-----------------------

===============================  ==========  ==============  ==============
metric                           clean       severity curve  baseline
===============================  ==========  ==============  ==============
``relative_robustness``          yes         yes             no
``absolute_relative_robustness`` yes         yes             no
``total_relative_robustness``    yes         yes             no
``error_inflation``              yes         yes             no
``error_based_robustness``       yes         yes             no
``total_error_based_robustness`` yes         yes             no
``mean_relative_error_inflation``yes         yes             **yes**
``corruption_error``             no          yes             **yes**
``mean_corruption_error``        no          yes             **yes**
``relative_corruption_error``    yes         yes             **yes**
``mean_relative_corruption_error`` yes       yes             **yes**
===============================  ==========  ==============  ==============

Asking for one of the baseline metrics on a sweep that had no baseline raises, naming
which metric and why, rather than substituting the model for its own baseline — a
substitution that yields exactly 1.0 and looks like a result.
"""

from __future__ import annotations

from collections.abc import Mapping

from themis_assess.assessment.robustness import SweepResult
from themis_assess.core.results import RobustnessResult
from themis_assess.metrics.registry import MetricRegistry, default_registry
from themis_assess.metrics.robustness import degradation

__all__ = ["score", "score_all", "BASELINE_METRICS", "SUPPORTED_METRICS"]


BASELINE_METRICS = frozenset(
    {
        "corruption_error",
        "mean_corruption_error",
        "relative_corruption_error",
        "mean_relative_corruption_error",
        "relative_error_inflation",
        "mean_relative_error_inflation",
    }
)

#: Robustness metrics this module can compute from a sweep.
#:
#: ``relative_error_inflation`` (Eq. 17) is deliberately absent: it is defined per
#: (perturbation, severity) cell, and a :class:`RobustnessResult` carries one score per
#: perturbation. Its aggregate, ``mean_relative_error_inflation`` (Eq. 18), is here
#: instead. An earlier revision listed Eq. 17 as supported while the dispatch had no
#: branch for it, so it fell through to the corruption-error branch and returned that
#: metric's value under Eq. 17's name — a wrong number with a plausible magnitude.
SUPPORTED_METRICS = frozenset(
    {
        "relative_robustness",
        "absolute_relative_robustness",
        "total_relative_robustness",
        "error_inflation",
        "error_based_robustness",
        "total_error_based_robustness",
        "mean_relative_error_inflation",
        "corruption_error",
        "mean_corruption_error",
        "relative_corruption_error",
        "mean_relative_corruption_error",
    }
)


#: Equations written in terms of an *error* E: Eqs. 13-22. Handing them an accuracy
#: produces a number with no error at all — every ratio comes out inverted — so the
#: orientation is checked rather than assumed.
_ERROR_METRICS = frozenset(
    {
        "error_inflation",
        "error_based_robustness",
        "total_error_based_robustness",
        "relative_error_inflation",
        "mean_relative_error_inflation",
        "corruption_error",
        "mean_corruption_error",
        "relative_corruption_error",
        "mean_relative_corruption_error",
    }
)

#: Equations written in terms of an accuracy A: Eqs. 11-12.
_ACCURACY_METRICS = frozenset(
    {
        "relative_robustness",
        "absolute_relative_robustness",
        "total_relative_robustness",
    }
)


def _check_orientation(
    robustness_metric: str, accuracy_metric: str, higher_is_better: bool | None
) -> None:
    """Refuse to compute an error equation over an accuracy, or the reverse.

    The survey writes each §5.1.3 equation over one or the other, and the two are not
    interchangeable: ``CE`` over accuracies still produces a number, and that number is
    upside down — a model whose accuracy holds up scores *above* 1, which reads as less
    robust. Nothing downstream can detect this, because the registry knows the
    orientation and the arithmetic does not.

    A metric with no direction (``higher_is_better=None`` — the signed and
    distance-style ones) is refused outright: a ratio of two quantities whose best value
    is zero is not defined in any useful sense.
    """
    if higher_is_better is None:
        raise ValueError(
            f"{accuracy_metric!r} has no direction (its best value is a target, not a "
            f"maximum or a minimum), so a ratio-based robustness score over it would be "
            f"meaningless. Use an accuracy or an error metric."
        )
    if robustness_metric in _ERROR_METRICS and higher_is_better:
        raise ValueError(
            f"{robustness_metric!r} is defined over an error (survey §5.1.3), but "
            f"{accuracy_metric!r} is higher-is-better. Computing it anyway would invert "
            f"the result — a model that held its accuracy would score as less robust. "
            f"Use an error metric, or convert: for a bounded accuracy, 1 - accuracy."
        )
    if robustness_metric in _ACCURACY_METRICS and not higher_is_better:
        raise ValueError(
            f"{robustness_metric!r} is defined over an accuracy (survey §5.1.3, Eqs. "
            f"11-12), but {accuracy_metric!r} is lower-is-better. Use an accuracy "
            f"metric, or one of {', '.join(sorted(_ERROR_METRICS))} for an error."
        )


def _mean(curve: Mapping[int, float]) -> float:
    return float(sum(curve.values()) / len(curve))


def score(
    sweep: SweepResult,
    robustness_metric: str,
    accuracy_metric: str,
    *,
    registry: MetricRegistry | None = None,
    tolerance: float | None = None,
) -> RobustnessResult:
    """Apply one robustness equation to one accuracy metric's curves.

    Parameters
    ----------
    accuracy_metric:
        Which underlying metric the robustness score is computed over. Two robustness
        scores over different underlying metrics are not comparable, which is why
        :class:`~themis_assess.core.results.RobustnessResult` records it.
    tolerance:
        How much degradation the deployment accepts. The survey (§8) calls this the one
        element no evaluation instrument can supply; passing it here is what makes the
        resulting claim complete.
    """
    reg = registry if registry is not None else default_registry()
    if robustness_metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"{robustness_metric!r} cannot be computed from a sweep. Supported: "
            f"{', '.join(sorted(SUPPORTED_METRICS))}."
        )
    if accuracy_metric not in sweep.clean.metrics:
        raise KeyError(
            f"{accuracy_metric!r} was not computed in this sweep. Available: "
            f"{sweep.clean.metric_names}."
        )

    curves = sweep.curves(accuracy_metric)
    clean = sweep.clean[accuracy_metric]
    baseline_curves = sweep.baseline_curves(accuracy_metric)
    baseline_clean = (
        None if sweep.baseline_clean is None else sweep.baseline_clean[accuracy_metric]
    )

    if robustness_metric in BASELINE_METRICS and baseline_curves is None:
        raise ValueError(
            f"{robustness_metric!r} is defined relative to a baseline model (survey "
            f"§5.1.3), but this sweep ran without one. Pass baseline=<handler> to the "
            f"runner — themis_assess.handlers.baselines has naive ones — and re-run."
        )
    if robustness_metric in {
        "relative_corruption_error",
        "mean_relative_corruption_error",
    } and baseline_curves is not None:
        flat = [
            name
            for name, curve in baseline_curves.items()
            if all(abs(v - baseline_clean) < 1e-12 for v in curve.values())
        ]
        if flat:
            raise ValueError(
                f"{robustness_metric!r} normalises by how much the perturbation hurt "
                f"the baseline, and this baseline was unaffected by "
                f"{', '.join(sorted(flat))} — its excess error over clean is exactly "
                f"zero, so the ratio is undefined. A baseline that ignores its input "
                f"(a majority-class or prior-probability model) can serve CE and mCE, "
                f"which need only its level, but never RCE or mRCE, which need it to "
                f"degrade. Use a weaker trained model as the baseline instead."
            )
    if robustness_metric in BASELINE_METRICS and baseline_clean is None:
        raise ValueError(
            f"{robustness_metric!r} needs the baseline's clean {accuracy_metric}, which "
            f"this sweep did not record. The baseline's access level may have excluded "
            f"that metric."
        )

    spec = reg.get(accuracy_metric)
    higher_is_better = spec.higher_is_better
    _check_orientation(robustness_metric, accuracy_metric, higher_is_better)
    scores: dict[str, float] = {}
    aggregate: float | None = None

    if robustness_metric in {"relative_robustness", "absolute_relative_robustness"}:
        fn = (
            degradation.relative_robustness
            if robustness_metric == "relative_robustness"
            else degradation.absolute_relative_robustness
        )
        # Eq. 11 compares one perturbed performance against clean; a severity curve has
        # several, so the mean over severities is used and recorded in the metadata.
        # Reporting one severity would let the caller pick the flattering one.
        scores = {name: fn(clean, _mean(curve)) for name, curve in curves.items()}
        aggregate = (
            degradation.total_relative_robustness(scores)
            if robustness_metric == "relative_robustness"
            # The absolute form measures distance from 1 in either direction, so the
            # product Eq. 12 uses would be meaningless (its best value is 0, and a
            # product of near-zeros is not a summary). The mean is the aggregate.
            else float(sum(scores.values()) / len(scores))
        )

    elif robustness_metric == "total_relative_robustness":
        scores = {
            name: degradation.relative_robustness(clean, _mean(curve))
            for name, curve in curves.items()
        }
        aggregate = degradation.total_relative_robustness(scores)

    elif robustness_metric == "error_inflation":
        scores = {
            name: degradation.error_inflation(clean, _mean(curve))
            for name, curve in curves.items()
        }
        aggregate = float(sum(scores.values()) / len(scores))

    elif robustness_metric in {"error_based_robustness", "total_error_based_robustness"}:
        scores = {
            name: degradation.error_based_robustness(clean, curve)
            for name, curve in curves.items()
        }
        aggregate = degradation.total_error_based_robustness(scores)

    elif robustness_metric == "mean_relative_error_inflation":
        scores = {
            name: degradation.mean_relative_error_inflation(
                clean, {name: curve}, baseline_clean, {name: baseline_curves[name]}
            )
            for name, curve in curves.items()
        }
        aggregate = degradation.mean_relative_error_inflation(
            clean, curves, baseline_clean, baseline_curves
        )

    elif robustness_metric in {"corruption_error", "mean_corruption_error"}:
        scores = {
            name: degradation.corruption_error(curve, baseline_curves[name])
            for name, curve in curves.items()
        }
        aggregate = degradation.mean_corruption_error(curves, baseline_curves)

    elif robustness_metric in {
        "relative_corruption_error",
        "mean_relative_corruption_error",
    }:
        scores = {
            name: degradation.relative_corruption_error(
                clean, curve, baseline_clean, baseline_curves[name]
            )
            for name, curve in curves.items()
        }
        aggregate = degradation.mean_relative_corruption_error(
            clean, curves, baseline_clean, baseline_curves
        )

    else:  # pragma: no cover - unreachable while SUPPORTED_METRICS and the branches agree
        raise AssertionError(
            f"{robustness_metric!r} is listed in SUPPORTED_METRICS but has no branch "
            f"here. Falling through to another metric's branch would return a wrong "
            f"number under the right name."
        )

    return RobustnessResult(
        metric_name=robustness_metric,
        accuracy_metric=accuracy_metric,
        clean=clean,
        per_perturbation=curves,
        scores=scores,
        aggregate=aggregate,
        baseline=baseline_curves,
        baseline_clean=baseline_clean,
        perturbation_set=tuple(sorted(curves)),
        access_used=sweep.plan.access,
        tolerance=tolerance,
        n_samples=sweep.clean.n_samples,
        seed=sweep.seed,
        metadata={
            "higher_is_better": higher_is_better,
            "severities": sorted({s for curve in curves.values() for s in curve}),
            "aggregation_over_severity": "mean",
            **dict(sweep.metadata),
        },
    )


def score_all(
    sweep: SweepResult,
    accuracy_metric: str,
    *,
    registry: MetricRegistry | None = None,
    tolerance: float | None = None,
) -> dict[str, RobustnessResult]:
    """Every robustness metric the sweep supports, over one accuracy metric.

    Metrics needing a baseline are skipped silently when the sweep has none, as are the
    ones whose orientation does not match — an accuracy metric yields the Eq. 11-12
    family, an error metric the Eq. 13-22 family, and asking for all of them is asking
    for whichever apply. This is the convenience path; use :func:`score` to be told why
    a specific metric is unavailable.
    """
    reg = registry if registry is not None else default_registry()
    higher_is_better = reg.get(accuracy_metric).higher_is_better
    has_baseline = sweep.baseline_perturbed is not None
    out = {}
    for name in sorted(SUPPORTED_METRICS):
        if name in BASELINE_METRICS and not has_baseline:
            continue
        if name in _ERROR_METRICS and higher_is_better is not False:
            continue
        if name in _ACCURACY_METRICS and higher_is_better is not True:
            continue
        out[name] = score(
            sweep, name, accuracy_metric, registry=registry, tolerance=tolerance
        )
    return out
