"""Performance-degradation metrics — survey §5.1.1 and §5.1.3.

Every function here is a pure function of numbers: no model, no data, no I/O. That is
what lets classification and forecasting share one implementation, and it is what the
four divergent ``compute_robustness`` implementations across repos A, B and C lacked.

Two conventions used throughout:

* **Severity curves.** A sequence indexed ``s = 1..S``; ``curve[s - 1]`` is the value
  at severity ``s``. Ordering matters — the survey's aggregate metrics sum over
  severities, and a report is expected to show the curve, not just its summary.
* **``A`` versus ``E``.** Following the survey's notation, ``a_*`` arguments are
  accuracy-type scores where higher is better, and ``e_*`` arguments are errors where
  lower is better. Passing an error where an accuracy is expected silently inverts the
  interpretation, so the argument names are explicit about which is which.

Reporting rules the survey attaches to these metrics (enforced later by ``report/``):
absolute performance must accompany any relative-drop metric; ``mCE`` alone is not
evidence of robustness without ``mRCE``; and ``total_relative_robustness`` is
comparable only across models evaluated on the same perturbation set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from themis_assess.core.errors import safe_divide
from themis_assess.core.types import PerturbationCurves, SeverityCurve

__all__ = [
    # §5.1.1
    "macro_error",
    "accuracy_drop",
    "relative_accuracy_drop",
    "error_ratio",
    # §5.1.3, accuracy-based
    "relative_robustness",
    "total_relative_robustness",
    "absolute_relative_robustness",
    # §5.1.3, error-based
    "relative_error_ratio",
    "error_based_robustness",
    "total_error_based_robustness",
    "error_inflation",
    "relative_error_inflation",
    "mean_relative_error_inflation",
    # §5.1.3, corruption error
    "corruption_error",
    "mean_corruption_error",
    "relative_corruption_error",
    "mean_relative_corruption_error",
]

#: Stabiliser for Equation 13, the one place the survey introduces one.
DEFAULT_EPS = 1e-6


def _as_list(curve: SeverityCurve, name: str) -> list[float]:
    """A severity curve as an ordered list of values.

    ``SeverityCurve`` is ``Mapping[int, float] | Sequence[float]``, and the two need
    different handling — iterating a mapping yields its *keys*. An earlier revision did
    exactly that, so every curve passed as ``{severity: value}`` silently became
    ``[1, 2, 3, 4, 5]``, identical for model and baseline, and ``CE`` came out at
    exactly 1.0: a plausible-looking "matches the baseline" rather than an error. The
    first end-to-end run found it.

    Mappings are sorted by severity, not left in insertion order: a curve assembled out
    of order would otherwise pair the model's severity 3 with the baseline's severity 1.
    """
    if isinstance(curve, Mapping):
        values = [float(curve[key]) for key in sorted(curve)]
    else:
        values = [float(v) for v in curve]
    if not values:
        raise ValueError(f"{name} is empty; a severity curve needs at least one level.")
    return values


def _check_aligned(a: Mapping[str, object], b: Mapping[str, object]) -> None:
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        raise ValueError(
            "Model and baseline must cover the same perturbations. "
            f"Only in model: {only_a or 'none'}. Only in baseline: {only_b or 'none'}."
        )


# --------------------------------------------------------------------------
# §5.1.1 — domain-level shift
# --------------------------------------------------------------------------


def macro_error(errors: Mapping[str, float] | Sequence[float]) -> float:
    """Unweighted mean error across domains. Survey §5.1.1, Eq. 2.

    ``Ē_macro = (1/|K|) Σ_k E_k``

    Unweighted on purpose: it gives every domain equal say regardless of how many
    samples it contributed, which is the point of reporting per-domain results at all.
    """
    values = list(errors.values()) if isinstance(errors, Mapping) else list(errors)
    if not values:
        raise ValueError("macro_error needs at least one domain.")
    return float(sum(values) / len(values))


def accuracy_drop(a_id: float, a_ood: float) -> float:
    """Absolute drop from in-distribution to shifted. Survey §5.1.1, Eq. 3.

    ``AccDrop = A_ID - A_OOD``. Lower is better; negative means the model did *better*
    under shift.

    Report ``a_id`` alongside this: a model with poor in-distribution accuracy can show
    a small drop simply because there was little left to lose.
    """
    return float(a_id) - float(a_ood)


def relative_accuracy_drop(a_id: float, a_ood: float) -> float:
    """Drop as a fraction of in-distribution accuracy. Survey §5.1.1, Eq. 4.

    ``RelAccDrop = (A_ID - A_OOD) / A_ID``
    """
    return safe_divide(float(a_id) - float(a_ood), float(a_id), what="in-distribution accuracy")


def error_ratio(e_id: float, e_ood: float) -> float:
    """Shifted error relative to in-distribution error. Survey §5.1.1, Eq. 5.

    ``ErrRatio = E_OOD / E_ID``. Best value is 1.

    The survey's caution applies directly: a value near one is consistent both with a
    robust model and with a model that fails equally on both splits, so absolute
    performance must be reported alongside it.
    """
    return safe_divide(float(e_ood), float(e_id), what="in-distribution error")


# --------------------------------------------------------------------------
# §5.1.3 — synthetic perturbations, accuracy-based
# --------------------------------------------------------------------------


def relative_robustness(a_clean: float, a_perturbed: float) -> float:
    """Perturbed accuracy as a fraction of clean accuracy. Survey §5.1.3, Eq. 11.

    ``R_φ = A_φ / A_clean``. Approximately 1 means high robustness; below 1 is
    degradation; above 1 (rare) means the perturbation helped.

    This is the metric that existed five times across repos A, B, C and D under three
    different names — ``rel_robustness_metric``, ``robustness_ratio``, and twice
    inline. This is the canonical implementation.
    """
    return safe_divide(float(a_perturbed), float(a_clean), what="clean accuracy")


def total_relative_robustness(per_perturbation: Mapping[str, float] | Sequence[float]) -> float:
    """Sum of per-perturbation robustness. Survey §5.1.3, Eq. 12.

    ``R_Φ = Σ_φ R_φ``

    Its magnitude grows with the number of perturbation types, so it is comparable only
    across models evaluated on the *same* set ``Φ``. Any report citing it must name the
    set.
    """
    values = (
        list(per_perturbation.values())
        if isinstance(per_perturbation, Mapping)
        else list(per_perturbation)
    )
    if not values:
        raise ValueError("total_relative_robustness needs at least one perturbation.")
    return float(sum(values))


def absolute_relative_robustness(a_clean: float, a_perturbed: float) -> float:
    """``|1 - R_φ|``. Repo-A extension, not in the survey.

    Zero is best. Unlike Eq. 11 this does not distinguish degradation from improvement,
    which is the point: it answers "how much did the perturbation move the metric",
    not "in which direction". Useful when aggregating across perturbations whose
    effects could cancel.

    Carried over from ``BaseRobustnessAssessment.abs_rel_robustness_metric``.
    """
    return abs(1.0 - relative_robustness(a_clean, a_perturbed))


# --------------------------------------------------------------------------
# §5.1.3 — synthetic perturbations, error-based
# --------------------------------------------------------------------------


def relative_error_ratio(
    e_clean: float, e_perturbed: float, *, eps: float = DEFAULT_EPS
) -> float:
    """Clean error over perturbed error. Survey §5.1.3, Eq. 13.

    ``E_rel,φ,s = (E_clean + ε) / (E_{s,φ} + ε)``

    Note the direction: clean over perturbed, so higher is better and 1 is unchanged.
    This is the one equation where the survey itself introduces ``ε``, which is why it
    is the only function here with a non-zero default.
    """
    return safe_divide(float(e_clean), float(e_perturbed), eps=eps, what="perturbed error")


def error_based_robustness(
    e_clean: float, perturbed_errors: SeverityCurve, *, eps: float = DEFAULT_EPS
) -> float:
    """Mean relative error ratio across severities. Survey §5.1.3, Eq. 14.

    ``R_φ = (1/S) Σ_s E_rel,φ,s``

    ``perturbed_errors`` is the severity curve for one perturbation type, indexed
    ``s = 1..S``.
    """
    ratios = [
        relative_error_ratio(e_clean, e, eps=eps)
        for e in _as_list(perturbed_errors, "perturbed_errors")
    ]
    return float(sum(ratios) / len(ratios))


def total_error_based_robustness(
    per_perturbation: Mapping[str, float] | Sequence[float],
) -> float:
    """Product of per-perturbation robustness values. Survey §5.1.3, Eq. 15.

    ``R_Φ = Π_φ R_φ``

    Multiplicative aggregation makes this a strict criterion: one weak perturbation
    substantially reduces the score, where a sum would let strong results elsewhere
    mask it.
    """
    values = (
        list(per_perturbation.values())
        if isinstance(per_perturbation, Mapping)
        else list(per_perturbation)
    )
    if not values:
        raise ValueError("total_error_based_robustness needs at least one perturbation.")
    return float(math.prod(values))


def error_inflation(e_clean: float, e_perturbed: float) -> float:
    """How much the error grew under perturbation. Survey §5.1.3, Eq. 16.

    ``EI_{s,φ} = E_{s,φ} / E_clean``. Lower is better; 1 means unchanged.

    The perturbation-space counterpart of :func:`error_ratio`, and subject to the same
    caveat — it must be reported alongside the clean error, which is what distinguishes
    "robust" from "uniformly bad".
    """
    return safe_divide(float(e_perturbed), float(e_clean), what="clean error")


def relative_error_inflation(
    e_clean: float,
    e_perturbed: float,
    e_clean_baseline: float,
    e_perturbed_baseline: float,
) -> float:
    """Error inflation in excess of a baseline model's. Survey §5.1.3, Eq. 17.

    ``REI_{s,φ} = EI^f_{s,φ} - EI^base_{s,φ}``

    Subtracting the baseline's inflation controls for how hard the perturbation is:
    a perturbation that doubles everyone's error scores 0 here, not 1. Zero is best.
    """
    return error_inflation(e_clean, e_perturbed) - error_inflation(
        e_clean_baseline, e_perturbed_baseline
    )


def mean_relative_error_inflation(
    e_clean: float,
    perturbed_errors: PerturbationCurves,
    e_clean_baseline: float,
    perturbed_errors_baseline: PerturbationCurves,
) -> float:
    """Mean REI over every perturbation type and severity. Survey §5.1.3, Eq. 18.

    ``mREI = 1/(N·S) Σ_i Σ_s REI_{s,φ_i}``

    Averaged over the flattened (type, severity) grid, so perturbation types with more
    severity levels carry proportionally more weight — which is the survey's definition,
    not an oversight here. Keep ``S`` equal across types (see the ``S = 5`` convention
    in ``docs/DECISIONS.md``, D16) and the two readings coincide.
    """
    _check_aligned(perturbed_errors, perturbed_errors_baseline)
    if not perturbed_errors:
        raise ValueError("mean_relative_error_inflation needs at least one perturbation.")

    total = 0.0
    count = 0
    for name, curve in perturbed_errors.items():
        model_curve = _as_list(curve, f"perturbed_errors[{name!r}]")
        base_curve = _as_list(
            perturbed_errors_baseline[name], f"perturbed_errors_baseline[{name!r}]"
        )
        if len(model_curve) != len(base_curve):
            raise ValueError(
                f"Severity curves for perturbation {name!r} differ in length: "
                f"model has {len(model_curve)}, baseline has {len(base_curve)}."
            )
        for e_model, e_base in zip(model_curve, base_curve, strict=True):
            total += relative_error_inflation(e_clean, e_model, e_clean_baseline, e_base)
            count += 1
    return total / count


# --------------------------------------------------------------------------
# §5.1.3 — corruption error
# --------------------------------------------------------------------------


def corruption_error(
    perturbed_errors: SeverityCurve, baseline_perturbed_errors: SeverityCurve
) -> float:
    """Summed error relative to a baseline's, over severities. Survey §5.1.3, Eq. 19.

    ``CE_φ = (Σ_s E^f_{s,φ}) / (Σ_s E^base_{s,φ})``

    1 matches the baseline, below 1 is more robust, above 1 less. Normalising by the
    baseline puts easy and hard corruptions on a comparable scale, so that averaging
    across types (:func:`mean_corruption_error`) is not dominated by the hardest ones.
    """
    model = _as_list(perturbed_errors, "perturbed_errors")
    base = _as_list(baseline_perturbed_errors, "baseline_perturbed_errors")
    if len(model) != len(base):
        raise ValueError(
            f"Severity curves differ in length: model has {len(model)}, "
            f"baseline has {len(base)}."
        )
    return safe_divide(sum(model), sum(base), what="summed baseline error")


def mean_corruption_error(
    perturbed_errors: PerturbationCurves, baseline_perturbed_errors: PerturbationCurves
) -> float:
    """Mean corruption error across perturbation types. Survey §5.1.3, Eq. 20.

    ``mCE = (1/N) Σ_i CE_{φ_i}``

    A low value indicates strong performance under corruption overall, but *not*
    robustness: it does not control for clean accuracy. Only
    :func:`mean_relative_corruption_error` does that, and the survey asks for both.
    """
    _check_aligned(perturbed_errors, baseline_perturbed_errors)
    if not perturbed_errors:
        raise ValueError("mean_corruption_error needs at least one perturbation.")
    values = [
        corruption_error(curve, baseline_perturbed_errors[name])
        for name, curve in perturbed_errors.items()
    ]
    return float(sum(values) / len(values))


def relative_corruption_error(
    e_clean: float,
    perturbed_errors: SeverityCurve,
    e_clean_baseline: float,
    baseline_perturbed_errors: SeverityCurve,
) -> float:
    """Corruption error with clean error subtracted first. Survey §5.1.3, Eq. 21.

    ``RCE_φ = Σ_s(E^f_{s,φ} - E^f_clean) / Σ_s(E^base_{s,φ} - E^base_clean)``

    Measures the *additional* error the corruption causes, rather than total error
    under corruption, so a model that is simply more accurate everywhere does not score
    as more robust.
    """
    model = _as_list(perturbed_errors, "perturbed_errors")
    base = _as_list(baseline_perturbed_errors, "baseline_perturbed_errors")
    if len(model) != len(base):
        raise ValueError(
            f"Severity curves differ in length: model has {len(model)}, "
            f"baseline has {len(base)}."
        )
    numerator = sum(e - e_clean for e in model)
    denominator = sum(e - e_clean_baseline for e in base)
    return safe_divide(
        numerator, denominator, what="baseline's summed excess error over clean"
    )


def mean_relative_corruption_error(
    e_clean: float,
    perturbed_errors: PerturbationCurves,
    e_clean_baseline: float,
    baseline_perturbed_errors: PerturbationCurves,
) -> float:
    """Mean relative corruption error across types. Survey §5.1.3, Eq. 22.

    ``mRCE = (1/N) Σ_i RCE_{φ_i}``

    This, not ``mCE``, is the one that provides evidence of robustness — it controls
    for differences in clean accuracy. Report both.
    """
    _check_aligned(perturbed_errors, baseline_perturbed_errors)
    if not perturbed_errors:
        raise ValueError("mean_relative_corruption_error needs at least one perturbation.")
    values = [
        relative_corruption_error(
            e_clean, curve, e_clean_baseline, baseline_perturbed_errors[name]
        )
        for name, curve in perturbed_errors.items()
    ]
    return float(sum(values) / len(values))
