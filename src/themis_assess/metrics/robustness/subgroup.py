"""Subgroup-disparity metrics — survey §5.1.2.

The design point here answers an objection raised in repo C:

    "I don't think this is so easy to abstract so that it is applicable to other
    use-cases. The medical use case has ohe features in a tabular format - which allow
    me to select a subset of the population very easily. But this is not the case for
    the media use case for instance."

That is right about *feature indexing*, and it is why the metrics below never see
features. They take ``{group_name: score}`` and nothing else. Producing that mapping is
:func:`group_scores`, which takes boolean masks — and every subgroup mechanism across
the four repos reduces to a mask:

===============================  =========================================
repo C, one-hot column           ``lambda ds: ds[col] == 1``
repo D, embedding-similar theme  ``lambda ds: sim(ds, theme) > tau``
repo B, callable category        ``lambda ds: category(ds) == label``
repo A, ``{"age": {"min": 30}}`` compiled to a mask
===============================  =========================================

As in :mod:`~themis_assess.metrics.robustness.degradation`, ``a_*`` names denote
accuracy-type scores (higher better) and ``e_*`` errors (lower better). Equations 6-8
are defined on accuracies, 9-10 on errors; passing the wrong kind inverts the meaning
without raising, so the argument names carry the distinction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.core.types import GroupScores

__all__ = [
    "group_scores",
    "worst_group_accuracy",
    "accuracy_gap",
    "accuracy_variance",
    "group_disparity",
    "equal_weight_risk",
    "welch_anova",
]


def _values(scores: GroupScores, fn_name: str, minimum: int = 1) -> list[float]:
    if len(scores) < minimum:
        raise ValueError(
            f"{fn_name} needs at least {minimum} group"
            f"{'s' if minimum > 1 else ''}; got {len(scores)}."
        )
    return [float(v) for v in scores.values()]


def group_scores(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    masks: Mapping[str, ArrayLike],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_group_size: int = 1,
) -> dict[str, float]:
    """Evaluate ``metric_fn`` within each group, given boolean masks over the samples.

    Parameters
    ----------
    y_true, y_pred:
        Aligned arrays of length ``n``.
    masks:
        ``{group_name: boolean array of length n}``. Groups may overlap and need not
        cover every sample — an age band and a job title are both legitimate groups over
        the same population.
    metric_fn:
        Called as ``metric_fn(y_true[mask], y_pred[mask])``.
    min_group_size:
        Groups with fewer members than this raise, rather than returning a metric
        computed on two samples that will then be compared against one computed on two
        thousand.

    Returns
    -------
    dict
        ``{group_name: score}``, in the order the masks were given.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape[0] != pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length; got {truth.shape[0]} and "
            f"{pred.shape[0]}."
        )
    if not masks:
        raise ValueError("group_scores needs at least one group mask.")

    out: dict[str, float] = {}
    for name, raw_mask in masks.items():
        mask = np.asarray(raw_mask, dtype=bool).ravel()
        if mask.shape[0] != truth.shape[0]:
            raise ValueError(
                f"Mask for group {name!r} has length {mask.shape[0]}, "
                f"but there are {truth.shape[0]} samples."
            )
        size = int(mask.sum())
        if size < min_group_size:
            raise ValueError(
                f"Group {name!r} has {size} sample(s), below min_group_size="
                f"{min_group_size}. Either merge it into a larger group or lower the "
                f"threshold deliberately — a metric on a handful of samples is not "
                f"comparable with one on the full population."
            )
        out[name] = float(metric_fn(truth[mask], pred[mask]))
    return out


def worst_group_accuracy(accuracies: GroupScores) -> float:
    """Lowest accuracy over all groups. Survey §5.1.2, Eq. 6.

    ``WGA = min_g A_g``

    The headline number for whether acceptable *average* performance is hiding a group
    the model fails on.
    """
    return min(_values(accuracies, "worst_group_accuracy"))


def accuracy_gap(accuracies: GroupScores) -> float:
    """Spread between the best- and worst-served groups. Survey §5.1.2, Eq. 7.

    ``Gap = max_g A_g - min_g A_g``. Lower is better.
    """
    values = _values(accuracies, "accuracy_gap", minimum=2)
    return max(values) - min(values)


def accuracy_variance(accuracies: GroupScores) -> float:
    """Population variance of the per-group accuracies. Survey §5.1.2, Eq. 8.

    ``VarAcc = (1/|G|) Σ_g (A_g - Ā)²``

    Population variance (divide by ``|G|``), not the sample variance — the groups are
    the whole population of interest, not a sample from a larger set of groups.
    """
    values = _values(accuracies, "accuracy_variance", minimum=2)
    mean = sum(values) / len(values)
    return float(sum((v - mean) ** 2 for v in values) / len(values))


def group_disparity(errors: GroupScores) -> float:
    """Largest error difference between any two groups. Survey §5.1.2, Eq. 9.

    ``Δ_group = max_{g,h} |E_g - E_h|``. Lower is better.

    The error-space counterpart of :func:`accuracy_gap`, and numerically equal to it
    when errors are ``1 - accuracy``.
    """
    values = _values(errors, "group_disparity", minimum=2)
    return max(values) - min(values)


def equal_weight_risk(errors: GroupScores) -> float:
    """Mean error with every group weighted equally. Survey §5.1.2, Eq. 10.

    ``E_equal = (1/|G|) Σ_g E_g``

    Weighting by group rather than by prevalence is the point: it stops large
    populations dominating the number, which is exactly what overall error already does.
    """
    values = _values(errors, "equal_weight_risk")
    return float(sum(values) / len(values))


def welch_anova(groups: Sequence[ArrayLike] | Mapping[str, ArrayLike]) -> dict[str, float]:
    """Welch's ANOVA: is the difference between group means statistically real?

    Repo-A extension, not in the survey. Carried over from repo B's
    ``compute_aggregated_accuracy_per_category``.

    Every metric above reports *how large* a subgroup difference is; none says whether
    it is distinguishable from sampling noise. Welch's variant is used rather than
    one-way ANOVA because it does not assume equal variances across groups, and
    per-group performance spreads routinely differ.

    Parameters
    ----------
    groups:
        Two or more sequences of per-sample (or per-series) metric values — the
        distribution within each group, not the group's aggregate.

    Returns
    -------
    dict
        ``{"statistic", "p_value", "df_between", "df_within", "n_groups"}``.

    Notes
    -----
    A small p-value says the group means differ, not that the difference matters. Read
    it alongside :func:`accuracy_gap`, which says by how much.
    """
    from scipy import stats  # scipy is a core dependency (via scikit-learn)

    samples = list(groups.values()) if isinstance(groups, Mapping) else list(groups)
    arrays = [np.asarray(s, dtype=float).ravel() for s in samples]
    if len(arrays) < 2:
        raise ValueError(f"welch_anova needs at least 2 groups; got {len(arrays)}.")
    for i, arr in enumerate(arrays):
        if arr.size < 2:
            raise ValueError(
                f"Group at position {i} has {arr.size} value(s). Welch's ANOVA needs at "
                f"least 2 per group to estimate a within-group variance."
            )

    k = len(arrays)
    n = np.array([a.size for a in arrays], dtype=float)
    means = np.array([a.mean() for a in arrays])
    var = np.array([a.var(ddof=1) for a in arrays])
    if np.any(var == 0):
        raise ValueError(
            "At least one group has zero variance, so Welch's ANOVA is undefined. "
            "This usually means a group's metric values are all identical."
        )

    w = n / var
    w_sum = w.sum()
    grand = float((w * means).sum() / w_sum)

    numerator = float((w * (means - grand) ** 2).sum()) / (k - 1)
    lam = float((((1 - w / w_sum) ** 2) / (n - 1)).sum())
    denominator = 1.0 + (2.0 * (k - 2) / (k**2 - 1)) * lam
    statistic = numerator / denominator

    df_between = float(k - 1)
    df_within = float((k**2 - 1) / (3 * lam)) if lam > 0 else float("inf")
    p_value = float(stats.f.sf(statistic, df_between, df_within))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "n_groups": float(k),
    }
