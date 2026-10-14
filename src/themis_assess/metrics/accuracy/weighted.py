"""Event-weighted accuracy metrics. Repo-A extension, not in the survey.

Ported from repo B's ``custom_accuracy_metrics.py``, which mixed genuinely Valencia-
specific event definitions with a weighting mechanism that is entirely general. The
weighting is here; the berthing-specific onset rule stays in the use case, injected as a
callable.

The problem it solves. Most operational forecasting cares far more about accuracy near a
rare transition than about average accuracy: when a vessel berths, when a patient
deteriorates, when a machine trips. A plain MAE over a long series is dominated by the
long quiet stretches, where any reasonable model does fine, and a model that is excellent
on average can be useless at the only moments anyone acts on.

Event weighting fixes the denominator. A Gaussian bump around each event onset, plus a
floor so the quiet periods still count for something, normalised to sum to one — so a
weighted error stays on the same scale as its unweighted counterpart and the two are
directly comparable.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "event_weights",
    "step_onsets",
    "weighted_mae",
    "weighted_mse",
    "mean_signed_deviation",
]


def step_onsets(actual: ArrayLike, *, min_duration: int = 1) -> NDArray[np.int_]:
    """Indices where a binary series switches on and stays on.

    Generalises repo B's ``_berthing_onsets``, whose docstring reads: "Indices where the
    status just switched 0 -> 1 and stays 1 for at least ``berth_period`` steps."

    The duration requirement is what makes it robust: a single flickering sample is not
    an event, and without the check every glitch would place a weight bump.

    A use case with a different notion of an event supplies its own callable instead —
    this is the default, not the only option.
    """
    y = np.asarray(actual).ravel()
    if y.size == 0:
        raise ValueError("actual is empty.")
    if min_duration < 1:
        raise ValueError(f"min_duration must be at least 1; got {min_duration}.")

    active = np.asarray(y, dtype=bool)
    switched = np.flatnonzero(active[1:] & ~active[:-1]) + 1
    if active[0]:
        switched = np.concatenate([[0], switched])

    kept = [i for i in switched if active[i : i + min_duration].all()]
    return np.asarray(kept, dtype=int)


def event_weights(
    actual: ArrayLike,
    *,
    onsets: ArrayLike | Callable[[NDArray], ArrayLike] | None = None,
    sigma: float | None = None,
    peak_to_floor: float = 5.0,
    min_duration: int = 1,
) -> NDArray[np.float64]:
    """Per-timestep weights emphasising the neighbourhood of each event.

    A Gaussian bump centred on every onset, plus a constant floor, normalised to sum
    to 1.

    Parameters
    ----------
    actual:
        The observed series. Used to locate events when ``onsets`` is not given.
    onsets:
        Event positions, or a callable mapping the series to them. Defaults to
        :func:`step_onsets`, which treats it as a binary state. **Supply your own for
        anything else** — a threshold crossing, a change point, an externally recorded
        incident time. This is the seam where use-case semantics enter.
    sigma:
        Width of each bump, in timesteps. Defaults to 5% of the series length, with a
        floor of 1, which is wide enough to cover the run-up to an event and narrow
        enough to stay local on a long series.
    peak_to_floor:
        How much more a timestep at an onset counts than one far from any event. Repo B
        used 5.0 for the weighted errors and 3.0 for the weight function's own default.
        1.0 recovers uniform weighting exactly, which is the useful null case.

    Returns
    -------
    ndarray
        Weights summing to 1, of the same length as ``actual``.
    """
    y = np.asarray(actual).ravel()
    n = y.size
    if n == 0:
        raise ValueError("actual is empty.")
    if peak_to_floor < 1.0:
        raise ValueError(
            f"peak_to_floor must be at least 1 (1.0 = uniform weighting); "
            f"got {peak_to_floor}."
        )

    if onsets is None:
        positions = step_onsets(y, min_duration=min_duration)
    elif callable(onsets):
        positions = np.asarray(onsets(y), dtype=int).ravel()
    else:
        positions = np.asarray(onsets, dtype=int).ravel()

    if positions.size and (positions.min() < 0 or positions.max() >= n):
        raise ValueError(
            f"Onset indices must lie in [0, {n - 1}]; got min={positions.min()}, "
            f"max={positions.max()}."
        )

    if positions.size == 0 or peak_to_floor == 1.0:
        # No events, or weighting explicitly disabled: uniform, so a weighted metric
        # degrades gracefully to its unweighted form rather than raising.
        return np.full(n, 1.0 / n)

    width = float(sigma) if sigma is not None else max(1.0, 0.05 * n)
    if width <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}.")

    t = np.arange(n, dtype=float)
    bump = np.zeros(n)
    for position in positions:
        bump = np.maximum(bump, np.exp(-0.5 * ((t - position) / width) ** 2))

    floor = 1.0 / (peak_to_floor - 1.0)
    weights = bump + floor
    return weights / weights.sum()


def _weighted_inputs(
    actual: ArrayLike, predicted: ArrayLike, weights: ArrayLike | None
) -> tuple[NDArray, NDArray, NDArray]:
    y = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(
            f"actual and predicted must align; got {y.size} and {p.size}."
        )
    if y.size == 0:
        raise ValueError("Need at least one observation.")
    if weights is None:
        w = np.full(y.size, 1.0 / y.size)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != y.shape:
            raise ValueError(f"weights must align; got {w.size} for {y.size} points.")
        if np.any(w < 0):
            raise ValueError("Weights must be non-negative.")
        total = w.sum()
        if total <= 0:
            raise ValueError("Weights sum to zero.")
        w = w / total
    return y, p, w


def weighted_mae(
    actual: ArrayLike, predicted: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Mean absolute error, weighted per timestep.

    With uniform weights this is exactly MAE, so a use case can move between the two by
    passing weights or not, and the numbers stay comparable.
    """
    y, p, w = _weighted_inputs(actual, predicted, weights)
    return float(np.sum(w * np.abs(y - p)))


def weighted_mse(
    actual: ArrayLike, predicted: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Mean squared error, weighted per timestep."""
    y, p, w = _weighted_inputs(actual, predicted, weights)
    return float(np.sum(w * (y - p) ** 2))


def mean_signed_deviation(
    actual: ArrayLike, predicted: ArrayLike, weights: ArrayLike | None = None
) -> float:
    """Weighted mean of ``predicted - actual``. Zero is best, either sign is bad.

    The metric that motivated ``MetricSpec.target``: it is neither higher-is-better nor
    lower-is-better, and normalising it needs a target value rather than a direction.

    What it catches that an absolute error cannot is *bias*. A forecaster that is
    consistently early and one that is consistently late can have identical MAE; only a
    signed measure distinguishes them, and for an operational forecast the distinction is
    usually the whole point — predicting an arrival too late and too early have different
    costs.

    Sign convention: positive means the model predicts **above** the truth. Repo B's
    notebook states it as "Positive numbers indicate that the prediction was corrected
    'upwards'".
    """
    y, p, w = _weighted_inputs(actual, predicted, weights)
    return float(np.sum(w * (p - y)))
