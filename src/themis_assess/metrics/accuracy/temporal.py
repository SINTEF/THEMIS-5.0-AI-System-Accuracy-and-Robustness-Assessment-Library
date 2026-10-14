"""Accuracy along the time axis: per origin, per horizon, and aggregated.

A single MAE over a rolling-origin evaluation hides the two things a forecasting team
actually asks about. *When* is the model bad — is the error steady, or concentrated around
particular events? And *how far ahead* does it stay usable — where does the error curve
cross the point at which nobody would act on it?

Ported from repo B's ``ForecastingAccuracyAssessment.compute_accuracy_over_time`` and
``compute_aggregated_accuracy``. Three things about that port.

*It ran on generic code that could not execute.* ``compute_accuracy_over_time`` called
``self.model_handler.historical_forecasts(series=..., covariates=..., forecast_horizon=...,
stride=...)`` while the handler it was written against takes ``past_covariates`` and reads
the horizon and stride from itself — so the call raises ``TypeError`` on the keyword before
doing anything. ``compute_aggregated_accuracy`` was an ``@abstractmethod`` with no body in
the general repo. The rolling evaluation existed only in the use case's notebooks.

*Argument order.* Repo B called ``metric(self.y_pred, self.y_true)`` — predictions first,
following darts. scikit-learn and this library are ``(actual, predicted)``. The order is
invisible for MAE and RMSE and inverts the sign of anything asymmetric, so
:func:`accuracy_over_time` takes the arguments by name and never positionally forwards
them.

*Median as well as mean.* Repo B offered ``aggregation_method`` of ``"mean"`` or
``"median"`` and defaulted to mean. Both are kept, and so is the reason to look at both:
they diverge exactly when a few origins dominate, which for event forecasting is the
normal case rather than the pathological one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "accuracy_over_time",
    "accuracy_by_horizon",
    "aggregate_over_time",
    "worst_windows",
]

_AGGREGATIONS: dict[str, Callable[[NDArray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
    "max": lambda v: float(np.max(v)),
    "min": lambda v: float(np.min(v)),
    "p90": lambda v: float(np.quantile(v, 0.9)),
}


def _rolling_inputs(
    forecasts: ArrayLike, actual: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Validate a rolling-origin forecast array against its truth.

    Accepts ``(n_origins, horizon)`` for a single entity or
    ``(n_entities, n_origins, horizon)`` for several — the shape
    :meth:`~themis_assess.handlers.forecasting.ForecastingHandler.historical_forecasts`
    produces — and normalises to the 3-D form.
    """
    f = np.asarray(forecasts, dtype=float)
    a = np.asarray(actual, dtype=float)
    if f.shape != a.shape:
        raise ValueError(
            f"forecasts and actual must have the same shape; got {f.shape} and {a.shape}. "
            f"These come from historical_forecasts() as a pair, so a mismatch usually "
            f"means one of them was sliced."
        )
    if f.ndim == 2:
        f, a = f[None, :, :], a[None, :, :]
    if f.ndim != 3:
        raise ValueError(
            f"Expected (n_origins, horizon) or (n_entities, n_origins, horizon); got "
            f"{f.shape}."
        )
    if f.size == 0:
        raise ValueError("No forecasts to score.")
    return f, a


def accuracy_over_time(
    forecasts: ArrayLike,
    actual: ArrayLike,
    metric: Callable[..., float],
    *,
    per_entity: bool = False,
) -> NDArray:
    """One metric value per forecast origin.

    The curve repo B's ``plot_accuracy_over_time`` drew, computed separately so it can be
    tested and reported without rendering anything.

    Parameters
    ----------
    metric:
        Called as ``metric(actual_window, forecast_window)`` — **actual first**, this
        library's convention throughout. A darts metric expects the opposite order; wrap
        it rather than passing it straight in.
    per_entity:
        Return ``(n_entities, n_origins)`` instead of averaging entities into
        ``(n_origins,)``. Keep the entities separate whenever they are not
        interchangeable — repo B's vessels are not — since averaging across them hides an
        entity the model handles badly.

    Returns
    -------
    ndarray
        ``(n_origins,)``, or ``(n_entities, n_origins)`` when ``per_entity``.
    """
    f, a = _rolling_inputs(forecasts, actual)
    n_entities, n_origins = f.shape[0], f.shape[1]

    values = np.empty((n_entities, n_origins))
    for e in range(n_entities):
        for o in range(n_origins):
            values[e, o] = float(metric(a[e, o], f[e, o]))
    return values if per_entity else values.mean(axis=0)


def accuracy_by_horizon(
    forecasts: ArrayLike,
    actual: ArrayLike,
    metric: Callable[..., float],
) -> NDArray:
    """One metric value per step ahead, pooled over origins and entities.

    The other axis of the same array, and the one repo B never computed — its notebooks
    reported a single aggregate over the whole horizon. For a forecast that is acted on at
    a particular lead time, this is the more decision-relevant of the two curves: it says
    how far ahead the model is still worth trusting.

    Almost always increasing. Where it is *not* — where a later horizon scores better than
    an earlier one — that is worth investigating rather than reporting, since it usually
    means a target leak or a misaligned origin.

    Returns
    -------
    ndarray
        ``(horizon,)``, indexed from one step ahead.
    """
    f, a = _rolling_inputs(forecasts, actual)
    horizon = f.shape[2]
    return np.array(
        [float(metric(a[:, :, h].ravel(), f[:, :, h].ravel())) for h in range(horizon)]
    )


def aggregate_over_time(
    values: ArrayLike, *, method: str = "mean", weights: ArrayLike | None = None
) -> float:
    """Collapse a per-origin curve to one number.

    Repo B's ``compute_aggregated_accuracy``, which was an abstract method with no body in
    the general repo.

    Parameters
    ----------
    method:
        ``"mean"``, ``"median"``, ``"max"``, ``"min"`` or ``"p90"``. Repo B offered the
        first two. ``"max"`` and ``"p90"`` are added because survey §8 asks for the worst
        case beside the average, and a mean over origins is precisely where a worst case
        goes missing.
    weights:
        Per-origin weights, for emphasising the origins that matter — the run-up to an
        event, say. :func:`themis_assess.metrics.accuracy.weighted.event_weights` produces
        them. Only valid with ``method="mean"``: a weighted median or maximum is either
        undefined or unaffected, and silently ignoring the weights would be worse than
        refusing.
    """
    curve = np.asarray(values, dtype=float).ravel()
    if curve.size == 0:
        raise ValueError("Nothing to aggregate.")
    if method not in _AGGREGATIONS:
        raise ValueError(
            f"method must be one of {sorted(_AGGREGATIONS)}; got {method!r}."
        )
    if weights is None:
        return _AGGREGATIONS[method](curve)

    if method != "mean":
        raise ValueError(
            f"weights are only meaningful with method='mean'; got {method!r}. A weighted "
            f"{method} is either undefined or ignores the weights, and quietly dropping "
            f"them would misreport what was computed."
        )
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != curve.shape:
        raise ValueError(f"Got {w.size} weight(s) for {curve.size} value(s).")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights sum to zero.")
    return float(np.sum(w * curve) / total)


def worst_windows(
    values: ArrayLike,
    *,
    n: int = 5,
    higher_is_better: bool = False,
    origins: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """The origins at which the model did worst.

    Where an aggregate says "the model is fine", this says which moments it was not, and
    those moments are what a forecasting team can go and look at. An accuracy curve
    without this is a shape with no way into the data behind it.

    Parameters
    ----------
    origins:
        Timestamps or indices to label the result with. Defaults to positions in the
        curve. Pass the real timestamps: "origin 47" sends nobody anywhere.
    """
    curve = np.asarray(values, dtype=float).ravel()
    if curve.size == 0:
        raise ValueError("Nothing to rank.")
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")
    labels = list(origins) if origins is not None else list(range(curve.size))
    if len(labels) != curve.size:
        raise ValueError(f"Got {len(labels)} origin label(s) for {curve.size} value(s).")

    order = np.argsort(curve)
    if not higher_is_better:
        order = order[::-1]
    return [
        {"origin": labels[i], "position": int(i), "value": float(curve[i])}
        for i in order[:n]
    ]
