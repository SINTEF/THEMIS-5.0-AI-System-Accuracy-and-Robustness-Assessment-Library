"""Baseline handlers — the reference model the §5.1.3 relative metrics need.

Four of the survey's metrics are defined against a baseline: ``CE`` and ``mCE`` (Eqs.
19-20), ``RCE`` and ``mRCE`` (21-22), ``REI`` and ``mREI`` (17-18), and the flip rates
(24-25). Under D14 the baseline is *a second model handler*, not a special abstract method
on the assessment class, which is how repo A's template had it. Making it an ordinary
handler means the baseline is evaluated by exactly the same code path as the model —
same perturbations, same seed, same severities — which is what makes the ratio meaningful.

These are here so that a user is not blocked from computing ``mCE`` on day one for want of
a second trained model. A naive baseline is also the *right* baseline more often than it
looks: the survey's framing is that a corruption error should be normalised by how hard
the corruption is, and a model that ignores the input entirely gives a clean reading of
that difficulty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import AccessLevel, Alignment, TaskType
from themis_assess.handlers.base import BaseHandler, Prediction
from themis_assess.handlers.forecasting import ForecastingConfig

__all__ = [
    "MajorityClassHandler",
    "PriorProbabilityHandler",
    "NaiveLastValueHandler",
    "MovingAverageHandler",
]


class MajorityClassHandler(BaseHandler):
    """Always predicts the most frequent training class.

    The floor any classifier must clear, and the natural denominator for a corruption
    error: its accuracy does not move under a perturbation of the *inputs*, so a
    ``CE`` computed against it is measuring the model's sensitivity alone.
    """

    def __init__(self, labels: Any, *, name: str = "majority_class") -> None:
        values, counts = np.unique(np.asarray(labels).ravel(), return_counts=True)
        if values.size == 0:
            raise ValueError("MajorityClassHandler needs at least one training label.")
        self.majority = values[np.argmax(counts)]
        self.classes = values
        super().__init__(
            None, name=name, task=TaskType.CLASSIFICATION, access=AccessLevel.BLACK_BOX
        )

    def predict(self, data: DataAdapter) -> Prediction:
        predicted = np.full(data.n_entities, self.majority)
        return self.check_prediction(
            Prediction(predicted=predicted, metadata={"model": self.name}), data
        )


class PriorProbabilityHandler(BaseHandler):
    """Predicts the training class distribution, identically for every sample.

    The probabilistic counterpart of :class:`MajorityClassHandler`, and the honest
    baseline for a calibration comparison: it is perfectly calibrated on data drawn from
    the training distribution and knows nothing at all. A model whose ECE under shift is
    worse than this one's is not merely degraded, it is misleading.
    """

    def __init__(self, labels: Any, *, name: str = "prior_probability") -> None:
        array = np.asarray(labels).ravel()
        values, counts = np.unique(array, return_counts=True)
        if values.size == 0:
            raise ValueError("PriorProbabilityHandler needs at least one training label.")
        self.classes = values
        self.prior = counts / counts.sum()
        self.majority = values[np.argmax(counts)]
        super().__init__(
            None,
            name=name,
            task=TaskType.CLASSIFICATION,
            access=AccessLevel.PROBABILISTIC,
        )

    def predict(self, data: DataAdapter) -> Prediction:
        n = data.n_entities
        return self.check_prediction(
            Prediction(
                predicted=np.full(n, self.majority),
                probabilities=np.tile(self.prior, (n, 1)),
                metadata={"model": self.name, "classes": self.classes.tolist()},
            ),
            data,
        )


class _SeriesBaseline(BaseHandler):
    """Shared machinery for the forecasting baselines: last window, flat forecast."""

    def __init__(
        self,
        config: ForecastingConfig,
        *,
        name: str,
        target: str | None = None,
    ) -> None:
        self.config = config
        self.target = target
        super().__init__(
            None, name=name, task=TaskType.FORECASTING, access=AccessLevel.BLACK_BOX
        )

    def _series(self, data: DataAdapter) -> list[np.ndarray]:
        values = data.values()
        j = 0 if self.target is None else data.feature_index(self.target)
        out = []
        for entity in range(data.n_entities):
            length = int(data.entity_lengths[entity])
            if data.alignment is Alignment.END:
                start, stop = data.n_timesteps - length, data.n_timesteps
            else:
                start, stop = 0, length
            out.append(values[entity, start:stop, j].astype(float))
        return out

    def _level(self, history: np.ndarray) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, data: DataAdapter) -> Prediction:
        horizon = self.config.forecast_horizon
        forecasts = []
        for series in self._series(data):
            if series.size <= horizon:
                raise ValueError(
                    f"A series of {series.size} step(s) cannot hold a {horizon}-step "
                    f"forecast plus history."
                )
            history = series[: series.size - horizon]
            forecasts.append(np.full(horizon, self._level(history)))
        return self.check_prediction(
            Prediction(
                predicted=np.vstack(forecasts),
                metadata={"model": self.name, "forecast_horizon": horizon},
            ),
            data,
        )


class NaiveLastValueHandler(_SeriesBaseline):
    """Predicts the last observed value, held flat across the horizon.

    The standard naive forecast, and a demanding baseline on anything with
    autocorrelation — which is most operational series.
    """

    def __init__(self, config: ForecastingConfig, *, name: str = "naive_last_value",
                 target: str | None = None) -> None:
        super().__init__(config, name=name, target=target)

    def _level(self, history: np.ndarray) -> float:
        finite = history[np.isfinite(history)]
        if finite.size == 0:
            raise ValueError("No finite history to forecast from.")
        return float(finite[-1])


class MovingAverageHandler(_SeriesBaseline):
    """Predicts the mean of the last ``window`` observations. Repo B's baseline.

    Repo B built this as ``ValenciaModelHandler_NaiveMovingAverage`` around darts'
    ``NaiveMovingAverage``, with ``retrain=True``, while the model it was compared
    against ran with ``retrain=False``. Here both run under one
    :class:`~themis_assess.handlers.forecasting.ForecastingConfig`, so that cannot
    silently differ again.
    """

    def __init__(
        self,
        config: ForecastingConfig,
        *,
        window: int | None = None,
        name: str = "moving_average",
        target: str | None = None,
    ) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be at least 1; got {window}.")
        self.window = window if window is not None else max(1, config.input_chunk_length)
        super().__init__(config, name=name, target=target)

    def _level(self, history: np.ndarray) -> float:
        finite = history[np.isfinite(history)]
        if finite.size == 0:
            raise ValueError("No finite history to forecast from.")
        return float(np.mean(finite[-self.window :]))
