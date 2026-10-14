"""Forecasting handlers, and the rolling-origin configuration they share.

Follows repo B's ``ForecastingModelHandler`` signature, with the four positional
parameters that always travelled together — ``input_chunk_length``,
``forecast_horizon``, ``stride``, and the retrain/last-points flags — collected into
:class:`ForecastingConfig` (docs/DECISIONS.md, D12). They travelled together because they
are one thing: a description of the evaluation protocol, not of the model. Passing them
separately is what let repo B's baseline handler default to ``retrain=True`` while its
TSMixer handler defaulted to ``retrain=False``, so the two were not evaluated under the
same protocol.

Nothing here imports darts. A darts model reaches this through
:class:`DartsForecastingHandler` in :mod:`themis_assess.integrations`, under the
``[timeseries]`` extra; everything else supplies a plain function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import AccessLevel, Alignment, TaskType
from themis_assess.handlers.base import BaseHandler, Prediction

__all__ = [
    "ForecastingConfig",
    "ForecastingHandler",
]


@dataclass(frozen=True, slots=True)
class ForecastingConfig:
    """How a forecast is produced and evaluated over time.

    Parameters
    ----------
    forecast_horizon:
        How many steps ahead to predict. ``n`` in darts' ``predict(n=...)``.
    input_chunk_length:
        How many past steps the model reads. Zero means "everything available", which
        suits a model with no fixed window. Where it is non-zero it is also the right
        value for :func:`~themis_assess.protocols.splits.temporal_split`'s ``gap``,
        since a test window reaching back across the split boundary reads training data.
    stride:
        How often a new forecast is issued along the timeline. 1 re-forecasts at every
        step.
    retrain:
        Whether the model is refitted at each origin. Almost always ``False`` for an
        assessment: refitting changes the model between severities, so a degradation
        curve would confound the perturbation with the refit.
    last_points_only:
        Keep only the final step of each forecast, giving one value per origin instead
        of a horizon-length vector. Repo B's flag, kept because it is the difference
        between "how good is the 24-hour-ahead forecast" and "how good is the whole
        trajectory".
    """

    forecast_horizon: int
    input_chunk_length: int = 0
    stride: int = 1
    retrain: bool = False
    last_points_only: bool = False

    def __post_init__(self) -> None:
        if self.forecast_horizon < 1:
            raise ValueError(
                f"forecast_horizon must be at least 1; got {self.forecast_horizon}."
            )
        if self.input_chunk_length < 0:
            raise ValueError(
                f"input_chunk_length cannot be negative; got {self.input_chunk_length}."
            )
        if self.stride < 1:
            raise ValueError(f"stride must be at least 1; got {self.stride}.")
        if self.retrain:
            # Not an error — repo B's baseline legitimately refits — but the caller
            # should have meant it.
            pass

    def origins(self, n_timesteps: int) -> list[int]:
        """Indices at which a forecast is issued, given a series of ``n_timesteps``.

        An origin ``t`` means: the model reads up to and including ``t - 1`` and predicts
        ``t .. t + forecast_horizon - 1``. Origins start once enough history exists and
        stop while a full horizon still fits, so every forecast this yields is one that
        can actually be scored — repo B's notebooks trimmed a ragged tail by hand.
        """
        if n_timesteps < 1:
            raise ValueError(f"n_timesteps must be positive; got {n_timesteps}.")
        first = max(self.input_chunk_length, 1)
        last = n_timesteps - self.forecast_horizon
        if last < first:
            raise ValueError(
                f"A series of {n_timesteps} step(s) cannot support a forecast of "
                f"{self.forecast_horizon} step(s) after {self.input_chunk_length} "
                f"step(s) of history: there is no origin where both fit. Shorten the "
                f"horizon or the input window, or supply a longer series."
            )
        return list(range(first, last + 1, self.stride))

    def n_forecasts(self, n_timesteps: int) -> int:
        """How many forecasts a series of this length yields. For cost estimation."""
        return len(self.origins(n_timesteps))


class ForecastingHandler(BaseHandler):
    """A forecaster evaluated over a rolling origin.

    The model is supplied as a function so that this stays free of any forecasting
    library: ``forecast_fn(history, horizon) -> array``, where ``history`` is
    ``(n_past_steps, n_features)`` for one entity and the return is
    ``(horizon,)`` or ``(horizon, n_targets)``.

    Parameters
    ----------
    forecast_fn:
        The model, as above. Called once per entity per origin, so a model that can
        batch should be wrapped with :meth:`predict` overridden instead.
    config:
        The rolling-origin protocol.
    target:
        Name of the feature being forecast. Defaults to the adapter's first feature.
        This is also the column the runner scores against.
    interval_fn, sample_fn:
        Optional, same shape as ``forecast_fn``: ``interval_fn`` returns
        ``(lower, upper)`` each of the horizon's shape, ``sample_fn`` returns
        ``(horizon, n_members)``. Either raises the declared access to
        ``PROBABILISTIC`` and unlocks the §5.4 interval and CRPS metrics.
    """

    def __init__(
        self,
        forecast_fn: Callable[..., Any],
        config: ForecastingConfig,
        *,
        name: str = "forecaster",
        target: str | None = None,
        interval_fn: Callable[..., Any] | None = None,
        sample_fn: Callable[..., Any] | None = None,
        access: AccessLevel | None = None,
        **metadata: Any,
    ) -> None:
        if not callable(forecast_fn):
            raise TypeError("forecast_fn must be callable.")
        self.forecast_fn = forecast_fn
        self.config = config
        self.target = target
        self.interval_fn = interval_fn
        self.sample_fn = sample_fn
        richer = interval_fn is not None or sample_fn is not None
        inferred = AccessLevel.PROBABILISTIC if richer else AccessLevel.BLACK_BOX
        super().__init__(
            forecast_fn,
            name=name,
            task=TaskType.FORECASTING,
            access=inferred if access is None else access,
            **metadata,
        )

    # -- history extraction -----------------------------------------------------

    def _target_index(self, data: DataAdapter) -> int:
        if self.target is None:
            return 0
        return data.feature_index(self.target)

    def _history(self, values: NDArray, entity: int, origin: int) -> NDArray:
        """The window the model reads before ``origin``, for one entity."""
        window = self.config.input_chunk_length
        start = 0 if window == 0 else max(0, origin - window)
        return values[entity, start:origin, :]

    def _valid_span(self, data: DataAdapter, entity: int) -> tuple[int, int]:
        """The ``[start, stop)`` of real (unpadded) timesteps for one entity.

        Ragged entities are padded with NaN, and a forecast issued from padding is
        meaningless. Repo B's data had one series per vessel of very unequal length, so
        this is the normal case rather than an edge one.
        """
        length = int(data.entity_lengths[entity])
        if data.alignment is Alignment.END:
            return data.n_timesteps - length, data.n_timesteps
        return 0, length

    # -- prediction -------------------------------------------------------------

    def forecast_at(self, data: DataAdapter, entity: int, origin: int) -> NDArray:
        """One forecast, for one entity, issued at one origin. Mostly for testing."""
        values = data.values()
        history = self._history(values, entity, origin)
        return np.asarray(
            self.forecast_fn(history, self.config.forecast_horizon), dtype=float
        ).ravel()

    def historical_forecasts(self, data: DataAdapter) -> dict[str, NDArray]:
        """Rolling-origin forecasts and the values they should be compared against.

        The generic form of repo B's ``historical_forecasts``, which delegated to darts.

        Returns
        -------
        dict
            ``forecasts`` and ``actual``, both ``(n_entities, n_origins, horizon)``, plus
            ``origins``. With ``last_points_only`` the horizon axis is dropped.
        """
        values = data.values()
        target = self._target_index(data)
        horizon = self.config.forecast_horizon

        per_entity_forecasts, per_entity_actual, origins_used = [], [], None
        for entity in range(data.n_entities):
            start, stop = self._valid_span(data, entity)
            span = stop - start
            origins = [start + o for o in self.config.origins(span)]
            if origins_used is None:
                origins_used = origins
            elif len(origins) != len(origins_used):
                raise ValueError(
                    f"Entity {entity} yields {len(origins)} forecast origin(s) but the "
                    f"first entity yields {len(origins_used)}. Rolling-origin evaluation "
                    f"needs entities of equal usable length; trim them, or evaluate one "
                    f"entity at a time."
                )
            forecasts = [self.forecast_at(data, entity, o) for o in origins]
            actual = [values[entity, o : o + horizon, target] for o in origins]
            per_entity_forecasts.append(np.vstack(forecasts))
            per_entity_actual.append(np.vstack(actual))

        forecasts_array = np.stack(per_entity_forecasts)
        actual_array = np.stack(per_entity_actual).astype(float)
        if self.config.last_points_only:
            forecasts_array = forecasts_array[:, :, -1]
            actual_array = actual_array[:, :, -1]
        return {
            "forecasts": forecasts_array,
            "actual": actual_array,
            "origins": np.asarray(origins_used or [], dtype=int),
        }

    def predict(self, data: DataAdapter) -> Prediction:
        """One forecast per entity, issued at the latest origin the series supports.

        This is what the perturbation sweep calls, and the reason it forecasts once per
        entity rather than at every origin is cost: a sweep is already
        ``perturbations x severities x variants`` model calls, and multiplying that by
        the number of origins turns a minute into an hour. Use
        :meth:`historical_forecasts` where the whole rolling evaluation is wanted.
        """
        target = self._target_index(data)
        values = data.values()
        horizon = self.config.forecast_horizon

        forecasts, lowers, uppers, samples = [], [], [], []
        for entity in range(data.n_entities):
            start, stop = self._valid_span(data, entity)
            origin = stop - horizon
            if origin <= start:
                raise ValueError(
                    f"Entity {entity} has {stop - start} usable timestep(s), which "
                    f"cannot hold a {horizon}-step forecast plus any history."
                )
            history = self._history(values, entity, origin)
            forecasts.append(
                np.asarray(self.forecast_fn(history, horizon), dtype=float).ravel()
            )
            if self.interval_fn is not None:
                low, high = self.interval_fn(history, horizon)
                lowers.append(np.asarray(low, dtype=float).ravel())
                uppers.append(np.asarray(high, dtype=float).ravel())
            if self.sample_fn is not None:
                samples.append(np.asarray(self.sample_fn(history, horizon), dtype=float))

        del target  # the target column selects `actual`, which the runner supplies
        return self.check_prediction(
            Prediction(
                predicted=np.vstack(forecasts),
                lower=np.vstack(lowers) if lowers else None,
                upper=np.vstack(uppers) if uppers else None,
                samples=np.stack(samples) if samples else None,
                metadata={
                    "model": self.name,
                    "forecast_horizon": horizon,
                    "input_chunk_length": self.config.input_chunk_length,
                },
            ),
            data,
        )

    def actual(self, data: DataAdapter) -> NDArray:
        """The values :meth:`predict`'s forecasts should be scored against.

        Kept beside ``predict`` deliberately. In a forecasting assessment the truth is a
        *slice of the input*, so which slice depends on the same origin arithmetic the
        forecast used — and getting it wrong by one step is both easy and invisible in
        the resulting numbers.
        """
        values = data.values()
        target = self._target_index(data)
        horizon = self.config.forecast_horizon
        out = []
        for entity in range(data.n_entities):
            start, stop = self._valid_span(data, entity)
            origin = stop - horizon
            out.append(values[entity, origin:stop, target])
        return np.vstack(out).astype(float)
