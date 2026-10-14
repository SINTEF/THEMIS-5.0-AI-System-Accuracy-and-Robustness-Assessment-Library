"""Tests for ForecastingConfig and ForecastingHandler.

Most of the risk in a forecasting assessment is off-by-one arithmetic around the origin:
a forecast compared against a slice one step out is still a plausible-looking number, and
nothing downstream can detect it. So the origin arithmetic is tested directly, and the
alignment between a forecast and its truth is tested by constructing a model that must
score exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import AccessLevel, Alignment, TaskType
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler


@pytest.fixture
def series():
    """One entity, 100 steps, a target that is simply the timestep index.

    Deliberately trivial: a model returning ``origin, origin+1, ...`` is exactly right,
    so any misalignment shows up as a non-zero error rather than as a slightly worse one.
    """
    t = np.arange(100, dtype=float)
    values = np.stack([t, np.zeros(100)], axis=1)[None, :, :]
    return ArrayAdapter(values, feature_names=["y", "pad"], categorical_features=[])


# --------------------------------------------------------------------------
# ForecastingConfig
# --------------------------------------------------------------------------


def test_config_validates_its_arguments():
    with pytest.raises(ValueError, match="forecast_horizon must be at least 1"):
        ForecastingConfig(forecast_horizon=0)
    with pytest.raises(ValueError, match="input_chunk_length cannot be negative"):
        ForecastingConfig(forecast_horizon=1, input_chunk_length=-1)
    with pytest.raises(ValueError, match="stride must be at least 1"):
        ForecastingConfig(forecast_horizon=1, stride=0)


def test_origins_leave_room_for_history_and_horizon():
    config = ForecastingConfig(forecast_horizon=10, input_chunk_length=20, stride=1)
    origins = config.origins(100)
    assert origins[0] == 20  # the first origin with 20 steps of history behind it
    assert origins[-1] == 90  # and the last with a full horizon ahead of it
    assert len(origins) == 71


def test_stride_thins_the_origins():
    config = ForecastingConfig(forecast_horizon=10, input_chunk_length=20, stride=5)
    assert config.origins(100) == list(range(20, 91, 5))
    assert config.n_forecasts(100) == 15


def test_every_origin_yields_a_scoreable_forecast():
    """Repo B's notebooks trimmed a ragged tail by hand; the arithmetic does it here."""
    config = ForecastingConfig(forecast_horizon=7, input_chunk_length=13, stride=3)
    n = 60
    for origin in config.origins(n):
        assert origin - config.input_chunk_length >= 0
        assert origin + config.forecast_horizon <= n


def test_a_series_too_short_for_the_protocol_says_so():
    config = ForecastingConfig(forecast_horizon=50, input_chunk_length=40)
    with pytest.raises(ValueError, match="no origin where both fit"):
        config.origins(60)


def test_zero_input_chunk_length_means_use_everything():
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=0)
    assert config.origins(20)[0] == 1  # one step of history is the minimum


# --------------------------------------------------------------------------
# ForecastingHandler
# --------------------------------------------------------------------------


def _perfect(history, horizon):
    """Knows the series is the timestep index, so continues it exactly."""
    return history[-1, 0] + np.arange(1, horizon + 1)


def test_forecast_and_truth_are_aligned(series):
    """The test that catches an off-by-one: a model that is exactly right scores zero."""
    config = ForecastingConfig(forecast_horizon=8, input_chunk_length=16)
    handler = ForecastingHandler(_perfect, config, target="y")
    prediction = handler.predict(series)
    actual = handler.actual(series)
    assert np.allclose(prediction.predicted, actual)


def test_the_forecast_covers_the_final_window(series):
    config = ForecastingConfig(forecast_horizon=8, input_chunk_length=16)
    handler = ForecastingHandler(_perfect, config, target="y")
    # The series is 0..99, so the last 8 steps are 92..99.
    assert handler.actual(series).ravel().tolist() == list(range(92, 100))


def test_historical_forecasts_align_at_every_origin(series):
    config = ForecastingConfig(forecast_horizon=6, input_chunk_length=10, stride=7)
    handler = ForecastingHandler(_perfect, config, target="y")
    out = handler.historical_forecasts(series)
    assert out["forecasts"].shape == out["actual"].shape
    assert np.allclose(out["forecasts"], out["actual"])
    assert out["forecasts"].shape[1] == len(out["origins"])


def test_last_points_only_drops_the_horizon_axis(series):
    config = ForecastingConfig(
        forecast_horizon=6, input_chunk_length=10, stride=7, last_points_only=True
    )
    handler = ForecastingHandler(_perfect, config, target="y")
    out = handler.historical_forecasts(series)
    assert out["forecasts"].ndim == 2  # (entities, origins)


def test_the_target_column_is_selected_by_name():
    t = np.arange(60, dtype=float)
    values = np.stack([np.zeros(60), t], axis=1)[None, :, :]
    data = ArrayAdapter(values, feature_names=["noise", "y"], categorical_features=[])
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=10)

    on_target = ForecastingHandler(
        lambda h, n: h[-1, 1] + np.arange(1, n + 1), config, target="y"
    )
    assert np.allclose(on_target.predict(data).predicted, on_target.actual(data))

    default_first_column = ForecastingHandler(_perfect, config)
    assert default_first_column.actual(data).ravel().tolist() == [0.0] * 5


def test_input_chunk_length_limits_what_the_model_sees(series):
    seen = []

    def recording(history, horizon):
        seen.append(history.shape[0])
        return np.zeros(horizon)

    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=12)
    ForecastingHandler(recording, config, target="y").predict(series)
    assert seen == [12]


def test_zero_input_chunk_length_passes_the_whole_history(series):
    seen = []

    def recording(history, horizon):
        seen.append(history.shape[0])
        return np.zeros(horizon)

    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=0)
    ForecastingHandler(recording, config, target="y").predict(series)
    assert seen == [95]  # everything before the final 5-step window


def test_intervals_raise_the_declared_access(series):
    config = ForecastingConfig(forecast_horizon=4, input_chunk_length=8)
    plain = ForecastingHandler(_perfect, config, target="y")
    with_intervals = ForecastingHandler(
        _perfect,
        config,
        target="y",
        interval_fn=lambda h, n: (np.zeros(n), np.full(n, 100.0)),
    )
    assert plain.access is AccessLevel.BLACK_BOX
    assert with_intervals.access is AccessLevel.PROBABILISTIC
    prediction = with_intervals.predict(series)
    assert prediction.lower.shape == prediction.predicted.shape


def test_the_task_is_always_forecasting(series):
    config = ForecastingConfig(forecast_horizon=4)
    assert ForecastingHandler(_perfect, config).task is TaskType.FORECASTING


def test_ragged_entities_forecast_from_their_own_real_data():
    """Repo B's vessels had very unequal series lengths; a forecast issued from NaN
    padding is meaningless."""
    values = np.full((2, 50, 1), np.nan)
    values[0, 20:, 0] = np.arange(30, dtype=float)  # 30 real steps, END-aligned
    values[1, 10:, 0] = np.arange(40, dtype=float)  # 40 real steps
    data = ArrayAdapter(
        values,
        feature_names=["y"],
        categorical_features=[],
        entity_lengths=[30, 40],
        alignment=Alignment.END,
    )
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=10)
    handler = ForecastingHandler(_perfect, config, target="y")
    prediction = handler.predict(data)
    assert np.isfinite(prediction.predicted).all()
    assert np.allclose(prediction.predicted, handler.actual(data))


def test_unequal_usable_lengths_are_refused_for_rolling_forecasts():
    values = np.full((2, 50, 1), np.nan)
    values[0, 20:, 0] = np.arange(30, dtype=float)
    values[1, 10:, 0] = np.arange(40, dtype=float)
    data = ArrayAdapter(
        values,
        feature_names=["y"],
        categorical_features=[],
        entity_lengths=[30, 40],
        alignment=Alignment.END,
    )
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=10, stride=3)
    handler = ForecastingHandler(_perfect, config, target="y")
    with pytest.raises(ValueError, match="entities of equal usable length"):
        handler.historical_forecasts(data)


def test_a_series_too_short_for_the_horizon_says_so():
    data = ArrayAdapter(
        np.arange(6, dtype=float).reshape(1, 6, 1),
        feature_names=["y"],
        categorical_features=[],
    )
    handler = ForecastingHandler(_perfect, ForecastingConfig(forecast_horizon=6))
    with pytest.raises(ValueError, match="cannot hold a 6-step forecast"):
        handler.predict(data)


def test_a_non_callable_model_is_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        ForecastingHandler("not a function", ForecastingConfig(forecast_horizon=1))
