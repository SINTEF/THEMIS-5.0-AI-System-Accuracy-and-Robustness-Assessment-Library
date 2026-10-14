"""Tests for the shipped baseline handlers.

A baseline is not a convenience here — four of the survey's §5.1.3 metrics are undefined
without one — so these check the properties those metrics rely on, chiefly that a
constant baseline really is constant under perturbation.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.handlers.baselines import (
    MajorityClassHandler,
    MovingAverageHandler,
    NaiveLastValueHandler,
    PriorProbabilityHandler,
)
from themis_assess.handlers.forecasting import ForecastingConfig


@pytest.fixture
def tabular():
    rng = np.random.default_rng(0)
    return ArrayAdapter(
        rng.normal(size=(40, 3)), feature_names=["a", "b", "c"], categorical_features=[]
    )


@pytest.fixture
def trending():
    """One entity whose target rises linearly, with a covariate."""
    t = np.arange(80, dtype=float)
    values = np.stack([t, np.zeros(80)], axis=1)[None, :, :]
    return ArrayAdapter(values, feature_names=["y", "pad"], categorical_features=[])


# --------------------------------------------------------------------------
# classification baselines
# --------------------------------------------------------------------------


def test_majority_class_predicts_the_most_frequent_label(tabular):
    labels = np.array([0] * 30 + [1] * 70)
    handler = MajorityClassHandler(labels)
    assert handler.majority == 1
    assert set(handler.predict(tabular).predicted.tolist()) == {1}


def test_majority_class_is_unaffected_by_the_input(tabular):
    """The property that makes it a valid CE denominator: it isolates the model's own
    sensitivity, because the baseline has none."""
    handler = MajorityClassHandler(np.array([0, 0, 1]))
    noisy = tabular.with_values(tabular.values() + 100.0)
    assert np.array_equal(
        handler.predict(tabular).predicted, handler.predict(noisy).predicted
    )


def test_prior_probability_returns_the_training_distribution(tabular):
    labels = np.array([0] * 25 + [1] * 75)
    prediction = PriorProbabilityHandler(labels).predict(tabular)
    assert np.allclose(prediction.probabilities[0], [0.25, 0.75])
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0)


def test_prior_probability_declares_probabilistic_access(tabular):
    handler = PriorProbabilityHandler(np.array([0, 1, 1]))
    assert handler.access is AccessLevel.PROBABILISTIC
    assert handler.task is TaskType.CLASSIFICATION


def test_prior_probability_is_well_calibrated_on_its_own_distribution():
    """It knows nothing and is honest about it, which is what makes it the right
    reference for a calibration comparison."""
    from themis_assess.metrics.robustness.calibration import expected_calibration_error

    rng = np.random.default_rng(1)
    labels = rng.choice([0, 1], size=2000, p=[0.3, 0.7])
    data = ArrayAdapter(
        rng.normal(size=(2000, 2)), feature_names=["a", "b"], categorical_features=[]
    )
    prediction = PriorProbabilityHandler(labels).predict(data)
    assert expected_calibration_error(prediction.probabilities, labels) < 0.02


def test_a_classification_baseline_needs_labels():
    with pytest.raises(ValueError, match="at least one training label"):
        MajorityClassHandler(np.array([]))
    with pytest.raises(ValueError, match="at least one training label"):
        PriorProbabilityHandler(np.array([]))


# --------------------------------------------------------------------------
# forecasting baselines
# --------------------------------------------------------------------------


def test_naive_last_value_holds_the_final_observation_flat(trending):
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=10)
    prediction = NaiveLastValueHandler(config, target="y").predict(trending)
    # The series is 0..79; the forecast window is the last 5, so history ends at 74.
    assert prediction.predicted.ravel().tolist() == [74.0] * 5


def test_moving_average_averages_the_window(trending):
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=10)
    prediction = MovingAverageHandler(config, window=10, target="y").predict(trending)
    # Mean of 65..74.
    assert prediction.predicted.ravel().tolist() == [69.5] * 5


def test_moving_average_defaults_its_window_to_the_input_chunk_length():
    config = ForecastingConfig(forecast_horizon=3, input_chunk_length=7)
    assert MovingAverageHandler(config).window == 7


def test_a_moving_average_lags_a_trend_more_than_the_naive_forecast(trending):
    """Which is the right ordering, and a useful sanity check on a baseline set: on a
    trending series the smoother baseline is the worse one."""
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=20)
    naive = NaiveLastValueHandler(config, target="y")
    smooth = MovingAverageHandler(config, window=20, target="y")
    actual = np.arange(75, 80, dtype=float)
    naive_error = np.mean(np.abs(naive.predict(trending).predicted - actual))
    smooth_error = np.mean(np.abs(smooth.predict(trending).predicted - actual))
    assert smooth_error > naive_error


def test_a_forecasting_baseline_reads_only_its_target_column():
    t = np.arange(40, dtype=float)
    values = np.stack([np.full(40, -999.0), t], axis=1)[None, :, :]
    data = ArrayAdapter(values, feature_names=["decoy", "y"], categorical_features=[])
    config = ForecastingConfig(forecast_horizon=4, input_chunk_length=8)
    prediction = NaiveLastValueHandler(config, target="y").predict(data)
    assert prediction.predicted.ravel().tolist() == [35.0] * 4


def test_a_forecasting_baseline_respects_ragged_lengths():
    values = np.full((2, 40, 1), np.nan)
    values[0, 10:, 0] = np.arange(30, dtype=float)
    values[1, 10:, 0] = np.arange(100, 130, dtype=float)
    data = ArrayAdapter(
        values,
        feature_names=["y"],
        categorical_features=[],
        entity_lengths=[30, 30],
    )
    config = ForecastingConfig(forecast_horizon=3, input_chunk_length=5)
    prediction = NaiveLastValueHandler(config, target="y").predict(data)
    assert prediction.predicted[0].tolist() == [26.0] * 3
    assert prediction.predicted[1].tolist() == [126.0] * 3


def test_a_moving_average_window_must_be_positive():
    with pytest.raises(ValueError, match="window must be at least 1"):
        MovingAverageHandler(ForecastingConfig(forecast_horizon=1), window=0)


def test_a_series_too_short_for_the_horizon_says_so():
    data = ArrayAdapter(
        np.arange(4, dtype=float).reshape(1, 4, 1),
        feature_names=["y"],
        categorical_features=[],
    )
    config = ForecastingConfig(forecast_horizon=4)
    with pytest.raises(ValueError, match="cannot hold a 4-step forecast"):
        NaiveLastValueHandler(config, target="y").predict(data)
