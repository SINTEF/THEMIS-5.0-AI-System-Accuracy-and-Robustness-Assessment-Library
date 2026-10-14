"""Tests for sensitivity maps.

The fixtures are built so the right answer is known from the generative process: one
feature with a large coefficient, one with a small one, two that are pure noise. A map
that works must recover that ordering, and the tipping-point test pins where each
feature starts to matter.

The strongest test here is the time map against a persistence forecaster: it reads only
the final value of its target, so exactly one cell of the grid may be non-zero.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from themis_assess.assessment import RobustnessEvaluator
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler
from themis_assess.handlers.sklearn import SklearnHandler
from themis_assess.sensitivity.maps import (
    SensitivityMap,
    SensitivityMapper,
    map_from_sweep,
)


@pytest.fixture(scope="module")
def graded():
    """Four features: one dominant, one minor, two irrelevant."""
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 4))
    y = (2.5 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X[:500], y[:500])
    data = ArrayAdapter(
        X[500:],
        feature_names=["important", "minor", "noise1", "noise2"],
        categorical_features=[],
    )
    return data, y[500:], SklearnHandler(model, name="rf")


@pytest.fixture
def persistence():
    """A forecaster that returns the last observed target value, held flat."""
    t = np.arange(200, dtype=float)
    values = np.stack([np.sin(t / 9.0), np.cos(t / 13.0)], axis=1)[None, :, :]
    data = ArrayAdapter(values, feature_names=["y", "x"], categorical_features=[])
    config = ForecastingConfig(forecast_horizon=6, input_chunk_length=20)
    handler = ForecastingHandler(
        lambda history, horizon: np.full(horizon, history[-1, 0]),
        config,
        name="persistence",
        target="y",
    )
    return data, handler


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------


def test_a_non_square_grid_is_fine_but_labels_must_match():
    with pytest.raises(ValueError, match="does not match"):
        SensitivityMap(
            grid=np.zeros((2, 3)),
            row_labels=("a", "b"),
            column_labels=("1", "2"),
            measure="prediction_shift",
        )


def test_a_one_dimensional_grid_is_rejected():
    with pytest.raises(ValueError, match="must be 2-D"):
        SensitivityMap(
            grid=np.zeros(3),
            row_labels=("a",),
            column_labels=("1",),
            measure="prediction_shift",
        )


def test_an_unknown_row_lists_the_available_ones():
    sensitivity_map = SensitivityMap(
        grid=np.zeros((1, 1)),
        row_labels=("a",),
        column_labels=("1",),
        measure="prediction_shift",
    )
    with pytest.raises(KeyError, match="Available"):
        sensitivity_map.row("b")


# --------------------------------------------------------------------------
# feature by severity
# --------------------------------------------------------------------------


def test_the_map_recovers_the_generative_importance(graded):
    """The map's whole claim, tested against data whose answer is known."""
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(
        handler, n_variants=5, seed=1
    ).feature_severity_map(data)

    ranked = [row["label"] for row in sensitivity_map.most_sensitive_rows(n=4)]
    assert ranked[0] == "important"
    assert ranked[1] == "minor"
    assert set(ranked[2:]) == {"noise1", "noise2"}


def test_sensitivity_grows_with_severity(graded):
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(
        handler, n_variants=5, seed=1
    ).feature_severity_map(data)
    curve = sensitivity_map.row("important")
    assert curve[0] < curve[-1]
    assert curve[-1] > 5 * curve[0]


def test_an_irrelevant_feature_barely_moves_the_prediction(graded):
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(
        handler, n_variants=5, seed=1
    ).feature_severity_map(data)
    assert sensitivity_map.row("noise1").max() < 0.05
    assert sensitivity_map.row("important").max() > 0.3


def test_the_tipping_point_orders_the_features(graded):
    """Not 'how bad does it get' but 'how much can this feature take before it matters'."""
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(
        handler, n_variants=5, seed=1
    ).feature_severity_map(data)
    assert sensitivity_map.tipping_point("important", 0.05) == "1"
    assert sensitivity_map.tipping_point("minor", 0.05) == "4"
    assert sensitivity_map.tipping_point("noise1", 0.05) is None


def test_the_hottest_cell_names_both_axes(graded):
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(
        handler, n_variants=3, seed=1
    ).feature_severity_map(data)
    hottest = sensitivity_map.hottest_cell()
    assert hottest["feature"] == "important"
    assert hottest["severity"] == "5"


def test_max_and_mean_aggregation_can_disagree():
    """A feature with one catastrophic severity and one uniformly mediocre are different
    problems, so the aggregate is a parameter."""
    sensitivity_map = SensitivityMap(
        grid=np.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.3, 0.3, 0.3, 0.3, 0.3]]),
        row_labels=("spiky", "steady"),
        column_labels=tuple("12345"),
        measure="prediction_shift",
    )
    assert sensitivity_map.most_sensitive_rows(n=1, aggregate="max")[0]["label"] == "spiky"
    assert (
        sensitivity_map.most_sensitive_rows(n=1, aggregate="mean")[0]["label"] == "steady"
    )


def test_the_plan_states_the_cost_before_it_is_incurred(graded):
    data, _, handler = graded
    plan = SensitivityMapper(
        handler, severities=(1, 2, 3), n_variants=4
    ).plan(data)
    assert plan["cells"] == 4 * 3
    assert plan["n_model_calls"] == 1 + 12 * 4
    assert plan["measure"] == "prediction_shift"


# --------------------------------------------------------------------------
# shift versus degradation
# --------------------------------------------------------------------------


def test_prediction_shift_needs_no_labels(graded):
    """The reason it is the default: it can be computed on production data."""
    data, _, handler = graded
    sensitivity_map = SensitivityMapper(handler, n_variants=2, seed=1).feature_severity_map(
        data
    )
    assert sensitivity_map.measure == "prediction_shift"
    assert sensitivity_map.metric is None


def test_degradation_needs_labels_and_a_metric(graded):
    data, actual, handler = graded
    with pytest.raises(ValueError, match="needs a metric"):
        SensitivityMapper(handler, measure="degradation")

    mapper = SensitivityMapper(
        handler, measure="degradation", metric="accuracy", n_variants=2
    )
    with pytest.raises(ValueError, match="needs the ground truth"):
        mapper.feature_severity_map(data)
    assert mapper.feature_severity_map(data, actual).measure == "degradation"


def test_a_degradation_map_is_oriented_so_higher_is_more_harm(graded):
    """Without this a map over accuracy and a map over an error would read in opposite
    directions."""
    data, actual, handler = graded
    accuracy_map = SensitivityMapper(
        handler, measure="degradation", metric="accuracy", n_variants=4, seed=1
    ).feature_severity_map(data, actual)
    error_map = SensitivityMapper(
        handler,
        measure="degradation",
        metric="negative_log_likelihood",
        n_variants=4,
        seed=1,
    ).feature_severity_map(data, actual)

    for grid in (accuracy_map, error_map):
        assert grid.row("important").max() > grid.row("noise1").max()
        assert grid.most_sensitive_rows(n=1)[0]["label"] == "important"


def test_an_unknown_measure_is_rejected(graded):
    _, _, handler = graded
    with pytest.raises(ValueError, match="measure must be"):
        SensitivityMapper(handler, measure="vibes")


# --------------------------------------------------------------------------
# feature by time
# --------------------------------------------------------------------------


def test_a_persistence_forecaster_is_sensitive_only_to_its_last_value(persistence):
    """The strongest available check on the time axis: this model reads exactly one
    number, so exactly one cell of the grid may be non-zero."""
    data, handler = persistence
    time_map = SensitivityMapper(
        handler, n_variants=4, seed=3
    ).feature_time_map(data, severity=3, time_granularity=10, window=60)

    assert time_map.row("x").max() == pytest.approx(0.0)
    y_row = time_map.row("y")
    assert y_row[-1] > 0.1  # the final group holds the value it reads
    assert y_row[:-1].max() == pytest.approx(0.0)


def test_the_time_window_is_a_parameter(persistence):
    """Repo B hardcoded 'only the last 120 time steps'."""
    data, handler = persistence
    mapper = SensitivityMapper(handler, n_variants=1, seed=3)
    narrow = mapper.feature_time_map(data, time_granularity=10, window=30)
    wide = mapper.feature_time_map(data, time_granularity=10, window=100)
    assert narrow.shape[1] == 3
    assert wide.shape[1] == 10


def test_the_time_columns_are_labelled_with_their_timesteps(persistence):
    data, handler = persistence
    time_map = SensitivityMapper(handler, n_variants=1, seed=3).feature_time_map(
        data, time_granularity=10, window=20
    )
    assert time_map.column_labels == ("t180-189", "t190-199")
    assert time_map.column_axis == "time"


def test_a_time_map_needs_sequential_data(graded):
    data, _, handler = graded
    with pytest.raises(ValueError, match="needs sequential data"):
        SensitivityMapper(handler).feature_time_map(data)


def test_time_granularity_must_be_positive(persistence):
    data, handler = persistence
    with pytest.raises(ValueError, match="time_granularity must be at least 1"):
        SensitivityMapper(handler).feature_time_map(data, time_granularity=0)


# --------------------------------------------------------------------------
# from a sweep
# --------------------------------------------------------------------------


def test_a_map_from_a_sweep_costs_no_model_calls(graded):
    """A robustness assessment produces this as a by-product."""
    data, actual, handler = graded
    sweep = RobustnessEvaluator(
        handler,
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 3, 5),
        n_variants=2,
        seed=2,
    ).run(data, actual)

    sensitivity_map = map_from_sweep(sweep, "accuracy")
    assert sensitivity_map.row_axis == "perturbation"
    assert sensitivity_map.row_labels == ("gaussian_noise", "uniform_bias")
    assert sensitivity_map.column_labels == ("1", "3", "5")
    assert sensitivity_map.metadata["from_sweep"] is True


def test_a_sweep_map_is_oriented_like_every_other_map(graded):
    data, actual, handler = graded
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1, 5), n_variants=3, seed=2
    ).run(data, actual)

    accuracy_map = map_from_sweep(sweep, "accuracy")
    error_map = map_from_sweep(sweep, "negative_log_likelihood")
    # Severity 5 is worse than severity 1 under both, so both grids increase rightwards.
    assert accuracy_map.grid[0, 1] > accuracy_map.grid[0, 0]
    assert error_map.grid[0, 1] > error_map.grid[0, 0]


def test_a_directionless_metric_cannot_be_mapped(graded):
    data, actual, handler = graded
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1,), seed=2
    ).run(data, actual)
    with pytest.raises(ValueError, match="has no direction"):
        map_from_sweep(sweep, "predictive_entropy")
