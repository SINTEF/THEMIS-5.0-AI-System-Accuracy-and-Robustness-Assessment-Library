"""Tests for accuracy along the time axis.

The fixtures are constructed so the right answer is known: a forecast that is exactly
right must score zero at every origin, and a series with a known regime break must show
that break in the curve. The argument-order test matters most — repo B called
``metric(y_pred, y_true)`` following darts, this library is ``(actual, predicted)``, and
the difference is invisible for MAE and inverts the sign of anything asymmetric.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error

from themis_assess.metrics.accuracy.temporal import (
    accuracy_by_horizon,
    accuracy_over_time,
    aggregate_over_time,
    worst_windows,
)


@pytest.fixture
def rolling():
    """(2 entities, 6 origins, 4 horizon), with a known error pattern.

    Entity 0 is perfect. Entity 1 is off by exactly the origin index, so its per-origin
    MAE is 0, 1, 2, 3, 4, 5 — every aggregate is then computable by hand.
    """
    actual = np.zeros((2, 6, 4))
    forecasts = actual.copy()
    for origin in range(6):
        forecasts[1, origin, :] = origin
    return forecasts, actual


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def test_a_single_entity_may_be_two_dimensional(rolling):
    forecasts, actual = rolling
    curve = accuracy_over_time(forecasts[1], actual[1], mean_absolute_error)
    assert curve.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        accuracy_over_time(np.zeros((3, 4)), np.zeros((3, 5)), mean_absolute_error)


def test_a_four_dimensional_array_is_rejected():
    with pytest.raises(ValueError, match="n_origins"):
        accuracy_over_time(np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 2, 2)), mean_absolute_error)


# --------------------------------------------------------------------------
# per origin
# --------------------------------------------------------------------------


def test_a_perfect_forecast_scores_zero_at_every_origin(rolling):
    forecasts, actual = rolling
    curve = accuracy_over_time(forecasts[0], actual[0], mean_absolute_error)
    assert curve == pytest.approx(np.zeros(6))


def test_entities_are_averaged_by_default(rolling):
    """Entity 0 contributes 0 at every origin, entity 1 contributes the origin index."""
    forecasts, actual = rolling
    curve = accuracy_over_time(forecasts, actual, mean_absolute_error)
    assert curve.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def test_entities_can_be_kept_separate(rolling):
    """Averaging across entities hides an entity the model handles badly; repo B's
    vessels are not interchangeable."""
    forecasts, actual = rolling
    curves = accuracy_over_time(forecasts, actual, mean_absolute_error, per_entity=True)
    assert curves.shape == (2, 6)
    assert curves[0] == pytest.approx(np.zeros(6))
    assert curves[1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_metric_receives_actual_first():
    """Repo B called metric(y_pred, y_true), following darts. Invisible for MAE, and it
    inverts the sign of anything asymmetric."""
    seen = []

    def recording(first, second):
        seen.append((float(first[0]), float(second[0])))
        return 0.0

    actual = np.full((1, 1, 1), 7.0)
    forecasts = np.full((1, 1, 1), 9.0)
    accuracy_over_time(forecasts, actual, recording)
    assert seen == [(7.0, 9.0)]  # actual, then predicted


def test_a_signed_metric_keeps_its_sign():
    """The consequence of the argument order, made concrete."""
    from themis_assess.metrics.accuracy.weighted import mean_signed_deviation

    actual = np.zeros((1, 2, 3))
    forecasts = np.full((1, 2, 3), 4.0)  # predicting above the truth
    curve = accuracy_over_time(forecasts, actual, mean_signed_deviation)
    assert curve.tolist() == [4.0, 4.0]  # positive, not -4


def test_a_regime_break_shows_up_in_the_curve():
    rng = np.random.default_rng(0)
    actual = np.zeros((1, 20, 5))
    forecasts = actual.copy()
    forecasts[0, 10:, :] = rng.normal(scale=3.0, size=(10, 5))  # calm, then not
    curve = accuracy_over_time(forecasts, actual, mean_absolute_error)
    assert curve[:10].mean() < 0.01
    assert curve[10:].mean() > 1.0


# --------------------------------------------------------------------------
# per horizon
# --------------------------------------------------------------------------


def test_error_grows_with_the_horizon():
    """Almost always true, and where it is not, that is a finding rather than a result."""
    rng = np.random.default_rng(1)
    horizon = 8
    actual = np.zeros((1, 30, horizon))
    # error growing linearly with steps ahead
    forecasts = np.arange(1, horizon + 1) * rng.normal(scale=0.3, size=(1, 30, horizon))
    curve = accuracy_by_horizon(forecasts, actual, mean_absolute_error)
    assert curve.shape == (horizon,)
    assert curve[-1] > curve[0]


def test_per_horizon_pools_over_origins_and_entities(rolling):
    """Entity 1's offset is constant across the horizon, so every step scores the same."""
    forecasts, actual = rolling
    curve = accuracy_by_horizon(forecasts, actual, mean_absolute_error)
    assert curve == pytest.approx(np.full(4, np.mean([0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])))


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_the_aggregations_are_the_hand_computed_ones():
    curve = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    assert aggregate_over_time(curve) == pytest.approx(21.2)
    assert aggregate_over_time(curve, method="median") == pytest.approx(2.0)
    assert aggregate_over_time(curve, method="max") == pytest.approx(100.0)
    assert aggregate_over_time(curve, method="min") == pytest.approx(0.0)


def test_mean_and_median_diverge_when_a_few_origins_dominate():
    """Which for event forecasting is the normal case, not the pathological one."""
    curve = np.r_[np.full(20, 0.1), np.full(2, 50.0)]
    assert aggregate_over_time(curve, method="median") == pytest.approx(0.1)
    assert aggregate_over_time(curve, method="mean") > 4.0


def test_the_worst_case_survives_aggregation():
    """Survey §8 asks for the worst case beside the average, and a mean over origins is
    exactly where a worst case goes missing."""
    curve = np.r_[np.full(50, 0.1), [12.0]]
    assert aggregate_over_time(curve, method="mean") < 0.35
    assert aggregate_over_time(curve, method="max") == pytest.approx(12.0)


def test_weights_emphasise_the_origins_that_matter():
    curve = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert aggregate_over_time(curve, weights=weights) == pytest.approx(10.0)
    assert aggregate_over_time(curve) == pytest.approx(2.8)


def test_weights_are_normalised():
    curve = np.array([2.0, 4.0])
    assert aggregate_over_time(curve, weights=[1.0, 1.0]) == pytest.approx(3.0)
    assert aggregate_over_time(curve, weights=[100.0, 100.0]) == pytest.approx(3.0)


def test_weights_with_a_non_mean_aggregation_are_refused():
    """Quietly dropping them would misreport what was computed."""
    with pytest.raises(ValueError, match="only meaningful with method='mean'"):
        aggregate_over_time(np.arange(3.0), method="median", weights=np.ones(3))


def test_aggregation_validates_its_arguments():
    with pytest.raises(ValueError, match="Nothing to aggregate"):
        aggregate_over_time(np.array([]))
    with pytest.raises(ValueError, match="method must be one of"):
        aggregate_over_time(np.arange(3.0), method="magic")
    with pytest.raises(ValueError, match="weight"):
        aggregate_over_time(np.arange(3.0), weights=np.ones(2))
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_over_time(np.arange(3.0), weights=[-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        aggregate_over_time(np.arange(3.0), weights=np.zeros(3))


# --------------------------------------------------------------------------
# worst windows
# --------------------------------------------------------------------------


def test_the_worst_origins_are_ranked_first():
    curve = np.array([0.1, 5.0, 0.2, 3.0, 0.3])
    worst = worst_windows(curve, n=2)
    assert [w["position"] for w in worst] == [1, 3]
    assert worst[0]["value"] == pytest.approx(5.0)


def test_direction_is_respected():
    curve = np.array([0.1, 5.0, 0.2])
    assert worst_windows(curve, n=1, higher_is_better=True)[0]["position"] == 0
    assert worst_windows(curve, n=1, higher_is_better=False)[0]["position"] == 1


def test_origins_are_labelled_with_real_timestamps():
    """'Origin 47' sends nobody anywhere."""
    curve = np.array([0.1, 9.0, 0.2])
    stamps = np.array(["2026-01-01", "2026-02-01", "2026-03-01"], dtype="datetime64[D]")
    worst = worst_windows(curve, n=1, origins=stamps.tolist())
    assert str(worst[0]["origin"]) == "2026-02-01"


def test_worst_windows_validates_its_arguments():
    with pytest.raises(ValueError, match="Nothing to rank"):
        worst_windows(np.array([]))
    with pytest.raises(ValueError, match="n must be at least 1"):
        worst_windows(np.arange(3.0), n=0)
    with pytest.raises(ValueError, match="origin label"):
        worst_windows(np.arange(3.0), origins=["a"])
