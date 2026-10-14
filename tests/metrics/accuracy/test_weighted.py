"""Tests for the event-weighted accuracy metrics ported from repo B.

Two properties matter most and are checked repeatedly here. First, the weighted forms
*reduce exactly* to their unweighted counterparts under uniform weights, so a use case can
move between them and the numbers stay comparable. Second, the weighting actually changes
the ranking of models in the way it is supposed to — otherwise there is no reason to have
it.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.metrics.accuracy.weighted import (
    event_weights,
    mean_signed_deviation,
    step_onsets,
    weighted_mae,
    weighted_mse,
)


# --------------------------------------------------------------------------
# onset detection
# --------------------------------------------------------------------------


def test_step_onsets_finds_each_rising_edge():
    series = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    assert step_onsets(series).tolist() == [2, 7]


def test_step_onsets_counts_a_series_that_starts_active():
    assert step_onsets(np.array([1, 1, 0, 0, 1])).tolist() == [0, 4]


def test_step_onsets_ignores_a_flicker_shorter_than_the_minimum():
    """Without the duration check every glitch would place a weight bump."""
    series = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0])
    assert step_onsets(series, min_duration=1).tolist() == [1, 4]
    assert step_onsets(series, min_duration=3).tolist() == [4]


def test_step_onsets_returns_nothing_for_a_series_that_never_activates():
    assert step_onsets(np.zeros(20)).size == 0


def test_step_onsets_rejects_an_empty_series():
    with pytest.raises(ValueError, match="empty"):
        step_onsets(np.array([]))


def test_step_onsets_rejects_a_non_positive_duration():
    with pytest.raises(ValueError, match="at least 1"):
        step_onsets(np.array([0, 1]), min_duration=0)


# --------------------------------------------------------------------------
# the weight function
# --------------------------------------------------------------------------


def test_weights_sum_to_one():
    series = np.zeros(100)
    series[50:60] = 1
    assert event_weights(series).sum() == pytest.approx(1.0)


def test_weights_peak_at_the_onset():
    series = np.zeros(100)
    series[50:60] = 1
    weights = event_weights(series, sigma=5.0)
    assert weights.argmax() == 50
    assert weights[50] > weights[30]
    assert weights[50] > weights[80]


def test_the_peak_to_floor_ratio_is_what_it_says():
    """A timestep at an onset counts ``peak_to_floor`` times one far from any event."""
    series = np.zeros(400)
    series[200:210] = 1
    for ratio in (2.0, 5.0, 10.0):
        weights = event_weights(series, sigma=2.0, peak_to_floor=ratio)
        assert weights[200] / weights[0] == pytest.approx(ratio, rel=1e-6)


def test_a_peak_to_floor_of_one_recovers_uniform_weighting_exactly():
    """The useful null case: the weighted metrics must collapse to the plain ones."""
    series = np.zeros(50)
    series[20:25] = 1
    weights = event_weights(series, peak_to_floor=1.0)
    assert np.allclose(weights, 1.0 / 50)


def test_a_series_with_no_events_falls_back_to_uniform_weights():
    """Degrading gracefully beats raising: a quiet period is a legitimate test set."""
    weights = event_weights(np.zeros(30))
    assert np.allclose(weights, 1.0 / 30)


def test_a_narrower_sigma_concentrates_the_weight():
    series = np.zeros(200)
    series[100:110] = 1
    narrow = event_weights(series, sigma=2.0)
    wide = event_weights(series, sigma=40.0)
    assert narrow[95:115].sum() > wide[95:115].sum()


def test_the_default_sigma_scales_with_the_series_length():
    short, long = np.zeros(100), np.zeros(1000)
    short[50] = long[500] = 1
    # 5% of the length: the bump stays proportionally the same width.
    assert event_weights(short)[40:61].sum() == pytest.approx(
        event_weights(long)[400:611].sum(), rel=0.05
    )


def test_onsets_can_be_supplied_directly():
    """The seam where use-case semantics enter."""
    series = np.zeros(100)
    weights = event_weights(series, onsets=[10, 60], sigma=3.0)
    assert weights[10] == pytest.approx(weights[60])
    assert weights[10] > weights[35]


def test_onsets_can_be_supplied_as_a_callable():
    series = np.linspace(0.0, 10.0, 100)
    crossings = lambda y: [int(np.argmax(y > 5.0))]  # noqa: E731 - the point is brevity
    weights = event_weights(series, onsets=crossings, sigma=3.0)
    assert weights.argmax() == int(np.argmax(series > 5.0))


def test_multiple_onsets_each_get_a_bump():
    series = np.zeros(300)
    weights = event_weights(series, onsets=[50, 150, 250], sigma=5.0)
    for position in (50, 150, 250):
        assert weights[position] > weights[position + 40]


def test_an_out_of_range_onset_is_rejected():
    with pytest.raises(ValueError, match=r"must lie in \[0, 49\]"):
        event_weights(np.zeros(50), onsets=[70])


def test_a_peak_to_floor_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        event_weights(np.zeros(10), onsets=[5], peak_to_floor=0.5)


# --------------------------------------------------------------------------
# the weighted errors
# --------------------------------------------------------------------------


@pytest.fixture
def series():
    """A quiet stretch with one event, and two models with the same plain MAE.

    ``near`` is wrong only around the event; ``far`` is wrong only in the quiet period.
    Their unweighted errors are identical by construction, which is what makes them the
    right pair for testing that weighting changes the ranking.
    """
    actual = np.zeros(200)
    actual[100:120] = 1.0
    near = actual.copy()
    near[95:105] += 1.0
    far = actual.copy()
    far[10:20] += 1.0
    return actual, near, far


def test_weighted_mae_with_uniform_weights_is_plain_mae():
    rng = np.random.default_rng(0)
    actual, predicted = rng.normal(size=100), rng.normal(size=100)
    expected = float(np.mean(np.abs(actual - predicted)))
    assert weighted_mae(actual, predicted) == pytest.approx(expected)


def test_weighted_mse_with_uniform_weights_is_plain_mse():
    rng = np.random.default_rng(1)
    actual, predicted = rng.normal(size=100), rng.normal(size=100)
    expected = float(np.mean((actual - predicted) ** 2))
    assert weighted_mse(actual, predicted) == pytest.approx(expected)


def test_weighting_changes_which_model_looks_better(series):
    """The whole reason the module exists: a plain MAE cannot tell these two apart."""
    actual, near, far = series
    assert weighted_mae(actual, near) == pytest.approx(weighted_mae(actual, far))

    weights = event_weights(actual, sigma=5.0, peak_to_floor=5.0)
    assert weighted_mae(actual, near, weights) > weighted_mae(actual, far, weights)


def test_weights_are_normalised_before_use():
    """Scaling every weight by a constant cannot change the answer."""
    rng = np.random.default_rng(2)
    actual, predicted = rng.normal(size=50), rng.normal(size=50)
    weights = rng.uniform(0.1, 1.0, 50)
    assert weighted_mae(actual, predicted, weights) == pytest.approx(
        weighted_mae(actual, predicted, weights * 1000.0)
    )


def test_a_weighted_error_stays_on_the_scale_of_its_unweighted_form(series):
    """Normalisation is what makes the two directly comparable."""
    actual, near, _ = series
    weights = event_weights(actual, sigma=5.0)
    plain = weighted_mae(actual, near)
    weighted = weighted_mae(actual, near, weights)
    assert 0.1 * plain < weighted < 20 * plain


def test_weighted_errors_are_zero_for_a_perfect_forecast():
    actual = np.linspace(0.0, 5.0, 40)
    weights = event_weights(actual, onsets=[10], sigma=3.0)
    assert weighted_mae(actual, actual, weights) == pytest.approx(0.0)
    assert weighted_mse(actual, actual, weights) == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must align"):
        weighted_mae(np.zeros(10), np.zeros(9))


def test_mismatched_weight_length_is_rejected():
    with pytest.raises(ValueError, match="weights must align"):
        weighted_mae(np.zeros(10), np.zeros(10), np.ones(3))


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_mae(np.zeros(3), np.zeros(3), np.array([1.0, -1.0, 1.0]))


def test_weights_summing_to_zero_are_rejected():
    with pytest.raises(ValueError, match="sum to zero"):
        weighted_mae(np.zeros(3), np.zeros(3), np.zeros(3))


# --------------------------------------------------------------------------
# the signed metric
# --------------------------------------------------------------------------


def test_mean_signed_deviation_is_positive_when_predicting_above_the_truth():
    """Repo B's convention: positive means the prediction was corrected upwards."""
    actual = np.zeros(10)
    assert mean_signed_deviation(actual, actual + 2.0) == pytest.approx(2.0)
    assert mean_signed_deviation(actual, actual - 2.0) == pytest.approx(-2.0)


def test_mean_signed_deviation_cancels_a_symmetric_error():
    """Zero is best, and this is the case an absolute error would score as bad."""
    actual = np.zeros(100)
    predicted = np.tile([1.0, -1.0], 50)
    assert mean_signed_deviation(actual, predicted) == pytest.approx(0.0)
    assert weighted_mae(actual, predicted) == pytest.approx(1.0)


def test_mean_signed_deviation_separates_two_models_with_identical_mae():
    """What it catches that an absolute error cannot: consistent bias."""
    actual = np.linspace(0.0, 10.0, 50)
    early, late = actual + 1.0, actual - 1.0
    assert weighted_mae(actual, early) == pytest.approx(weighted_mae(actual, late))
    assert mean_signed_deviation(actual, early) == pytest.approx(1.0)
    assert mean_signed_deviation(actual, late) == pytest.approx(-1.0)


def test_mean_signed_deviation_respects_weights():
    actual = np.zeros(10)
    predicted = np.array([5.0] + [0.0] * 9)
    weights = np.zeros(10)
    weights[0] = 1.0
    assert mean_signed_deviation(actual, predicted, weights) == pytest.approx(5.0)
    assert mean_signed_deviation(actual, predicted) == pytest.approx(0.5)
