"""Tests for survey §5.4, calibration under shift.

Several expected values here are computed by hand from the survey's equations rather than
captured from this implementation, so the tests would catch a change of behaviour that a
regression test recorded from the code itself would silently bless.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.metrics.robustness.calibration import (
    apply_temperature,
    brier_score,
    crps_ensemble,
    crps_gaussian,
    energy_score,
    expected_calibration_error,
    fit_temperature,
    interval_score,
    mean_prediction_interval_width,
    negative_log_likelihood,
    prediction_interval_coverage,
    predictive_entropy,
    reliability_bins,
)


# --------------------------------------------------------------------------
# input validation, shared by the classification metrics
# --------------------------------------------------------------------------


def test_a_one_dimensional_probability_is_read_as_binary():
    """Handlers commonly return only P(class 1); both spellings must agree."""
    p1 = np.array([0.2, 0.9, 0.6])
    matrix = np.column_stack([1 - p1, p1])
    labels = np.array([0, 1, 1])
    assert brier_score(p1, labels) == pytest.approx(brier_score(matrix, labels))


def test_probabilities_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match="must sum to 1"):
        expected_calibration_error(np.array([[0.5, 0.9], [0.3, 0.2]]), np.array([0, 1]))


def test_logits_are_rejected_with_a_pointer_to_softmax():
    with pytest.raises(ValueError, match="softmax"):
        expected_calibration_error(np.array([[-2.0, 3.0]]), np.array([1]))


def test_mismatched_labels_are_rejected():
    with pytest.raises(ValueError, match="label"):
        brier_score(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0]))


# --------------------------------------------------------------------------
# ECE and reliability bins
# --------------------------------------------------------------------------


def test_a_perfectly_calibrated_model_has_zero_ece():
    """Ten samples at confidence 0.8, exactly eight correct: gap zero, ECE zero."""
    probs = np.tile([0.2, 0.8], (10, 1))
    labels = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_ece_equals_the_hand_computed_gap():
    """One occupied bin: confidence 0.9, accuracy 0.5, so ECE = 1.0 * |0.5 - 0.9| = 0.4."""
    probs = np.tile([0.1, 0.9], (10, 1))
    labels = np.array([1] * 5 + [0] * 5)
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4)


def test_ece_weights_bins_by_occupancy():
    """80 samples at gap 0 and 20 at gap 0.5 give 0.2 * 0.5 = 0.1, not 0.25."""
    calibrated = np.tile([0.2, 0.8], (80, 1))
    calibrated_labels = np.array([1] * 64 + [0] * 16)  # 80% correct at confidence 0.8
    overconfident = np.tile([0.05, 0.95], (20, 1))
    overconfident_labels = np.array([1] * 9 + [0] * 11)  # 45% correct at confidence 0.95
    probs = np.vstack([calibrated, overconfident])
    labels = np.concatenate([calibrated_labels, overconfident_labels])
    # bin gaps: |0.8-0.8| = 0 over 80 samples, |0.45-0.95| = 0.5 over 20
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.1)


def test_an_overconfident_model_scores_worse_than_a_calibrated_one():
    rng = np.random.default_rng(0)
    n = 4000
    confidence = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < confidence  # honest by construction
    labels = np.where(correct, 1, 0)
    calibrated = np.column_stack([1 - confidence, confidence])
    sharpened = np.clip(confidence + 0.15, 0.0, 1.0)
    overconfident = np.column_stack([1 - sharpened, sharpened])
    assert expected_calibration_error(calibrated, labels) < 0.03
    assert expected_calibration_error(overconfident, labels) > 0.10


def test_reliability_bins_keep_empty_bins_visible():
    """A gap in the diagram is informative, so empty bins are NaN rather than dropped."""
    probs = np.tile([0.05, 0.95], (10, 1))
    bins = reliability_bins(probs, np.ones(10, dtype=int), n_bins=10)
    assert bins["count"].sum() == 10
    assert bins["count"][-1] == 10
    assert np.isnan(bins["confidence"][0])
    assert len(bins["bin_lower"]) == 10


def test_reliability_bins_place_confidence_of_exactly_one_in_the_last_bin():
    bins = reliability_bins(np.array([[0.0, 1.0]]), np.array([1]), n_bins=5)
    assert bins["count"][-1] == 1


# --------------------------------------------------------------------------
# proper scoring rules
# --------------------------------------------------------------------------


def test_brier_score_of_a_perfect_confident_prediction_is_zero():
    assert brier_score(np.array([[0.0, 1.0]]), np.array([1])) == pytest.approx(0.0)


def test_brier_score_of_a_confidently_wrong_prediction_is_two():
    """The multi-class form's maximum: (1-0)^2 + (0-1)^2."""
    assert brier_score(np.array([[0.0, 1.0]]), np.array([0])) == pytest.approx(2.0)


def test_brier_score_of_a_uniform_guess_on_two_classes_is_a_half():
    assert brier_score(np.array([[0.5, 0.5]]), np.array([0])) == pytest.approx(0.5)


def test_nll_of_a_uniform_guess_is_log_two():
    assert negative_log_likelihood(
        np.array([[0.5, 0.5]]), np.array([0])
    ) == pytest.approx(np.log(2.0))


def test_nll_punishes_a_confident_error_far_harder_than_brier():
    """The survey's reason for keeping both: they disagree about confident mistakes."""
    probs = np.array([[0.999, 0.001]])
    labels = np.array([1])
    assert brier_score(probs, labels) < 2.0
    assert negative_log_likelihood(probs, labels) > 6.0


def test_nll_is_finite_for_a_zero_probability_on_the_truth():
    value = negative_log_likelihood(np.array([[1.0, 0.0]]), np.array([1]))
    assert np.isfinite(value)
    assert value > 20.0


def test_predictive_entropy_is_maximal_for_a_uniform_distribution():
    uniform = predictive_entropy(np.array([[0.25, 0.25, 0.25, 0.25]]))
    certain = predictive_entropy(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert uniform[0] == pytest.approx(np.log(4.0))
    assert certain[0] == pytest.approx(0.0, abs=1e-9)  # the 1e-12 clip leaves a trace


def test_predictive_entropy_is_per_sample():
    entropy = predictive_entropy(np.array([[0.5, 0.5], [1.0, 0.0], [0.9, 0.1]]))
    assert entropy.shape == (3,)
    assert entropy[0] > entropy[2] > entropy[1]


# --------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------


def test_temperature_scaling_cannot_change_the_prediction():
    """The property that makes it safe: accuracy is exactly unchanged."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 4)) * 3.0
    for temperature in (0.2, 1.0, 5.0):
        scaled = apply_temperature(logits, temperature)
        assert np.array_equal(scaled.argmax(axis=1), logits.argmax(axis=1))


def test_a_higher_temperature_softens_confidence():
    logits = np.array([[0.0, 5.0]])
    assert apply_temperature(logits, 5.0).max() < apply_temperature(logits, 1.0).max()
    assert apply_temperature(logits, 0.5).max() > apply_temperature(logits, 1.0).max()


def test_apply_temperature_rows_sum_to_one():
    probs = apply_temperature(np.random.default_rng(0).normal(size=(50, 3)), 2.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_fitting_temperature_on_overconfident_logits_returns_above_one():
    rng = np.random.default_rng(2)
    n = 2000
    logits = rng.normal(size=(n, 3))
    labels = np.array(
        [rng.choice(3, p=np.exp(row / 4.0) / np.exp(row / 4.0).sum()) for row in logits]
    )
    # The labels follow the logits at temperature 4, so the raw logits are overconfident.
    temperature = fit_temperature(logits, labels)
    assert temperature > 2.0
    before = expected_calibration_error(apply_temperature(logits, 1.0), labels)
    after = expected_calibration_error(apply_temperature(logits, temperature), labels)
    assert after < before


def test_fitting_temperature_leaves_accuracy_untouched():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(500, 3)) * 2.0
    labels = rng.integers(0, 3, 500)
    temperature = fit_temperature(logits, labels)
    raw = apply_temperature(logits, 1.0).argmax(axis=1)
    tuned = apply_temperature(logits, temperature).argmax(axis=1)
    assert np.array_equal(raw, tuned)


def test_apply_temperature_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="must be positive"):
        apply_temperature(np.zeros((2, 2)), 0.0)


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------


def test_coverage_counts_the_observations_inside_their_interval():
    lower = np.zeros(10)
    upper = np.ones(10)
    actual = np.array([0.5] * 7 + [1.5] * 3)
    assert prediction_interval_coverage(lower, upper, actual) == pytest.approx(0.7)


def test_coverage_includes_the_endpoints():
    assert prediction_interval_coverage([0.0], [1.0], [1.0]) == pytest.approx(1.0)
    assert prediction_interval_coverage([0.0], [1.0], [0.0]) == pytest.approx(1.0)


def test_coverage_of_a_correctly_calibrated_normal_interval():
    rng = np.random.default_rng(4)
    y = rng.normal(size=5000)
    half = 1.6449  # the 90% two-sided normal interval
    covered = prediction_interval_coverage(np.full(5000, -half), np.full(5000, half), y)
    assert covered == pytest.approx(0.9, abs=0.02)


def test_coverage_alone_is_gameable_which_is_why_width_is_reported_with_it():
    """The survey's objection, made concrete: absurd intervals score perfectly."""
    y = np.random.default_rng(5).normal(size=200)
    lower, upper = np.full(200, -1e6), np.full(200, 1e6)
    assert prediction_interval_coverage(lower, upper, y) == 1.0
    assert mean_prediction_interval_width(lower, upper) == pytest.approx(2e6)


def test_mean_interval_width_is_the_mean_gap():
    assert mean_prediction_interval_width([0.0, 1.0], [2.0, 5.0]) == pytest.approx(3.0)


def test_interval_score_of_a_covered_observation_is_the_width():
    assert interval_score([0.0], [2.0], [1.0], alpha=0.1) == pytest.approx(2.0)


def test_interval_score_adds_the_scaled_miss():
    """y = 3 above an upper bound of 2, alpha = 0.1: 2 + (2/0.1)*1 = 22."""
    assert interval_score([0.0], [2.0], [3.0], alpha=0.1) == pytest.approx(22.0)


def test_interval_score_prefers_a_tight_honest_interval_to_an_absurd_one():
    """Where PICP ranks them equal, the interval score does not."""
    rng = np.random.default_rng(6)
    y = rng.normal(size=500)
    tight_lo, tight_hi = np.full(500, -1.6449), np.full(500, 1.6449)
    absurd_lo, absurd_hi = np.full(500, -1000.0), np.full(500, 1000.0)
    assert interval_score(tight_lo, tight_hi, y) < interval_score(absurd_lo, absurd_hi, y)


def test_intervals_with_an_upper_below_its_lower_are_rejected():
    for fn in (prediction_interval_coverage, interval_score):
        with pytest.raises(ValueError, match="at or above"):
            fn([1.0], [0.0], [0.5])


# --------------------------------------------------------------------------
# CRPS and the energy score
# --------------------------------------------------------------------------


def test_crps_of_a_single_member_ensemble_equals_mae():
    """The survey's stated property, and the reason CRPS is comparable with MAE."""
    rng = np.random.default_rng(7)
    y = rng.normal(size=300)
    point = y + rng.normal(scale=0.8, size=300)
    mae = float(np.mean(np.abs(point - y)))
    assert crps_ensemble(point[:, None], y) == pytest.approx(mae)


def test_crps_of_a_perfect_deterministic_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert crps_ensemble(y[:, None], y) == pytest.approx(0.0)


def test_crps_rewards_a_spread_that_covers_the_truth():
    """A point forecast that is off by 1 scores worse than an ensemble straddling it."""
    y = np.array([0.0])
    point = np.array([[1.0]])
    straddling = np.array([[-1.0, 0.0, 1.0]])
    assert crps_ensemble(straddling, y) < crps_ensemble(point, y)


def test_crps_penalises_a_needlessly_wide_ensemble():
    y = np.zeros(1)
    tight = np.array([[-0.1, 0.0, 0.1]])
    wide = np.array([[-50.0, 0.0, 50.0]])
    assert crps_ensemble(tight, y) < crps_ensemble(wide, y)


def test_crps_ensemble_agrees_with_the_gaussian_closed_form():
    rng = np.random.default_rng(8)
    y = np.array([0.3, -1.2, 2.0])
    mu = np.array([0.0, -1.0, 2.5])
    sigma = np.array([1.0, 0.5, 2.0])
    draws = rng.normal(mu[:, None], sigma[:, None], size=(3, 20000))
    assert crps_ensemble(draws, y) == pytest.approx(
        crps_gaussian(mu, sigma, y), abs=0.01
    )


def test_crps_gaussian_at_the_mean_is_the_known_constant():
    """For y = mu, CRPS = sigma * (2*phi(0) - 1/sqrt(pi)) = sigma * (1/sqrt(pi))...

    Written out: z = 0, so the bracket is 0 + 2/sqrt(2*pi) - 1/sqrt(pi), which is
    sigma * (sqrt(2) - 1) / sqrt(pi).
    """
    expected = (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)
    assert crps_gaussian([0.0], [1.0], [0.0]) == pytest.approx(expected)
    assert crps_gaussian([0.0], [3.0], [0.0]) == pytest.approx(3.0 * expected)


def test_crps_gaussian_rejects_a_zero_standard_deviation():
    with pytest.raises(ValueError, match="must be positive"):
        crps_gaussian([0.0], [0.0], [0.0])


@pytest.mark.parametrize("fair", [False, True])
def test_energy_score_reduces_to_crps_in_one_dimension(fair):
    """Regression: the two used different spread-term divisors and disagreed."""
    rng = np.random.default_rng(9)
    y = rng.normal(size=(50, 1))
    draws = rng.normal(size=(50, 30, 1))
    assert energy_score(draws, y, fair=fair) == pytest.approx(
        crps_ensemble(draws[:, :, 0], y.ravel(), fair=fair)
    )


def test_the_plug_in_estimator_credits_a_small_ensemble_with_less_spread():
    """Where the two differ, and why the choice matters below ~50 members.

    The plug-in spread term divides the pair total by ``m^2``, counting the ``m`` zero
    self-pairs, so it *understates* the ensemble's spread. Since CRPS subtracts half the
    spread, understating it leaves the score higher — and a model can shave that penalty
    by being narrower than it should be, which is the under-dispersion bias the fair
    estimator removes.
    """
    rng = np.random.default_rng(11)
    y = rng.normal(size=200)
    draws = rng.normal(size=(200, 5))
    assert crps_ensemble(draws, y, fair=False) > crps_ensemble(draws, y, fair=True)


def test_only_the_fair_estimator_is_minimised_by_an_honest_ensemble():
    """The under-dispersion bias, shown directly rather than asserted.

    The truth is standard normal. Ensembles are drawn at several scales; a proper score
    should be best at scale 1.0. The fair estimator is. The plug-in estimator keeps
    improving as the ensemble is made *narrower* than the truth, which is exactly the
    incentive it should not create — and the reason to pass ``fair=True`` when the
    ensemble is small.
    """
    rng = np.random.default_rng(14)
    n, m = 20000, 5
    y = rng.normal(size=n)
    scales = [0.6, 0.8, 1.0, 1.2]
    draws = {s: rng.normal(scale=s, size=(n, m)) for s in scales}

    fair = [crps_ensemble(draws[s], y, fair=True) for s in scales]
    assert min(fair) == fair[scales.index(1.0)]

    plug_in = [crps_ensemble(draws[s], y, fair=False) for s in scales]
    assert plug_in[scales.index(0.6)] < plug_in[scales.index(1.0)]


def test_the_two_estimators_converge_for_a_large_ensemble():
    rng = np.random.default_rng(12)
    y = rng.normal(size=100)
    draws = rng.normal(size=(100, 2000))
    assert crps_ensemble(draws, y, fair=True) == pytest.approx(
        crps_ensemble(draws, y, fair=False), rel=1e-3
    )


def test_both_estimators_reduce_to_mae_at_one_member():
    rng = np.random.default_rng(13)
    y = rng.normal(size=50)
    point = (y + rng.normal(size=50))[:, None]
    mae = float(np.mean(np.abs(point[:, 0] - y)))
    assert crps_ensemble(point, y, fair=False) == pytest.approx(mae)
    assert crps_ensemble(point, y, fair=True) == pytest.approx(mae)


def test_energy_score_sees_joint_incoherence_that_per_horizon_crps_misses():
    """The survey's reason for including it: each margin can be right and the joint wrong.

    Truth is perfectly correlated across the two horizons. One ensemble reproduces that
    correlation; the other has identical marginals with the correlation destroyed. Scoring
    each horizon separately cannot tell them apart; the energy score can.
    """
    rng = np.random.default_rng(10)
    n, m = 200, 40
    base = rng.normal(size=n)
    y = np.column_stack([base, base])

    shared = rng.normal(size=(n, m))
    coherent = np.stack([base[:, None] + shared, base[:, None] + shared], axis=2)
    independent = np.stack(
        [base[:, None] + shared, base[:, None] + rng.permutation(shared, axis=1)], axis=2
    )

    # The marginals are identical by construction, so per-horizon CRPS agrees.
    for horizon in (0, 1):
        assert crps_ensemble(coherent[:, :, horizon], y[:, horizon]) == pytest.approx(
            crps_ensemble(independent[:, :, horizon], y[:, horizon])
        )
    assert energy_score(coherent, y) < energy_score(independent, y)


def test_energy_score_rejects_a_two_dimensional_sample_array():
    with pytest.raises(ValueError, match="n_dimensions"):
        energy_score(np.zeros((5, 3)), np.zeros((5, 1)))
