"""Tests for survey §5.3, distribution distance.

The three measures are kept for different reasons, and most of these tests are about the
differences the survey draws between them rather than about any one of them in isolation:
Wasserstein is in the data's units, MMD sees joint structure, KL is blind to geometry.
A test suite that only checked each in isolation would not notice if one silently acquired
another's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.metrics.robustness.distance import (
    kl_divergence,
    maximum_mean_discrepancy,
    median_heuristic_bandwidth,
    wasserstein_distance,
)


# --------------------------------------------------------------------------
# shared input handling
# --------------------------------------------------------------------------


def test_one_dimensional_samples_are_accepted_as_a_single_feature():
    a = np.linspace(0.0, 1.0, 50)
    b = a + 0.5
    assert wasserstein_distance(a, b) == pytest.approx(
        wasserstein_distance(a[:, None], b[:, None])
    )


def test_samples_with_different_feature_counts_are_rejected():
    with pytest.raises(ValueError, match="same number of features"):
        wasserstein_distance(np.zeros((10, 2)), np.zeros((10, 3)))


def test_non_finite_values_are_rejected_with_a_pointer_to_imputation():
    with pytest.raises(ValueError, match="Drop or impute"):
        wasserstein_distance(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


# --------------------------------------------------------------------------
# Wasserstein
# --------------------------------------------------------------------------


def test_wasserstein_of_a_constant_offset_is_the_offset():
    """The survey's own example: under sensor drift W_p equals delta exactly."""
    a = np.linspace(-3.0, 3.0, 500)
    for delta in (0.5, 3.2, 100.0):
        assert wasserstein_distance(a, a + delta) == pytest.approx(delta, rel=1e-9)


def test_wasserstein_of_a_sample_against_itself_is_zero():
    a = np.random.default_rng(0).normal(size=200)
    assert wasserstein_distance(a, a) == pytest.approx(0.0)


def test_wasserstein_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(loc=1.0, size=300)
    assert wasserstein_distance(a, b) == pytest.approx(wasserstein_distance(b, a))


def test_wasserstein_scales_with_the_data_units():
    """What makes it the interpretable one: a distance of 3.2 means 3.2 degrees."""
    a = np.linspace(0.0, 1.0, 200)
    near = wasserstein_distance(a, a + 1.0)
    far = wasserstein_distance(a, a + 10.0)
    assert far == pytest.approx(10.0 * near)


def test_order_two_weights_large_displacements_more():
    """W_2 >= W_1, with equality only for a uniform displacement."""
    a = np.linspace(0.0, 1.0, 400)
    uneven = np.concatenate([a[:200], a[200:] + 5.0])
    w1 = wasserstein_distance(a, uneven, order=1)
    w2 = wasserstein_distance(a, uneven, order=2)
    assert w2 > w1
    shifted = a + 2.0
    assert wasserstein_distance(a, shifted, order=2) == pytest.approx(2.0, abs=1e-2)


def test_aggregation_across_features():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(400, 3))
    b = a + np.array([0.0, 1.0, 5.0])
    assert wasserstein_distance(a, b, aggregate="max") == pytest.approx(5.0, abs=0.05)
    assert wasserstein_distance(a, b, aggregate="sum") == pytest.approx(6.0, abs=0.1)
    assert wasserstein_distance(a, b, aggregate="mean") == pytest.approx(2.0, abs=0.05)


def test_wasserstein_rejects_an_unknown_aggregation():
    with pytest.raises(ValueError, match="mean, max or sum"):
        wasserstein_distance(np.zeros(5), np.ones(5), aggregate="median")


def test_wasserstein_is_blind_to_a_pure_dependence_change():
    """The documented limitation: per-feature distances cannot see a rotation.

    Both samples have identical marginals; only the correlation between the two features
    differs. The per-feature construction reports almost nothing.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    correlated = np.column_stack([x, x])
    independent = np.column_stack([x, rng.permutation(x)])
    assert wasserstein_distance(correlated, independent) < 0.1


# --------------------------------------------------------------------------
# MMD
# --------------------------------------------------------------------------


def test_mmd_of_identical_distributions_is_approximately_zero():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    assert abs(maximum_mean_discrepancy(a, b)) < 0.01


def test_the_unbiased_estimator_may_go_slightly_negative():
    """A property of the estimator, deliberately left visible rather than clipped."""
    rng = np.random.default_rng(5)
    values = [
        maximum_mean_discrepancy(
            rng.normal(size=(200, 1)), rng.normal(size=(200, 1)), unbiased=True
        )
        for _ in range(20)
    ]
    assert min(values) < 0.0
    assert max(values) < 0.02


def test_the_biased_estimator_never_goes_negative():
    rng = np.random.default_rng(6)
    for _ in range(10):
        value = maximum_mean_discrepancy(
            rng.normal(size=(100, 1)), rng.normal(size=(100, 1)), unbiased=False
        )
        assert value >= 0.0


def test_mmd_grows_with_the_separation():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(400, 1))
    bandwidth = 1.0
    values = [
        maximum_mean_discrepancy(a, a + shift, bandwidth=bandwidth)
        for shift in (0.0, 0.5, 1.0, 2.0)
    ]
    assert values == sorted(values)


def test_mmd_sees_a_dependence_change_that_wasserstein_misses():
    """The reason MMD is kept: with a characteristic kernel it sees joint structure."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=1000)
    correlated = np.column_stack([x, x])
    independent = np.column_stack([x, rng.permutation(x)])
    w = wasserstein_distance(correlated, independent)
    mmd = maximum_mean_discrepancy(correlated, independent)
    assert mmd > 5 * w


def test_mmd_is_symmetric():
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(150, 2)), rng.normal(loc=0.5, size=(150, 2))
    assert maximum_mean_discrepancy(a, b, bandwidth=1.0) == pytest.approx(
        maximum_mean_discrepancy(b, a, bandwidth=1.0)
    )


def test_mmd_depends_on_the_bandwidth_which_is_why_it_must_be_reported():
    rng = np.random.default_rng(10)
    a = rng.normal(size=(300, 1))
    b = a + 1.0
    narrow = maximum_mean_discrepancy(a, b, bandwidth=0.1)
    wide = maximum_mean_discrepancy(a, b, bandwidth=20.0)
    assert narrow > 10 * wide  # the same shift, two incomparable numbers


def test_the_unbiased_estimator_needs_two_samples_a_side():
    with pytest.raises(ValueError, match="at least 2 samples"):
        maximum_mean_discrepancy(np.zeros((1, 1)), np.zeros((5, 1)))


def test_a_non_positive_bandwidth_is_rejected():
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        maximum_mean_discrepancy(np.zeros((5, 1)), np.ones((5, 1)), bandwidth=0.0)


def test_the_median_heuristic_tracks_the_data_scale():
    rng = np.random.default_rng(11)
    small = rng.normal(scale=1.0, size=(200, 1))
    large = small * 100.0
    assert median_heuristic_bandwidth(large, large) == pytest.approx(
        100.0 * median_heuristic_bandwidth(small, small), rel=1e-6
    )


def test_the_median_heuristic_is_deterministic():
    """A bandwidth chosen by a random subsample must still be reproducible."""
    rng = np.random.default_rng(12)
    a = rng.normal(size=(2000, 2))
    b = rng.normal(size=(2000, 2))
    assert median_heuristic_bandwidth(a, b) == median_heuristic_bandwidth(a, b)


# --------------------------------------------------------------------------
# KL, and the reasons the survey warns about it
# --------------------------------------------------------------------------


def test_kl_of_a_distribution_against_itself_is_approximately_zero():
    a = np.random.default_rng(13).normal(size=2000)
    assert kl_divergence(a, a) == pytest.approx(0.0, abs=1e-9)


def test_kl_is_asymmetric():
    """Not a distance, despite being used as one."""
    rng = np.random.default_rng(14)
    a = rng.normal(size=3000)
    b = rng.normal(loc=0.7, scale=2.0, size=3000)
    forward = kl_divergence(a, b)
    backward = kl_divergence(b, a)
    assert forward != pytest.approx(backward, rel=0.1)


def test_kl_is_blind_to_how_far_apart_the_mismatch_is():
    """The survey's objection, made numeric.

    Two disjoint shifts, one small and one enormous. Wasserstein separates them by the
    same factor as the shift; KL saturates, because once the supports no longer overlap
    there is no more probability mass to disagree about.
    """
    a = np.random.default_rng(15).normal(size=3000)
    near, far = a + 8.0, a + 100.0

    w_near, w_far = wasserstein_distance(a, near), wasserstein_distance(a, far)
    assert w_far / w_near == pytest.approx(100.0 / 8.0, rel=0.01)

    kl_near, kl_far = kl_divergence(a, near), kl_divergence(a, far)
    assert kl_far / kl_near < 1.5  # nowhere near 12.5


def test_kl_stays_finite_where_the_reference_has_no_mass():
    """eps floors the reference density, so a disjoint support is expensive, not infinite."""
    a = np.zeros(100) + 0.1
    b = np.zeros(100) + 50.0
    value = kl_divergence(np.concatenate([a, a + 1]), np.concatenate([b, b + 1]))
    assert np.isfinite(value)
    assert value > 10.0


def test_kl_ignores_a_feature_constant_across_both_samples():
    rng = np.random.default_rng(16)
    varying = rng.normal(size=(500, 1))
    constant = np.full((500, 1), 3.0)
    with_constant = np.hstack([varying, constant])
    shifted = np.hstack([varying + 1.0, constant])
    assert kl_divergence(with_constant, shifted) == pytest.approx(
        kl_divergence(varying, varying + 1.0)
    )


def test_kl_needs_at_least_two_bins():
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        kl_divergence(np.zeros(10), np.ones(10), n_bins=1)


def test_kl_depends_on_the_bin_count():
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=2000), rng.normal(loc=0.5, size=2000)
    assert kl_divergence(a, b, n_bins=5) != pytest.approx(
        kl_divergence(a, b, n_bins=100), rel=0.05
    )


# --------------------------------------------------------------------------
# the survey's framing: context, not a score
# --------------------------------------------------------------------------


def test_a_larger_shift_gives_a_larger_distance_for_all_three():
    """What the report layer relies on: these order shifts, so a degradation can be
    read against the size of the shift that caused it."""
    rng = np.random.default_rng(18)
    a = rng.normal(size=(600, 1))
    small, large = a + 0.3, a + 2.0
    assert wasserstein_distance(a, small) < wasserstein_distance(a, large)
    assert maximum_mean_discrepancy(a, small, bandwidth=1.0) < maximum_mean_discrepancy(
        a, large, bandwidth=1.0
    )
    assert kl_divergence(a, small) < kl_divergence(a, large)
