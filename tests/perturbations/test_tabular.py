"""Tests for the feature-space perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import Scope, TimeScope
from themis_assess.perturbations.tabular import (
    AmplitudeScaling,
    CategorySwap,
    GaussianNoise,
    MissingValues,
    UniformBias,
    feature_reference_scale,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sequence():
    """Two entities x three timesteps; speed varies over time, vessel_type does not."""
    values = np.array(
        [
            [[10.0, 1.0], [12.0, 1.0], [14.0, 1.0]],
            [[20.0, 2.0], [22.0, 2.0], [24.0, 2.0]],
        ]
    )
    return ArrayAdapter(
        values, ["speed", "vessel_type"], categorical_features=["vessel_type"]
    )


@pytest.fixture
def ragged():
    values = np.full((2, 4, 1), np.nan)
    values[0, 1:, 0] = [1.0, 2.0, 3.0]
    values[1, :, 0] = [10.0, 20.0, 30.0, 40.0]
    return ArrayAdapter(values, ["x"], entity_lengths=[3, 4])


# --------------------------------------------------------------------------
# the relative reference scale
# --------------------------------------------------------------------------


def test_reference_scale_is_the_per_feature_std_over_all_entities_and_timesteps(sequence):
    """Repo B: 'std ... computed over all vessels and timesteps'."""
    expected = float(np.std([10, 12, 14, 20, 22, 24]))
    assert feature_reference_scale(sequence, ["speed"])["speed"] == pytest.approx(expected)


def test_reference_scale_ignores_padding(ragged):
    expected = float(np.std([1, 2, 3, 10, 20, 30, 40]))
    assert feature_reference_scale(ragged, ["x"])["x"] == pytest.approx(expected)


def test_reference_scale_falls_back_for_a_constant_feature():
    """A zero-std feature would otherwise be immune, and misreported as robust."""
    constant = ArrayAdapter(np.full((3, 1), 7.0), ["c"])
    assert feature_reference_scale(constant, ["c"])["c"] == pytest.approx(7.0)


def test_reference_scale_handles_an_all_zero_feature():
    zeros = ArrayAdapter(np.zeros((3, 1)), ["z"])
    assert feature_reference_scale(zeros, ["z"])["z"] > 0.0


def test_relative_severity_transfers_between_differently_scaled_datasets(rng):
    """The lesson from the two tutorial notebooks' 100x-apart sigma ladders.

    Same data, one copy multiplied by 100. A relative perturbation degrades both by the
    same proportion; an absolute one would wreck the small-scale copy and barely touch
    the large one.
    """
    base = rng.normal(0.0, 1.0, (200, 1))
    small = ArrayAdapter(base, ["x"])
    large = ArrayAdapter(base * 100.0, ["x"])

    p = GaussianNoise(relative=True)
    a = p.perturb(small, p.spec(severity=3), np.random.default_rng(1))[0].values()
    b = p.perturb(large, p.spec(severity=3), np.random.default_rng(1))[0].values()

    rel_shift_small = np.std(a - small.values()) / np.std(small.values())
    rel_shift_large = np.std(b - large.values()) / np.std(large.values())
    assert rel_shift_small == pytest.approx(rel_shift_large, rel=1e-9)


def test_absolute_mode_reproduces_the_notebook_behaviour(rng):
    p = GaussianNoise(relative=False)
    data = ArrayAdapter(np.zeros((5000, 1)), ["x"])
    out = p.perturb(data, p.spec(severity=3, time_scope=TimeScope.CONSTANT), rng)[0]
    # severity 3 -> factor 1.0, applied absolutely
    assert np.std(out.values()) == pytest.approx(1.0, rel=0.05)
    assert p.metadata(p.spec(severity=3))["relative_to_feature_std"] is False


# --------------------------------------------------------------------------
# GaussianNoise
# --------------------------------------------------------------------------


def test_gaussian_constant_time_scope_is_a_sensor_offset(sequence, rng):
    """One draw per entity, held across the series — repo B's perturb_numerical_data."""
    out = rng and GaussianNoise().perturb(
        sequence,
        GaussianNoise().spec(severity=3, scope=Scope.NUMERICAL),
        np.random.default_rng(2),
    )[0]
    delta = out.values()[:, :, 0] - sequence.values()[:, :, 0]
    # each row's shift is identical across time
    assert np.allclose(delta[0], delta[0][0])
    assert np.allclose(delta[1], delta[1][0])
    # and the two entities got different shifts
    assert delta[0][0] != pytest.approx(delta[1][0])


def test_gaussian_per_timestep_is_jitter(sequence):
    """Independent draw at each step — repo B's perturb_numerical_data_over_time."""
    p = GaussianNoise()
    spec = p.spec(severity=3, scope=Scope.NUMERICAL, time_scope=TimeScope.PER_TIMESTEP)
    out = p.perturb(sequence, spec, np.random.default_rng(2))[0]
    delta = out.values()[:, :, 0] - sequence.values()[:, :, 0]
    assert not np.allclose(delta[0], delta[0][0])


def test_gaussian_leaves_categorical_features_untouched(sequence, rng):
    p = GaussianNoise()
    out = p.perturb(sequence, p.spec(severity=5), rng)[0]
    assert np.array_equal(out.values()[:, :, 1], sequence.values()[:, :, 1])


def test_gaussian_severity_increases_the_perturbation(rng):
    data = ArrayAdapter(np.random.default_rng(9).normal(0, 1, (4000, 1)), ["x"])
    p = GaussianNoise()
    shifts = []
    for s in range(1, 6):
        out = p.perturb(data, p.spec(severity=s), np.random.default_rng(3))[0]
        shifts.append(float(np.std(out.values() - data.values())))
    assert shifts == sorted(shifts)
    # the declared ladder is 0.25, 0.5, 1, 2, 4 -> each step doubles
    assert shifts[3] / shifts[2] == pytest.approx(2.0, rel=0.1)


def test_gaussian_preserves_padding(ragged, rng):
    p = GaussianNoise()
    out = p.perturb(ragged, p.spec(severity=4), rng)[0]
    assert np.isnan(out.values()[0, 0, 0])
    assert np.array_equal(out.valid_mask(), ragged.valid_mask())


def test_n_variants_produces_independent_draws(sequence):
    p = GaussianNoise()
    out = p.perturb(sequence, p.spec(severity=4, n_variants=3), np.random.default_rng(4))
    firsts = [v.values()[0, 0, 0] for v in out]
    assert len(set(firsts)) == 3


def test_the_same_seed_reproduces_the_same_perturbation(sequence):
    p = GaussianNoise()
    a = p.perturb(sequence, p.spec(severity=3), np.random.default_rng(7))[0]
    b = p.perturb(sequence, p.spec(severity=3), np.random.default_rng(7))[0]
    assert np.array_equal(a.values(), b.values())


def test_perturbation_draws_only_from_the_supplied_generator(sequence):
    """No global numpy state — repo D's MaskingPerturbation used np.random.rand()."""
    p = GaussianNoise()
    np.random.seed(0)
    a = p.perturb(sequence, p.spec(severity=3), np.random.default_rng(5))[0]
    np.random.seed(12345)
    b = p.perturb(sequence, p.spec(severity=3), np.random.default_rng(5))[0]
    assert np.array_equal(a.values(), b.values())


# --------------------------------------------------------------------------
# UniformBias
# --------------------------------------------------------------------------


def test_uniform_bias_is_a_single_positive_offset(sequence, rng):
    """What the notebooks' uniform_* entries actually did: a scalar shift."""
    p = UniformBias()
    out = p.perturb(sequence, p.spec(severity=4, scope=Scope.NUMERICAL), rng)[0]
    delta = out.values()[:, :, 0] - sequence.values()[:, :, 0]
    assert np.allclose(delta, delta.flat[0])  # identical everywhere
    assert delta.flat[0] > 0.0  # one-directional


def test_uniform_bias_is_not_zero_mean_unlike_gaussian(rng):
    data = ArrayAdapter(np.zeros((500, 1)), ["x"])
    p = UniformBias()
    means = [
        float(np.mean(p.perturb(data, p.spec(severity=5), np.random.default_rng(i))[0].values()))
        for i in range(20)
    ]
    assert all(m > 0 for m in means)


# --------------------------------------------------------------------------
# AmplitudeScaling
# --------------------------------------------------------------------------


def test_amplitude_scaling_is_multiplicative(sequence, rng):
    p = AmplitudeScaling()
    out = p.perturb(sequence, p.spec(severity=3, scope=Scope.NUMERICAL), rng)[0]
    ratio = out.values()[:, :, 0] / sequence.values()[:, :, 0]
    assert np.allclose(ratio, ratio.flat[0])
    assert abs(ratio.flat[0] - 1.0) == pytest.approx(0.30)  # severity 3 -> 30%


def test_amplitude_scaling_needs_no_reference_scale(sequence):
    """A ratio is dimensionless, so `relative` is meaningless here."""
    p = AmplitudeScaling()
    assert "relative_to_feature_std" not in p.metadata(p.spec(severity=1))


def test_amplitude_scaling_reports_the_fractional_change():
    p = AmplitudeScaling()
    assert p.metadata(p.spec(severity=5))["fractional_change"] == pytest.approx(0.80)


# --------------------------------------------------------------------------
# MissingValues
# --------------------------------------------------------------------------


def test_missing_values_rate_matches_the_severity(rng):
    data = ArrayAdapter(np.ones((4000, 1)), ["x"])
    p = MissingValues()
    out = p.perturb(data, p.spec(severity=3), rng)[0]
    assert float(np.mean(np.isnan(out.values()))) == pytest.approx(0.25, abs=0.03)


def test_missing_values_severity_ladder_matches_the_plovdiv_sweep():
    """Condensed from repo C's 100-rate sweep."""
    assert MissingValues.severity_scale == (0.05, 0.10, 0.25, 0.50, 0.90)


def test_missing_values_applies_to_categorical_features_too(sequence, rng):
    """A missing category is as real as a missing measurement."""
    p = MissingValues()
    out = p.perturb(sequence, p.spec(severity=5, scope=Scope.CATEGORICAL), rng)[0]
    assert np.isnan(out.values()[:, :, 1]).any()


def test_missing_values_constant_scope_drops_a_whole_entity(rng):
    """A sensor that fails outright, rather than intermittently."""
    values = np.ones((200, 5, 1))
    data = ArrayAdapter(values, ["x"])
    p = MissingValues()
    out = p.perturb(data, p.spec(severity=3, time_scope=TimeScope.CONSTANT), rng)[0]
    per_entity = np.isnan(out.values()[:, :, 0]).mean(axis=1)
    assert set(np.unique(per_entity)) <= {0.0, 1.0}  # all-or-nothing per entity


def test_missing_values_named_group_targets_only_that_group(rng):
    """Repo C's real question: what if *this group of features* goes missing."""
    data = ArrayAdapter(np.ones((100, 4)), ["a", "b", "c", "d"])
    p = MissingValues()
    spec = p.spec(severity=5, scope=Scope.NAMED, features=("b", "c"))
    out = p.perturb(data, spec, rng)[0]
    assert not np.isnan(out.values()[:, :, 0]).any()
    assert np.isnan(out.values()[:, :, 1]).any()
    assert np.isnan(out.values()[:, :, 2]).any()
    assert not np.isnan(out.values()[:, :, 3]).any()


# --------------------------------------------------------------------------
# CategorySwap
# --------------------------------------------------------------------------


def test_category_swap_only_uses_observed_categories(sequence, rng):
    """The perturbed input must stay a value the model could have seen."""
    p = CategorySwap()
    out = p.perturb(sequence, p.spec(severity=5, scope=Scope.CATEGORICAL), rng)[0]
    assert set(np.unique(out.values()[:, :, 1])) <= {1.0, 2.0}


def test_category_swap_constant_scope_holds_across_time(sequence, rng):
    """Repo B: 'categorical features are constant per vessel ... applied uniformly'."""
    p = CategorySwap()
    out = p.perturb(sequence, p.spec(severity=5, scope=Scope.CATEGORICAL), rng)[0]
    for row in out.values()[:, :, 1]:
        assert len(set(row)) == 1


def test_category_swap_per_timestep_varies_within_an_entity():
    """Severity 3 (probability 0.5), not 5: at probability 1.0 with two categories every
    timestep flips, so each row stays uniform and the test would prove nothing."""
    p = CategorySwap()
    values = np.array([[[1.0]] * 12])
    data = ArrayAdapter(
        np.concatenate([values, np.full_like(values, 2.0)], axis=0),
        ["kind"],
        categorical_features=["kind"],
    )
    spec = p.spec(severity=3, scope=Scope.CATEGORICAL, time_scope=TimeScope.PER_TIMESTEP)
    out = p.perturb(data, spec, np.random.default_rng(11))[0]
    assert any(len(set(row)) > 1 for row in out.values()[:, :, 0])


def test_category_swap_at_probability_one_flips_every_value(rng):
    """With two categories and probability 1.0 the mapping is deterministic."""
    p = CategorySwap()
    values = np.array([[[1.0]] * 4, [[2.0]] * 4])
    data = ArrayAdapter(
        values, ["kind"], categorical_features=["kind"]
    )
    spec = p.spec(severity=5, scope=Scope.CATEGORICAL, time_scope=TimeScope.PER_TIMESTEP)
    out = p.perturb(data, spec, rng)[0].values()[:, :, 0]
    assert np.allclose(out[0], 2.0)
    assert np.allclose(out[1], 1.0)


def test_category_swap_always_picks_a_different_category(rng):
    values = np.array([[0.0], [1.0], [2.0]])
    data = ArrayAdapter(values, ["kind"], categorical_features=["kind"])
    p = CategorySwap()
    out = p.perturb(data, p.spec(severity=5), rng)[0]  # severity 5 -> probability 1.0
    assert not np.array_equal(out.values(), data.values())
    assert np.all(out.values() != data.values())


def test_category_swap_leaves_numerical_features_untouched(sequence, rng):
    p = CategorySwap()
    out = p.perturb(sequence, p.spec(severity=5), rng)[0]
    assert np.array_equal(out.values()[:, :, 0], sequence.values()[:, :, 0])


def test_category_swap_needs_at_least_two_categories(rng):
    data = ArrayAdapter(np.ones((3, 1)), ["kind"], categorical_features=["kind"])
    p = CategorySwap()
    with pytest.raises(ValueError, match="swapping needs at least 2"):
        p.perturb(data, p.spec(severity=3), rng)


def test_category_swap_probability_rises_with_severity():
    values = np.arange(3.0).repeat(300).reshape(-1, 1)
    data = ArrayAdapter(values, ["kind"], categorical_features=["kind"])
    p = CategorySwap()
    rates = []
    for s in range(1, 6):
        out = p.perturb(data, p.spec(severity=s), np.random.default_rng(6))[0]
        rates.append(float(np.mean(out.values() != data.values())))
    assert rates == sorted(rates)


# --------------------------------------------------------------------------
# scope eligibility
# --------------------------------------------------------------------------


def test_a_named_scope_naming_an_ineligible_feature_raises(sequence, rng):
    """Explicit request, explicit mistake."""
    p = GaussianNoise()
    spec = p.spec(severity=3, scope=Scope.NAMED, features=("vessel_type",))
    with pytest.raises(ValueError, match="does not apply to categorical features"):
        p.perturb(sequence, spec, rng)


def test_a_broad_scope_narrows_silently_but_records_it(sequence):
    p = GaussianNoise()
    spec = p.spec(severity=3, scope=Scope.ALL_FEATURES)
    assert p.eligible_features(sequence, spec) == ["speed"]
    assert p.metadata(spec)["accepts_categorical"] is False


def test_metadata_states_what_the_severity_meant(sequence):
    """A report cannot say 'severity 3' and stop."""
    p = GaussianNoise()
    meta = p.metadata(p.spec(severity=3))
    assert meta["severity"] == 3
    assert meta["severity_factor"] == pytest.approx(1.0)
    assert meta["severity_scale"] == [0.25, 0.5, 1.0, 2.0, 4.0]


def test_severity_beyond_the_ladder_is_rejected(sequence, rng):
    p = GaussianNoise()
    with pytest.raises(ValueError, match="exposes 5 severity level"):
        p.perturb(sequence, p.spec(severity=6), rng)
