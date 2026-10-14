"""Tests for the time-series value-domain perturbations (survey §4.2, Figure 3)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import Alignment
from themis_assess.perturbations.timeseries.value_domain import (
    AlphaStableNoise,
    ExponentialDamping,
    FlatSensor,
    MissingSegment,
    OutlierSpikes,
    SensorDrift,
    sample_alpha_stable,
)

ALL = (
    SensorDrift,
    ExponentialDamping,
    FlatSensor,
    OutlierSpikes,
    MissingSegment,
    AlphaStableNoise,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ramp():
    """One entity, a clean 20-step ramp — easy to see what a perturbation did."""
    return ArrayAdapter(
        (np.arange(20, dtype=float) + 10.0).reshape(1, 20, 1), ["x"], sequential=True
    )


@pytest.fixture
def ragged():
    """Lengths 5 and 12, end-aligned — the shape repo B's vessels have."""
    values = np.full((2, 12, 1), np.nan)
    values[0, 7:, 0] = np.arange(5, dtype=float) + 100.0
    values[1, :, 0] = np.arange(12, dtype=float)
    return ArrayAdapter(values, ["x"], entity_lengths=[5, 12], alignment=Alignment.END)


# --------------------------------------------------------------------------
# properties shared by the whole family
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
def test_all_are_sequence_only(cls, rng):
    """A drift across a single timestep is not a milder drift, it is a meaningless one."""
    tabular = ArrayAdapter(np.ones((3, 2)), ["a", "b"])
    p = cls()
    assert not p.applies_to(tabular)
    with pytest.raises(ValueError, match="applies to \\['sequence'\\]"):
        p.perturb(tabular, p.spec(severity=3), rng)


@pytest.mark.parametrize("cls", ALL)
def test_all_change_the_data(cls, ramp, rng):
    p = cls()
    out = p.perturb(ramp, p.spec(severity=4), rng)[0]
    changed = ~np.isclose(
        out.values(), ramp.values(), equal_nan=True
    ) | np.isnan(out.values()) != np.isnan(ramp.values())
    assert changed.any()


@pytest.mark.parametrize("cls", ALL)
def test_all_preserve_padding(cls, ragged, rng):
    """A perturbation must not turn padding into data, or data into padding elsewhere."""
    p = cls()
    out = p.perturb(ragged, p.spec(severity=3), rng)[0]
    assert np.array_equal(out.valid_mask(), ragged.valid_mask())
    assert np.isnan(out.values()[0, :7, 0]).all()


@pytest.mark.parametrize("cls", ALL)
def test_all_are_reproducible_from_a_seed(cls, ramp):
    p = cls()
    a = p.perturb(ramp, p.spec(severity=3), np.random.default_rng(1))[0]
    b = p.perturb(ramp, p.spec(severity=3), np.random.default_rng(1))[0]
    assert np.array_equal(a.values(), b.values(), equal_nan=True)


@pytest.mark.parametrize("cls", ALL)
def test_all_report_their_severity_mapping(cls):
    p = cls()
    meta = p.metadata(p.spec(severity=2))
    assert meta["severity"] == 2
    assert "severity_factor" in meta
    assert len(meta["severity_scale"]) == p.severity_levels


@pytest.mark.parametrize("cls", ALL)
def test_all_act_within_each_entitys_own_stretch(cls, ragged, rng):
    """Entity 0's data starts at position 7; a perturbation must not reach before it."""
    p = cls()
    out = p.perturb(ragged, p.spec(severity=5), rng)[0]
    assert np.isnan(out.values()[0, :7, 0]).all()


# --------------------------------------------------------------------------
# SensorDrift
# --------------------------------------------------------------------------


def test_sensor_drift_grows_monotonically_over_time(ramp, rng):
    p = SensorDrift()
    out = p.perturb(ramp, p.spec(severity=4), rng)[0]
    delta = out.values()[0, :, 0] - ramp.values()[0, :, 0]
    assert delta[0] == pytest.approx(0.0)
    assert np.all(np.diff(np.abs(delta)) >= -1e-12)  # magnitude never decreases


def test_sensor_drift_reaches_severity_times_the_feature_std(ramp, rng):
    p = SensorDrift()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]  # factor 1.0
    delta = out.values()[0, :, 0] - ramp.values()[0, :, 0]
    std = float(np.std(ramp.values()[0, :, 0]))
    assert abs(delta[-1]) == pytest.approx(std, rel=1e-9)


def test_sensor_drift_differs_from_a_constant_offset(ramp, rng):
    """The distinction the docstring claims: a drift is not a calibration error."""
    p = SensorDrift()
    out = p.perturb(ramp, p.spec(severity=4), rng)[0]
    delta = out.values()[0, :, 0] - ramp.values()[0, :, 0]
    assert not np.allclose(delta, delta[0])


def test_sensor_drift_direction_varies_across_entities():
    values = np.tile(np.arange(30, dtype=float).reshape(1, 30, 1), (40, 1, 1))
    data = ArrayAdapter(values, ["x"], sequential=True)
    p = SensorDrift()
    out = p.perturb(data, p.spec(severity=4), np.random.default_rng(2))[0]
    ends = out.values()[:, -1, 0] - values[:, -1, 0]
    assert (ends > 0).any() and (ends < 0).any()


# --------------------------------------------------------------------------
# ExponentialDamping
# --------------------------------------------------------------------------


def test_damping_decays_toward_the_entitys_own_level(ramp, rng):
    p = ExponentialDamping()
    original = ramp.values()[0, :, 0]
    level = float(np.mean(original))
    out = p.perturb(ramp, p.spec(severity=5), rng)[0].values()[0, :, 0]
    # deviation from the level shrinks monotonically in relative terms
    assert abs(out[-1] - level) < abs(original[-1] - level)
    assert out[0] == pytest.approx(original[0])


def test_damping_does_not_inject_an_offset(rng):
    """Decaying toward zero would shift a non-zero-centred feature; toward the mean does not."""
    values = np.full((1, 15, 1), 500.0)
    values[0, :, 0] += np.arange(15) * 0.01  # essentially constant at 500
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = ExponentialDamping()
    out = p.perturb(data, p.spec(severity=5), rng)[0].values()[0, :, 0]
    assert np.all(np.abs(out - 500.0) < 1.0)


def test_damping_reports_the_retained_deviation():
    p = ExponentialDamping()
    meta = p.metadata(p.spec(severity=3))
    assert meta["decay_exponent"] == pytest.approx(1.0)
    assert meta["retained_deviation_at_end"] == pytest.approx(np.exp(-1.0))


def test_damping_needs_no_reference_scale():
    assert "relative_to_feature_std" not in ExponentialDamping().metadata()


# --------------------------------------------------------------------------
# FlatSensor
# --------------------------------------------------------------------------


def test_flat_sensor_freezes_a_contiguous_window(ramp, rng):
    p = FlatSensor()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0].values()[0, :, 0]
    original = ramp.values()[0, :, 0]
    frozen = np.flatnonzero(~np.isclose(out, original))
    assert frozen.size > 0
    assert np.array_equal(frozen, np.arange(frozen[0], frozen[-1] + 1))  # contiguous


def test_flat_sensor_holds_the_last_known_good_value(ramp, rng):
    p = FlatSensor()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0].values()[0, :, 0]
    original = ramp.values()[0, :, 0]
    changed = np.flatnonzero(~np.isclose(out, original))
    held = out[changed]
    assert np.allclose(held, held[0])
    if changed[0] > 0:
        assert held[0] == pytest.approx(original[changed[0] - 1])


def test_flat_sensor_window_grows_with_severity(rng):
    values = np.arange(100, dtype=float).reshape(1, 100, 1)
    data = ArrayAdapter(values, ["x"], sequential=True)
    p = FlatSensor()
    widths = []
    for s in range(1, 6):
        out = p.perturb(data, p.spec(severity=s), np.random.default_rng(4))[0]
        widths.append(int(np.sum(~np.isclose(out.values()[0, :, 0], values[0, :, 0]))))
    assert widths == sorted(widths)


def test_flat_sensor_keeps_the_value_range_plausible(ramp, rng):
    """Why it is easy to miss: distribution distances will not flag it."""
    p = FlatSensor()
    out = p.perturb(ramp, p.spec(severity=4), rng)[0].values()[0, :, 0]
    original = ramp.values()[0, :, 0]
    assert out.min() >= original.min()
    assert out.max() <= original.max()


# --------------------------------------------------------------------------
# OutlierSpikes
# --------------------------------------------------------------------------


def test_outlier_spikes_are_sparse_and_isolated(rng):
    # categorical_features=[] is required: an all-constant feature is inferred
    # categorical by the sequential heuristic, and a numeric perturbation would then
    # have nothing eligible to touch.
    values = np.zeros((1, 500, 1))
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = OutlierSpikes()
    out = p.perturb(data, p.spec(severity=2), rng)[0].values()[0, :, 0]
    hit = np.flatnonzero(out != 0.0)
    assert 0 < hit.size < 100  # rate 0.02 of 500 -> around 10
    assert np.all(np.diff(hit) > 1) or hit.size < 3  # mostly isolated


def test_outlier_spikes_guarantee_at_least_one_spike(rng):
    """Severity 1 has rate 0.01, so a short series would otherwise often be untouched."""
    values = np.zeros((1, 5, 1))
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = OutlierSpikes()
    for seed in range(10):
        out = p.perturb(data, p.spec(severity=1), np.random.default_rng(seed))[0]
        assert (out.values() != 0.0).any()


def test_outlier_spike_height_is_in_reference_units(rng):
    values = np.zeros((1, 200, 1))
    values[0, :100, 0] = 1.0  # std of the feature is 0.5
    data = ArrayAdapter(values, ["x"], sequential=True)
    std = float(np.std(values))
    p = OutlierSpikes()
    out = p.perturb(data, p.spec(severity=3), rng)[0].values()[0, :, 0]
    deltas = np.abs(out - values[0, :, 0])
    spikes = deltas[deltas > 1e-9]
    assert np.allclose(spikes, 5.0 * std)  # severity 3 -> height 5 std


def test_outlier_spikes_have_random_sign_so_they_do_not_bias_the_mean():
    values = np.zeros((200, 100, 1))
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = OutlierSpikes()
    out = p.perturb(data, p.spec(severity=5), np.random.default_rng(5))[0]
    assert abs(float(np.mean(out.values()))) < 0.2


def test_outlier_spikes_report_both_severity_dimensions():
    p = OutlierSpikes()
    meta = p.metadata(p.spec(severity=4))
    assert meta["spike_rate"] == pytest.approx(0.10)
    assert meta["spike_height_in_std"] == pytest.approx(6.0)


# --------------------------------------------------------------------------
# MissingSegment
# --------------------------------------------------------------------------


def test_missing_segment_is_contiguous(ramp, rng):
    """The difference from MissingValues, which drops entries independently."""
    p = MissingSegment()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0].values()[0, :, 0]
    gap = np.flatnonzero(np.isnan(out))
    assert gap.size > 0
    assert np.array_equal(gap, np.arange(gap[0], gap[-1] + 1))


def test_missing_segment_size_matches_the_severity(rng):
    values = np.arange(100, dtype=float).reshape(1, 100, 1)
    data = ArrayAdapter(values, ["x"], sequential=True)
    p = MissingSegment()
    out = p.perturb(data, p.spec(severity=3), rng)[0]
    assert int(np.sum(np.isnan(out.values()))) == 25  # 0.25 of 100


def test_missing_segment_respects_each_entitys_own_length(ragged, rng):
    p = MissingSegment()
    out = p.perturb(ragged, p.spec(severity=5), rng)[0]
    # entity 0 has 5 real steps; 0.8 of 5 = 4 of them go missing, not 4 of 12
    real = out.values()[0, 7:, 0]
    assert int(np.sum(np.isnan(real))) == 4


def test_missing_segment_applies_to_categorical_features_too(rng):
    values = np.zeros((1, 10, 2))
    values[0, :, 1] = 3.0
    data = ArrayAdapter(
        values, ["num", "kind"], categorical_features=["kind"], sequential=True
    )
    p = MissingSegment()
    out = p.perturb(data, p.spec(severity=3), rng)[0]
    assert np.isnan(out.values()[0, :, 1]).any()


# --------------------------------------------------------------------------
# alpha-stable noise
# --------------------------------------------------------------------------


def test_alpha_two_is_gaussian():
    """The survey: Gaussian is the light-tailed special case of the family."""
    draws = sample_alpha_stable(2.0, 0.0, 1.0, 50_000, np.random.default_rng(0))
    # alpha=2 gives N(0, sqrt(2)) under the standard parameterisation
    assert float(np.mean(np.abs(draws) > 6.0)) < 0.001


def test_smaller_alpha_gives_heavier_tails():
    fractions = []
    for alpha in (1.9, 1.5, 1.1):
        draws = sample_alpha_stable(alpha, 0.0, 1.0, 40_000, np.random.default_rng(1))
        fractions.append(float(np.mean(np.abs(draws) > 10.0)))
    assert fractions == sorted(fractions)


def test_alpha_one_is_handled_as_its_own_limit():
    """abs(alpha - 1) is a separate branch of the CMS formula, not a substitution."""
    draws = sample_alpha_stable(1.0, 0.0, 1.0, 5_000, np.random.default_rng(2))
    assert np.isfinite(draws).all()
    assert float(np.median(np.abs(draws))) > 0.0


def test_alpha_stable_is_symmetric_when_beta_is_zero():
    draws = sample_alpha_stable(1.5, 0.0, 1.0, 40_000, np.random.default_rng(3))
    assert float(np.median(draws)) == pytest.approx(0.0, abs=0.05)


def test_alpha_stable_skews_with_beta():
    right = sample_alpha_stable(1.5, 0.9, 1.0, 40_000, np.random.default_rng(4))
    left = sample_alpha_stable(1.5, -0.9, 1.0, 40_000, np.random.default_rng(4))
    assert float(np.median(right)) > float(np.median(left))


def test_alpha_stable_rejects_out_of_range_parameters():
    with pytest.raises(ValueError, match="alpha must lie in"):
        sample_alpha_stable(2.5, 0.0, 1.0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="beta must lie in"):
        sample_alpha_stable(1.5, 2.0, 1.0, 10, np.random.default_rng(0))


def test_alpha_stable_perturbation_gets_heavier_tailed_with_severity():
    values = np.zeros((1, 4000, 1))
    values[0, :, 0] = np.random.default_rng(0).normal(0, 1, 4000)
    data = ArrayAdapter(values, ["x"], sequential=True)
    p = AlphaStableNoise()
    extremes = []
    for s in (1, 5):
        out = p.perturb(data, p.spec(severity=s), np.random.default_rng(6))[0]
        delta = out.values()[0, :, 0] - values[0, :, 0]
        extremes.append(float(np.max(np.abs(delta)) / np.median(np.abs(delta))))
    assert extremes[1] > extremes[0]


def test_alpha_stable_tail_index_is_overridable_via_params():
    p = AlphaStableNoise()
    spec = p.spec(severity=3, params={"alpha": 1.05, "beta": 0.5})
    meta = p.metadata(spec)
    assert meta["alpha"] == pytest.approx(1.05)
    assert meta["beta"] == pytest.approx(0.5)


def test_alpha_stable_default_tail_index_follows_severity():
    p = AlphaStableNoise()
    assert p.metadata(p.spec(severity=1))["alpha"] == pytest.approx(1.9)
    assert p.metadata(p.spec(severity=5))["alpha"] == pytest.approx(1.1)


# --------------------------------------------------------------------------
# an empty eligible feature set is an error, not a silent pass
# --------------------------------------------------------------------------


def test_a_numeric_perturbation_on_all_categorical_data_raises(rng):
    """Silently returning the data unchanged would score as perfect robustness.

    This is easy to hit by accident: a feature that never varies within an entity is
    inferred categorical, so a constant sensor reading takes every numeric perturbation
    out of scope.
    """
    values = np.zeros((2, 6, 1))
    data = ArrayAdapter(values, ["stuck_sensor"], sequential=True)
    assert data.categorical_features == ["stuck_sensor"]

    p = SensorDrift()
    with pytest.raises(ValueError, match="nothing to perturb"):
        p.perturb(data, p.spec(severity=3), rng)


def test_the_error_explains_the_inference_and_the_consequence(rng):
    values = np.zeros((2, 6, 1))
    data = ArrayAdapter(values, ["stuck_sensor"], sequential=True)
    p = SensorDrift()
    with pytest.raises(ValueError) as excinfo:
        p.perturb(data, p.spec(severity=3), rng)
    message = str(excinfo.value)
    assert "inferred categorical" in message
    assert "perfect robustness" in message


def test_such_data_is_also_reported_as_not_applicable(rng):
    values = np.zeros((2, 6, 1))
    data = ArrayAdapter(values, ["stuck_sensor"], sequential=True)
    assert not SensorDrift().applies_to(data)
