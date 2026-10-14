"""Tests for STL component-selective perturbation (survey §4.2, structure-aware)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import TimeScope
from themis_assess.perturbations.timeseries.structural import (
    Component,
    STLComponentPerturbation,
    decompose,
)

PERIOD = 24
LENGTH = 96


def make_signal(seed: int = 0) -> np.ndarray:
    """A trend plus a clean daily cycle plus noise — the case STL is designed for."""
    gen = np.random.default_rng(seed)
    t = np.arange(LENGTH)
    return 0.05 * t + 3.0 * np.sin(2 * np.pi * t / PERIOD) + gen.normal(0, 0.3, LENGTH)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def seasonal():
    return ArrayAdapter(make_signal().reshape(1, LENGTH, 1), ["x"], sequential=True)


# --------------------------------------------------------------------------
# decomposition
# --------------------------------------------------------------------------


def test_components_sum_back_to_the_series():
    signal = make_signal()
    trend, seasonal_part, residual = decompose(signal, PERIOD)
    assert np.allclose(trend + seasonal_part + residual, signal)


def test_decomposition_separates_trend_from_cycle():
    signal = make_signal()
    trend, seasonal_part, residual = decompose(signal, PERIOD)
    assert np.all(np.diff(trend) > -0.5)  # the trend is essentially monotone rising
    assert seasonal_part.std() > residual.std()  # the cycle dominates the noise


def test_decompose_needs_two_full_periods():
    with pytest.raises(ValueError, match="at least 2 full periods"):
        decompose(np.zeros(30), 24)


def test_decompose_rejects_a_degenerate_period():
    with pytest.raises(ValueError, match="period must be at least 2"):
        decompose(np.zeros(100), 1)


# --------------------------------------------------------------------------
# the survey's actual point: one component at a time
# --------------------------------------------------------------------------


def test_residual_perturbation_preserves_trend_and_seasonality(seasonal, rng):
    """The survey's first example: 'noise into residual preserving trend and seasonality'.

    Compared in *relative* terms — each component's drift against its own scale.
    Re-decomposing the perturbed series re-attributes a little of the injected noise to
    the other components, so an absolute comparison would be measuring STL's estimator
    rather than the perturbation.
    """
    original = seasonal.values()[0, :, 0]
    trend, season, resid = decompose(original, PERIOD)

    p = STLComponentPerturbation(period=PERIOD, component=Component.RESIDUAL)
    out = p.perturb(seasonal, p.spec(severity=4), rng)[0].values()[0, :, 0]
    trend2, season2, resid2 = decompose(out, PERIOD)

    relative_resid = np.std(resid2 - resid) / resid.std()
    relative_season = np.std(season2 - season) / season.std()
    relative_trend = np.std(trend2 - trend) / trend.std()

    assert relative_resid > 5 * relative_season
    assert relative_resid > 5 * relative_trend
    assert relative_season < 0.25  # the cycle survives essentially intact


def test_seasonal_perturbation_changes_amplitude_and_keeps_phase(seasonal, rng):
    """The survey's second example: 'alter the seasonal amplitude only'."""
    original = seasonal.values()[0, :, 0]
    trend, season, resid = decompose(original, PERIOD)

    p = STLComponentPerturbation(period=PERIOD, component=Component.SEASONAL)
    out = p.perturb(seasonal, p.spec(severity=4), rng)[0].values()[0, :, 0]
    trend2, season2, resid2 = decompose(out, PERIOD)

    # the amplitude is scaled by exactly 1 +- the severity factor...
    ratio = season2.std() / season.std()
    factor = p.severity_factor(p.spec(severity=4))
    assert ratio == pytest.approx(1.0 - factor, abs=0.05) or ratio == pytest.approx(
        1.0 + factor, abs=0.05
    )
    # ...the cycle still lines up in phase with the original...
    assert abs(float(np.corrcoef(season, season2)[0, 1])) > 0.95
    # ...and trend and noise are barely touched, relative to their own scale
    assert np.std(trend2 - trend) / trend.std() < 0.1
    assert np.std(resid2 - resid) / resid.std() < 0.6


def test_trend_perturbation_is_a_level_shift(seasonal, rng):
    """A constant offset, so it must be measured by its mean — its std is zero."""
    p = STLComponentPerturbation(period=PERIOD, component=Component.TREND)
    original = seasonal.values()[0, :, 0]
    out = p.perturb(seasonal, p.spec(severity=4), rng)[0].values()[0, :, 0]
    delta = out - original
    assert np.allclose(delta, delta[0])  # constant offset
    assert abs(delta[0]) > 0.0

    trend, season, resid = decompose(original, PERIOD)
    trend2, season2, resid2 = decompose(out, PERIOD)
    # the whole shift lands in the trend; the cycle and the noise are untouched
    assert abs(np.mean(trend2 - trend)) == pytest.approx(abs(delta[0]), rel=1e-6)
    assert np.allclose(season2, season, atol=1e-9)
    assert np.allclose(resid2, resid, atol=1e-9)


def test_each_component_reports_what_it_preserves():
    for component in Component:
        p = STLComponentPerturbation(period=PERIOD, component=component)
        meta = p.metadata(p.spec(severity=3))
        assert meta["component"] == component.value
        assert component.value not in meta["preserves"]
        assert len(meta["preserves"]) == 2


def test_severity_means_something_different_per_component():
    """And each says so, so a report can state what severity 3 was."""
    keys = {
        Component.RESIDUAL: "noise_in_residual_std",
        Component.SEASONAL: "seasonal_amplitude_change",
        Component.TREND: "trend_offset_in_trend_std",
    }
    for component, key in keys.items():
        p = STLComponentPerturbation(period=PERIOD, component=component)
        assert key in p.metadata(p.spec(severity=3))


def test_structure_preserving_noise_is_gentler_than_broadband_noise(seasonal, rng):
    """Why the family exists.

    Adding noise to the whole series blurs the seasonal structure the model extrapolates
    from; putting the same-sized noise into the residual leaves that structure exact.
    """
    from themis_assess.perturbations.tabular import GaussianNoise

    original = seasonal.values()[0, :, 0]
    _, season, _ = decompose(original, PERIOD)

    # PER_TIMESTEP, not the default CONSTANT: a constant offset is absorbed whole into
    # the trend and leaves the seasonal component untouched, which would make this
    # comparison vacuous.
    broadband = GaussianNoise().perturb(
        seasonal,
        GaussianNoise().spec(severity=3, time_scope=TimeScope.PER_TIMESTEP),
        np.random.default_rng(1),
    )[0]
    structural = STLComponentPerturbation(
        period=PERIOD, component=Component.RESIDUAL
    ).perturb(
        seasonal,
        STLComponentPerturbation(period=PERIOD).spec(severity=3),
        np.random.default_rng(1),
    )[0]

    _, season_broad, _ = decompose(broadband.values()[0, :, 0], PERIOD)
    _, season_struct, _ = decompose(structural.values()[0, :, 0], PERIOD)

    assert np.std(season_struct - season) < 0.25 * np.std(season_broad - season)


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------


def test_severity_increases_the_disturbance(seasonal):
    original = seasonal.values()[0, :, 0]
    p = STLComponentPerturbation(period=PERIOD, component=Component.RESIDUAL)
    shifts = []
    for s in range(1, 6):
        out = p.perturb(seasonal, p.spec(severity=s), np.random.default_rng(2))[0]
        shifts.append(float(np.std(out.values()[0, :, 0] - original)))
    assert shifts == sorted(shifts)


def test_reproducible_from_a_seed(seasonal):
    p = STLComponentPerturbation(period=PERIOD)
    a = p.perturb(seasonal, p.spec(severity=3), np.random.default_rng(5))[0]
    b = p.perturb(seasonal, p.spec(severity=3), np.random.default_rng(5))[0]
    assert np.array_equal(a.values(), b.values())


def test_series_too_short_are_skipped_and_counted():
    """A ragged dataset will usually have a few; failing the run for them would be worse."""
    values = np.full((2, LENGTH, 1), np.nan)
    values[0, -20:, 0] = make_signal()[:20]  # only 20 steps, under 2 periods
    values[1, :, 0] = make_signal(1)
    data = ArrayAdapter(
        values, ["x"], entity_lengths=[20, LENGTH], categorical_features=[]
    )
    p = STLComponentPerturbation(period=PERIOD)
    out = p.perturb(data, p.spec(severity=3), np.random.default_rng(6))[0]

    assert np.array_equal(out.values()[0], data.values()[0], equal_nan=True)  # untouched
    assert not np.array_equal(out.values()[1], data.values()[1], equal_nan=True)
    assert p.metadata()["entities_skipped_too_short"] == 1


def test_it_is_sequence_only(rng):
    tabular = ArrayAdapter(np.random.default_rng(7).normal(0, 1, (20, 2)), ["a", "b"])
    p = STLComponentPerturbation(period=PERIOD)
    assert not p.applies_to(tabular)


def test_it_leaves_categorical_features_alone(rng):
    values = np.zeros((1, LENGTH, 2))
    values[0, :, 0] = make_signal()
    values[0, :, 1] = 5.0
    data = ArrayAdapter(values, ["x", "kind"], categorical_features=["kind"])
    p = STLComponentPerturbation(period=PERIOD)
    out = p.perturb(data, p.spec(severity=4), rng)[0]
    assert np.array_equal(out.values()[:, :, 1], data.values()[:, :, 1])


def test_construction_rejects_a_degenerate_period():
    with pytest.raises(ValueError, match="period must be at least 2"):
        STLComponentPerturbation(period=1)


def test_component_accepts_a_string():
    p = STLComponentPerturbation(period=PERIOD, component="seasonal")
    assert p.component is Component.SEASONAL


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError):
        STLComponentPerturbation(period=PERIOD, component="cyclical")


def test_it_is_not_in_the_default_registry():
    """It needs a seasonal period, which cannot be inferred from an unlabelled array."""
    from themis_assess.perturbations.registry import default_registry

    assert "stl_component" not in default_registry()


def test_it_can_be_registered_by_the_caller(seasonal):
    from themis_assess.perturbations.registry import default_registry

    registry = default_registry()
    registry.register(STLComponentPerturbation(period=PERIOD, component=Component.SEASONAL))
    assert "stl_component" in registry.applicable(seasonal)


def test_seasonal_severity_never_annihilates_or_inverts_the_cycle(seasonal):
    """Why the seasonal ladder is capped below 1.0.

    Scaling by ``1 - severity`` would zero the cycle at severity 1.0 and flip its sign
    above it. An inverted cycle is a different signal, not a quieter one.
    """
    p = STLComponentPerturbation(period=PERIOD, component=Component.SEASONAL)
    assert max(p.severity_scale) < 1.0

    original = seasonal.values()[0, :, 0]
    _, season, _ = decompose(original, PERIOD)
    for s in range(1, 6):
        for seed in range(6):  # cover both draw directions
            out = p.perturb(seasonal, p.spec(severity=s), np.random.default_rng(seed))[0]
            _, season2, _ = decompose(out.values()[0, :, 0], PERIOD)
            assert season2.std() > 0.2 * season.std()  # never annihilated
            assert float(np.corrcoef(season, season2)[0, 1]) > 0.9  # never inverted


def test_each_component_has_its_own_severity_ladder():
    residual = STLComponentPerturbation(period=PERIOD, component=Component.RESIDUAL)
    seasonal_p = STLComponentPerturbation(period=PERIOD, component=Component.SEASONAL)
    assert residual.severity_scale != seasonal_p.severity_scale
    assert residual.metadata()["severity_scale"] == list(residual.severity_scale)
    assert seasonal_p.metadata()["severity_scale"] == list(seasonal_p.severity_scale)
