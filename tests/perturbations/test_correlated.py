"""Tests for the correlation-respecting perturbation (survey §7)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import Scope, TimeScope
from themis_assess.perturbations.correlated import CorrelatedNoise, estimate_covariance
from themis_assess.perturbations.tabular import GaussianNoise


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def correlated():
    """Two features with correlation near 1 — y is essentially 3x."""
    gen = np.random.default_rng(0)
    x = gen.normal(0.0, 1.0, 600)
    y = 3.0 * x + gen.normal(0.0, 0.05, 600)
    return ArrayAdapter(np.column_stack([x, y]), ["x", "y"])


# --------------------------------------------------------------------------
# covariance estimation
# --------------------------------------------------------------------------


def test_estimate_covariance_recovers_the_structure(correlated):
    covariance, factor = estimate_covariance(correlated, ["x", "y"], shrinkage=0.0)
    assert covariance.shape == (2, 2)
    corr = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    assert corr == pytest.approx(1.0, abs=0.01)
    assert np.allclose(factor @ factor.T, covariance, atol=1e-8)


def test_shrinkage_pulls_toward_the_diagonal(correlated):
    plain, _ = estimate_covariance(correlated, ["x", "y"], shrinkage=0.0)
    shrunk, _ = estimate_covariance(correlated, ["x", "y"], shrinkage=0.5)
    assert abs(shrunk[0, 1]) < abs(plain[0, 1])
    assert shrunk[0, 0] == pytest.approx(plain[0, 0])  # diagonal untouched


def test_estimate_covariance_survives_exactly_collinear_features():
    """The raw covariance is singular here and has no Cholesky factor."""
    x = np.random.default_rng(1).normal(0, 1, 300)
    data = ArrayAdapter(np.column_stack([x, x]), ["a", "b"])
    covariance, factor = estimate_covariance(data, ["a", "b"])
    assert np.all(np.isfinite(factor))
    assert np.allclose(factor @ factor.T, covariance, atol=1e-6)


def test_estimate_covariance_ignores_padding():
    values = np.full((2, 5, 2), np.nan)
    values[0, 2:, :] = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
    values[1, :, :] = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0], [5.0, 10.0]]
    data = ArrayAdapter(
        values, ["a", "b"], entity_lengths=[3, 5], categorical_features=[]
    )
    covariance, _ = estimate_covariance(data, ["a", "b"], shrinkage=0.0)
    assert np.all(np.isfinite(covariance))


def test_estimate_covariance_needs_two_complete_observations():
    data = ArrayAdapter(np.array([[1.0, 2.0]]), ["a", "b"])
    with pytest.raises(ValueError, match="at least 2 complete observations"):
        estimate_covariance(data, ["a", "b"])


def test_estimate_covariance_rejects_shrinkage_out_of_range(correlated):
    with pytest.raises(ValueError, match="shrinkage must lie in"):
        estimate_covariance(correlated, ["x", "y"], shrinkage=1.5)


# --------------------------------------------------------------------------
# the survey's actual point
# --------------------------------------------------------------------------


def test_independent_noise_destroys_the_correlation_and_correlated_noise_does_not(
    correlated,
):
    """Survey §7, made measurable.

    "perturbing individual features independently can produce physically implausible
    combinations that violate the statistical structure of real data."

    On data with correlation ~1.0, independent Gaussian noise at the same severity
    collapses the relationship — every perturbed point is a state the world does not
    produce. Correlated noise moves along the data's own direction of variation and
    leaves the relationship largely intact.
    """
    original = float(np.corrcoef(correlated.values()[:, 0, 0], correlated.values()[:, 0, 1])[0, 1])
    assert original > 0.99

    def corr_after(perturbation):
        out = perturbation.perturb(
            correlated, perturbation.spec(severity=4), np.random.default_rng(1)
        )[0]
        v = out.values()[:, 0, :]
        return float(np.corrcoef(v[:, 0], v[:, 1])[0, 1])

    independent = corr_after(GaussianNoise())
    respecting = corr_after(CorrelatedNoise())

    assert independent < 0.5
    assert respecting > 0.9
    assert respecting > independent


def test_the_gap_between_the_two_is_a_diagnostic(correlated):
    """A large gap says the independent-noise result was an artefact of implausible inputs."""
    spec_kwargs = {"severity": 4}
    gaussian = GaussianNoise()
    corr_noise = CorrelatedNoise()
    g = gaussian.perturb(correlated, gaussian.spec(**spec_kwargs), np.random.default_rng(2))[0]
    c = corr_noise.perturb(correlated, corr_noise.spec(**spec_kwargs), np.random.default_rng(2))[0]
    # Both move the data by a comparable amount...
    shift_g = float(np.std(g.values() - correlated.values()))
    shift_c = float(np.std(c.values() - correlated.values()))
    assert shift_c > 0.3 * shift_g
    # ...but only one of them lands on plausible states.
    assert not np.allclose(g.values(), c.values())


def test_uncorrelated_data_gives_similar_results_either_way():
    """No correlation to respect means no advantage — the honest null case."""
    gen = np.random.default_rng(3)
    data = ArrayAdapter(gen.normal(0, 1, (800, 2)), ["a", "b"])

    def corr_after(p):
        out = p.perturb(data, p.spec(severity=4), np.random.default_rng(4))[0]
        v = out.values()[:, 0, :]
        return float(np.corrcoef(v[:, 0], v[:, 1])[0, 1])

    assert abs(corr_after(GaussianNoise())) < 0.15
    assert abs(corr_after(CorrelatedNoise())) < 0.15


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------


def test_severity_scales_the_perturbation(correlated):
    p = CorrelatedNoise()
    shifts = []
    for s in range(1, 6):
        out = p.perturb(correlated, p.spec(severity=s), np.random.default_rng(5))[0]
        shifts.append(float(np.std(out.values() - correlated.values())))
    assert shifts == sorted(shifts)


def test_constant_time_scope_holds_the_draw_across_the_series():
    gen = np.random.default_rng(6)
    x = gen.normal(0, 1, (3, 8))
    values = np.stack([x, 2 * x], axis=-1)
    data = ArrayAdapter(values, ["a", "b"], categorical_features=[])
    p = CorrelatedNoise()
    out = p.perturb(data, p.spec(severity=4, time_scope=TimeScope.CONSTANT), gen)[0]
    delta = out.values() - data.values()
    for e in range(3):
        assert np.allclose(delta[e, :, 0], delta[e, 0, 0])


def test_per_timestep_scope_draws_independently():
    gen = np.random.default_rng(7)
    x = gen.normal(0, 1, (3, 8))
    values = np.stack([x, 2 * x], axis=-1)
    data = ArrayAdapter(values, ["a", "b"], categorical_features=[])
    p = CorrelatedNoise()
    out = p.perturb(data, p.spec(severity=4, time_scope=TimeScope.PER_TIMESTEP), gen)[0]
    delta = out.values() - data.values()
    assert not np.allclose(delta[0, :, 0], delta[0, 0, 0])


def test_needs_at_least_two_features(rng):
    data = ArrayAdapter(np.random.default_rng(8).normal(0, 1, (100, 1)), ["only"])
    p = CorrelatedNoise()
    with pytest.raises(ValueError, match="at least 2 numerical features"):
        p.perturb(data, p.spec(severity=3), rng)


def test_leaves_categorical_features_untouched(rng):
    gen = np.random.default_rng(9)
    values = np.column_stack([gen.normal(0, 1, 200), gen.normal(0, 1, 200), np.ones(200)])
    data = ArrayAdapter(values, ["a", "b", "kind"], categorical_features=["kind"])
    p = CorrelatedNoise()
    out = p.perturb(data, p.spec(severity=4), rng)[0]
    assert np.array_equal(out.values()[:, :, 2], data.values()[:, :, 2])


def test_preserves_padding():
    gen = np.random.default_rng(10)
    values = np.full((2, 6, 2), np.nan)
    values[0, 2:, :] = gen.normal(0, 1, (4, 2))
    values[1, :, :] = gen.normal(0, 1, (6, 2))
    data = ArrayAdapter(
        values, ["a", "b"], entity_lengths=[4, 6], categorical_features=[]
    )
    p = CorrelatedNoise()
    out = p.perturb(data, p.spec(severity=3), gen)[0]
    assert np.isnan(out.values()[0, :2, :]).all()
    assert np.array_equal(out.valid_mask(), data.valid_mask())


def test_metadata_records_the_caveat_and_the_shrinkage():
    p = CorrelatedNoise(shrinkage=0.1)
    meta = p.metadata(p.spec(severity=3))
    assert meta["shrinkage"] == pytest.approx(0.1)
    assert meta["respects_feature_correlation"] is True
    assert "linear dependence only" in meta["caveat"]


def test_reproducible_from_a_seed(correlated):
    p = CorrelatedNoise()
    a = p.perturb(correlated, p.spec(severity=3), np.random.default_rng(11))[0]
    b = p.perturb(correlated, p.spec(severity=3), np.random.default_rng(11))[0]
    assert np.array_equal(a.values(), b.values())


def test_scope_named_restricts_which_features_move(rng):
    gen = np.random.default_rng(12)
    values = np.column_stack([gen.normal(0, 1, 300) for _ in range(3)])
    data = ArrayAdapter(values, ["a", "b", "c"])
    p = CorrelatedNoise()
    spec = p.spec(severity=4, scope=Scope.NAMED, features=("a", "b"))
    out = p.perturb(data, spec, rng)[0]
    assert np.array_equal(out.values()[:, :, 2], data.values()[:, :, 2])
    assert not np.array_equal(out.values()[:, :, 0], data.values()[:, :, 0])
