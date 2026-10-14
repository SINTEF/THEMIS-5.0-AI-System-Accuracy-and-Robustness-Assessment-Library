"""Tests for normalize_metric, including the argument-shadowing bug fixed on the port."""

from __future__ import annotations

import pytest

from themis_assess.metrics.normalize import normalize_metric, normalize_metrics
from themis_assess.metrics.registry import MetricSpec, default_registry


@pytest.fixture
def reg():
    return default_registry()


def test_bounded_higher_is_better_is_min_max(reg):
    assert normalize_metric(0.0, "accuracy", registry=reg) == pytest.approx(0.0)
    assert normalize_metric(0.75, "accuracy", registry=reg) == pytest.approx(0.75)
    assert normalize_metric(1.0, "accuracy", registry=reg) == pytest.approx(1.0)


def test_bounded_metric_with_negative_lower_bound(reg):
    # matthews_corrcoef spans [-1, 1]; 0 sits halfway
    assert normalize_metric(0.0, "matthews_corrcoef", registry=reg) == pytest.approx(0.5)
    assert normalize_metric(-1.0, "matthews_corrcoef", registry=reg) == pytest.approx(0.0)


def test_bounded_lower_is_better_is_flipped(reg):
    reg.register(
        "bounded_error", MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0)
    )
    assert normalize_metric(0.0, "bounded_error", registry=reg) == pytest.approx(1.0)
    assert normalize_metric(1.0, "bounded_error", registry=reg) == pytest.approx(0.0)
    assert normalize_metric(0.25, "bounded_error", registry=reg) == pytest.approx(0.75)


def test_unbounded_lower_is_better_decays(reg):
    # rmse: lower_bound=0, no upper bound -> exp(-value / scale)
    assert normalize_metric(0.0, "rmse", registry=reg) == pytest.approx(1.0)
    assert normalize_metric(1.0, "rmse", registry=reg) == pytest.approx(pytest.approx(0.3679, abs=1e-3))
    assert normalize_metric(10.0, "rmse", registry=reg) < 0.01


def test_unbounded_lower_is_better_is_monotone(reg):
    values = [normalize_metric(v, "mae", registry=reg) for v in (0.1, 1.0, 5.0, 50.0)]
    assert values == sorted(values, reverse=True)


def test_unbounded_higher_is_better_squashes(reg):
    assert normalize_metric(0.0, "log_likelihood", registry=reg) == pytest.approx(0.5)
    assert normalize_metric(10.0, "log_likelihood", registry=reg) > 0.99
    assert normalize_metric(-10.0, "log_likelihood", registry=reg) < 0.01


def test_symmetric_metric_peaks_at_its_target(reg):
    """mean_signed_deviation: best at 0, worse in either direction."""
    assert normalize_metric(0.0, "mean_signed_deviation", registry=reg) == pytest.approx(1.0)
    above = normalize_metric(0.5, "mean_signed_deviation", registry=reg)
    below = normalize_metric(-0.5, "mean_signed_deviation", registry=reg)
    assert above == pytest.approx(below)
    assert above < 1.0


def test_symmetric_branch_wins_over_min_max_even_when_bounds_exist(reg):
    """The repo B ordering bug: a symmetric metric with bounds fell into min-max."""
    reg.register(
        "signed_bounded",
        MetricSpec(higher_is_better=None, target=0.0, lower_bound=-1.0, upper_bound=1.0),
    )
    assert normalize_metric(0.0, "signed_bounded", registry=reg) == pytest.approx(1.0)
    # min-max would have given 0.5 at the target, and 1.0 at the upper bound
    assert normalize_metric(1.0, "signed_bounded", registry=reg) < 0.5


def test_explicit_bounds_are_not_shadowed_by_the_spec(reg):
    """The repo B defect: `best = spec.best if spec.best != None else best`.

    An explicit upper_bound argument must win, and must not be silently replaced by a
    target the spec happens to declare.
    """
    assert normalize_metric(
        1.0, "accuracy", registry=reg, lower_bound=0.0, upper_bound=2.0
    ) == pytest.approx(0.5)


def test_explicit_scale_overrides_the_spec(reg):
    reg.amend("rmse", scale=1.0)
    tight = normalize_metric(2.0, "rmse", registry=reg, scale=1.0)
    loose = normalize_metric(2.0, "rmse", registry=reg, scale=10.0)
    assert loose > tight


def test_registered_scale_is_used_when_none_is_given(reg):
    reg.amend("rmse", scale=10.0)
    assert normalize_metric(2.0, "rmse", registry=reg) == pytest.approx(
        normalize_metric(2.0, "rmse", registry=reg, scale=10.0)
    )


def test_result_is_always_clamped_to_zero_one(reg):
    assert normalize_metric(5.0, "accuracy", registry=reg) == pytest.approx(1.0)
    assert normalize_metric(-5.0, "accuracy", registry=reg) == pytest.approx(0.0)


def test_degenerate_bounds_raise(reg):
    reg.register(
        "flat", MetricSpec(higher_is_better=True, lower_bound=1.0, upper_bound=1.0 + 1e-12)
    )
    with pytest.raises(ValueError, match="lower and upper bound"):
        normalize_metric(1.0, "flat", registry=reg, lower_bound=2.0, upper_bound=2.0)


def test_non_positive_scale_raises(reg):
    with pytest.raises(ValueError, match="scale must be positive"):
        normalize_metric(1.0, "rmse", registry=reg, scale=0.0)


def test_unknown_metric_raises(reg):
    with pytest.raises(KeyError):
        normalize_metric(1.0, "not_a_metric", registry=reg)


def test_normalize_metrics_preserves_key_order(reg):
    values = {"accuracy": 0.8, "f1": 0.6, "rmse": 0.5}
    out = normalize_metrics(values, registry=reg)
    assert list(out) == list(values)
    assert out["accuracy"] == pytest.approx(0.8)


def test_normalization_makes_mixed_metrics_comparable(reg):
    """Why this exists: an accuracy and an error on one polar chart, 1 = best for both."""
    out = normalize_metrics({"accuracy": 1.0, "rmse": 0.0}, registry=reg)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["rmse"] == pytest.approx(1.0)
