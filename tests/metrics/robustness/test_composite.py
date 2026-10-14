"""Tests for the composite scores (survey §5.1.4) and their component-reporting contract."""

from __future__ import annotations

import pytest

from themis_assess.metrics.robustness.composite import (
    composite_reliability_score,
    unified_robustness_metric,
)


def test_urm_eq26():
    # 1 - (0.5*0.2 + 0.5*10/100) = 1 - 0.15 = 0.85
    result = unified_robustness_metric(acdd=0.2, sd=10.0, lam=0.5)
    assert result.score == pytest.approx(0.85)


def test_urm_is_one_for_a_perfectly_stable_model():
    assert unified_robustness_metric(acdd=0.0, sd=0.0).score == pytest.approx(1.0)


def test_urm_depends_on_lambda_which_is_why_it_must_be_reported():
    degradation_heavy = unified_robustness_metric(acdd=0.4, sd=2.0, lam=1.0)
    variability_heavy = unified_robustness_metric(acdd=0.4, sd=2.0, lam=0.0)
    assert degradation_heavy.score != pytest.approx(variability_heavy.score)
    assert degradation_heavy.components()["lambda"] == 1.0


def test_urm_carries_its_components():
    result = unified_robustness_metric(acdd=0.25, sd=8.0, lam=0.3)
    assert result.components() == {"ACDD": 0.25, "SD": 8.0, "lambda": 0.3}


def test_urm_rejects_lambda_outside_zero_one():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="lam must be"):
            unified_robustness_metric(acdd=0.1, sd=1.0, lam=bad)


def test_composite_reliability_score_eq27_defaults_to_uniform_weights():
    scores = {"s_ID": 0.9, "s_DS": 0.7, "s_ADV": 0.5, "s_CAL": 0.8, "s_OOD": 0.6}
    result = composite_reliability_score(scores)
    assert result.score == pytest.approx((0.9 + 0.7 + 0.5 + 0.8 + 0.6) / 5)
    assert all(w == pytest.approx(0.2) for w in result.weights.values())


def test_composite_reliability_score_applies_given_weights():
    scores = {"a": 1.0, "b": 0.0}
    result = composite_reliability_score(scores, weights={"a": 3.0, "b": 1.0})
    assert result.score == pytest.approx(0.75)
    assert result.weights == {"a": 0.75, "b": 0.25}


def test_composite_reliability_score_normalises_weights_that_do_not_sum_to_one():
    result = composite_reliability_score({"a": 1.0, "b": 0.0}, weights={"a": 30, "b": 10})
    assert result.score == pytest.approx(0.75)


def test_composite_reliability_score_carries_per_property_results():
    """'it cannot show which property failed, so the per-property results should
    always accompany it'."""
    scores = {"s_ID": 0.95, "s_DS": 0.95, "s_ADV": 0.95, "s_CAL": 0.05, "s_OOD": 0.95}
    result = composite_reliability_score(scores)
    assert result.score > 0.7  # the aggregate looks healthy
    assert min(result.components(), key=result.components().get) == "s_CAL"


def test_composite_reliability_score_accepts_a_reduced_property_set():
    """A project may not measure adversarial robustness at all."""
    result = composite_reliability_score({"s_ID": 0.8, "s_DS": 0.6})
    assert result.score == pytest.approx(0.7)


def test_composite_reliability_score_rejects_unnormalised_components():
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        composite_reliability_score({"s_ID": 1.4, "s_DS": 0.6})


def test_composite_reliability_score_rejects_mismatched_weight_keys():
    with pytest.raises(ValueError, match="Missing weights"):
        composite_reliability_score({"a": 0.5, "b": 0.5}, weights={"a": 1.0})
    with pytest.raises(ValueError, match="Unexpected weights"):
        composite_reliability_score({"a": 0.5}, weights={"a": 1.0, "b": 1.0})


def test_composite_reliability_score_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one property"):
        composite_reliability_score({})


def test_composite_results_are_immutable():
    result = unified_robustness_metric(acdd=0.1, sd=1.0)
    with pytest.raises(AttributeError):
        result.score = 0.0  # type: ignore[misc]
