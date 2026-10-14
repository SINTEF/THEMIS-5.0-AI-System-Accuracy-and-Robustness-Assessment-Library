"""Tests for the result containers — what keeps the survey's reporting rules enforceable."""

from __future__ import annotations

import pytest

from themis_assess.core.results import EvaluationResult, RobustnessResult, severity_curve
from themis_assess.core.types import AccessLevel, TaskType


def test_severity_curve_is_one_based():
    assert severity_curve([0.1, 0.2, 0.3]) == {1: 0.1, 2: 0.2, 3: 0.3}


def test_severity_curve_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one level"):
        severity_curve([])


# --------------------------------------------------------------------------
# EvaluationResult
# --------------------------------------------------------------------------


def test_evaluation_result_indexes_by_metric_name():
    result = EvaluationResult(metrics={"mae": 0.3, "rmse": 0.5}, n_samples=100)
    assert result["mae"] == pytest.approx(0.3)
    assert result.metric_names == ["mae", "rmse"]


def test_evaluation_result_names_available_metrics_when_one_is_missing():
    result = EvaluationResult(metrics={"mae": 0.3}, n_samples=10)
    with pytest.raises(KeyError, match=r"Available: \['mae'\]"):
        result["rmse"]


def test_evaluation_result_carries_its_label_and_task():
    result = EvaluationResult(
        metrics={"accuracy": 0.9},
        n_samples=50,
        label="gaussian_noise@3",
        task=TaskType.CLASSIFICATION,
    )
    assert result.label == "gaussian_noise@3"
    assert result.task is TaskType.CLASSIFICATION


def test_evaluation_result_rejects_an_empty_metric_set():
    with pytest.raises(ValueError, match="carries no metrics"):
        EvaluationResult(metrics={}, n_samples=10)


def test_evaluation_result_rejects_a_negative_sample_count():
    with pytest.raises(ValueError, match="cannot be negative"):
        EvaluationResult(metrics={"mae": 0.1}, n_samples=-1)


def test_evaluation_result_is_immutable():
    result = EvaluationResult(metrics={"mae": 0.1}, n_samples=1)
    with pytest.raises(AttributeError):
        result.n_samples = 2  # type: ignore[misc]


# --------------------------------------------------------------------------
# RobustnessResult
# --------------------------------------------------------------------------


@pytest.fixture
def result():
    return RobustnessResult(
        metric_name="mean_corruption_error",
        accuracy_metric="mae",
        clean=0.10,
        per_perturbation={
            "jitter": {1: 0.12, 2: 0.18, 3: 0.30},
            "drift": {1: 0.11, 2: 0.14, 3: 0.20},
        },
        baseline={
            "jitter": {1: 0.20, 2: 0.30, 3: 0.50},
            "drift": {1: 0.22, 2: 0.28, 3: 0.40},
        },
        baseline_clean=0.18,
        scores={"jitter": 0.6, "drift": 0.55},
        aggregate=0.575,
        n_samples=500,
        seed=7,
    )


def test_result_keeps_the_clean_value(result):
    """Required by every relative-drop reporting rule."""
    assert result.clean == pytest.approx(0.10)


def test_result_keeps_the_full_severity_curve_not_just_a_summary(result):
    """'reports performance across a range of severities' — survey §8."""
    assert result.curve("jitter") == {1: 0.12, 2: 0.18, 3: 0.30}
    assert result.severities == {"jitter": [1, 2, 3], "drift": [1, 2, 3]}


def test_result_keeps_the_baseline(result):
    assert result.has_baseline
    assert result.baseline_clean == pytest.approx(0.18)


def test_perturbation_set_defaults_to_the_perturbations_present(result):
    """R_Phi is comparable only within a fixed set, so the set must be recorded."""
    assert set(result.perturbation_set) == {"jitter", "drift"}


def test_worst_severity_follows_the_metric_direction(result):
    assert result.worst_severity("jitter", higher_is_worse=True) == 3
    assert result.worst_severity("jitter", higher_is_worse=False) == 1


def test_curve_names_available_perturbations_when_one_is_missing(result):
    with pytest.raises(KeyError, match="Available:"):
        result.curve("blur")


def test_a_claim_is_incomplete_until_a_tolerance_is_supplied(result):
    """Survey §8: the tolerance is the one element no instrument provides."""
    assert result.is_complete_claim is False
    with_tolerance = RobustnessResult(
        metric_name=result.metric_name,
        accuracy_metric=result.accuracy_metric,
        clean=result.clean,
        per_perturbation=result.per_perturbation,
        tolerance=0.05,
    )
    assert with_tolerance.is_complete_claim is True


def test_result_rejects_having_no_perturbations():
    with pytest.raises(ValueError, match="not a measurement"):
        RobustnessResult(
            metric_name="mCE", accuracy_metric="mae", clean=0.1, per_perturbation={}
        )


def test_result_rejects_an_empty_severity_curve():
    with pytest.raises(ValueError, match="empty severity curve"):
        RobustnessResult(
            metric_name="mCE",
            accuracy_metric="mae",
            clean=0.1,
            per_perturbation={"jitter": {}},
        )


def test_result_rejects_zero_based_severities():
    with pytest.raises(ValueError, match="1-based"):
        RobustnessResult(
            metric_name="mCE",
            accuracy_metric="mae",
            clean=0.1,
            per_perturbation={"jitter": {0: 0.2, 1: 0.3}},
        )


def test_result_rejects_a_baseline_covering_different_perturbations():
    with pytest.raises(ValueError, match="same perturbations"):
        RobustnessResult(
            metric_name="mCE",
            accuracy_metric="mae",
            clean=0.1,
            per_perturbation={"jitter": {1: 0.2}},
            baseline={"drift": {1: 0.3}},
        )


def test_result_records_the_access_it_needed(result):
    assert result.access_used is AccessLevel.BLACK_BOX


def test_result_is_immutable(result):
    with pytest.raises(AttributeError):
        result.clean = 0.0  # type: ignore[misc]
