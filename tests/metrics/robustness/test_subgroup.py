"""Golden-value tests for the subgroup metrics (survey §5.1.2) and the mask design."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from themis_assess.metrics.robustness.subgroup import (
    accuracy_gap,
    accuracy_variance,
    equal_weight_risk,
    group_disparity,
    group_scores,
    welch_anova,
    worst_group_accuracy,
)

ACC = {"a": 0.9, "b": 0.6, "c": 0.75}
ERR = {"a": 0.1, "b": 0.4, "c": 0.25}


def test_worst_group_accuracy_eq6():
    assert worst_group_accuracy(ACC) == pytest.approx(0.6)


def test_accuracy_gap_eq7():
    assert accuracy_gap(ACC) == pytest.approx(0.3)


def test_accuracy_variance_eq8_is_the_population_variance():
    # mean = 0.75 ; deviations 0.15, -0.15, 0.0 ; sum sq = 0.045 ; / 3 = 0.015
    assert accuracy_variance(ACC) == pytest.approx(0.015)


def test_accuracy_variance_divides_by_g_not_g_minus_one():
    """Population, not sample: the groups are the whole population of interest."""
    values = {"a": 0.0, "b": 1.0}
    assert accuracy_variance(values) == pytest.approx(0.25)  # sample variance would be 0.5


def test_group_disparity_eq9():
    assert group_disparity(ERR) == pytest.approx(0.3)


def test_group_disparity_equals_accuracy_gap_when_errors_are_one_minus_accuracy():
    assert group_disparity(ERR) == pytest.approx(accuracy_gap(ACC))


def test_equal_weight_risk_eq10():
    assert equal_weight_risk(ERR) == pytest.approx(0.25)


def test_equal_weight_risk_is_not_prevalence_weighted():
    """The point of Eq. 10: a large well-served group must not hide a small failing one."""
    errors = {"majority": 0.01, "minority": 0.99}
    assert equal_weight_risk(errors) == pytest.approx(0.5)


def test_average_accuracy_can_hide_a_failing_group():
    """Why §5.1.2 exists at all."""
    accuracies = {"large": 0.95, "small": 0.20}
    assert equal_weight_risk({k: 1 - v for k, v in accuracies.items()}) > 0.4
    assert worst_group_accuracy(accuracies) == pytest.approx(0.20)


def test_metrics_needing_two_groups_reject_one():
    for fn in (accuracy_gap, accuracy_variance, group_disparity):
        with pytest.raises(ValueError, match="at least 2 group"):
            fn({"only": 0.5})


def test_metrics_reject_no_groups():
    for fn in (worst_group_accuracy, equal_weight_risk):
        with pytest.raises(ValueError, match="at least 1 group"):
            fn({})


# --------------------------------------------------------------------------
# group_scores — the mask design (docs/ARCHITECTURE.md §5.6)
# --------------------------------------------------------------------------


def test_group_scores_evaluates_within_each_mask():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 1])
    masks = {
        "first_half": np.array([True, True, True, False, False, False]),
        "second_half": np.array([False, False, False, True, True, True]),
    }
    scores = group_scores(y_true, y_pred, masks, accuracy_score)
    assert scores["first_half"] == pytest.approx(2 / 3)
    assert scores["second_half"] == pytest.approx(2 / 3)


def test_group_scores_allows_overlapping_and_partial_masks():
    """An age band and a job title are both legitimate groups over the same people."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    masks = {
        "over_30": np.array([True, True, False, False]),
        "nurses": np.array([True, False, True, False]),  # overlaps, covers neither fully
    }
    scores = group_scores(y_true, y_pred, masks, accuracy_score)
    assert scores["over_30"] == pytest.approx(1.0)
    assert scores["nurses"] == pytest.approx(0.5)


def test_group_scores_preserves_mask_order():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    masks = {"z": np.array([True, False]), "a": np.array([False, True])}
    assert list(group_scores(y_true, y_pred, masks, accuracy_score)) == ["z", "a"]


def test_group_scores_rejects_a_group_below_min_size():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    masks = {"tiny": np.array([True, False, False, False])}
    with pytest.raises(ValueError, match="below min_group_size"):
        group_scores(y_true, y_pred, masks, accuracy_score, min_group_size=10)


def test_group_scores_rejects_a_mask_of_the_wrong_length():
    with pytest.raises(ValueError, match="but there are 3 samples"):
        group_scores(
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            {"g": np.array([True, False])},
            accuracy_score,
        )


def test_the_three_repo_subgroup_mechanisms_all_reduce_to_masks():
    """One metric, three ways of naming a group — repo C's objection, answered."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    onehot_column = np.array([1, 1, 1, 0, 0, 0])  # repo C: tabular one-hot
    similarity = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.0])  # repo D: embedding themes
    category = np.array(["x", "x", "x", "y", "y", "y"])  # repo B: callable category

    masks = {
        "onehot": onehot_column == 1,
        "similar": similarity > 0.5,
        "category_x": category == "x",
    }
    scores = group_scores(y_true, y_pred, masks, accuracy_score)
    assert len(set(scores.values())) == 1  # all three select the same three samples


# --------------------------------------------------------------------------
# Welch's ANOVA — repo-A extension
# --------------------------------------------------------------------------


def test_welch_anova_finds_no_difference_between_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 200)
    b = rng.normal(0.8, 0.05, 200)
    result = welch_anova([a, b])
    assert result["p_value"] > 0.05
    assert result["n_groups"] == 2


def test_welch_anova_detects_a_real_difference():
    rng = np.random.default_rng(1)
    good = rng.normal(0.90, 0.03, 150)
    poor = rng.normal(0.60, 0.03, 150)
    result = welch_anova({"good": good, "poor": poor})
    assert result["p_value"] < 1e-10
    assert result["statistic"] > 100


def test_welch_anova_tolerates_unequal_variances():
    """The reason for Welch's variant rather than one-way ANOVA."""
    rng = np.random.default_rng(2)
    tight = rng.normal(0.8, 0.01, 100)
    loose = rng.normal(0.8, 0.30, 100)
    result = welch_anova([tight, loose])
    assert result["p_value"] > 0.01  # same mean, wildly different spread


def test_welch_anova_reports_degrees_of_freedom():
    rng = np.random.default_rng(3)
    groups = [rng.normal(0.5, 0.1, 50) for _ in range(3)]
    result = welch_anova(groups)
    assert result["df_between"] == pytest.approx(2.0)
    assert result["df_within"] > 0


def test_welch_anova_needs_two_groups():
    with pytest.raises(ValueError, match="at least 2 groups"):
        welch_anova([[0.1, 0.2]])


def test_welch_anova_needs_at_least_two_values_per_group():
    with pytest.raises(ValueError, match="at least 2 per group"):
        welch_anova([[0.1], [0.2, 0.3]])


def test_welch_anova_rejects_a_zero_variance_group():
    with pytest.raises(ValueError, match="zero variance"):
        welch_anova([[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
