"""Tests for MetricSpec and MetricRegistry, including the two defects fixed on the port."""

from __future__ import annotations

import pytest

from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.metrics.registry import MetricRegistry, MetricSpec, default_registry


def test_default_registry_has_the_survey_metrics():
    reg = default_registry()
    for name in (
        "accuracy_drop",
        "relative_robustness",
        "corruption_error",
        "mean_relative_corruption_error",
        "flip_probability",
        "flip_rate",
        "worst_group_accuracy",
        "auroc",
        "autc",
    ):
        assert name in reg


def test_each_registry_instance_is_independent():
    """A caller registering a custom metric must not mutate anyone else's registry."""
    a, b = default_registry(), default_registry()
    a.register("my_metric", MetricSpec(higher_is_better=True))
    assert "my_metric" in a
    assert "my_metric" not in b


def test_survey_metrics_carry_their_equation_reference():
    reg = default_registry()
    assert reg.get("corruption_error").paper_reference == "survey §5.1.3 Eq. 19"
    assert reg.get("flip_rate").paper_reference == "survey §5.1.3 Eq. 24"


def test_flip_rate_is_the_baseline_normalised_metric_not_the_flip_proportion():
    """D19: the name now means Eq. 24, not what repo A and repo D called flip_rate."""
    reg = default_registry()
    assert reg.get("flip_rate").requires_baseline is True
    assert reg.get("flip_probability").requires_baseline is False


def test_target_and_upper_bound_are_separate_fields():
    """The repo B defect: `best` conflated the ideal value with the min-max bound."""
    reg = default_registry()

    signed = reg.get("mean_signed_deviation")
    assert signed.target == 0.0
    assert signed.upper_bound is None
    assert signed.higher_is_better is None

    f1 = reg.get("f1")
    assert f1.upper_bound == 1.0
    assert f1.target is None


def test_a_directionless_metric_must_declare_a_target():
    with pytest.raises(ValueError, match="must declare a target"):
        MetricSpec(higher_is_better=None)


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="must be below"):
        MetricSpec(higher_is_better=True, lower_bound=1.0, upper_bound=0.0)


def test_scale_must_be_positive():
    with pytest.raises(ValueError, match="scale must be positive"):
        MetricSpec(higher_is_better=True, scale=0.0)


def test_specs_are_immutable():
    spec = MetricSpec(higher_is_better=True)
    with pytest.raises(AttributeError):
        spec.higher_is_better = False  # type: ignore[misc]


def test_registering_a_duplicate_requires_overwrite():
    reg = MetricRegistry(with_defaults=False)
    reg.register("m", MetricSpec(higher_is_better=True))
    with pytest.raises(ValueError, match="already registered"):
        reg.register("m", MetricSpec(higher_is_better=False))
    reg.register("m", MetricSpec(higher_is_better=False), overwrite=True)
    assert reg.get("m").higher_is_better is False


def test_amend_changes_selected_fields():
    reg = default_registry()
    amended = reg.amend("rmse", scale=2.5)
    assert amended.scale == 2.5
    assert amended.higher_is_better is False  # unchanged
    assert reg.get("rmse").scale == 2.5


def test_unknown_metric_suggests_near_matches():
    reg = default_registry()
    with pytest.raises(KeyError, match="Did you mean"):
        reg.get("corruption")


def test_unknown_metric_without_near_matches_still_names_itself():
    reg = default_registry()
    with pytest.raises(KeyError, match="zzz_nonsense"):
        reg.get("zzz_nonsense")


def test_for_task_filters_by_applicability():
    reg = default_registry()
    clf_only = set(reg.for_task(TaskType.CLASSIFICATION))
    fc_only = set(reg.for_task(TaskType.FORECASTING))
    assert "flip_probability" in clf_only
    assert "flip_probability" not in fc_only
    assert "rmse" in fc_only
    assert "corruption_error" in clf_only and "corruption_error" in fc_only


def test_requiring_baseline_lists_the_right_family():
    reg = default_registry()
    needs_baseline = set(reg.requiring_baseline())
    assert {
        "corruption_error",
        "mean_corruption_error",
        "relative_corruption_error",
        "mean_relative_corruption_error",
        "relative_error_inflation",
        "mean_relative_error_inflation",
        "flip_rate",
        "mean_flip_rate",
    } <= needs_baseline
    assert "accuracy_drop" not in needs_baseline


def test_unavailable_at_reports_what_a_handler_cannot_compute():
    """The up-front capability check: tell the user before a long run, not during it."""
    reg = default_registry()
    blocked = set(reg.unavailable_at(AccessLevel.BLACK_BOX))
    assert "auroc" in blocked  # needs probabilities
    assert "confidence_retention" in blocked
    assert "accuracy_drop" not in blocked

    assert reg.unavailable_at(AccessLevel.GRADIENT) == []


def test_access_levels_are_ordered():
    assert AccessLevel.PROBABILISTIC.satisfies(AccessLevel.BLACK_BOX)
    assert not AccessLevel.BLACK_BOX.satisfies(AccessLevel.PROBABILISTIC)
    assert AccessLevel.GRADIENT.satisfies(AccessLevel.EMBEDDING)


def test_names_are_sorted():
    reg = default_registry()
    assert reg.names() == sorted(reg.names())
    assert len(reg) == len(reg.names())
