"""Tests for acceptance criteria — the accuracy-side counterpart of the robustness tolerance.

The design commitment being tested is that a threshold is a *requirement*, not a property
of a metric. So: the registry supplies the direction and the caller never does; a metric
with no requirement is reported rather than dropped; and an assessment with no criteria at
all reports ``None``, not a pass.
"""

from __future__ import annotations

import pytest

from themis_assess.assessment.criteria import (
    AcceptanceCriteria,
    Requirement,
    themis_50_criteria,
)
from themis_assess.core.results import EvaluationResult


# --------------------------------------------------------------------------
# Requirement
# --------------------------------------------------------------------------


def test_a_requirement_needs_something_to_check():
    with pytest.raises(ValueError, match="neither an acceptable value nor a tolerance"):
        Requirement("accuracy")


def test_a_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        Requirement("mean_signed_deviation", tolerance=-0.1)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_a_plain_mapping_is_accepted():
    """What a user reaches for first, and what all four repos used."""
    criteria = AcceptanceCriteria({"accuracy": 0.8, "f1": 0.7})
    assert criteria.metrics() == ["accuracy", "f1"]
    assert criteria.get("accuracy").acceptable == 0.8


def test_empty_criteria_are_rejected():
    """An empty criteria set would report every assessment as passing."""
    with pytest.raises(ValueError, match="at least one requirement"):
        AcceptanceCriteria({})


def test_an_unknown_metric_is_rejected_at_construction():
    with pytest.raises(KeyError, match="Unknown metric"):
        AcceptanceCriteria({"not_a_metric": 0.5})


def test_a_duplicate_requirement_is_rejected():
    with pytest.raises(ValueError, match="Duplicate requirement"):
        AcceptanceCriteria(
            [Requirement("accuracy", acceptable=0.8), Requirement("accuracy", acceptable=0.9)]
        )


def test_a_directionless_metric_needs_a_tolerance_not_a_value():
    """mean_signed_deviation is best at zero and worse either side, so 'at least X'
    cannot be checked against it."""
    with pytest.raises(ValueError, match="has no direction"):
        AcceptanceCriteria({"mean_signed_deviation": 0.1})
    criteria = AcceptanceCriteria([Requirement("mean_signed_deviation", tolerance=0.1)])
    assert criteria.get("mean_signed_deviation").tolerance == 0.1


def test_a_directional_metric_needs_a_value_not_a_tolerance():
    with pytest.raises(ValueError, match="needs acceptable=, not tolerance="):
        AcceptanceCriteria([Requirement("accuracy", tolerance=0.1)])


# --------------------------------------------------------------------------
# direction comes from the registry
# --------------------------------------------------------------------------


def test_a_higher_is_better_metric_passes_at_or_above():
    criteria = AcceptanceCriteria({"accuracy": 0.8})
    assert criteria.verdict("accuracy", 0.85).passed is True
    assert criteria.verdict("accuracy", 0.80).passed is True  # at the bar passes
    assert criteria.verdict("accuracy", 0.79).passed is False


def test_a_lower_is_better_metric_passes_at_or_below():
    """The caller never says which way round the metric runs — that is the whole point."""
    criteria = AcceptanceCriteria({"mae": 0.5})
    assert criteria.verdict("mae", 0.42).passed is True
    assert criteria.verdict("mae", 0.61).passed is False


def test_the_margin_is_signed_so_positive_is_always_good():
    """A table of margins can be read and sorted without checking each row's direction."""
    high = AcceptanceCriteria({"accuracy": 0.8}).verdict("accuracy", 0.85)
    low = AcceptanceCriteria({"mae": 0.5}).verdict("mae", 0.45)
    assert high.margin == pytest.approx(0.05)
    assert low.margin == pytest.approx(0.05)  # not -0.05


def test_a_tolerance_measures_distance_from_the_target():
    criteria = AcceptanceCriteria([Requirement("mean_signed_deviation", tolerance=0.1)])
    assert criteria.verdict("mean_signed_deviation", 0.05).passed is True
    assert criteria.verdict("mean_signed_deviation", -0.05).passed is True  # either side
    assert criteria.verdict("mean_signed_deviation", -0.25).passed is False
    assert criteria.verdict("mean_signed_deviation", -0.25).margin == pytest.approx(-0.15)


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def test_a_measured_metric_with_no_requirement_is_reported_not_dropped():
    """A table that silently omitted it would read as though everything shown was vetted."""
    report = AcceptanceCriteria({"accuracy": 0.8}).check({"accuracy": 0.9, "f1": 0.5})
    assert report.unchecked == ["f1"]
    assert report["f1"].passed is None
    assert report["f1"].value == 0.5


def test_a_requirement_whose_metric_was_not_measured_appears_in_the_provenance():
    """It is not a pass and not a model failure — it is a gap in the evaluation."""
    report = AcceptanceCriteria({"accuracy": 0.8, "recall": 0.8}).check({"accuracy": 0.9})
    assert report.provenance["not_measured"] == ["recall"]
    assert report.passed is True  # nothing checked has failed


def test_the_overall_verdict_is_three_valued():
    """No criteria is not the same as passing, which is why this is not a bool."""
    from themis_assess.assessment.criteria import AcceptanceReport

    assert AcceptanceReport(verdicts=()).passed is None
    checked = AcceptanceCriteria({"accuracy": 0.8}).check({"accuracy": 0.9})
    assert checked.passed is True
    assert AcceptanceCriteria({"accuracy": 0.8}).check({"accuracy": 0.7}).passed is False


def test_only_critical_failures_fail_the_assessment():
    """Not everything is a gate: a team may require accuracy and merely monitor F1."""
    criteria = AcceptanceCriteria(
        [
            Requirement("accuracy", acceptable=0.8, critical=True),
            Requirement("f1", acceptable=0.9, critical=False),
        ]
    )
    report = criteria.check({"accuracy": 0.85, "f1": 0.5})
    assert report.passed is True
    assert [v.metric for v in report.failures] == ["f1"]


def test_a_report_accepts_an_evaluation_result():
    result = EvaluationResult(metrics={"accuracy": 0.9}, n_samples=100, label="clean")
    report = AcceptanceCriteria({"accuracy": 0.8}).check(result)
    assert report.label == "clean"
    assert report.passed is True


def test_the_provenance_records_which_thresholds_have_no_rationale():
    """A threshold with no rationale is indistinguishable from one somebody copied."""
    criteria = AcceptanceCriteria(
        [
            Requirement("accuracy", acceptable=0.8, rationale="clinical sign-off 2026-03"),
            Requirement("f1", acceptable=0.7),
        ],
        source="requirements doc v2",
    )
    provenance = criteria.provenance()
    assert provenance["source"] == "requirements doc v2"
    assert provenance["without_rationale"] == ["f1"]


def test_the_summary_states_the_overall_outcome():
    summary = AcceptanceCriteria({"accuracy": 0.8}).check({"accuracy": 0.7}).summary()
    assert "FAIL" in summary
    assert "overall: FAIL" in summary


def test_the_table_carries_the_verdict():
    """The difference from the four repos' get_metrics_table: the verdict is in it."""
    frame = AcceptanceCriteria({"accuracy": 0.8}).check(
        {"accuracy": 0.9, "f1": 0.6}
    ).to_frame()
    assert list(frame.columns) == [
        "value",
        "acceptable",
        "passed",
        "margin",
        "critical",
        "rationale",
    ]
    assert frame.loc["accuracy", "passed"] is True
    assert frame.loc["f1", "passed"] is None


def test_an_unknown_metric_in_a_report_lookup_lists_what_is_present():
    report = AcceptanceCriteria({"accuracy": 0.8}).check({"accuracy": 0.9})
    with pytest.raises(KeyError, match="Present:"):
        report["recall"]


# --------------------------------------------------------------------------
# the inherited example
# --------------------------------------------------------------------------


def test_the_example_criteria_match_the_repos_values():
    criteria = themis_50_criteria()
    assert criteria.get("accuracy").acceptable == 0.8
    assert criteria.get("balanced_accuracy").acceptable == 0.8
    assert criteria.get("precision").acceptable == 0.8
    assert criteria.get("recall").acceptable == 0.8
    assert criteria.get("matthews_corrcoef").acceptable == 0.6


def test_the_example_criteria_omit_f1_as_the_repos_did():
    """Repo C listed f1 in its ranges but set no acceptable value, so none is invented."""
    assert "f1" not in themis_50_criteria()


def test_the_example_criteria_say_where_they_came_from():
    """They must not be mistakable for derived thresholds."""
    criteria = themis_50_criteria()
    assert "no recorded derivation" in criteria.source
    assert criteria.provenance()["without_rationale"] == []
