"""Tests for the findings layer.

The properties that matter are not "a finding was produced" but:

* every finding names the numbers it rests on, so a reader can disagree with it;
* the absence of criteria produces a finding rather than silence, because silence reads
  as approval;
* the subgroup threshold is scale-free — the defect that a raw 0.05 gap threshold is
  meaningless on an unbounded metric;
* findings serialise to JSON, since a summarising model is the intended consumer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from themis_assess.report.findings import (
    Finding,
    _gap_fraction,
    findings_from_accuracy,
    findings_from_groups,
    findings_from_robustness,
    rank_findings,
)


def rules(findings):
    return [f.rule for f in findings]


# --------------------------------------------------------------------------
# the record itself
# --------------------------------------------------------------------------


def test_a_finding_serialises_with_its_evidence():
    finding = Finding(
        severity="serious",
        rule="test.rule",
        headline="something",
        detail="because of numbers",
        evidence={"value": np.float64(0.5), "curve": np.arange(3)},
    )
    payload = finding.as_dict()
    assert json.loads(json.dumps(payload))["evidence"] == {
        "value": 0.5,
        "curve": [0, 1, 2],
    }
    assert set(payload) == {
        "severity",
        "rule",
        "area",
        "headline",
        "detail",
        "evidence",
        "suggestion",
    }


def test_ranking_is_by_severity_and_stable_within_it():
    made = [
        Finding("info", "a", "", ""),
        Finding("critical", "b", "", ""),
        Finding("warning", "c", "", ""),
        Finding("critical", "d", "", ""),
    ]
    assert rules(rank_findings(made)) == ["b", "d", "c", "a"]


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


def test_a_missed_requirement_is_critical_and_carries_its_rationale(accuracy):
    found = findings_from_accuracy(accuracy)
    failures = [f for f in found if f.rule == "accuracy.criterion_failed"]
    assert failures, rules(found)
    failure = failures[0]
    assert failure.severity == "critical"
    assert failure.evidence["metric"] == "accuracy"
    assert failure.evidence["rationale"] == "deliberately out of reach"
    # the value in the finding is the value in the result, not a rounded restatement
    assert failure.evidence["value"] == accuracy["accuracy"]


def test_a_requirement_on_an_uncomputed_metric_is_a_gap_not_a_failure(accuracy):
    """auprc has a threshold and was not measured. That cannot read as a pass."""
    found = findings_from_accuracy(accuracy)
    gaps = [f for f in found if f.rule == "accuracy.requirement_not_measured"]
    assert [f.evidence["metric"] for f in gaps] == ["auprc"]
    assert gaps[0].severity == "warning"


def test_no_criteria_produces_a_finding(accuracy_bare):
    """Silence would read as approval, which is the failure survey §8 is about."""
    found = findings_from_accuracy(accuracy_bare)
    assert "accuracy.no_criteria" in rules(found)


def test_the_worst_subgroup_is_named_once_not_once_per_metric(accuracy):
    """Ten metrics showing the same gap is one fact, and belongs in one finding."""
    found = [f for f in findings_from_accuracy(accuracy) if f.rule == "accuracy.subgroup_gap"]
    subgroups = [f.evidence["subgroup"] for f in found]
    assert len(subgroups) == len(set(subgroups))
    worst = max(found, key=lambda f: len(f.evidence["metrics"]))
    assert worst.evidence["subgroup"] == "oslo"
    assert worst.evidence["n_samples"] == accuracy.subgroups["oslo"].n_samples
    # and the per-metric detail survives the grouping
    assert worst.evidence["metrics"]["accuracy"]["in_subgroup"] == (
        accuracy.subgroups["oslo"].metrics["accuracy"]
    )


def test_the_subgroup_finding_reports_the_direction_the_metric_defines(accuracy):
    """oslo is worse on accuracy (lower) and worse on log-likelihood (higher).

    A finding that used one direction for both would name the wrong subgroup for half
    the metrics, which is the sign error this library has had to fix repeatedly.
    """
    oslo = accuracy.subgroups["oslo"].metrics
    plovdiv = accuracy.subgroups["plovdiv"].metrics
    assert oslo["accuracy"] < plovdiv["accuracy"]
    assert oslo["negative_log_likelihood"] > plovdiv["negative_log_likelihood"]

    found = [f for f in findings_from_accuracy(accuracy) if f.rule == "accuracy.subgroup_gap"]
    by_subgroup = {f.evidence["subgroup"]: f.evidence["metrics"] for f in found}
    assert "accuracy" in by_subgroup["oslo"]
    assert "negative_log_likelihood" in by_subgroup["oslo"]


# --------------------------------------------------------------------------
# the gap fraction — the scale-free threshold
# --------------------------------------------------------------------------


def test_a_bounded_gap_is_a_fraction_of_the_metric_range():
    """Matthews' correlation runs -1 to 1, so a 0.15 gap is 7.5% of it, not 15%."""
    assert _gap_fraction("accuracy", 0.15, 0.8) == pytest.approx(0.15)
    assert _gap_fraction("matthews_corrcoef", 0.15, 0.7) == pytest.approx(0.075)


def test_an_unbounded_gap_is_relative_to_the_aggregate():
    """A 0.164 spread in log-likelihood means nothing until it is 37% of 0.44."""
    assert _gap_fraction("negative_log_likelihood", 0.164, 0.444) == pytest.approx(
        0.164 / 0.444, rel=1e-9
    )


def test_an_unbounded_gap_with_no_scale_declines_to_report():
    assert _gap_fraction("negative_log_likelihood", 0.1, 0.0) == pytest.approx(0.1)
    assert _gap_fraction("mae", 0.1, 0.0) is None
    assert _gap_fraction("not_a_registered_metric", 0.1, None) is None


def test_the_threshold_is_applied_to_the_fraction_not_the_raw_gap(accuracy):
    """The regression this guards: an absolute 0.05 threshold on every metric alike.

    With that threshold, an unbounded metric fires on any gap at all while a metric on a
    two-wide range needs twice the gap to fire. Every reported gap here clears the
    threshold *as a fraction*, and at least one of them would have been classified
    differently by the raw gap.
    """
    found = [f for f in findings_from_accuracy(accuracy) if f.rule == "accuracy.subgroup_gap"]
    reported = {m: e for f in found for m, e in f.evidence["metrics"].items()}
    assert reported
    assert all(entry["fraction"] >= 0.05 for entry in reported.values())
    assert any(entry["fraction"] != entry["gap"] for entry in reported.values())


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_an_exceeded_tolerance_names_the_perturbation_that_broke_it(robustness):
    found = findings_from_robustness(robustness)
    failures = [f for f in found if f.rule == "robustness.tolerance_exceeded"]
    assert failures, rules(found)
    assert failures[0].severity == "critical"
    assert failures[0].evidence["worst_at"]
    assert failures[0].evidence["basis"] == "worst_cell"


def test_no_tolerance_is_an_incomplete_claim_not_a_pass(robustness_bare):
    found = findings_from_robustness(robustness_bare)
    assert "robustness.no_tolerance" in rules(found)
    assert robustness_bare.passed is None


def test_the_worst_cell_is_reported_against_the_clean_value(robustness):
    found = [f for f in findings_from_robustness(robustness) if f.rule == "robustness.worst_cell"]
    assert found
    evidence = found[0].evidence
    perturbation, severity, value = robustness.worst_cell()
    assert evidence["perturbation"] == perturbation
    assert evidence["severity"] == severity
    assert evidence["worst"] == value
    assert evidence["clean"] == robustness.clean
    assert evidence["retained"] == pytest.approx(value / robustness.clean)


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------


def test_a_group_comparison_that_finds_nothing_says_so_without_claiming_equivalence():
    from themis_assess.metrics.accuracy.significance import compare_groups

    rng = np.random.default_rng(1)
    values = rng.normal(size=90)
    groups = np.tile(["a", "b", "c"], 30)
    found = findings_from_groups(compare_groups(values, groups, metric="accuracy"))
    assert rules(found) == ["groups.no_significant_difference"]
    assert "equivalence" in found[0].suggestion


def test_a_real_group_difference_is_serious_and_names_the_pairs():
    from themis_assess.metrics.accuracy.significance import compare_groups

    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(0, 1, 40), rng.normal(3, 1, 40)])
    groups = np.array(["a"] * 40 + ["b"] * 40)
    found = findings_from_groups(compare_groups(values, groups, metric="accuracy"))
    difference = [f for f in found if f.rule == "groups.significant_difference"][0]
    assert difference.severity == "serious"
    assert difference.evidence["significant_pairs"]
    assert difference.evidence["p_value"] < 0.05
