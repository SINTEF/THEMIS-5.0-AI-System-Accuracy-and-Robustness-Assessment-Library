"""Tests for survey §8's ten reporting rules.

Each rule needs two tests and gets them: one payload that trips it and one that does not.
A rule that only ever fires is as useless as one that never does — a report layer whose
warnings are always present is a report layer nobody reads.

The payloads are built by hand rather than from assessments. That is the point of the
rules reading the serialised dictionary: a report assembled from loose numbers is checked
on exactly the same terms as one the evaluators built, and the tests can construct the
incomplete report that a real evaluator would never produce.
"""

from __future__ import annotations

from themis_assess.report.rules import check_report


def fired(payload) -> set[str]:
    return {f.rule for f in check_report(payload)}


def robustness(**overrides) -> dict:
    """A complete robustness section: every rule quiet unless an override breaks one."""
    section = {
        "accuracy_metric": "accuracy",
        "clean": 0.9,
        "perturbations": ["gaussian_noise", "uniform_bias"],
        "severities": [1, 2, 3],
        "curves": {"gaussian_noise": {"1": 0.88, "2": 0.83, "3": 0.7}},
        "plan": {"metrics": ["accuracy", "expected_calibration_error"]},
        "scored": {
            "relative_robustness": {
                "accuracy_metric": "accuracy",
                "perturbation_set": ["gaussian_noise", "uniform_bias"],
                "metadata": {},
            }
        },
    }
    section.update(overrides)
    return {"robustness": section}


def accuracy(**overrides) -> dict:
    section = {
        "overall": {"metrics": {"accuracy": 0.9}, "metadata": {}},
        "per_class": {0: {"recall": 0.9}, 1: {"recall": 0.9}},
    }
    section.update(overrides)
    return {"accuracy": section}


# --------------------------------------------------------------------------
# a complete report trips nothing — the control
# --------------------------------------------------------------------------


def test_a_complete_report_raises_no_issues():
    payload = {**accuracy(), **robustness()}
    assert fired(payload) == set()


def test_an_empty_report_raises_no_issues():
    """Nothing was claimed, so there is nothing to be incomplete about."""
    assert fired({}) == set()


# --------------------------------------------------------------------------
# 1 — relative without absolute
# --------------------------------------------------------------------------


def test_a_relative_metric_without_a_clean_value_is_serious():
    payload = robustness(clean=None)
    issues = check_report(payload)
    assert "reporting.relative_without_absolute" in {f.rule for f in issues}
    assert next(
        f for f in issues if f.rule == "reporting.relative_without_absolute"
    ).severity == "serious"


def test_a_relative_metric_with_a_clean_value_is_fine():
    assert "reporting.relative_without_absolute" not in fired(robustness())


def test_a_clean_value_of_zero_still_counts_as_reported():
    """Zero is a value. Treating a falsy clean value as missing would fire on a model
    with no clean error at all, which is the one case where the relative metric is
    genuinely undefined and needs a different complaint."""
    assert "reporting.relative_without_absolute" not in fired(robustness(clean=0.0))


# --------------------------------------------------------------------------
# 2 — mCE without mRCE
# --------------------------------------------------------------------------


def test_the_corruption_error_alone_asks_for_its_relative_form():
    payload = robustness(
        scored={
            "mean_corruption_error": {
                "perturbation_set": ["gaussian_noise"],
                "metadata": {},
            }
        }
    )
    assert "reporting.corruption_error_without_relative" in fired(payload)


def test_the_pair_together_is_fine():
    payload = robustness(
        scored={
            "mean_corruption_error": {"perturbation_set": ["a"], "metadata": {}},
            "mean_relative_corruption_error": {
                "perturbation_set": ["a"],
                "metadata": {},
            },
        }
    )
    assert "reporting.corruption_error_without_relative" not in fired(payload)


# --------------------------------------------------------------------------
# 3 — an aggregate over an unnamed set
# --------------------------------------------------------------------------


def test_a_set_aggregated_metric_must_name_its_set():
    payload = robustness(
        scored={"total_relative_robustness": {"perturbation_set": [], "metadata": {}}}
    )
    assert "reporting.aggregate_without_perturbation_set" in fired(payload)


def test_a_named_set_satisfies_the_rule():
    payload = robustness(
        scored={
            "total_relative_robustness": {
                "perturbation_set": ["gaussian_noise", "uniform_bias"],
                "metadata": {},
            }
        }
    )
    assert "reporting.aggregate_without_perturbation_set" not in fired(payload)


# --------------------------------------------------------------------------
# 4 — a composite standing alone
# --------------------------------------------------------------------------


def test_a_composite_reported_alone_is_serious():
    payload = robustness(
        scored={
            "composite_reliability_score": {
                "perturbation_set": ["a"],
                "metadata": {},
            }
        }
    )
    issues = {f.rule: f for f in check_report(payload)}
    assert "reporting.composite_without_components" in issues
    assert issues["reporting.composite_without_components"].severity == "serious"


def test_a_composite_beside_its_components_is_fine():
    payload = robustness(
        scored={
            "composite_reliability_score": {
                "perturbation_set": ["a"],
                "metadata": {},
            },
            "relative_robustness": {"perturbation_set": ["a"], "metadata": {}},
        }
    )
    assert "reporting.composite_without_components" not in fired(payload)


def test_a_composite_that_records_its_own_weights_is_fine():
    payload = robustness(
        scored={
            "unified_robustness_metric": {
                "perturbation_set": ["a"],
                "metadata": {
                    "components": {"accuracy": 0.9, "flip_rate": 0.1},
                    "weights": {"accuracy": 0.5, "flip_rate": 0.5},
                },
            }
        }
    )
    assert "reporting.composite_without_components" not in fired(payload)


# --------------------------------------------------------------------------
# 5 — one severity in place of a curve
# --------------------------------------------------------------------------


def test_a_single_severity_is_not_a_curve():
    payload = robustness(severities=[3], curves={"gaussian_noise": {"3": 0.7}})
    assert "reporting.single_severity" in fired(payload)


def test_a_swept_axis_is_a_curve():
    assert "reporting.single_severity" not in fired(robustness())


def test_a_curve_wider_than_the_declared_severities_counts():
    """The declared severities can be wrong where a report was assembled by hand; the
    curve is the evidence, so it wins."""
    payload = robustness(
        severities=[], curves={"gaussian_noise": {"1": 0.9, "2": 0.8, "3": 0.7}}
    )
    assert "reporting.single_severity" not in fired(payload)


# --------------------------------------------------------------------------
# 6 — an aggregate with no breakdown
# --------------------------------------------------------------------------


def test_accuracy_with_no_breakdown_asks_for_one():
    payload = {"accuracy": {"overall": {"metrics": {"accuracy": 0.9}, "metadata": {}}}}
    assert "reporting.aggregate_without_breakdown" in fired(payload)


def test_a_per_class_breakdown_satisfies_the_rule():
    assert "reporting.aggregate_without_breakdown" not in fired(accuracy())


def test_a_subgroup_breakdown_satisfies_the_rule():
    payload = {
        "accuracy": {
            "overall": {"metrics": {"accuracy": 0.9}, "metadata": {}},
            "subgroups": {"oslo": {"metrics": {"accuracy": 0.88}}},
        }
    }
    assert "reporting.aggregate_without_breakdown" not in fired(payload)


# --------------------------------------------------------------------------
# 7 — calibration measured only on clean data
# --------------------------------------------------------------------------


def test_clean_calibration_with_an_unswept_calibration_metric_fires():
    payload = {
        **accuracy(
            overall={
                "metrics": {"accuracy": 0.9, "expected_calibration_error": 0.05},
                "metadata": {},
            }
        ),
        **robustness(plan={"metrics": ["accuracy"]}),
    }
    assert "reporting.calibration_on_clean_only" in fired(payload)


def test_sweeping_the_calibration_metric_satisfies_the_rule():
    payload = {
        **accuracy(
            overall={
                "metrics": {"accuracy": 0.9, "expected_calibration_error": 0.05},
                "metadata": {},
            }
        ),
        **robustness(),
    }
    assert "reporting.calibration_on_clean_only" not in fired(payload)


def test_confidence_retention_answers_the_same_question():
    section = robustness(plan={"metrics": ["accuracy"]})
    section["robustness"]["scored"]["confidence_retention"] = {
        "perturbation_set": ["a"],
        "metadata": {},
    }
    payload = {
        **accuracy(
            overall={
                "metrics": {"accuracy": 0.9, "expected_calibration_error": 0.05},
                "metadata": {},
            }
        ),
        **section,
    }
    assert "reporting.calibration_on_clean_only" not in fired(payload)


def test_calibration_with_no_sweep_at_all_does_not_fire():
    """There is no shifted measurement to be missing if nothing was shifted."""
    payload = accuracy(
        overall={
            "metrics": {"accuracy": 0.9, "expected_calibration_error": 0.05},
            "metadata": {},
        }
    )
    assert "reporting.calibration_on_clean_only" not in fired(payload)


# --------------------------------------------------------------------------
# 8 — an operating-point metric with no operating point
# --------------------------------------------------------------------------


def test_an_operating_point_metric_must_name_its_threshold():
    payload = accuracy(
        overall={"metrics": {"accuracy": 0.9, "fpr_at_tpr": 0.12}, "metadata": {}}
    )
    assert "reporting.threshold_metric_without_threshold" in fired(payload)


def test_a_recorded_target_rate_satisfies_the_rule():
    payload = accuracy(
        overall={
            "metrics": {"accuracy": 0.9, "fpr_at_tpr": 0.12},
            "metadata": {"target_tpr": 0.95},
        }
    )
    assert "reporting.threshold_metric_without_threshold" not in fired(payload)


# --------------------------------------------------------------------------
# 9 — a distance presented as a robustness score
# --------------------------------------------------------------------------


def test_a_distance_among_the_robustness_scores_is_serious():
    payload = robustness(
        scored={
            "wasserstein_distance": {"perturbation_set": ["a"], "metadata": {}},
            "relative_robustness": {"perturbation_set": ["a"], "metadata": {}},
        }
    )
    issues = {f.rule: f for f in check_report(payload)}
    assert issues["reporting.distance_as_robustness"].severity == "serious"
    assert issues["reporting.distance_as_robustness"].evidence["metrics"] == [
        "wasserstein_distance"
    ]


def test_a_distance_in_its_own_section_is_fine():
    payload = {**robustness(), "distances": {"wasserstein_distance": 0.4}}
    assert "reporting.distance_as_robustness" not in fired(payload)


# --------------------------------------------------------------------------
# 10 — coverage without width
# --------------------------------------------------------------------------


def test_interval_coverage_alone_asks_for_the_width():
    payload = accuracy(
        overall={
            "metrics": {"mae": 1.2, "prediction_interval_coverage": 0.91},
            "metadata": {},
        }
    )
    assert "reporting.coverage_without_width" in fired(payload)


def test_coverage_with_width_is_fine():
    payload = accuracy(
        overall={
            "metrics": {
                "mae": 1.2,
                "prediction_interval_coverage": 0.91,
                "mean_prediction_interval_width": 3.4,
            },
            "metadata": {},
        }
    )
    assert "reporting.coverage_without_width" not in fired(payload)


def test_the_interval_score_also_satisfies_the_rule():
    payload = accuracy(
        overall={
            "metrics": {
                "mae": 1.2,
                "prediction_interval_coverage": 0.91,
                "interval_score": 2.2,
            },
            "metadata": {},
        }
    )
    assert "reporting.coverage_without_width" not in fired(payload)


# --------------------------------------------------------------------------
# the layer's own contract
# --------------------------------------------------------------------------


def test_every_issue_is_in_the_reporting_area_and_carries_a_suggestion():
    payload = {
        "accuracy": {"overall": {"metrics": {"accuracy": 0.9}, "metadata": {}}},
        "robustness": {
            "accuracy_metric": "accuracy",
            "clean": None,
            "severities": [3],
            "curves": {},
            "plan": {"metrics": ["accuracy"]},
            "scored": {
                "mean_corruption_error": {"perturbation_set": [], "metadata": {}}
            },
        },
    }
    issues = check_report(payload)
    assert len(issues) >= 4
    assert all(f.area == "reporting" for f in issues)
    assert all(f.suggestion for f in issues)
    assert all(f.rule.startswith("reporting.") for f in issues)


def test_the_rules_warn_and_never_raise():
    """A library that raised here would stop an honest exploratory run. §8 is about what
    gets published, which this code cannot observe."""
    for payload in ({}, {"accuracy": None}, {"robustness": "not a mapping"}):
        assert isinstance(check_report(payload), list)
