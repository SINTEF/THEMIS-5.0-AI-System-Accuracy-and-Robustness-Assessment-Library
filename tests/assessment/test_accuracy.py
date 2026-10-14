"""Tests for the accuracy evaluator — the first-class accuracy path.

Three things carry the weight. It must be cheap: one model call, two with a baseline, and
nothing perturbed. Its verdict must be three-valued, so an assessment with no criteria
cannot read as a pass. And the baseline comparison must be sign-corrected, because for an
error metric "better" is a smaller number and a raw subtraction gets that backwards.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from themis_assess.assessment.accuracy import AccuracyEvaluator
from themis_assess.assessment.criteria import (
    AcceptanceCriteria,
    Requirement,
    themis_50_criteria,
)
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.subgroups import SubgroupSpec
from themis_assess.core.types import TaskType
from themis_assess.handlers.base import BaseHandler, Prediction
from themis_assess.handlers.baselines import MajorityClassHandler, PriorProbabilityHandler
from themis_assess.handlers.sklearn import CallableHandler, SklearnHandler


@pytest.fixture(scope="module")
def problem():
    """Three ordinal classes, harder at the top of the age range, with a site label."""
    rng = np.random.default_rng(0)
    n = 900
    age = rng.uniform(20, 85, n)
    severity = rng.normal(size=n)
    site = rng.choice(["oslo", "plovdiv", "valencia"], n, p=[0.5, 0.3, 0.2])
    logit = 0.06 * (age - 50) + 1.4 * severity + rng.normal(scale=0.9, size=n)
    y = np.digitize(logit, [-1.0, 2.0])
    X = np.column_stack([age, severity, rng.normal(size=n)])

    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X[:600], y[:600])
    data = ArrayAdapter(
        X[600:], feature_names=["age", "severity", "noise"], categorical_features=[]
    )
    return {
        "data": data,
        "actual": y[600:],
        "train_labels": y[:600],
        "site": site[600:],
        "handler": SklearnHandler(model, name="random_forest"),
    }


@pytest.fixture
def groups(problem):
    site = problem["site"]
    return [
        SubgroupSpec(name, (lambda name: lambda d: site == name)(name))
        for name in ("oslo", "plovdiv", "valencia")
    ]


# --------------------------------------------------------------------------
# cost and planning
# --------------------------------------------------------------------------


def test_the_assessment_costs_one_model_call(problem):
    """The point of a separate evaluator: cheap enough to run on every commit."""
    calls = []

    class Counting(BaseHandler):
        def predict(self, d):
            calls.append(d.n_entities)
            return Prediction(
                predicted=np.zeros(d.n_entities, dtype=int),
                probabilities=np.full((d.n_entities, 2), 0.5),
            )

    handler = Counting(None, name="counting")
    AccuracyEvaluator(handler, metrics=["accuracy"], per_class=False).run(
        problem["data"], np.zeros(problem["data"].n_entities, dtype=int)
    )
    assert len(calls) == 1


def test_a_baseline_costs_a_second_call(problem):
    plan = AccuracyEvaluator(
        problem["handler"], baseline=MajorityClassHandler(problem["train_labels"])
    ).plan(problem["data"])
    assert plan["n_model_calls"] == 2


def test_the_plan_says_what_will_be_skipped_and_why(problem):
    labels_only = CallableHandler(
        lambda X: (X[:, 1] > 0).astype(int), name="labels_only"
    )
    plan = AccuracyEvaluator(
        labels_only, metrics=["accuracy", "auc", "expected_calibration_error"]
    ).plan(problem["data"])
    assert plan["metrics"] == ("accuracy",)
    assert "PROBABILISTIC" in plan["skipped"]["auc"]


def test_the_plan_flags_a_criterion_with_no_matching_metric(problem):
    """A requirement whose metric is not computed is a gap in the evaluation, and it
    should look like one before the run rather than after."""
    labels_only = CallableHandler(
        lambda X: (X[:, 1] > 0).astype(int), name="labels_only"
    )
    plan = AccuracyEvaluator(
        labels_only, metrics=["accuracy"], criteria=themis_50_criteria()
    ).plan(problem["data"])
    assert "matthews_corrcoef" in plan["criteria_without_a_metric"]


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------


def test_the_overall_metrics_are_computed(problem):
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert result.overall.n_samples == 300
    assert 0.6 < result["accuracy"] < 0.9
    assert "expected_calibration_error" in result.overall.metrics


def test_no_criteria_means_no_verdict_not_a_pass(problem):
    """The three-valued outcome, at the level a caller actually touches."""
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert result.acceptance is None
    assert result.passed is None


def test_criteria_produce_a_verdict(problem):
    result = AccuracyEvaluator(
        problem["handler"], criteria=themis_50_criteria()
    ).run(problem["data"], problem["actual"])
    assert result.passed is False  # this model is around 0.74 against a bar of 0.8
    assert {v.metric for v in result.acceptance.failures} <= set(
        themis_50_criteria().metrics()
    )


def test_lenient_criteria_pass(problem):
    result = AccuracyEvaluator(
        problem["handler"],
        criteria=AcceptanceCriteria(
            [Requirement("accuracy", acceptable=0.6, rationale="triage queue")],
            source="test",
        ),
    ).run(problem["data"], problem["actual"])
    assert result.passed is True


def test_the_confusion_matrix_and_per_class_breakdown_are_produced(problem):
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert result.confusion is not None
    assert result.confusion.n_classes == 3
    assert set(result.per_class) == {0, 1, 2}
    assert result.confusion.support().sum() == 300


def test_per_class_can_be_switched_off(problem):
    result = AccuracyEvaluator(problem["handler"], per_class=False).run(
        problem["data"], problem["actual"]
    )
    assert result.confusion is None
    assert result.per_class == {}


def test_top_confusions_appear_beyond_two_classes(problem):
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert "top_confusions" in result.diagnostics
    assert result.diagnostics["top_confusions"][0]["true"] != (
        result.diagnostics["top_confusions"][0]["predicted"]
    )


# --------------------------------------------------------------------------
# baseline comparison
# --------------------------------------------------------------------------


def test_the_baseline_improvement_is_sign_corrected(problem):
    """For an error metric 'better' is a smaller number; a raw subtraction inverts it."""
    result = AccuracyEvaluator(
        problem["handler"], baseline=PriorProbabilityHandler(problem["train_labels"])
    ).run(problem["data"], problem["actual"])

    assert result.improvement_over_baseline("accuracy") > 0
    assert result.improvement_over_baseline("negative_log_likelihood") > 0
    assert result.improvement_over_baseline("brier_score") > 0
    # and the raw differences confirm the direction really did need correcting
    assert result["negative_log_likelihood"] < result.baseline["negative_log_likelihood"]


def test_a_metric_the_baseline_could_not_produce_gives_no_improvement(problem):
    """A majority-class baseline is black-box, so it has no ECE to compare against."""
    result = AccuracyEvaluator(
        problem["handler"], baseline=MajorityClassHandler(problem["train_labels"])
    ).run(problem["data"], problem["actual"])
    assert result.improvement_over_baseline("expected_calibration_error") is None
    assert result.improvement_over_baseline("accuracy") is not None


def test_no_baseline_gives_no_improvement(problem):
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert result.baseline is None
    assert result.improvement_over_baseline("accuracy") is None


# --------------------------------------------------------------------------
# subgroups
# --------------------------------------------------------------------------


def test_subgroups_are_broken_out(problem, groups):
    result = AccuracyEvaluator(problem["handler"], subgroups=groups).run(
        problem["data"], problem["actual"]
    )
    assert set(result.subgroups) == {"oslo", "plovdiv", "valencia"}
    assert sum(r.n_samples for r in result.subgroups.values()) == 300


def test_the_disparity_metrics_are_computed_not_left_to_the_caller(problem, groups):
    """The survey's rule is that an aggregate must be accompanied by them; a result that
    made it optional would make the rule unenforceable."""
    result = AccuracyEvaluator(problem["handler"], subgroups=groups).run(
        problem["data"], problem["actual"]
    )
    accuracy = result.disparity["accuracy"]
    assert "accuracy_gap" in accuracy
    assert "worst_group_accuracy" in accuracy
    assert accuracy["accuracy_gap"] >= 0


def test_disparity_uses_the_right_family_for_an_error_metric(problem, groups):
    """worst_group_accuracy is meaningless for an error; group_disparity is its analogue."""
    result = AccuracyEvaluator(problem["handler"], subgroups=groups).run(
        problem["data"], problem["actual"]
    )
    error_side = result.disparity["negative_log_likelihood"]
    assert "group_disparity" in error_side
    assert "equal_weight_risk" in error_side
    assert "worst_group_accuracy" not in error_side


def test_the_worst_subgroup_is_identifiable(problem, groups):
    result = AccuracyEvaluator(problem["handler"], subgroups=groups).run(
        problem["data"], problem["actual"]
    )
    worst = result.worst_subgroup("accuracy")
    assert worst in result.subgroups
    assert result.subgroups[worst]["accuracy"] == min(
        r["accuracy"] for r in result.subgroups.values()
    )


def test_the_worst_subgroup_respects_direction(problem, groups):
    result = AccuracyEvaluator(problem["handler"], subgroups=groups).run(
        problem["data"], problem["actual"]
    )
    worst_error = result.worst_subgroup(
        "negative_log_likelihood", higher_is_better=False
    )
    assert result.subgroups[worst_error]["negative_log_likelihood"] == max(
        r["negative_log_likelihood"] for r in result.subgroups.values()
    )


def test_no_subgroups_gives_no_disparity(problem):
    result = AccuracyEvaluator(problem["handler"]).run(problem["data"], problem["actual"])
    assert result.subgroups == {}
    assert result.disparity == {}


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def test_outlier_diagnostics_are_opt_in(problem):
    """They fit three detectors, and that cost should be opted into."""
    without = AccuracyEvaluator(problem["handler"]).run(
        problem["data"], problem["actual"]
    )
    assert "outliers" not in without.diagnostics

    with_them = AccuracyEvaluator(problem["handler"]).run(
        problem["data"], problem["actual"], diagnose_outliers=True
    )
    assert with_them.diagnostics["outliers"].n_outliers > 0


def test_feature_bin_diagnostics_name_their_feature(problem):
    result = AccuracyEvaluator(problem["handler"]).run(
        problem["data"], problem["actual"], diagnose_features=["age", "severity"]
    )
    bins = result.diagnostics["feature_bins"]
    assert set(bins) == {"age", "severity"}
    assert bins["age"]["error_rate"].shape == bins["age"]["count"].shape


def test_an_unknown_diagnostic_feature_is_rejected(problem):
    with pytest.raises(ValueError, match="not in the data"):
        AccuracyEvaluator(problem["handler"]).run(
            problem["data"], problem["actual"], diagnose_features=["height"]
        )


def test_the_ambiguity_flag_reports_whether_it_helps(problem):
    """If flagged and unflagged error rates match, the model's uncertainty carries no
    information and a review queue built on it sorts at random."""
    result = AccuracyEvaluator(problem["handler"]).run(
        problem["data"], problem["actual"], diagnose_ambiguity=True
    )
    ambiguity = result.diagnostics["ambiguity"]
    assert ambiguity["flagged_fraction"] == pytest.approx(0.2, abs=0.02)
    assert ambiguity["error_rate_flagged"] is not None
    assert ambiguity["error_rate_unflagged"] is not None


def test_ambiguity_needs_probabilities(problem):
    labels_only = CallableHandler(
        lambda X: (X[:, 1] > 0).astype(int), name="labels_only"
    )
    with pytest.raises(ValueError, match="Ambiguity needs probabilities"):
        AccuracyEvaluator(labels_only, metrics=["accuracy"]).run(
            problem["data"], problem["actual"], diagnose_ambiguity=True
        )


def test_the_diagnostics_are_classification_only():
    """They compare predicted labels against true labels. Forecasting has temporal.py."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -1.0])
    data = ArrayAdapter(X, feature_names=["a", "b"], categorical_features=[])
    handler = SklearnHandler(
        LinearRegression().fit(X, y), task=TaskType.FORECASTING, name="ols"
    )
    with pytest.raises(ValueError, match="classification-only"):
        AccuracyEvaluator(handler, per_class=False).run(
            data, y, diagnose_outliers=True
        )


# --------------------------------------------------------------------------
# forecasting, and the summary
# --------------------------------------------------------------------------


def test_a_forecasting_model_gets_the_forecasting_metrics():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.5, -0.5, 0.2]) + rng.normal(scale=0.2, size=200)
    data = ArrayAdapter(X, feature_names=["a", "b", "c"], categorical_features=[])
    handler = SklearnHandler(
        LinearRegression().fit(X, y), task=TaskType.FORECASTING, name="ols"
    )
    result = AccuracyEvaluator(
        handler,
        criteria=AcceptanceCriteria(
            [Requirement("mae", acceptable=0.5, rationale="ops budget")], source="test"
        ),
    ).run(data, y)
    assert "mae" in result.overall.metrics
    assert "accuracy" not in result.overall.metrics
    assert result.confusion is None  # not a classifier
    assert result.passed is True


def test_the_summary_states_the_outcome_and_the_failures(problem, groups):
    result = AccuracyEvaluator(
        problem["handler"], criteria=themis_50_criteria(), subgroups=groups
    ).run(problem["data"], problem["actual"])
    summary = result.summary()
    assert "random_forest" in summary
    assert "acceptance: FAIL" in summary
    assert "subgroup gap" in summary


def test_an_undefined_metric_is_named_in_the_summary(problem):
    """A single-class dataset makes AUC undefined; the summary must say so rather than
    quietly showing one metric fewer."""
    n = problem["data"].n_entities
    single_class = np.zeros(n, dtype=int)
    handler = CallableHandler(
        lambda X: np.zeros(X.shape[0], dtype=int),
        probability_fn=lambda X: np.tile([0.7, 0.3], (X.shape[0], 1)),
        name="constant",
    )
    result = AccuracyEvaluator(handler, per_class=False).run(problem["data"], single_class)
    assert "auc" in result.overall.metadata["undefined"]
    assert "undefined" in result.summary()
