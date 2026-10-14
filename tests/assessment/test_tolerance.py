"""Tests for robustness tolerances and the symmetric one-call assessment.

The design commitment is that both halves of the library return the same shape of answer:
a rich result with a three-valued verdict whose ``None`` means "nothing was checked".
Survey §8 is explicit that neither an accuracy threshold nor a robustness tolerance can
come from the instrument, so neither may be defaulted.

The tolerance side has one wrinkle the accuracy side does not: a tolerance is checked
against a *shape*, and the three defensible readings of "acceptable degradation" disagree.
Several tests here construct a case where they disagree and check that each reports its
own answer rather than the flattering one.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from themis_assess.assessment import scoring
from themis_assess.assessment.robustness import RobustnessEvaluator
from themis_assess.assessment.tolerance import (
    RobustnessCriteria,
    ToleranceRequirement,
)
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.baselines import PriorProbabilityHandler
from themis_assess.handlers.sklearn import SklearnHandler


@pytest.fixture(scope="module")
def assessed():
    """One sweep, scored, with a baseline — reused across the tolerance tests."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X[:400], y[:400])
    data = ArrayAdapter(
        X[400:], feature_names=[f"f{i}" for i in range(4)], categorical_features=[]
    )
    evaluator = RobustnessEvaluator(
        SklearnHandler(model, name="rf"),
        baseline=PriorProbabilityHandler(y[:400]),
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 3, 5),
        n_variants=3,
        seed=4,
    )
    return evaluator, data, y[400:]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_a_plain_mapping_is_accepted():
    criteria = RobustnessCriteria({"relative_robustness": 0.85})
    assert criteria.metrics() == ["relative_robustness"]
    assert criteria.get("relative_robustness").basis == "worst_perturbation"


def test_empty_criteria_are_rejected():
    with pytest.raises(ValueError, match="at least one tolerance"):
        RobustnessCriteria({})


def test_a_duplicate_tolerance_is_rejected():
    with pytest.raises(ValueError, match="Duplicate tolerance"):
        RobustnessCriteria(
            [
                ToleranceRequirement("relative_robustness", acceptable=0.8),
                ToleranceRequirement("relative_robustness", acceptable=0.9),
            ]
        )


def test_an_unknown_basis_is_rejected():
    """Each basis answers a different question, so there is no sensible fallback."""
    with pytest.raises(ValueError, match="basis must be one of"):
        ToleranceRequirement("relative_robustness", acceptable=0.8, basis="average")


def test_the_provenance_records_the_bases_and_missing_rationales():
    criteria = RobustnessCriteria(
        [
            ToleranceRequirement(
                "relative_robustness", acceptable=0.85, rationale="ops sign-off"
            ),
            ToleranceRequirement(
                "mean_corruption_error", acceptable=1.2, basis="aggregate"
            ),
        ],
        source="requirements v3",
    )
    provenance = criteria.provenance()
    assert provenance["source"] == "requirements v3"
    assert provenance["bases"]["mean_corruption_error"] == "aggregate"
    assert provenance["without_rationale"] == ["mean_corruption_error"]


# --------------------------------------------------------------------------
# the three bases
# --------------------------------------------------------------------------


def test_the_three_bases_disagree_and_each_reports_its_own(assessed):
    """The whole reason the basis is explicit: reporting only the flattering one is the
    failure mode this module exists to prevent."""
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "relative_robustness", "accuracy")

    values = {}
    for basis in ("aggregate", "worst_perturbation", "worst_cell"):
        criteria = RobustnessCriteria(
            [
                ToleranceRequirement(
                    "relative_robustness", acceptable=0.0, basis=basis
                )
            ]
        )
        values[basis] = criteria.verdict(result).value

    # The aggregate is Eq. 12's *sum* over two perturbations, so it is the largest;
    # the worst cell is at severity 5, so it is the smallest.
    assert values["aggregate"] > values["worst_perturbation"] > values["worst_cell"]


def test_the_worst_perturbation_basis_names_the_culprit(assessed):
    """A failed robustness claim is actionable only if you know what broke it."""
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "relative_robustness", "accuracy")
    criteria = RobustnessCriteria({"relative_robustness": 0.99})
    verdict = criteria.verdict(result)
    assert verdict.passed is False
    assert verdict.worst_at in ("gaussian_noise", "uniform_bias")
    assert verdict.worst_at in verdict.describe()


def test_the_worst_cell_basis_names_the_severity_too(assessed):
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "relative_robustness", "accuracy")
    criteria = RobustnessCriteria(
        [
            ToleranceRequirement(
                "relative_robustness", acceptable=0.5, basis="worst_cell"
            )
        ]
    )
    verdict = criteria.verdict(result)
    assert "@" in verdict.worst_at
    assert verdict.worst_at.endswith("@5")  # the sweep's severities end at 5


def test_the_aggregate_basis_needs_an_aggregate(assessed):
    """Refuses rather than silently falling back to a different reading."""
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "relative_robustness", "accuracy")

    from dataclasses import replace

    no_aggregate = replace(result, aggregate=None)
    criteria = RobustnessCriteria(
        [ToleranceRequirement("relative_robustness", acceptable=0.8, basis="aggregate")]
    )
    with pytest.raises(ValueError, match="no aggregate"):
        criteria.verdict(no_aggregate)


def test_the_worst_cell_basis_refuses_a_zero_clean_value(assessed):
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "relative_robustness", "accuracy")

    from dataclasses import replace

    criteria = RobustnessCriteria(
        [
            ToleranceRequirement(
                "relative_robustness", acceptable=0.8, basis="worst_cell"
            )
        ]
    )
    with pytest.raises(ValueError, match="clean .* is zero"):
        criteria.verdict(replace(result, clean=0.0))


# --------------------------------------------------------------------------
# direction and margins
# --------------------------------------------------------------------------


def test_the_margin_is_signed_so_positive_is_always_good(assessed):
    """As on the accuracy side: a table of margins is sortable without checking each
    row's direction."""
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)

    higher = scoring.score(sweep, "relative_robustness", "accuracy")
    lower = scoring.score(sweep, "mean_corruption_error", "negative_log_likelihood")

    generous = RobustnessCriteria(
        [
            ToleranceRequirement("relative_robustness", acceptable=0.0),
            ToleranceRequirement(
                "mean_corruption_error", acceptable=100.0, basis="aggregate"
            ),
        ]
    )
    assert generous.verdict(higher).margin > 0
    assert generous.verdict(lower).margin > 0  # not negative, though lower is better


def test_a_lower_is_better_robustness_metric_passes_below_its_bar(assessed):
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    result = scoring.score(sweep, "mean_corruption_error", "negative_log_likelihood")
    tight = RobustnessCriteria(
        [ToleranceRequirement("mean_corruption_error", acceptable=0.0, basis="aggregate")]
    )
    loose = RobustnessCriteria(
        [
            ToleranceRequirement(
                "mean_corruption_error", acceptable=1000.0, basis="aggregate"
            )
        ]
    )
    assert tight.verdict(result).passed is False
    assert loose.verdict(result).passed is True


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------


def test_a_tolerance_whose_metric_was_not_scored_is_reported_as_unchecked(assessed):
    """'overall: PASS' beside an unchecked critical tolerance is the most misleading
    thing this report could say."""
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data,
        actual,
        accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
        criteria=RobustnessCriteria(
            [
                ToleranceRequirement("relative_robustness", acceptable=0.1),
                ToleranceRequirement(
                    "mean_corruption_error", acceptable=1.2, basis="aggregate"
                ),
            ]
        ),
    )
    assert assessment.tolerance.unchecked == ["mean_corruption_error"]
    assert "NOT CHECKED" in assessment.summary()
    assert assessment.passed is True  # what *was* checked passed


def test_only_critical_failures_fail_the_assessment(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data,
        actual,
        accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
        criteria=RobustnessCriteria(
            [
                ToleranceRequirement(
                    "relative_robustness", acceptable=0.99, critical=False
                )
            ]
        ),
    )
    assert assessment.passed is True
    assert len(assessment.tolerance.failures) == 1


def test_the_table_carries_the_verdict_and_the_culprit(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data,
        actual,
        accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
        criteria=RobustnessCriteria({"relative_robustness": 0.99}),
    )
    frame = assessment.tolerance.to_frame()
    assert "worst_at" in frame.columns
    # bool(...) because a homogeneous column comes back as numpy.bool_, where the
    # accuracy side's table has None values mixed in and stays object dtype.
    assert bool(frame.loc["relative_robustness", "passed"]) is False
    assert frame.loc["relative_robustness", "worst_at"] in (
        "gaussian_noise",
        "uniform_bias",
    )


def test_an_unknown_metric_lookup_lists_what_is_present(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data, actual, accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
        criteria=RobustnessCriteria({"relative_robustness": 0.1}),
    )
    with pytest.raises(KeyError, match="Present:"):
        assessment.tolerance["mean_corruption_error"]


# --------------------------------------------------------------------------
# the symmetric assessment
# --------------------------------------------------------------------------


def test_assess_returns_the_sweep_the_scores_and_the_verdict(assessed):
    """One call, the same shape of answer as the accuracy side."""
    evaluator, data, actual = assessed
    assessment = evaluator.assess(data, actual, accuracy_metric="accuracy")
    assert assessment.sweep.perturbation_names == ["gaussian_noise", "uniform_bias"]
    assert "relative_robustness" in assessment.metric_names
    assert assessment.clean == assessment.sweep.clean["accuracy"]


def test_no_criteria_means_no_verdict_not_a_pass(assessed):
    """A robustness claim without a tolerance is incomplete under survey §8."""
    evaluator, data, actual = assessed
    assessment = evaluator.assess(data, actual, accuracy_metric="accuracy")
    assert assessment.tolerance is None
    assert assessment.passed is None
    assert "incomplete claim" in assessment.summary()


def test_the_sweep_stays_attached(assessed):
    """The survey's reporting rules need it: mCE may not be stated without mRCE, an
    aggregate not without the breakdown, one severity not without the curve."""
    evaluator, data, actual = assessed
    assessment = evaluator.assess(data, actual, accuracy_metric="accuracy")
    curves = assessment.curves()
    assert set(curves) == {"gaussian_noise", "uniform_bias"}
    assert sorted(curves["gaussian_noise"]) == [1, 3, 5]


def test_the_worst_cell_is_reported(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(data, actual, accuracy_metric="accuracy")
    perturbation, severity, value = assessment.worst_cell()
    assert perturbation in ("gaussian_noise", "uniform_bias")
    assert severity == 5
    assert value == min(
        v for curve in assessment.curves().values() for v in curve.values()
    )


def test_the_most_damaging_perturbation_is_identifiable(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data, actual, accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
    )
    worst = assessment.most_damaging_perturbation("relative_robustness")
    scores = assessment["relative_robustness"].scores
    assert scores[worst] == min(scores.values())


def test_a_defaulted_underlying_metric_says_so(assessed):
    """Two robustness scores over different underlying metrics are not comparable, so a
    default must not look like a choice someone made."""
    evaluator, data, actual = assessed
    defaulted = evaluator.assess(
        data, actual, robustness_metrics=["relative_robustness"]
    )
    chosen = evaluator.assess(
        data, actual, accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
    )
    assert defaulted.metadata["accuracy_metric_chosen_by"] == "default"
    assert chosen.metadata["accuracy_metric_chosen_by"] == "caller"
    assert "chosen by default" in defaulted.summary()
    assert "chosen by default" not in chosen.summary()


def test_the_default_prefers_an_error_metric_for_the_error_family(assessed):
    evaluator, data, actual = assessed
    error_side = evaluator.assess(data, actual, robustness_metrics=["error_inflation"])
    accuracy_side = evaluator.assess(
        data, actual, robustness_metrics=["relative_robustness"]
    )
    assert error_side.accuracy_metric == "negative_log_likelihood"
    assert accuracy_side.accuracy_metric == "accuracy"


def test_run_still_returns_the_unscored_sweep(assessed):
    """Kept for the sensitivity and attribution layers, which need the raw cells."""
    evaluator, data, actual = assessed
    sweep = evaluator.run(data, actual)
    assert not hasattr(sweep, "scored")
    assert sweep.curve("gaussian_noise", "accuracy")


def test_an_unscored_metric_lookup_lists_what_is_available(assessed):
    evaluator, data, actual = assessed
    assessment = evaluator.assess(
        data, actual, accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
    )
    with pytest.raises(KeyError, match="Available:"):
        assessment["mean_corruption_error"]
