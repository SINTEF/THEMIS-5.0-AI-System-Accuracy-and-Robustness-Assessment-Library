"""Tests for turning a sweep into robustness scores.

The scoring layer's job is applying the right survey equation to the right numbers, and
its main risk is applying a *plausible* equation to the wrong ones — an error equation to
an accuracy, a baseline metric to a baseline that cannot support it. Both produce numbers
that look fine. Most of these tests are about refusing them.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from themis_assess.assessment.robustness import RobustnessEvaluator
from themis_assess.assessment.scoring import SUPPORTED_METRICS, score, score_all
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.base import BaseHandler, Prediction
from themis_assess.handlers.baselines import PriorProbabilityHandler
from themis_assess.handlers.sklearn import SklearnHandler
from themis_assess.core.types import AccessLevel, TaskType


@pytest.fixture(scope="module")
def sweeps():
    """One sweep with a constant baseline, one with a baseline that degrades."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    data = ArrayAdapter(
        X, feature_names=[f"f{i}" for i in range(4)], categorical_features=[]
    )
    handler = SklearnHandler(LogisticRegression().fit(X, y), name="logreg")

    class TwoFeature(BaseHandler):
        """A weaker model that still reads its input, so it degrades under perturbation."""

        def predict(self, d):
            matrix = d.values().reshape(d.n_entities, -1)[:, :1]
            return self.check_prediction(
                Prediction(
                    predicted=self.model.predict(matrix),
                    probabilities=self.model.predict_proba(matrix),
                ),
                d,
            )

    weak = TwoFeature(
        LogisticRegression().fit(X[:, :1], y),
        name="one_feature",
        task=TaskType.CLASSIFICATION,
        access=AccessLevel.PROBABILISTIC,
    )

    common = {
        "perturbations": ["gaussian_noise", "uniform_bias"],
        "severities": (1, 3, 5),
        "n_variants": 3,
        "seed": 4,
    }
    return {
        "none": RobustnessEvaluator(handler, **common).run(data, y),
        "constant": RobustnessEvaluator(
            handler, baseline=PriorProbabilityHandler(y), **common
        ).run(data, y),
        "degrading": RobustnessEvaluator(handler, baseline=weak, **common).run(data, y),
    }


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------


def test_an_error_equation_refuses_an_accuracy(sweeps):
    """CE over accuracies still produces a number, and that number is upside down."""
    with pytest.raises(ValueError, match="defined over an error"):
        score(sweeps["degrading"], "corruption_error", "accuracy")


def test_an_accuracy_equation_refuses_an_error(sweeps):
    with pytest.raises(ValueError, match="defined over an accuracy"):
        score(sweeps["none"], "relative_robustness", "brier_score")


def test_a_metric_with_no_direction_is_refused(sweeps):
    """A ratio of two quantities whose best value is zero means nothing."""
    sweep = sweeps["none"]
    if "predictive_entropy" in sweep.clean.metrics:  # pragma: no cover - defensive
        with pytest.raises(ValueError, match="no direction"):
            score(sweep, "relative_robustness", "predictive_entropy")


def test_score_all_returns_only_the_matching_family(sweeps):
    accuracy_side = score_all(sweeps["degrading"], "accuracy")
    error_side = score_all(sweeps["degrading"], "negative_log_likelihood")
    assert set(accuracy_side) == {
        "relative_robustness",
        "total_relative_robustness",
        "absolute_relative_robustness",
    }
    assert "corruption_error" in error_side
    assert "relative_robustness" not in error_side


# --------------------------------------------------------------------------
# the baseline requirement
# --------------------------------------------------------------------------


def test_a_baseline_metric_without_a_baseline_says_so(sweeps):
    with pytest.raises(ValueError, match="defined relative to a baseline"):
        score(sweeps["none"], "mean_corruption_error", "negative_log_likelihood")


def test_score_all_skips_baseline_metrics_when_there_is_no_baseline(sweeps):
    names = set(score_all(sweeps["none"], "negative_log_likelihood"))
    assert "corruption_error" not in names
    assert "error_inflation" in names


def test_a_constant_baseline_can_serve_mce_but_not_mrce(sweeps):
    """CE needs only the baseline's level; RCE needs it to degrade. A prior-probability
    baseline never degrades, so its excess error is exactly zero."""
    sweep = sweeps["constant"]
    assert score(sweep, "mean_corruption_error", "negative_log_likelihood").aggregate > 0

    with pytest.raises(ValueError, match="unaffected by"):
        score(sweep, "mean_relative_corruption_error", "negative_log_likelihood")


def test_a_degrading_baseline_supports_the_whole_family(sweeps):
    results = score_all(sweeps["degrading"], "negative_log_likelihood")
    for name in ("mean_corruption_error", "mean_relative_corruption_error",
                 "mean_relative_error_inflation"):
        assert np.isfinite(results[name].aggregate)


# --------------------------------------------------------------------------
# the numbers themselves
# --------------------------------------------------------------------------


def test_relative_robustness_is_the_retained_fraction(sweeps):
    sweep = sweeps["none"]
    result = score(sweep, "relative_robustness", "accuracy")
    for name, value in result.scores.items():
        curve = sweep.curve(name, "accuracy")
        expected = (sum(curve.values()) / len(curve)) / sweep.clean["accuracy"]
        assert value == pytest.approx(expected)


def test_total_relative_robustness_sums_over_the_perturbation_set(sweeps):
    """Eq. 12 is a sum, so its magnitude grows with the set — which is why the set is
    recorded on the result."""
    result = score(sweeps["none"], "total_relative_robustness", "accuracy")
    assert result.aggregate == pytest.approx(sum(result.scores.values()))
    assert set(result.perturbation_set) == {"gaussian_noise", "uniform_bias"}


def test_error_inflation_exceeds_one_for_a_degraded_model(sweeps):
    result = score(sweeps["none"], "error_inflation", "negative_log_likelihood")
    assert result.aggregate > 1.0


def test_the_result_keeps_everything_the_report_needs(sweeps):
    result = score(
        sweeps["degrading"], "mean_corruption_error", "negative_log_likelihood"
    )
    assert result.clean == sweeps["degrading"].clean["negative_log_likelihood"]
    assert result.baseline_clean is not None
    assert result.accuracy_metric == "negative_log_likelihood"
    assert set(result.per_perturbation) == {"gaussian_noise", "uniform_bias"}
    assert result.severities["gaussian_noise"] == [1, 3, 5]


def test_a_claim_is_incomplete_without_a_tolerance(sweeps):
    """Survey §8: the tolerance is the one element no evaluation instrument supplies."""
    without = score(sweeps["none"], "relative_robustness", "accuracy")
    with_tolerance = score(
        sweeps["none"], "relative_robustness", "accuracy", tolerance=0.9
    )
    assert not without.is_complete_claim
    assert with_tolerance.is_complete_claim


def test_scoring_does_not_re_run_the_model(sweeps):
    """Six robustness metrics over one sweep must cost six pieces of arithmetic."""
    results = score_all(sweeps["degrading"], "negative_log_likelihood")
    assert len(results) >= 6
    assert all(r.n_samples == sweeps["degrading"].clean.n_samples for r in results.values())


def test_an_unknown_robustness_metric_lists_the_supported_ones(sweeps):
    with pytest.raises(ValueError, match="Supported:"):
        score(sweeps["none"], "not_a_metric", "accuracy")


def test_an_uncomputed_accuracy_metric_says_what_is_available(sweeps):
    with pytest.raises(KeyError, match="Available:"):
        score(sweeps["none"], "relative_robustness", "mae")


def test_every_supported_metric_is_reachable(sweeps):
    """Nothing in SUPPORTED_METRICS may be unreachable through score().

    A name listed as supported with no dispatch branch would fall through to the next
    branch and return that metric's number under this one's name — which is how
    ``relative_error_inflation`` behaved before this test existed.
    """
    reached = set()
    for accuracy_metric in ("accuracy", "negative_log_likelihood"):
        reached |= set(score_all(sweeps["degrading"], accuracy_metric))
    assert reached == SUPPORTED_METRICS


def test_the_per_cell_error_inflation_is_not_offered_as_a_sweep_score(sweeps):
    """Eq. 17 is per (perturbation, severity); only its Eq. 18 aggregate fits here."""
    assert "relative_error_inflation" not in SUPPORTED_METRICS
    with pytest.raises(ValueError, match="Supported:"):
        score(sweeps["degrading"], "relative_error_inflation", "negative_log_likelihood")
