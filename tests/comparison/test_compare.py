"""Tests for the model comparison.

The tests here are mostly about what the comparison *refuses*. Producing a table of
numbers for three models is easy; the value is in not putting two incomparable numbers on
one axis, and every refusal below corresponds to a way the four repos could have drawn a
misleading chart.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.comparison import ModelComparison, compare_models


# --------------------------------------------------------------------------
# orientation and scale
# --------------------------------------------------------------------------


def test_an_error_metric_is_flipped_so_one_is_always_best():
    """The defect a naive polar chart has: on a raw axis the worse model is further out."""
    c = compare_models({"low_error": {"mae": 1.0}, "high_error": {"mae": 4.0}})
    assert c.raw["high_error"]["mae"] > c.raw["low_error"]["mae"]
    assert c.normalised["high_error"]["mae"] < c.normalised["low_error"]["mae"]
    assert c.best("mae") == "low_error"


def test_metrics_of_different_ranges_end_up_on_a_common_axis():
    c = compare_models(
        {"a": {"accuracy": 0.9, "mae": 0.5}, "b": {"accuracy": 0.6, "mae": 3.0}}
    )
    for model in c.models:
        for metric in c.metrics:
            assert 0.0 <= c.normalised[model][metric] <= 1.0


def test_the_raw_values_survive_normalisation():
    """A normalised score is not a number anyone can act on, so both are kept."""
    c = compare_models({"a": {"accuracy": 0.9}, "b": {"accuracy": 0.6}, })
    assert c.raw["a"]["accuracy"] == 0.9
    assert c.to_frame().loc["a", "accuracy"] == 0.9


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


def test_one_model_is_not_a_comparison():
    with pytest.raises(ValueError, match="at least two models"):
        compare_models({"only": {"accuracy": 0.9}})


def test_a_single_models_metrics_are_not_mistaken_for_several_models():
    """``{"accuracy": 0.9, "mae": 1.0}`` is one model, and reading it as two models named
    'accuracy' and 'mae' would produce a comparison of two scalars."""
    with pytest.raises(ValueError, match="one model's metrics"):
        compare_models({"accuracy": 0.9, "mae": 1.0})


def test_a_metric_one_model_lacks_is_dropped_with_the_reason():
    c = compare_models(
        {
            "probabilistic": {"accuracy": 0.9, "expected_calibration_error": 0.05},
            "black_box": {"accuracy": 0.88},
        }
    )
    assert c.metrics == ("accuracy",)
    assert "black_box" in c.dropped["expected_calibration_error"]


def test_asking_for_an_unavailable_metric_is_visible_not_silent():
    c = compare_models(
        {"a": {"accuracy": 0.9, "f1": 0.9}, "b": {"accuracy": 0.8}},
        metrics=["accuracy", "f1"],
    )
    assert c.metrics == ("accuracy",)
    assert "f1" in c.dropped


def test_an_unregistered_metric_cannot_share_an_axis():
    c = compare_models(
        {"a": {"accuracy": 0.9, "house_score": 3.0}, "b": {"accuracy": 0.8, "house_score": 4.0}}
    )
    assert c.metrics == ("accuracy",)
    assert "registry" in c.dropped["house_score"]


def test_nothing_in_common_raises_rather_than_drawing_an_empty_chart():
    with pytest.raises(ValueError, match="nothing to compare"):
        compare_models({"a": {"accuracy": 0.9}, "b": {"mae": 1.0}})


def test_a_non_finite_value_is_dropped():
    c = compare_models(
        {"a": {"accuracy": 0.9, "auc": float("nan")}, "b": {"accuracy": 0.8, "auc": 0.9}}
    )
    assert c.metrics == ("accuracy",)
    assert "finite" in c.dropped["auc"]


def test_two_reports_with_the_same_name_are_refused():
    class Fake:
        model_name = "same"
        accuracy = None
        robustness = None

    with pytest.raises(ValueError, match="both named"):
        compare_models([Fake(), Fake()])


# --------------------------------------------------------------------------
# robustness comparability — the mistake the repos were most exposed to
# --------------------------------------------------------------------------


def test_robustness_over_different_accuracy_metrics_is_not_one_axis(problem, other_model):
    """A relative_robustness over accuracy and one over balanced accuracy are different
    quantities under one name, and nothing in the number says which was used — the
    underlying metric is chosen by a default a caller need never look at."""
    from themis_assess.assessment.robustness import RobustnessEvaluator
    from themis_assess.report import build_report

    kwargs = dict(perturbations=["gaussian_noise"], severities=(1, 2))
    over_accuracy = RobustnessEvaluator(problem["handler"], **kwargs).assess(
        problem["data"], problem["actual"], accuracy_metric="accuracy"
    )
    over_balanced = RobustnessEvaluator(other_model, **kwargs).assess(
        problem["data"], problem["actual"], accuracy_metric="balanced_accuracy"
    )
    reports = {
        "over_accuracy": build_report(model_name="x", robustness=over_accuracy),
        "over_balanced": build_report(model_name="y", robustness=over_balanced),
    }
    # relative_robustness is present for both models and is still not one axis, so with
    # it as the only requested metric there is nothing left to compare — which is the
    # right outcome, and better than a chart.
    with pytest.raises(ValueError, match="different underlying metrics"):
        compare_models(reports, metrics=["relative_robustness"])


def test_an_aggregate_over_different_perturbation_sets_is_not_one_axis(
    problem, other_model
):
    """Survey §8: a sum-based aggregate scales with the size of the perturbation set, so
    the same metric over two different sets is two quantities."""
    from themis_assess.assessment.robustness import RobustnessEvaluator
    from themis_assess.report import build_report

    one = RobustnessEvaluator(
        problem["handler"], perturbations=["gaussian_noise"], severities=(1, 2)
    ).assess(problem["data"], problem["actual"], accuracy_metric="accuracy")
    two = RobustnessEvaluator(
        other_model,
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 2),
    ).assess(problem["data"], problem["actual"], accuracy_metric="accuracy")

    c = compare_models(
        {
            "one_perturbation": build_report(model_name="x", robustness=one),
            "two_perturbations": build_report(model_name="y", robustness=two),
        }
    )
    assert "perturbation sets" in c.dropped["total_relative_robustness"]
    # accuracy itself is still comparable: it does not depend on the set
    assert "accuracy" in c.metrics


def test_robustness_can_be_left_out_entirely(problem, other_model):
    from themis_assess.assessment.accuracy import AccuracyEvaluator
    from themis_assess.assessment.robustness import RobustnessEvaluator
    from themis_assess.report import build_report

    reports = {}
    for name, handler in (("x", problem["handler"]), ("y", other_model)):
        reports[name] = build_report(
            model_name=name,
            accuracy=AccuracyEvaluator(handler).run(problem["data"], problem["actual"]),
            robustness=RobustnessEvaluator(
                handler, perturbations=["gaussian_noise"], severities=(1,)
            ).assess(problem["data"], problem["actual"], accuracy_metric="accuracy"),
        )
    with_robustness = compare_models(reports)
    without = compare_models(reports, include_robustness=False)
    assert "relative_robustness" in with_robustness.metrics
    assert "relative_robustness" not in without.metrics
    assert "accuracy" in without.metrics


# --------------------------------------------------------------------------
# reading the comparison
# --------------------------------------------------------------------------


def test_the_disagreements_are_the_output_worth_reading():
    c = compare_models(
        {"accurate": {"accuracy": 0.92, "mae": 3.0}, "precise": {"accuracy": 0.80, "mae": 1.0}}
    )
    assert not c.is_unanimous
    described = [d.describe() for d in c.disagreements()]
    assert len(described) == 1
    assert "accuracy prefers accurate" in described[0]
    assert "mae prefers precise" in described[0]


def test_unanimity_is_reported_when_it_holds():
    c = compare_models(
        {"better": {"accuracy": 0.92, "f1": 0.9}, "worse": {"accuracy": 0.80, "f1": 0.7}}
    )
    assert c.is_unanimous
    assert c.disagreements() == []
    assert "prefers the same model" in c.summary()


def test_the_ranking_carries_its_components():
    """The composite never appears alone — survey §8 rule 4, applied to this package's
    own ranking device."""
    c = compare_models(
        {"a": {"accuracy": 0.9, "f1": 0.9}, "b": {"accuracy": 0.8, "f1": 0.8}}
    )
    top = c.ranking()[0]
    assert top["model"] == "a"
    assert set(top["per_metric_normalised"]) == set(c.metrics)
    assert set(top["per_metric_raw"]) == set(c.metrics)
    assert top["wins"] == 2


def test_the_mean_normalised_score_is_the_mean_of_the_normalised_values():
    c = compare_models(
        {"a": {"accuracy": 0.9, "f1": 0.7}, "b": {"accuracy": 0.8, "f1": 0.8}}
    )
    assert c.mean_normalised()["a"] == pytest.approx(np.mean([0.9, 0.7]))


def test_a_metric_that_is_not_an_axis_is_a_clear_error():
    c = compare_models({"a": {"accuracy": 0.9}, "b": {"accuracy": 0.8}})
    with pytest.raises(KeyError, match="not a comparison axis"):
        c.best("f1")


def test_the_comparison_serialises_with_its_glossary():
    import json

    c = compare_models(
        {"a": {"accuracy": 0.9, "mae": 1.0}, "b": {"accuracy": 0.8, "mae": 2.0}}
    )
    payload = c.to_dict()
    json.dumps(payload)
    assert payload["schema"] == "themis-assess/comparison@1"
    assert payload["glossary"]["mae"]["higher_is_better"] is False
    assert payload["ranking"][0]["model"] == c.ranking()[0]["model"]


def test_model_order_is_preserved_so_colours_do_not_move():
    """Colour follows the entity, not its rank: adding a model must not repaint the
    others, so the comparison keeps the order it was given rather than sorting."""
    c = compare_models(
        {"z": {"accuracy": 0.7}, "a": {"accuracy": 0.9}, "m": {"accuracy": 0.8}}
    )
    assert c.models == ("z", "a", "m")
    assert c.ranking()[0]["model"] == "a"


def test_the_construction_refuses_an_empty_axis_set():
    with pytest.raises(ValueError, match="nothing to compare"):
        ModelComparison(models=("a", "b"), metrics=(), raw={}, normalised={})
