"""Tests for the robustness evaluator — the perturbation sweep.

Three things carry most of the weight here.

*The plan is accurate.* Its whole purpose is telling the user, before a long run, what
will be computed and what will be skipped and why. A plan that were merely optimistic
would be worse than none.

*The ground truth is not recomputed from perturbed data.* In forecasting the target is a
column of the input, so a perturbation touching it would move the truth with the
prediction. There is a test that constructs exactly that situation.

*Cells are reproducible individually.* Re-running one severity must give the numbers that
severity had inside the full sweep, which is what repo B's global seeding could not do.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.subgroups import SubgroupSpec
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.handlers.base import BaseHandler, Prediction
from themis_assess.handlers.baselines import MajorityClassHandler, NaiveLastValueHandler
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler
from themis_assess.handlers.sklearn import CallableHandler, SklearnHandler
from themis_assess.assessment.robustness import RobustnessEvaluator
from themis_assess.perturbations.registry import default_registry as default_perturbations


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    data = ArrayAdapter(
        X, feature_names=[f"f{i}" for i in range(4)], categorical_features=[]
    )
    return data, y, SklearnHandler(model, name="logreg")


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def test_the_plan_lists_what_will_be_computed(problem):
    data, y, handler = problem
    plan = RobustnessEvaluator(handler, perturbations=["gaussian_noise"]).plan(data, y)
    assert "accuracy" in plan.metrics
    assert "expected_calibration_error" in plan.metrics
    assert plan.perturbations == ("gaussian_noise",)
    assert plan.task is TaskType.CLASSIFICATION


def test_the_plan_says_why_a_metric_will_be_skipped(problem):
    """The thing none of the four repos could do: tell the user before the run."""
    data, y, _ = problem
    labels_only = CallableHandler(
        lambda X: (X[:, 0] > 0).astype(int), name="labels_only"
    )
    plan = RobustnessEvaluator(
        labels_only,
        metrics=["accuracy", "auc", "expected_calibration_error"],
        perturbations=["gaussian_noise"],
    ).plan(data, y)

    assert plan.metrics == ("accuracy",)
    assert "auc" in plan.skipped
    assert "PROBABILISTIC" in plan.skipped["auc"]
    assert "expected_calibration_error" in plan.skipped


def test_the_plan_reports_a_metric_defined_for_the_other_task(problem):
    data, y, handler = problem
    plan = RobustnessEvaluator(
        handler, metrics=["accuracy", "mae"], perturbations=["gaussian_noise"]
    ).plan(data, y)
    assert plan.metrics == ("accuracy",)
    assert "forecasting" in plan.skipped["mae"]


def test_the_plan_reports_an_unknown_metric_rather_than_raising(problem):
    data, y, handler = problem
    plan = RobustnessEvaluator(
        handler, metrics=["accuracy", "not_a_metric"], perturbations=["gaussian_noise"]
    ).plan(data, y)
    assert plan.skipped["not_a_metric"] == "not in the metric catalog"


def test_the_plan_counts_the_model_calls(problem):
    data, y, handler = problem
    plan = RobustnessEvaluator(
        handler,
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 2, 3),
        n_variants=2,
    ).plan(data, y)
    # 1 clean + 2 perturbations x 3 severities x 2 variants
    assert plan.n_model_calls == 13


def test_a_baseline_doubles_the_model_calls(problem):
    data, y, handler = problem
    plan = RobustnessEvaluator(
        handler,
        baseline=MajorityClassHandler(y),
        perturbations=["gaussian_noise"],
        severities=(1, 2),
    ).plan(data, y)
    assert plan.n_model_calls == 2 * (1 + 2)


def test_a_handler_that_can_compute_nothing_raises(problem):
    data, y, _ = problem
    labels_only = CallableHandler(lambda X: np.zeros(X.shape[0]), name="labels_only")
    runner = RobustnessEvaluator(
        labels_only, metrics=["auc", "brier_score"], perturbations=["gaussian_noise"]
    )
    with pytest.raises(ValueError, match="No metric can be computed"):
        runner.plan(data, y)


def test_the_plan_summary_mentions_the_skipped_metrics(problem):
    data, y, _ = problem
    labels_only = CallableHandler(
        lambda X: (X[:, 0] > 0).astype(int), name="labels_only"
    )
    summary = RobustnessEvaluator(
        labels_only, metrics=["accuracy", "auc"], perturbations=["gaussian_noise"]
    ).plan(data, y).summary()
    assert "skipped" in summary
    assert "auc" in summary


def test_naming_an_inapplicable_perturbation_raises(problem):
    """Silently skipping a named perturbation would overstate robustness."""
    data, y, handler = problem
    runner = RobustnessEvaluator(handler, perturbations=["category_swap"])
    with pytest.raises(ValueError, match="do not apply to this data"):
        runner.plan(data, y)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def test_a_sweep_produces_a_curve_per_perturbation(problem):
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise", "uniform_bias"], severities=(1, 3, 5)
    ).run(data, y)
    assert sweep.perturbation_names == ["gaussian_noise", "uniform_bias"]
    assert sorted(sweep.curve("gaussian_noise", "accuracy")) == [1, 3, 5]
    assert sweep.clean["accuracy"] > 0.9


def test_accuracy_degrades_as_severity_rises(problem):
    """The most basic thing the whole library exists to measure."""
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], n_variants=5, seed=1
    ).run(data, y)
    curve = sweep.curve("gaussian_noise", "accuracy")
    assert curve[1] > curve[5]
    assert sweep.clean["accuracy"] >= curve[1]


def test_averaging_over_variants_smooths_a_stochastic_perturbation(problem):
    """A single draw of a signed perturbation can help as easily as hurt, so a
    one-variant curve is not always monotone and a many-variant one is."""
    data, y, handler = problem
    single = RobustnessEvaluator(
        handler, perturbations=["uniform_bias"], n_variants=1, seed=7
    ).run(data, y).curve("uniform_bias", "accuracy")
    many = RobustnessEvaluator(
        handler, perturbations=["uniform_bias"], n_variants=15, seed=7
    ).run(data, y).curve("uniform_bias", "accuracy")

    def descents(curve):
        values = [curve[s] for s in sorted(curve)]
        return sum(a >= b for a, b in zip(values, values[1:]))  # noqa: B905

    assert descents(many) >= descents(single)
    assert descents(many) == 4  # strictly decreasing across five severities


def test_a_cell_is_reproducible_on_its_own(problem):
    """Repo B seeded globally once per run, so a severity could not be re-run in
    isolation and give the numbers it had inside the sweep."""
    data, y, handler = problem
    full = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1, 2, 3, 4, 5), seed=42
    ).run(data, y)
    just_one = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(4,), seed=42
    ).run(data, y)
    assert full.curve("gaussian_noise", "accuracy")[4] == pytest.approx(
        just_one.curve("gaussian_noise", "accuracy")[4]
    )


def test_two_runs_with_the_same_seed_agree(problem):
    data, y, handler = problem
    kwargs = {"perturbations": ["gaussian_noise"], "severities": (2,), "seed": 5}
    a = RobustnessEvaluator(handler, **kwargs).run(data, y)
    b = RobustnessEvaluator(handler, **kwargs).run(data, y)
    assert a.curve("gaussian_noise", "accuracy") == b.curve("gaussian_noise", "accuracy")


def test_different_seeds_give_different_draws(problem):
    """Compared on the perturbed data rather than on accuracy: two different noise draws
    can coincidentally give the same accuracy, since accuracy is quantised to 1/n."""
    data, y, handler = problem
    perturbation = default_perturbations()["gaussian_noise"]
    spec = perturbation.spec(severity=3)

    draws = []
    for seed in (1, 2):
        evaluator = RobustnessEvaluator(
            handler, perturbations=["gaussian_noise"], severities=(3,), seed=seed
        )
        rng = evaluator._cell_rng("gaussian_noise", 3, 0)
        draws.append(perturbation.perturb(data, spec, rng)[0].values())

    assert not np.allclose(draws[0], draws[1])


def test_worst_reports_the_worst_cell(problem):
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], n_variants=3, seed=1
    ).run(data, y)
    name, severity, value = sweep.worst("accuracy", higher_is_better=True)
    assert name == "gaussian_noise"
    assert severity == 5
    assert value == min(sweep.curve("gaussian_noise", "accuracy").values())


def test_the_severity_curve_is_recorded_with_its_parameter_mapping(problem):
    """Survey §8: a report must be able to say what 'severity 3' actually was."""
    data, y, handler = problem
    sweep = RobustnessEvaluator(handler, perturbations=["gaussian_noise"]).run(data, y)
    metadata = sweep.perturbed["gaussian_noise"][3].metadata
    assert metadata["n_variants"] == 1
    assert "severity" in metadata or "severity_factor" in metadata


def test_a_model_that_cannot_consume_a_perturbation_names_the_cell(problem):
    """An sklearn NaN error 40 cells into a run says nothing about which cell."""
    data, y, _ = problem
    def nan_intolerant(X):
        if np.isnan(X).any():
            raise ValueError("Input contains NaN")
        return (X[:, 0] > 0).astype(int)

    handler = CallableHandler(nan_intolerant, name="nan_intolerant")
    runner = RobustnessEvaluator(handler, perturbations=["missing_values"], severities=(3,))
    with pytest.raises(RuntimeError, match=r"missing_values@3"):
        runner.run(data, y)


# --------------------------------------------------------------------------
# baseline and subgroups
# --------------------------------------------------------------------------


def test_a_baseline_is_swept_under_the_same_perturbations(problem):
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler,
        baseline=MajorityClassHandler(y),
        perturbations=["gaussian_noise"],
        severities=(1, 2),
    ).run(data, y)
    assert set(sweep.baseline_curves("accuracy")) == set(sweep.curves("accuracy"))
    assert sweep.baseline_clean is not None


def test_a_baseline_only_gets_the_metrics_its_access_supports(problem):
    """A black-box baseline cannot produce ECE, and the sweep must not pretend it did."""
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler,
        baseline=MajorityClassHandler(y),
        perturbations=["gaussian_noise"],
        severities=(1,),
    ).run(data, y)
    assert "expected_calibration_error" in sweep.clean.metrics
    assert "expected_calibration_error" not in sweep.baseline_clean.metrics


def test_a_baseline_of_the_wrong_task_is_rejected(problem):
    _, y, handler = problem
    baseline = NaiveLastValueHandler(ForecastingConfig(forecast_horizon=2))
    with pytest.raises(ValueError, match="not a comparison"):
        RobustnessEvaluator(handler, baseline=baseline)


def test_subgroups_are_scored_from_the_same_predictions(problem):
    data, y, handler = problem
    groups = [
        SubgroupSpec("positive", lambda d: np.asarray(y) == 1),
        SubgroupSpec("negative", lambda d: np.asarray(y) == 0),
    ]
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1,), subgroups=groups
    ).run(data, y)
    assert set(sweep.subgroups) == {"positive", "negative"}
    counts = [r.n_samples for r in sweep.subgroups.values()]
    assert sum(counts) == sweep.clean.n_samples
    assert sweep.subgroups["positive"].metadata["group"] == "positive"


# --------------------------------------------------------------------------
# the rule that is not obvious
# --------------------------------------------------------------------------


def test_the_truth_is_not_recomputed_from_perturbed_data():
    """A perturbation touching the target must not move the truth with the prediction.

    The model here is a cheat: it returns the *last observed value of the target*, so if
    the runner rescored against the perturbed series it would look near-perfect however
    badly the data was corrupted. Holding ``actual`` fixed is what exposes it.
    """
    t = np.arange(120, dtype=float)
    values = np.stack([t, np.zeros(120)], axis=1)[None, :, :]
    data = ArrayAdapter(values, feature_names=["y", "pad"], categorical_features=[])
    config = ForecastingConfig(forecast_horizon=6, input_chunk_length=12)

    handler = ForecastingHandler(
        lambda history, horizon: np.full(horizon, history[-1, 0]),
        config,
        name="repeats_last_seen",
        target="y",
    )
    actual = handler.actual(data)  # captured once, from the clean series

    sweep = RobustnessEvaluator(
        handler,
        metrics=["mae"],
        perturbations=["gaussian_noise"],
        severities=(5,),
        seed=0,
    ).run(data, actual)

    # Severity 5 noise on a series with a large spread must show up as a large error.
    assert sweep.curve("gaussian_noise", "mae")[5] > sweep.clean["mae"]


def test_scoring_a_subset_does_not_re_predict(problem):
    """Re-predicting a subgroup on perturbed data would draw different noise and
    compare two different datasets."""
    data, y, _ = problem
    calls = []

    class Counting(BaseHandler):
        def predict(self, d):
            calls.append(d.n_entities)
            return Prediction(
                predicted=np.zeros(d.n_entities),
                probabilities=np.full((d.n_entities, 2), 0.5),
            )

    handler = Counting(
        None, name="counting", task=TaskType.CLASSIFICATION,
        access=AccessLevel.PROBABILISTIC,
    )
    groups = [SubgroupSpec("half", lambda d: np.arange(len(y)) < len(y) // 2)]
    RobustnessEvaluator(
        handler,
        metrics=["accuracy"],
        perturbations=["gaussian_noise"],
        severities=(1,),
        subgroups=groups,
    ).run(data, y)
    # Every call saw the full dataset: the subgroup was taken from a prediction.
    assert set(calls) == {len(y)}


def test_the_runner_validates_its_own_arguments(problem):
    _, _, handler = problem
    with pytest.raises(ValueError, match="n_variants must be at least 1"):
        RobustnessEvaluator(handler, n_variants=0)
    with pytest.raises(ValueError, match="1-based"):
        RobustnessEvaluator(handler, severities=(0, 1))


def test_a_metric_undefined_on_a_subgroup_is_recorded_not_returned_as_nan(problem):
    """AUC needs both classes present. A single-class subgroup makes it undefined —
    scikit-learn warns and returns NaN, which would propagate silently through every
    mean and ratio downstream, so it is recorded as undefined instead."""
    data, y, handler = problem
    groups = [SubgroupSpec("only_positive", lambda d: np.asarray(y) == 1)]
    sweep = RobustnessEvaluator(
        handler, perturbations=["gaussian_noise"], severities=(1,), subgroups=groups
    ).run(data, y)

    result = sweep.subgroups["only_positive"]
    assert "auc" not in result.metrics
    assert "auc" in result.metadata["undefined"]
    assert "accuracy" in result.metrics  # the rest of the subgroup still scores


def test_a_metric_that_is_fine_leaves_no_undefined_note(problem):
    data, y, handler = problem
    sweep = RobustnessEvaluator(
        handler, metrics=["accuracy", "f1"], perturbations=["gaussian_noise"],
        severities=(1,),
    ).run(data, y)
    assert "undefined" not in sweep.clean.metadata


def test_a_custom_perturbation_can_be_selected_by_name(problem):
    """``default_perturbations()`` builds a fresh registry on every call, so registering
    a perturbation into one instance used to be invisible to a name lookup — the only way
    to reach a custom perturbation was to sweep the whole registry."""
    import numpy as np

    from themis_assess.perturbations.registry import default_registry as default_perturbations
    from themis_assess.perturbations.tabular import _NumericFeaturePerturbation

    class Quantisation(_NumericFeaturePerturbation):
        name = "quantisation"
        severity_scale = (0.5, 1.0, 2.0, 4.0, 8.0)

        def _apply_column(self, column, valid, timesteps, magnitude, spec, rng):
            step = max(magnitude, 1e-12)
            column[:, timesteps] = np.round(column[:, timesteps] / step) * step
            return np.where(valid, column, np.nan)

    data, y, handler = problem
    registry = default_perturbations()
    registry.register(Quantisation())

    sweep = RobustnessEvaluator(
        handler,
        perturbations=["quantisation"],
        perturbation_registry=registry,
        severities=(1, 3, 5),
    ).run(data, y)

    assert sweep.perturbation_names == ["quantisation"]
    curve = sweep.curve("quantisation", "accuracy")
    assert sorted(curve) == [1, 3, 5]
    # coarser quantisation costs more accuracy than finer
    assert curve[5] <= curve[1]


def test_two_registries_are_refused_rather_than_one_silently_winning(problem):
    from themis_assess.perturbations.registry import default_registry as default_perturbations

    data, y, handler = problem
    with pytest.raises(ValueError, match="not two registries"):
        RobustnessEvaluator(
            handler,
            perturbations=default_perturbations(),
            perturbation_registry=default_perturbations(),
        )
