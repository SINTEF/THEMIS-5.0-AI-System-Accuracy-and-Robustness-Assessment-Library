"""Tests for the learned error attribution.

Two properties carry the weight. The gate must work in both directions: a surrogate that
learned nothing must refuse to hand out attributions, and one that learned something must
recover the feature the error was generated from. And the gate must use the right statistic
for the target — R² for a magnitude, AUROC for a 0/1 misclassification indicator, because
whether a particular borderline sample flips is close to irreducibly random even when the
*risk* of flipping is highly predictable.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from themis_assess.attribution.error_model import (
    DEFAULT_AUROC_THRESHOLD,
    DEFAULT_TRUST_THRESHOLD,
    ErrorModel,
    per_sample_error,
)
from themis_assess.attribution.features import (
    TabularFeatures,
    WindowSummaryFeatures,
)
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.base import Prediction
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler
from themis_assess.handlers.sklearn import SklearnHandler

# SHAP is behind the [explain] extra, so the tests that need it are skipped where it is
# absent. That the rest of the suite passes without it is the point of the design: the
# core path — a scikit-learn surrogate and permutation importance — needs no extra.
requires_shap = pytest.mark.skipif(
    importlib.util.find_spec("shap") is None,
    reason="local SHAP attribution needs the [explain] extra",
)


@pytest.fixture(scope="module")
def volatile_series():
    """300 series whose forecastability is set by a per-series volatility.

    A persistence forecaster's error is then predictable from the *spread* of its input
    window, which is the ground truth the attribution must recover.
    """
    rng = np.random.default_rng(0)
    n_series, n_steps = 300, 40
    volatility = rng.uniform(0.05, 1.5, n_series)
    series = np.stack(
        [
            np.cumsum(rng.normal(scale=v, size=n_steps))
            + np.sin(np.arange(n_steps) / 5)
            for v in volatility
        ]
    )
    values = np.stack([series, np.tile(np.arange(n_steps), (n_series, 1))], axis=2)
    data = ArrayAdapter(values, feature_names=["y", "clock"], categorical_features=[])
    config = ForecastingConfig(forecast_horizon=5, input_chunk_length=20)
    handler = ForecastingHandler(
        lambda history, horizon: np.full(horizon, history[-1, 0]),
        config,
        name="persistence",
        target="y",
    )
    prediction = handler.predict(data)
    return data, handler.actual(data), prediction.predicted


@pytest.fixture(scope="module")
def noisy_classification():
    """A classifier whose error *rate* is driven by a feature it cannot see."""
    rng = np.random.default_rng(0)
    n = 2000
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    noise_level = rng.uniform(0.1, 3.0, n)
    y = (x0 + 0.3 * x1 + noise_level * rng.normal(size=n) > 0).astype(int)
    X = np.column_stack([x0, x1, noise_level, rng.normal(size=n)])
    model = RandomForestClassifier(n_estimators=80, random_state=0).fit(X[:1200, :2], y[:1200])

    class TwoColumns(SklearnHandler):
        def predict(self, adapter):
            matrix = adapter.values().reshape(adapter.n_entities, -1)[:, :2]
            return self.check_prediction(
                Prediction(
                    predicted=self.model.predict(matrix),
                    probabilities=self.model.predict_proba(matrix),
                ),
                adapter,
            )

    data = ArrayAdapter(
        X[1200:],
        feature_names=["x0", "x1", "noise_level", "irrelevant"],
        categorical_features=[],
    )
    handler = TwoColumns(model, name="rf")
    return data, y[1200:], handler.predict(data).predicted


# --------------------------------------------------------------------------
# the per-sample error
# --------------------------------------------------------------------------


def test_misclassification_is_a_zero_one_indicator():
    error = per_sample_error(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
    assert error.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_a_continuous_target_gives_an_absolute_error():
    error = per_sample_error(np.array([1.5, 2.5]), np.array([1.0, 4.0]), kind="absolute")
    assert error == pytest.approx([0.5, 1.5])


def test_a_multi_horizon_forecast_averages_over_the_horizon():
    actual = np.zeros((2, 4))
    predicted = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    assert per_sample_error(actual, predicted, kind="absolute") == pytest.approx([1.0, 1.0])


def test_squared_error_weights_the_large_errors():
    error = per_sample_error(np.zeros(2), np.array([1.0, 3.0]), kind="squared")
    assert error == pytest.approx([1.0, 9.0])


def test_auto_picks_misclassification_for_labels():
    labels = np.array([0, 1, 2, 1, 0])
    assert set(per_sample_error(labels, labels[::-1]).tolist()) <= {0.0, 1.0}


def test_auto_picks_absolute_for_a_continuous_target():
    actual = np.linspace(0.0, 100.0, 50)
    assert per_sample_error(actual, actual + 2.0) == pytest.approx(np.full(50, 2.0))


def test_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError, match="same samples"):
        per_sample_error(np.zeros(5), np.zeros(4))


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind must be"):
        per_sample_error(np.zeros(3), np.zeros(3), kind="magic")


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------


def test_tabular_features_are_the_raw_columns():
    data = ArrayAdapter(
        np.arange(6.0).reshape(3, 2), feature_names=["a", "b"], categorical_features=[]
    )
    matrix, names = TabularFeatures().extract(data)
    assert names == ["a", "b"]
    assert matrix.tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]


def test_tabular_features_flatten_a_sequence_with_timestamped_names():
    data = ArrayAdapter(
        np.zeros((2, 3, 2)), feature_names=["a", "b"], categorical_features=[]
    )
    _, names = TabularFeatures().extract(data)
    assert names[:2] == ["a@t0", "b@t0"]
    assert len(names) == 6


def test_window_summaries_describe_a_known_window():
    """Every statistic checked by hand against a ramp from 0 to 9."""
    values = np.arange(10.0).reshape(1, 10, 1)
    data = ArrayAdapter(values, feature_names=["y"], categorical_features=[])
    matrix, names = WindowSummaryFeatures().extract(data)
    summary = dict(zip(names, matrix[0], strict=True))
    assert summary["y__mean"] == pytest.approx(4.5)
    assert summary["y__last"] == pytest.approx(9.0)
    assert summary["y__range"] == pytest.approx(9.0)
    assert summary["y__slope"] == pytest.approx(1.0)
    assert summary["y__last_minus_mean"] == pytest.approx(4.5)
    assert summary["y__abs_diff_mean"] == pytest.approx(1.0)
    assert summary["y__flat_fraction"] == pytest.approx(0.0)
    assert summary["y__missing_fraction"] == pytest.approx(0.0)


def test_missing_fraction_counts_the_gaps():
    """A model whose errors follow how much data was absent has an availability problem
    rather than a modelling one."""
    values = np.array([[[1.0], [np.nan], [3.0], [np.nan]]])
    data = ArrayAdapter(values, feature_names=["y"], categorical_features=[])
    matrix, names = WindowSummaryFeatures().extract(data)
    summary = dict(zip(names, matrix[0], strict=True))
    assert summary["y__missing_fraction"] == pytest.approx(0.5)
    assert summary["y__mean"] == pytest.approx(2.0)  # NaN-aware


def test_a_flat_window_is_reported_as_flat():
    values = np.full((1, 8, 1), 3.0)
    data = ArrayAdapter(values, feature_names=["y"], categorical_features=[])
    matrix, names = WindowSummaryFeatures().extract(data)
    summary = dict(zip(names, matrix[0], strict=True))
    assert summary["y__flat_fraction"] == pytest.approx(1.0)
    assert summary["y__std"] == pytest.approx(0.0)


def test_a_statistic_subset_is_allowed():
    data = ArrayAdapter(np.zeros((3, 4, 1)), feature_names=["y"], categorical_features=[])
    _, names = WindowSummaryFeatures(statistics=("mean", "std")).extract(data)
    assert names == ["y__mean", "y__std"]


def test_an_unknown_statistic_is_rejected():
    with pytest.raises(ValueError, match="Unknown statistic"):
        WindowSummaryFeatures(statistics=("vibes",))


def test_the_extractor_defaults_by_modality():
    from themis_assess.attribution.features import resolve_extractor

    tabular = ArrayAdapter(np.zeros((3, 2)), feature_names=["a", "b"], categorical_features=[])
    sequential = ArrayAdapter(
        np.zeros((3, 5, 2)), feature_names=["a", "b"], categorical_features=[]
    )
    assert resolve_extractor(None, tabular).name == "tabular"
    assert resolve_extractor(None, sequential).name == "window_summary"


def test_an_unknown_extractor_name_is_rejected():
    from themis_assess.attribution.features import resolve_extractor

    data = ArrayAdapter(np.zeros((3, 2)), feature_names=["a", "b"], categorical_features=[])
    with pytest.raises(ValueError, match="Unknown extractor"):
        resolve_extractor("magic", data)


# --------------------------------------------------------------------------
# fitting, and the gate
# --------------------------------------------------------------------------


def test_a_forecast_error_model_recovers_the_generative_feature(volatile_series):
    """The capability's whole claim: the error is predictable from the input's spread,
    and the attribution says so."""
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    fit = model.fit(data, actual, predicted)

    assert fit.gate == "test_r2"
    assert fit.test_r2 > DEFAULT_TRUST_THRESHOLD
    assert model.is_trustworthy

    top = [entry["feature"] for entry in model.importance(n_repeats=8).top(4)]
    assert any(name.startswith("y__") for name in top)
    assert any(
        statistic in name
        for name in top
        for statistic in ("range", "std", "slope", "last_minus_mean")
    )


def test_a_binary_error_target_is_gated_on_auroc(noisy_classification):
    """R² is the wrong yardstick for a 0/1 indicator: measured here it sits below zero
    while the surrogate ranks error risk usefully above chance."""
    data, actual, predicted = noisy_classification
    model = ErrorModel(seed=1)
    fit = model.fit(data, actual, predicted)

    assert fit.is_binary_error
    assert fit.gate == "test_auroc"
    assert fit.test_r2 < DEFAULT_TRUST_THRESHOLD  # the R² gate would have refused
    assert fit.test_auroc >= DEFAULT_AUROC_THRESHOLD
    assert model.is_trustworthy


def test_the_classification_attribution_finds_the_hidden_driver(noisy_classification):
    """noise_level sets the error rate and the model cannot even see it."""
    data, actual, predicted = noisy_classification
    model = ErrorModel(seed=1)
    model.fit(data, actual, predicted)
    top = [entry["feature"] for entry in model.importance(n_repeats=15).top(2)]
    assert "noise_level" in top


def test_an_untrustworthy_surrogate_refuses_to_attribute():
    """Reading noise attributions is worse than computing none: the output is identical."""
    rng = np.random.default_rng(3)
    n = 200
    data = ArrayAdapter(
        rng.normal(size=(n, 4)),
        feature_names=[f"f{i}" for i in range(4)],
        categorical_features=[],
    )
    actual = rng.normal(size=n)
    predicted = actual + rng.normal(size=n)  # error independent of every feature

    model = ErrorModel(seed=0)
    fit = model.fit(data, actual, predicted)
    assert not fit.is_trustworthy
    with pytest.raises(ValueError, match="not predicting the error"):
        model.importance()


def test_the_gate_can_be_overridden_explicitly():
    rng = np.random.default_rng(3)
    n = 200
    data = ArrayAdapter(
        rng.normal(size=(n, 4)),
        feature_names=[f"f{i}" for i in range(4)],
        categorical_features=[],
    )
    actual = rng.normal(size=n)
    model = ErrorModel(seed=0)
    model.fit(data, actual, actual + rng.normal(size=n))
    assert model.importance(force=True).method == "permutation"


def test_overfitting_is_flagged(noisy_classification):
    """Few entities, many features is the normal regime here."""
    data, actual, predicted = noisy_classification
    fit = ErrorModel(seed=1).fit(data, actual, predicted)
    assert fit.looks_overfitted
    assert "far above test" in fit.describe()


def test_attribution_before_fitting_is_refused():
    with pytest.raises(RuntimeError, match="Call fit"):
        ErrorModel().importance()


def test_too_few_samples_is_refused():
    data = ArrayAdapter(
        np.zeros((5, 3)), feature_names=["a", "b", "c"], categorical_features=[]
    )
    with pytest.raises(ValueError, match="needs more than"):
        ErrorModel().fit(data, np.arange(5.0), np.zeros(5))


def test_a_constant_error_has_nothing_to_explain():
    rng = np.random.default_rng(4)
    data = ArrayAdapter(
        rng.normal(size=(50, 3)), feature_names=["a", "b", "c"], categorical_features=[]
    )
    actual = rng.normal(size=50)
    with pytest.raises(ValueError, match="no variation to explain"):
        ErrorModel().fit(data, actual, actual)


def test_constant_features_are_dropped(volatile_series):
    """The clock column is identical across series, so it can inform nothing."""
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    fit = model.fit(data, actual, predicted)
    assert fit.n_features < 24  # 12 statistics x 2 features before dropping
    assert not any(name.startswith("clock__") for name in model._features)


def test_the_split_is_reproducible(volatile_series):
    data, actual, predicted = volatile_series
    a = ErrorModel(extractor=WindowSummaryFeatures(), seed=7).fit(data, actual, predicted)
    b = ErrorModel(extractor=WindowSummaryFeatures(), seed=7).fit(data, actual, predicted)
    assert a.test_r2 == pytest.approx(b.test_r2)


# --------------------------------------------------------------------------
# local attribution
# --------------------------------------------------------------------------


@requires_shap
def test_a_held_out_sample_can_be_explained(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    index = int(model._test_index[0])

    attribution = model.explain(index)
    assert attribution.scope == "local"
    assert attribution.sample_role == "test"
    assert attribution.signed is not None
    assert attribution.actual_error == pytest.approx(
        float(np.abs(actual[index] - predicted[index]).mean())
    )


def test_an_in_sample_explanation_is_refused(volatile_series):
    """Repo B printed a warning. A print in a library reaches nobody, and the output is
    indistinguishable either way.

    The refusal is core behaviour and needs no optional dependency: it happens before
    SHAP is reached.
    """
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    with pytest.raises(ValueError, match="training set"):
        model.explain(int(model._train_index[0]))


@requires_shap
def test_an_in_sample_explanation_can_be_forced(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    index = int(model._train_index[0])
    assert model.explain(index, allow_in_sample=True).sample_role == "train"


def test_the_sample_role_is_reported(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    assert model.role_of(int(model._test_index[0])) == "test"
    assert model.role_of(int(model._train_index[0])) == "train"


def test_an_out_of_range_sample_is_refused(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    with pytest.raises(IndexError, match="out of range"):
        model.explain(10_000)


@requires_shap
def test_shap_attribution_carries_a_direction(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    attribution = model.importance(method="shap")
    assert attribution.signed is not None
    assert {entry["direction"] for entry in attribution.top(5)} <= {"worse", "better"}


def test_the_shap_path_names_the_extra_when_it_is_absent(volatile_series):
    """The optional dependency must fail with a pointer, not an ImportError traceback."""
    if importlib.util.find_spec("shap") is not None:
        pytest.skip("shap is installed here; the absent-dependency path cannot be reached")
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    with pytest.raises(ImportError, match=r"\[explain\] extra"):
        model.importance(method="shap")


def test_permutation_attribution_has_no_direction(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    attribution = model.importance()
    assert attribution.signed is None
    assert "direction" not in attribution.top(1)[0]


def test_an_unknown_attribution_method_is_rejected(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    with pytest.raises(ValueError, match="method must be"):
        model.importance(method="magic")


# --------------------------------------------------------------------------
# the scatter sanity check
# --------------------------------------------------------------------------


def test_the_scatter_shows_the_marginal_relationship(volatile_series):
    """The plainest possible check on an attribution: a highly ranked feature should
    show something here."""
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    scatter = model.feature_error_scatter("y__std")
    correlation = float(
        np.corrcoef(scatter["feature_value"], scatter["error"])[0, 1]
    )
    assert correlation > 0.3
    assert scatter["is_test"].sum() == model.fit_report.n_test


def test_an_unknown_scatter_feature_lists_what_exists(volatile_series):
    data, actual, predicted = volatile_series
    model = ErrorModel(extractor=WindowSummaryFeatures(), seed=2)
    model.fit(data, actual, predicted)
    with pytest.raises(KeyError, match="Available"):
        model.feature_error_scatter("not_a_feature")
