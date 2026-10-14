"""Tests for the metric catalog.

The catalog replaces four parallel dictionaries spread across the three use-case repos,
so the tests that matter most are the ones about *consistency*: every catalogued metric is
a registered metric, every entry declares what it needs, and asking for something the data
cannot support fails with a message naming the missing quantity rather than with a
TypeError from deep inside scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.metrics.accuracy.catalog import MetricCatalog, default_catalog
from themis_assess.metrics.registry import default_registry


# --------------------------------------------------------------------------
# consistency with the registry
# --------------------------------------------------------------------------


def test_every_catalogued_metric_is_a_registered_metric():
    """The defect the four parallel dictionaries had: they drifted out of sync."""
    catalog, registry = default_catalog(), default_registry()
    unregistered = [name for name in catalog.names() if name not in registry]
    assert unregistered == []


def test_catalog_and_registry_agree_on_task_applicability():
    catalog, registry = default_catalog(), default_registry()
    for name in catalog.names():
        assert catalog.get(name).tasks == registry.get(name).tasks, name


def test_catalog_and_registry_agree_on_access_level():
    catalog, registry = default_catalog(), default_registry()
    for name in catalog.names():
        assert catalog.get(name).access == registry.get(name).access, name


def test_each_catalog_instance_is_independent():
    a, b = default_catalog(), default_catalog()
    a.add("my_metric", lambda actual, predicted: 0.0)
    assert "my_metric" in a
    assert "my_metric" not in b


def test_registering_an_existing_name_needs_an_explicit_overwrite():
    catalog = default_catalog()
    with pytest.raises(ValueError, match="already in the catalog"):
        catalog.add("accuracy", lambda actual, predicted: 1.0)
    catalog.add("accuracy", lambda actual, predicted: 1.0, overwrite=True)
    assert catalog.compute("accuracy", {"actual": [0], "predicted": [1]}) == 1.0


def test_an_unknown_name_suggests_a_near_match():
    catalog = default_catalog()
    with pytest.raises(KeyError, match="Did you mean"):
        catalog.get("crps_gauss")


# --------------------------------------------------------------------------
# computing from a bag of quantities
# --------------------------------------------------------------------------


@pytest.fixture
def classification_bag():
    rng = np.random.default_rng(0)
    actual = rng.integers(0, 2, 200)
    noisy = np.where(rng.random(200) < 0.2, 1 - actual, actual)
    confidence = np.where(noisy == 1, 0.8, 0.2)
    return {
        "actual": actual,
        "predicted": noisy,
        "probabilities": np.column_stack([1 - confidence, confidence]),
    }


def test_label_metrics_compute_from_labels_alone(classification_bag):
    bag = {k: v for k, v in classification_bag.items() if k != "probabilities"}
    catalog = default_catalog()
    values = catalog.compute_all(["accuracy", "f1", "matthews_corrcoef"], bag)
    assert set(values) == {"accuracy", "f1", "matthews_corrcoef"}
    assert values["accuracy"] == pytest.approx(0.805)


def test_a_probability_metric_asks_for_what_is_missing(classification_bag):
    bag = {k: v for k, v in classification_bag.items() if k != "probabilities"}
    catalog = default_catalog()
    with pytest.raises(KeyError, match="missing probabilities"):
        catalog.compute("auc", bag)


def test_computable_reports_what_the_data_supports(classification_bag):
    catalog = default_catalog()
    labels_only = {k: v for k, v in classification_bag.items() if k != "probabilities"}

    without = catalog.computable(labels_only, task=TaskType.CLASSIFICATION)
    with_probs = catalog.computable(classification_bag, task=TaskType.CLASSIFICATION)

    assert "accuracy" in without
    assert "auc" not in without
    assert "expected_calibration_error" not in without
    assert set(with_probs) - set(without) == {
        "auc",
        "brier_score",
        "expected_calibration_error",
        "negative_log_likelihood",
    }


def test_computable_filters_by_task(classification_bag):
    catalog = default_catalog()
    assert "mae" not in catalog.computable(
        classification_bag, task=TaskType.CLASSIFICATION
    )
    assert "accuracy" not in catalog.computable(
        classification_bag, task=TaskType.FORECASTING
    )


def test_a_present_but_none_valued_quantity_counts_as_missing():
    """Handlers return None for what they cannot produce; that must not be passed on."""
    catalog = default_catalog()
    bag = {"actual": np.array([0, 1]), "predicted": np.array([0, 1]), "probabilities": None}
    assert "auc" not in catalog.computable(bag)


def test_compute_all_can_skip_what_it_cannot_compute(classification_bag):
    bag = {k: v for k, v in classification_bag.items() if k != "probabilities"}
    catalog = default_catalog()
    values = catalog.compute_all(["accuracy", "auc"], bag, skip_uncomputable=True)
    assert set(values) == {"accuracy"}


# --------------------------------------------------------------------------
# the call-convention wrappers
# --------------------------------------------------------------------------


def test_precision_and_recall_work_on_more_than_two_classes():
    """All three repos called precision_score with no average, which raises here."""
    catalog = default_catalog()
    bag = {
        "actual": np.array([0, 1, 2, 0, 1, 2]),
        "predicted": np.array([0, 1, 2, 0, 2, 1]),
    }
    assert 0.0 <= catalog.compute("precision", bag) <= 1.0
    assert 0.0 <= catalog.compute("recall", bag) <= 1.0


def test_binary_precision_scores_the_positive_class_only():
    """Two classes keep the binary convention, so numbers stay comparable with repo D's."""
    catalog = default_catalog()
    bag = {
        "actual": np.array([0, 0, 1, 1]),
        "predicted": np.array([0, 1, 1, 1]),
    }
    # Positive class: 2 true positives, 1 false positive.
    assert catalog.compute("precision", bag) == pytest.approx(2 / 3)
    assert catalog.compute("recall", bag) == pytest.approx(1.0)


def test_a_class_never_predicted_scores_zero_rather_than_raising():
    catalog = default_catalog()
    bag = {"actual": np.array([0, 0, 1, 1]), "predicted": np.zeros(4, dtype=int)}
    assert catalog.compute("precision", bag) == pytest.approx(0.0)


def test_auc_accepts_a_probability_matrix_in_the_binary_case():
    catalog = default_catalog()
    actual = np.array([0, 0, 1, 1])
    confidence = np.array([0.1, 0.3, 0.7, 0.9])
    matrix = {"actual": actual, "probabilities": np.column_stack([1 - confidence, confidence])}
    vector = {"actual": actual, "probabilities": confidence}
    assert catalog.compute("auc", matrix) == pytest.approx(1.0)
    assert catalog.compute("auc", vector) == pytest.approx(1.0)


def test_auc_handles_more_than_two_classes():
    rng = np.random.default_rng(1)
    actual = rng.integers(0, 3, 90)
    probs = np.zeros((90, 3))
    probs[np.arange(90), actual] = 0.7
    probs[probs == 0.0] = 0.15
    value = default_catalog().compute("auc", {"actual": actual, "probabilities": probs})
    assert value == pytest.approx(1.0)


def test_rmse_is_the_square_root_of_mse():
    catalog = default_catalog()
    bag = {"actual": np.array([0.0, 0.0, 0.0]), "predicted": np.array([3.0, 4.0, 0.0])}
    assert catalog.compute("mse", bag) == pytest.approx(25 / 3)
    assert catalog.compute("rmse", bag) == pytest.approx(np.sqrt(25 / 3))


# --------------------------------------------------------------------------
# forecasting entries
# --------------------------------------------------------------------------


def test_interval_metrics_need_bounds_not_predictions():
    catalog = default_catalog()
    bag = {"actual": np.zeros(5), "predicted": np.zeros(5)}
    with pytest.raises(KeyError, match="missing lower, upper"):
        catalog.compute("prediction_interval_coverage", bag)


def test_interval_metrics_compute_from_bounds():
    catalog = default_catalog()
    bag = {
        "actual": np.array([0.5, 0.5, 5.0]),
        "lower": np.zeros(3),
        "upper": np.ones(3),
    }
    assert catalog.compute("prediction_interval_coverage", bag) == pytest.approx(2 / 3)
    assert catalog.compute("mean_prediction_interval_width", bag) == pytest.approx(1.0)


def test_crps_computes_from_an_ensemble():
    catalog = default_catalog()
    y = np.array([0.0, 1.0])
    bag = {"actual": y, "samples": np.column_stack([y, y])}
    assert catalog.compute("crps", bag) == pytest.approx(0.0)


def test_weights_are_optional_and_used_when_present():
    """The reason MetricEntry has an ``optional`` field at all."""
    catalog = default_catalog()
    actual = np.zeros(4)
    predicted = np.array([4.0, 0.0, 0.0, 0.0])
    without = catalog.compute("weighted_mae", {"actual": actual, "predicted": predicted})
    weights = np.array([1.0, 0.0, 0.0, 0.0])
    with_weights = catalog.compute(
        "weighted_mae", {"actual": actual, "predicted": predicted, "weights": weights}
    )
    assert without == pytest.approx(1.0)
    assert with_weights == pytest.approx(4.0)


# --------------------------------------------------------------------------
# extension by a use case
# --------------------------------------------------------------------------


def test_a_use_case_can_add_its_own_metric():
    """Repo B's berthing metrics are not general, but must run through the same
    machinery so the report and the normalisation treat them identically."""
    catalog = MetricCatalog()
    catalog.add(
        "berth_timing_error",
        lambda actual, predicted: float(np.mean(predicted - actual)),
        tasks=frozenset({TaskType.FORECASTING}),
    )
    bag = {"actual": np.zeros(3), "predicted": np.array([1.0, 2.0, 3.0])}
    assert catalog.compute("berth_timing_error", bag) == pytest.approx(2.0)
    assert "berth_timing_error" in catalog.for_task(TaskType.FORECASTING)


def test_a_custom_metric_can_declare_its_own_requirements():
    catalog = MetricCatalog(with_defaults=False)
    catalog.add(
        "embedding_drift",
        lambda reference, probabilities: 0.5,
        requires=("reference", "probabilities"),
        access=AccessLevel.EMBEDDING,
    )
    assert catalog.computable({"actual": np.zeros(3)}) == []
    assert catalog.computable({"reference": np.zeros(3), "probabilities": np.zeros(3)}) == [
        "embedding_drift"
    ]


def test_an_empty_catalog_starts_empty():
    assert len(MetricCatalog(with_defaults=False)) == 0
