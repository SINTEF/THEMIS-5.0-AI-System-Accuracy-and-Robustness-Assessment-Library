"""Tests for the scikit-learn and callable handlers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.handlers.sklearn import CallableHandler, SklearnHandler


@pytest.fixture
def classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = ArrayAdapter(
        X, feature_names=[f"f{i}" for i in range(4)], categorical_features=[]
    )
    return X, y, data


def test_access_is_inferred_from_predict_proba(classification):
    X, y, _ = classification
    with_proba = SklearnHandler(LogisticRegression().fit(X, y))
    without = SklearnHandler(SVC(probability=False).fit(X, y))
    assert with_proba.access is AccessLevel.PROBABILISTIC
    assert without.access is AccessLevel.BLACK_BOX


def test_access_can_be_lowered_but_not_raised(classification):
    X, y, _ = classification
    model = LogisticRegression().fit(X, y)
    lowered = SklearnHandler(model, access=AccessLevel.BLACK_BOX)
    assert lowered.access is AccessLevel.BLACK_BOX

    with pytest.raises(ValueError, match="cannot serve access=PROBABILISTIC"):
        SklearnHandler(SVC(probability=False).fit(X, y), access=AccessLevel.PROBABILISTIC)


def test_lowering_access_actually_withholds_the_probabilities(classification):
    """Declaring black-box must mean the run behaves as black-box, not merely say so."""
    X, y, data = classification
    handler = SklearnHandler(LogisticRegression().fit(X, y), access=AccessLevel.BLACK_BOX)
    assert handler.predict(data).probabilities is None


def test_a_model_without_predict_is_rejected():
    with pytest.raises(TypeError, match="has no predict method"):
        SklearnHandler(object())


def test_predictions_and_probabilities_come_back_together(classification):
    """Repo C returned labels and left probabilities on self.y_pred_raw as a side
    effect; both belong to the same call and now travel together."""
    X, y, data = classification
    prediction = SklearnHandler(RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)).predict(data)
    assert prediction.predicted.shape == (120,)
    assert prediction.probabilities.shape == (120, 2)
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0)


def test_a_decision_function_is_not_passed_off_as_a_probability(classification):
    """Squashing a margin through a sigmoid would make every calibration metric a
    measurement of that line rather than of the model.

    A binary ``decision_function`` returns a 1-D array of margins, which is shaped
    exactly like the ``P(class 1)`` vector a handler may legitimately return — only the
    range tells them apart, so the range is what is checked.
    """
    X, y, data = classification
    handler = SklearnHandler(
        SVC(probability=False).fit(X, y),
        probability_source="decision_function",
        access=AccessLevel.PROBABILISTIC,
    )
    with pytest.raises(ValueError, match="not a probability distribution"):
        handler.predict(data)


def test_a_calibrated_classifier_passes_the_probability_check(classification):
    """The fix the error message recommends must actually work."""
    from sklearn.calibration import CalibratedClassifierCV

    X, y, data = classification
    model = CalibratedClassifierCV(SVC(probability=False), cv=3).fit(X, y)
    prediction = SklearnHandler(model, name="calibrated_svc").predict(data)
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0)


def test_sequential_data_is_flattened_for_a_tabular_model():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(30, 6, 2))
    flat = values.reshape(30, -1)
    y = (flat[:, 0] > 0).astype(int)
    data = ArrayAdapter(values, feature_names=["a", "b"], categorical_features=[])
    handler = SklearnHandler(LogisticRegression().fit(flat, y))
    prediction = handler.predict(data)
    assert len(prediction) == 30
    assert prediction.metadata["n_features_in"] == 12


def test_a_regressor_declares_forecasting():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    data = ArrayAdapter(X, feature_names=["a", "b", "c"], categorical_features=[])
    handler = SklearnHandler(
        LinearRegression().fit(X, y), task=TaskType.FORECASTING, name="ols"
    )
    assert handler.task is TaskType.FORECASTING
    assert handler.access is AccessLevel.BLACK_BOX
    assert handler.predict(data).predicted.shape == (50,)


# --------------------------------------------------------------------------
# CallableHandler
# --------------------------------------------------------------------------


def test_a_plain_function_is_a_valid_model():
    data = ArrayAdapter(
        np.zeros((7, 2)), feature_names=["a", "b"], categorical_features=[]
    )
    handler = CallableHandler(lambda X: np.ones(X.shape[0]))
    assert handler.access is AccessLevel.BLACK_BOX
    assert handler.predict(data).predicted.tolist() == [1.0] * 7


def test_supplying_extras_raises_the_declared_access():
    handler = CallableHandler(
        lambda X: np.zeros(X.shape[0]),
        probability_fn=lambda X: np.tile([0.4, 0.6], (X.shape[0], 1)),
    )
    assert handler.access is AccessLevel.PROBABILISTIC


def test_an_interval_function_produces_both_bounds():
    data = ArrayAdapter(
        np.zeros((5, 1)), feature_names=["a"], categorical_features=[]
    )
    handler = CallableHandler(
        lambda X: np.zeros(X.shape[0]),
        interval_fn=lambda X: (np.full(X.shape[0], -1.0), np.full(X.shape[0], 1.0)),
        task=TaskType.FORECASTING,
    )
    prediction = handler.predict(data)
    assert prediction.lower.tolist() == [-1.0] * 5
    assert prediction.upper.tolist() == [1.0] * 5


def test_a_non_callable_is_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        CallableHandler("not a function")
