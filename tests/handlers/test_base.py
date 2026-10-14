"""Tests for the handler contract: Prediction and BaseHandler.

The point of this layer is that a handler *declares* what it can produce and the
declaration is then trustworthy, because everything downstream — which metrics get
computed, what the user is told will be skipped — is derived from it. So most of these
tests are about the declaration being enforced rather than believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.handlers.base import BaseHandler, ModelHandler, Prediction


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return ArrayAdapter(
        rng.normal(size=(20, 3)), feature_names=["a", "b", "c"], categorical_features=[]
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


def test_a_prediction_needs_at_least_one_value():
    with pytest.raises(ValueError, match="at least one predicted value"):
        Prediction(predicted=np.array([]))


def test_every_field_must_describe_the_same_samples():
    """Misaligned fields would silently score one sample's probability against
    another's label."""
    with pytest.raises(ValueError, match="must describe the same samples"):
        Prediction(predicted=np.zeros(10), probabilities=np.zeros((8, 2)))


def test_an_interval_needs_both_bounds():
    with pytest.raises(ValueError, match="need both bounds"):
        Prediction(predicted=np.zeros(5), lower=np.zeros(5))


def test_an_inverted_interval_is_rejected():
    with pytest.raises(ValueError, match="at or above"):
        Prediction(predicted=np.zeros(3), lower=np.ones(3), upper=np.zeros(3))


def test_available_reports_only_what_is_present():
    labels_only = Prediction(predicted=np.zeros(4))
    assert sorted(labels_only.available()) == ["predicted"]

    richer = Prediction(
        predicted=np.zeros(4),
        probabilities=np.full((4, 2), 0.5),
        embeddings=np.zeros((4, 8)),
    )
    assert sorted(richer.available()) == ["embeddings", "predicted", "probabilities"]


def test_implied_access_reads_the_outputs():
    assert Prediction(predicted=np.zeros(3)).implied_access() is AccessLevel.BLACK_BOX
    assert (
        Prediction(predicted=np.zeros(3), probabilities=np.full((3, 2), 0.5))
        .implied_access()
        is AccessLevel.PROBABILISTIC
    )
    assert (
        Prediction(predicted=np.zeros(3), lower=np.zeros(3), upper=np.ones(3))
        .implied_access()
        is AccessLevel.PROBABILISTIC
    )
    assert (
        Prediction(predicted=np.zeros(3), embeddings=np.zeros((3, 4)))
        .implied_access()
        is AccessLevel.EMBEDDING
    )


def test_take_subsets_every_field_together():
    prediction = Prediction(
        predicted=np.arange(6),
        probabilities=np.tile([0.3, 0.7], (6, 1)),
        lower=np.zeros(6),
        upper=np.ones(6),
    )
    subset = prediction.take([0, 2, 4])
    assert len(subset) == 3
    assert subset.predicted.tolist() == [0, 2, 4]
    assert subset.probabilities.shape == (3, 2)
    assert subset.lower.shape == (3,)


def test_take_accepts_a_boolean_mask():
    prediction = Prediction(predicted=np.arange(5))
    mask = np.array([True, False, True, False, True])
    assert prediction.take(mask).predicted.tolist() == [0, 2, 4]


def test_take_rejects_a_mask_of_the_wrong_length():
    with pytest.raises(ValueError, match="must cover all 5"):
        Prediction(predicted=np.arange(5)).take(np.ones(3, dtype=bool))


def test_take_rejects_an_empty_subset():
    with pytest.raises(ValueError, match="empty subset"):
        Prediction(predicted=np.arange(5)).take(np.zeros(5, dtype=bool))


def test_take_does_not_re_predict():
    """Subsetting rather than re-predicting is what keeps a stochastic perturbation's
    subgroup comparison honest — a second call would draw different noise."""
    calls = []

    class Counting(BaseHandler):
        def predict(self, data):
            calls.append(1)
            return Prediction(predicted=np.zeros(data.n_entities))

    handler = Counting(None, name="counting")
    prediction = handler.predict(
        ArrayAdapter(np.zeros((10, 2)), feature_names=["a", "b"], categorical_features=[])
    )
    prediction.take(np.arange(5))
    assert len(calls) == 1


# --------------------------------------------------------------------------
# BaseHandler
# --------------------------------------------------------------------------


def test_a_handler_satisfying_the_protocol_needs_no_inheritance():
    """Repo D's RoBERTa handler did not inherit its own base class. Under a Protocol
    that is fine, which is the point."""

    class Standalone:
        name = "standalone"
        task = TaskType.CLASSIFICATION
        access = AccessLevel.BLACK_BOX

        def predict(self, data):
            return Prediction(predicted=np.zeros(data.n_entities))

    assert isinstance(Standalone(), ModelHandler)


def test_check_prediction_catches_a_length_mismatch(data):
    class Wrong(BaseHandler):
        def predict(self, d):
            return self.check_prediction(Prediction(predicted=np.zeros(3)), d)

    with pytest.raises(ValueError, match="returned 3 prediction"):
        Wrong(None, name="wrong").predict(data)


def test_check_prediction_catches_an_over_claimed_access_level(data):
    """The declaration is what the runner promises the user, so an over-claim is a lie
    the runner would repeat."""

    class Overclaiming(BaseHandler):
        def predict(self, d):
            return self.check_prediction(
                Prediction(predicted=np.zeros(d.n_entities)), d
            )

    handler = Overclaiming(None, name="overclaimer", access=AccessLevel.PROBABILISTIC)
    with pytest.raises(ValueError, match="declares access=PROBABILISTIC"):
        handler.predict(data)


def test_check_prediction_allows_returning_more_than_declared(data):
    """Declaring less than you deliver is legitimate — evaluating a probabilistic model
    under black-box conditions is a thing someone might want."""

    class Modest(BaseHandler):
        def predict(self, d):
            return self.check_prediction(
                Prediction(
                    predicted=np.zeros(d.n_entities),
                    probabilities=np.full((d.n_entities, 2), 0.5),
                ),
                d,
            )

    handler = Modest(None, name="modest", access=AccessLevel.BLACK_BOX)
    assert handler.predict(data).probabilities is not None


def test_base_handler_defaults_its_name_to_the_model_type():
    class Dummy:
        pass

    assert BaseHandler(Dummy()).name == "Dummy"


def test_the_base_predict_is_not_implemented():
    with pytest.raises(NotImplementedError, match="must implement predict"):
        BaseHandler(None).predict(None)
