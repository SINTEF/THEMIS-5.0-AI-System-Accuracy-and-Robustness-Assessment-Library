"""Tests for the metric glossary.

The cross-check is the point: the glossary and the registry are two halves of the same
statement about a metric, and the failure mode is drift — a metric registered with no
description, or a description for a metric nobody can compute. Both are caught here
rather than by a reader meeting a bare name in a report.
"""

from __future__ import annotations

import pytest

from themis_assess.metrics.glossary import (
    METRIC_GLOSSARY,
    describe_metric,
    glossary_for,
    missing_descriptions,
)
from themis_assess.metrics.registry import default_registry


def test_every_registered_metric_has_a_description():
    assert missing_descriptions() == []


def test_every_description_names_a_registered_metric():
    registry = default_registry()
    assert sorted(name for name in METRIC_GLOSSARY if name not in registry) == []


def test_a_card_carries_the_sentence_and_the_orientation():
    card = describe_metric("expected_calibration_error")
    assert card["higher_is_better"] is False
    assert card["lower_bound"] == 0.0
    assert "confidence" in card["description"]
    assert card["reference"]


def test_a_card_reports_the_access_level_as_a_name_not_a_number():
    """AccessLevel is an IntEnum. A card carrying access=1 is true and unreadable."""
    assert describe_metric("auc")["access"] == "PROBABILISTIC"


def test_an_unregistered_metric_has_no_card():
    with pytest.raises(KeyError):
        describe_metric("not_a_metric")


def test_a_bulk_lookup_skips_what_it_cannot_describe():
    """A report must not fail to serialise over a locally registered custom metric."""
    cards = glossary_for(["accuracy", "not_a_metric"])
    assert set(cards) == {"accuracy"}


def test_a_strict_bulk_lookup_says_so():
    with pytest.raises(KeyError):
        glossary_for(["accuracy", "not_a_metric"], strict=True)


def test_no_card_leaves_the_direction_to_be_guessed():
    """The invariant the glossary exists for: a metric value never arrives at a reader
    as a bare number. Every card pairs a description with a machine-readable direction —
    ``higher_is_better``, or a ``target`` for the metrics that are best at a value and
    worse either side of it — so nothing downstream has to parse the prose to know
    whether 0.3 is good news."""
    registry = default_registry()
    for name in registry.names():
        card = describe_metric(name, registry=registry)
        assert len(card["description"].split()) >= 12, name
        assert (
            card["higher_is_better"] is not None or card["target"] is not None
        ), name


def test_a_description_says_what_the_metric_is_not_just_what_it_is_called():
    """The failure this guards is a glossary of restated names — "accuracy: the accuracy
    of the model". Every description says more than its own name does."""
    for name, text in METRIC_GLOSSARY.items():
        words = set(text.lower().replace("-", " ").split())
        assert not words <= set(name.split("_")), name


def test_specificity_and_npv_are_deliberately_absent():
    """Both are per-class recall and precision under other names. Registering them would
    mean two names for one computation, which is how the four repos ended up unable to
    compare their own numbers."""
    registry = default_registry()
    assert "specificity" not in registry
    assert "negative_predictive_value" not in registry
    assert "negative class" in METRIC_GLOSSARY["recall"]
