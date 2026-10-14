"""Tests for signature-based dispatch (ported from repo D and repo A's main)."""

from __future__ import annotations

import pytest

from themis_assess.core.dispatch import (
    call_with_available,
    can_call,
    compute_available,
    missing_arguments,
    optional_parameters,
    required_parameters,
    unavailable,
)


def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


def roc_auc(y_true, y_prob, average="macro"):
    return 0.5 + 0.1 * len(average)


def takes_kwargs(y_true, **kwargs):
    return sorted(kwargs)


POOL = {"y_true": [0, 1, 1], "y_pred": [0, 1, 0], "sample_weight": [1, 1, 1]}


def test_required_parameters_excludes_defaults_and_varargs():
    assert required_parameters(roc_auc) == ["y_true", "y_prob"]
    assert optional_parameters(roc_auc) == ["average"]
    assert required_parameters(takes_kwargs) == ["y_true"]


def test_required_parameters_skips_self_on_a_method():
    class Metric:
        def __call__(self, y_true, y_pred):
            return 0.0

        def score(self, y_true, y_pred):
            return 0.0

    assert required_parameters(Metric.score) == ["y_true", "y_pred"]
    assert required_parameters(Metric().score) == ["y_true", "y_pred"]


def test_missing_arguments_names_what_is_absent():
    assert missing_arguments(accuracy, POOL) == []
    assert missing_arguments(roc_auc, POOL) == ["y_prob"]


def test_can_call_agrees_with_missing_arguments():
    assert can_call(accuracy, POOL)
    assert not can_call(roc_auc, POOL)


def test_call_with_available_passes_only_accepted_arguments():
    """`sample_weight` is in the pool but not in the signature; it must not leak through."""
    assert call_with_available(accuracy, POOL) == pytest.approx(2 / 3)


def test_call_with_available_supplies_optional_arguments_when_present():
    pool = {"y_true": [0], "y_prob": [0.9], "average": "weighted"}
    assert call_with_available(roc_auc, pool) == pytest.approx(0.5 + 0.1 * len("weighted"))


def test_call_with_available_gives_everything_to_a_kwargs_function():
    assert call_with_available(takes_kwargs, POOL) == ["sample_weight", "y_pred"]


def test_call_with_available_names_the_missing_argument():
    with pytest.raises(TypeError, match=r"missing required argument\(s\) \['y_prob'\]"):
        call_with_available(roc_auc, POOL)


def test_compute_available_is_strict_by_default():
    """A silently omitted metric produces a quietly incomplete report."""
    with pytest.raises(TypeError, match="y_prob"):
        compute_available({"acc": accuracy, "auc": roc_auc}, POOL)


def test_compute_available_can_skip_and_preserves_order():
    out = compute_available(
        {"acc": accuracy, "auc": roc_auc, "kw": takes_kwargs},
        POOL,
        skip_unavailable=True,
    )
    assert list(out) == ["acc", "kw"]


def test_unavailable_reports_what_was_skipped_and_why():
    report = unavailable({"acc": accuracy, "auc": roc_auc}, POOL)
    assert report == {"auc": ["y_prob"]}


def test_unavailable_is_empty_when_everything_can_run():
    assert unavailable({"acc": accuracy}, POOL) == {}


def test_uninspectable_objects_raise_a_clear_error():
    with pytest.raises(TypeError, match="Cannot inspect the signature"):
        required_parameters(object())  # type: ignore[arg-type]


def test_a_callable_instance_works_like_a_function():
    """sklearn-style metric objects and functools.partial both turn up in practice."""

    class WeightedAccuracy:
        def __call__(self, y_true, y_pred):
            return 1.0

    metric = WeightedAccuracy()
    assert required_parameters(metric) == ["y_true", "y_pred"]
    assert call_with_available(metric, POOL) == pytest.approx(1.0)


def test_functools_partial_reports_the_remaining_parameters():
    from functools import partial

    fixed = partial(roc_auc, average="weighted")
    assert required_parameters(fixed) == ["y_true", "y_prob"]


def test_dispatch_complements_rather_than_replaces_the_declared_access_level():
    """The declaration warns up front; this dispatches once the run is under way.

    Here the pool lacks probabilities, so the probability-hungry metric is skipped —
    the same conclusion the registry's `unavailable_at(BLACK_BOX)` reaches statically.
    """
    from themis_assess.core.types import AccessLevel
    from themis_assess.metrics.registry import default_registry

    statically_blocked = set(default_registry().unavailable_at(AccessLevel.BLACK_BOX))
    assert "auroc" in statically_blocked

    dynamically_skipped = set(
        unavailable({"accuracy": accuracy, "auroc_like": roc_auc}, POOL)
    )
    assert dynamically_skipped == {"auroc_like"}
