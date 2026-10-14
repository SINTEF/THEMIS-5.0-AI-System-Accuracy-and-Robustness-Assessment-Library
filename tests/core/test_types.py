"""Tests for PerturbationSpec and the enums it is built from."""

from __future__ import annotations

import pytest

from themis_assess.core.types import (
    DEFAULT_SEVERITY_LEVELS,
    AccessLevel,
    Alignment,
    Modality,
    PerturbationSpec,
    Scope,
    TaskType,
    TimeScope,
)


def test_default_severity_levels_is_five():
    """ImageNet-C's convention; docs/DECISIONS.md, D16."""
    assert DEFAULT_SEVERITY_LEVELS == 5


def test_access_levels_are_ordered_and_comparable():
    assert AccessLevel.BLACK_BOX < AccessLevel.PROBABILISTIC < AccessLevel.GRADIENT
    assert AccessLevel.EMBEDDING.satisfies(AccessLevel.PROBABILISTIC)
    assert not AccessLevel.PROBABILISTIC.satisfies(AccessLevel.EMBEDDING)


def test_enums_are_string_valued_for_readable_config():
    assert Modality.SEQUENCE.value == "sequence"
    assert Scope.NUMERICAL.value == "numerical"
    assert TimeScope.PER_TIMESTEP.value == "per_timestep"
    assert TaskType.FORECASTING.value == "forecasting"
    assert Alignment.END.value == "end"


# --------------------------------------------------------------------------
# PerturbationSpec
# --------------------------------------------------------------------------


def test_minimal_spec_has_sensible_defaults():
    spec = PerturbationSpec(name="gaussian_noise")
    assert spec.severity == 1
    assert spec.scope is Scope.ALL_FEATURES
    assert spec.time_scope is TimeScope.CONSTANT
    assert spec.n_variants == 1


def test_key_is_a_label_not_something_to_parse():
    """Replaces the "normal_0_01" keys the old compute_robustness split on "_"."""
    spec = PerturbationSpec(name="perturb_only_numerical", severity=3)
    assert spec.key == "perturb_only_numerical@3"
    # The point: the underscore-laden name is preserved intact, and severity is a field.
    assert spec.name == "perturb_only_numerical"
    assert spec.severity == 3


def test_key_survives_a_name_with_no_underscore():
    """The old parser raised IndexError on this."""
    assert PerturbationSpec(name="jitter", severity=2).key == "jitter@2"


def test_at_severity_returns_a_copy_with_everything_else_intact():
    spec = PerturbationSpec(
        name="jitter",
        severity=1,
        scope=Scope.NAMED,
        features=("speed",),
        time_scope=TimeScope.AT_INDICES,
        time_indices=(0, 2),
        n_variants=7,
        params={"distribution": "alpha_stable"},
    )
    other = spec.at_severity(4)
    assert other.severity == 4
    assert other.name == spec.name
    assert other.features == spec.features
    assert other.time_indices == spec.time_indices
    assert other.n_variants == 7
    assert other.params == spec.params
    assert spec.severity == 1  # original untouched


def test_specs_are_immutable():
    spec = PerturbationSpec(name="jitter")
    with pytest.raises(AttributeError):
        spec.severity = 2  # type: ignore[misc]


def test_params_are_frozen_so_a_shared_spec_cannot_be_mutated():
    params = {"distribution": "gaussian"}
    spec = PerturbationSpec(name="jitter", params=params)
    with pytest.raises(TypeError):
        spec.params["distribution"] = "alpha_stable"  # type: ignore[index]
    params["distribution"] = "alpha_stable"  # mutating the original must not leak in
    assert spec.params["distribution"] == "gaussian"


def test_fingerprint_is_hashable_and_respects_equality():
    """Specs are not hashable (params is a mapping); fingerprint is the cache key."""
    a = PerturbationSpec(name="jitter", severity=2, params={"b": 1, "a": 2})
    b = PerturbationSpec(name="jitter", severity=2, params={"a": 2, "b": 1})
    assert a == b
    assert a.fingerprint == b.fingerprint
    assert len({a.fingerprint, b.fingerprint}) == 1
    assert a.fingerprint != a.at_severity(3).fingerprint


def test_severity_is_one_based():
    with pytest.raises(ValueError, match="1-based"):
        PerturbationSpec(name="jitter", severity=0)


def test_n_variants_must_be_positive():
    with pytest.raises(ValueError, match="n_variants must be at least 1"):
        PerturbationSpec(name="jitter", n_variants=0)


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty name"):
        PerturbationSpec(name="")


def test_named_scope_requires_features():
    with pytest.raises(ValueError, match="scope=NAMED requires"):
        PerturbationSpec(name="jitter", scope=Scope.NAMED)


def test_features_without_named_scope_is_rejected():
    """Silently ignoring `features` would make a mis-specified run look like it worked."""
    with pytest.raises(ValueError, match="only meaningful with scope=NAMED"):
        PerturbationSpec(name="jitter", scope=Scope.NUMERICAL, features=("speed",))


def test_at_indices_requires_indices():
    with pytest.raises(ValueError, match="AT_INDICES requires"):
        PerturbationSpec(name="jitter", time_scope=TimeScope.AT_INDICES)


def test_indices_without_at_indices_scope_is_rejected():
    with pytest.raises(ValueError, match="only meaningful with time_scope=AT_INDICES"):
        PerturbationSpec(name="jitter", time_indices=(0, 1))


def test_negative_time_indices_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        PerturbationSpec(
            name="jitter", time_scope=TimeScope.AT_INDICES, time_indices=(0, -1)
        )


def test_scope_and_time_scope_express_repo_bs_sixteen_methods():
    """A sample of repo B's method names, each as one specification."""
    cases = {
        "perturb_only_numerical": PerturbationSpec(
            name="gaussian_noise", scope=Scope.NUMERICAL, time_scope=TimeScope.CONSTANT
        ),
        "perturb_only_numerical_over_time": PerturbationSpec(
            name="gaussian_noise", scope=Scope.NUMERICAL, time_scope=TimeScope.PER_TIMESTEP
        ),
        "perturb_only_categorical": PerturbationSpec(
            name="category_swap", scope=Scope.CATEGORICAL, time_scope=TimeScope.CONSTANT
        ),
        "perturb_all_data": PerturbationSpec(
            name="gaussian_noise", scope=Scope.ALL_FEATURES
        ),
        "perturb_one_feature_at_k_times": PerturbationSpec(
            name="gaussian_noise",
            scope=Scope.NAMED,
            features=("speed",),
            time_scope=TimeScope.AT_INDICES,
            time_indices=(0, 3, 7),
        ),
    }
    # All valid, all distinct, none needing its own class.
    assert len({s.key for s in cases.values()}) == 2  # two methods, five configurations
    assert len(cases) == 5
