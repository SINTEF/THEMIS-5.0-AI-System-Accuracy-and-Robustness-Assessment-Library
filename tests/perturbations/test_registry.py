"""Tests for the built-in registry, applicability filtering and cost estimation."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter, TextAdapter
from themis_assess.core.types import TaskType
from themis_assess.perturbations.registry import (
    BUILTIN_PERTURBATIONS,
    default_registry,
    estimate_cost,
)


@pytest.fixture
def tabular():
    gen = np.random.default_rng(0)
    kind = np.repeat([0.0, 1.0], 100)
    values = np.column_stack([gen.normal(0, 1, 200), gen.normal(0, 1, 200), kind])
    return ArrayAdapter(values, ["a", "b", "kind"], categorical_features=["kind"])


@pytest.fixture
def sequence():
    gen = np.random.default_rng(1)
    values = gen.normal(0, 1, (5, 20, 3))
    values[:, :, 2] = np.arange(5).reshape(-1, 1)  # constant per entity -> categorical
    return ArrayAdapter(
        values, ["speed", "load", "kind"], categorical_features=["kind"]
    )


def test_every_builtin_registers_without_a_name_clash():
    registry = default_registry()
    assert len(registry) == len(BUILTIN_PERTURBATIONS)
    assert len(registry.names()) == len(set(registry.names()))


def test_each_call_returns_an_independent_registry():
    a, b = default_registry(), default_registry()
    a.register(a["gaussian_noise"], overwrite=True)
    assert len(a) == len(b)
    assert a is not b
    assert a["gaussian_noise"] is not None


def test_every_builtin_declares_a_severity_scale_of_the_right_length():
    for perturbation in default_registry().perturbations():
        assert len(perturbation.severity_scale) == perturbation.severity_levels


def test_every_builtin_maps_all_five_severities_to_a_number():
    for perturbation in default_registry().perturbations():
        factors = [
            perturbation.severity_factor(perturbation.spec(severity=s))
            for s in range(1, perturbation.severity_levels + 1)
        ]
        assert all(isinstance(f, float) for f in factors)
        assert factors == sorted(factors), perturbation.name


def test_every_builtin_reports_metadata_without_a_spec():
    for perturbation in default_registry().perturbations():
        meta = perturbation.metadata()
        assert meta["name"] == perturbation.name
        assert "severity_scale" in meta


# --------------------------------------------------------------------------
# applicability
# --------------------------------------------------------------------------


def test_sequence_only_perturbations_are_not_offered_for_tabular_data(tabular):
    applicable = set(default_registry().applicable(tabular))
    for name in (
        "sensor_drift",
        "exponential_damping",
        "flat_sensor",
        "outlier_spikes",
        "missing_segment",
        "alpha_stable_noise",
    ):
        assert name not in applicable


def test_every_non_text_perturbation_is_offered_for_sequence_data(sequence):
    registry = default_registry()
    text_only = {"character_noise", "token_masking", "word_dropout"}
    assert set(registry.applicable(sequence)) == set(registry.names()) - text_only


def test_categorical_perturbations_are_not_offered_without_categorical_features():
    """Otherwise the caller only finds out when perturb() raises."""
    numeric_only = ArrayAdapter(np.random.default_rng(2).normal(0, 1, (50, 2)), ["a", "b"])
    assert "category_swap" not in default_registry().applicable(numeric_only)


def test_numeric_perturbations_are_not_offered_without_numerical_features():
    values = np.column_stack([np.repeat([0.0, 1.0], 15), np.repeat([2.0, 3.0], 15)])
    all_categorical = ArrayAdapter(values, ["p", "q"], categorical_features=["p", "q"])
    applicable = set(default_registry().applicable(all_categorical))
    assert "gaussian_noise" not in applicable
    assert "category_swap" in applicable


def test_only_text_perturbations_are_offered_for_text():
    applicable = default_registry().applicable(TextAdapter(["a document", "another"]))
    assert set(applicable) == {"character_noise", "token_masking", "word_dropout"}


def test_text_perturbations_are_not_offered_for_arrays():
    numeric = ArrayAdapter(np.random.default_rng(13).normal(0, 1, (40, 2)), ["a", "b"])
    applicable = set(default_registry().applicable(numeric))
    assert applicable.isdisjoint({"character_noise", "token_masking", "word_dropout"})


def test_task_filtering_passes_through(sequence):
    registry = default_registry()
    for task in (TaskType.CLASSIFICATION, TaskType.FORECASTING):
        assert registry.applicable(sequence, task) == registry.applicable(sequence)


# --------------------------------------------------------------------------
# every applicable perturbation actually runs
# --------------------------------------------------------------------------


def test_every_applicable_perturbation_runs_on_sequence_data(sequence):
    registry = default_registry()
    rng = np.random.default_rng(3)
    for name in registry.applicable(sequence):
        perturbation = registry[name]
        for severity in (1, 3, 5):
            out = perturbation.perturb(
                sequence, perturbation.spec(severity=severity), rng
            )
            assert len(out) == 1
            assert out[0].values().shape == sequence.values().shape


def test_every_applicable_perturbation_runs_on_tabular_data(tabular):
    registry = default_registry()
    rng = np.random.default_rng(4)
    for name in registry.applicable(tabular):
        perturbation = registry[name]
        out = perturbation.perturb(tabular, perturbation.spec(severity=3), rng)
        assert out[0].values().shape == tabular.values().shape


def test_every_applicable_perturbation_actually_changes_the_data(sequence):
    registry = default_registry()
    rng = np.random.default_rng(5)
    for name in registry.applicable(sequence):
        perturbation = registry[name]
        out = perturbation.perturb(sequence, perturbation.spec(severity=5), rng)[0]
        same = np.array_equal(out.values(), sequence.values(), equal_nan=True)
        assert not same, f"{name} left the data unchanged at severity 5"


def test_severity_sequences_work_for_every_applicable_perturbation(sequence):
    """Required by the flip-rate family (survey Eqs. 23-25)."""
    registry = default_registry()
    rng = np.random.default_rng(6)
    for name in registry.applicable(sequence):
        perturbation = registry[name]
        frames = perturbation.sequence(sequence, perturbation.spec(), rng)
        assert len(frames) == perturbation.severity_levels


# --------------------------------------------------------------------------
# cost estimation
# --------------------------------------------------------------------------


def test_estimate_cost_sums_model_calls(tabular):
    registry = default_registry()
    specs = [
        registry["gaussian_noise"].spec(severity=s, n_variants=10) for s in range(1, 6)
    ]
    cost = estimate_cost(registry, tabular, specs)
    assert cost["total"] == 50
    assert cost["gaussian_noise@1"] == 10


def test_estimate_cost_omits_inapplicable_perturbations(tabular):
    """So the total is what will actually be spent, not an optimistic count."""
    registry = default_registry()
    specs = [
        registry["gaussian_noise"].spec(severity=1, n_variants=4),
        registry["sensor_drift"].spec(severity=1, n_variants=100),  # sequence-only
    ]
    cost = estimate_cost(registry, tabular, specs)
    assert "sensor_drift@1" not in cost
    assert cost["total"] == 4


def test_estimate_cost_covers_a_full_severity_sweep(sequence):
    """The number repo C's notebook wished it had before the 15-minute run."""
    registry = default_registry()
    specs = [
        registry[name].spec(severity=s, n_variants=10)
        for name in registry.applicable(sequence)
        for s in range(1, 6)
    ]
    cost = estimate_cost(registry, sequence, specs)
    applicable = registry.applicable(sequence)
    # input_window_shift is deterministic and costs one call whatever n_variants says
    expected = sum(
        5 * (1 if name == "input_window_shift" else 10) for name in applicable
    )
    assert cost["total"] == expected


def test_estimate_cost_keys_are_the_spec_keys(tabular):
    registry = default_registry()
    spec = registry["gaussian_noise"].spec(severity=2)
    cost = estimate_cost(registry, tabular, [spec])
    assert spec.key in cost


# --------------------------------------------------------------------------
# applicable() must never promise something perturb() refuses
# --------------------------------------------------------------------------


def test_correlated_noise_is_not_offered_for_single_feature_data():
    one = ArrayAdapter(np.random.default_rng(7).normal(0, 1, (100, 1)), ["only"])
    assert "correlated_noise" not in default_registry().applicable(one)


def test_category_swap_is_not_offered_when_a_category_has_no_alternative():
    values = np.column_stack([np.random.default_rng(8).normal(0, 1, 60), np.ones(60)])
    data = ArrayAdapter(values, ["a", "kind"], categorical_features=["kind"])
    assert data.categorical_features == ["kind"]
    assert "category_swap" not in default_registry().applicable(data)


@pytest.mark.parametrize(
    "data",
    [
        ArrayAdapter(np.random.default_rng(9).normal(0, 1, (80, 1)), ["only"]),
        ArrayAdapter(
            np.column_stack([np.random.default_rng(10).normal(0, 1, 80), np.ones(80)]),
            ["a", "kind"],
            categorical_features=["kind"],
        ),
        ArrayAdapter(np.random.default_rng(11).normal(0, 1, (6, 15, 2)), ["a", "b"]),
    ],
)
def test_whatever_applicable_lists_can_actually_run(data):
    """The registry's contract, checked on awkward datasets."""
    registry = default_registry()
    rng = np.random.default_rng(12)
    for name in registry.applicable(data):
        perturbation = registry[name]
        perturbation.perturb(data, perturbation.spec(severity=3), rng)
