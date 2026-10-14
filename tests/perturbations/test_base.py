"""Tests for the perturbation contract.

No perturbation implementations exist yet, so these exercise the contract through a
minimal in-test subclass. That is the point of a frozen contract: it can be verified
before anything is built on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter, TextAdapter
from themis_assess.core.types import (
    Modality,
    PerturbationSpec,
    Scope,
    TaskType,
    TimeScope,
)
from themis_assess.perturbations.base import Perturbation, PerturbationRegistry


class AddSeverity(Perturbation):
    """Adds ``severity`` to every selected feature at every selected timestep."""

    name = "add_severity"
    severity_levels = 5

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        features = self.resolve_features(data, spec)
        timesteps = self.resolve_timesteps(data, spec)
        out = []
        for _ in range(spec.n_variants):
            values = data.values()
            for feature in features:
                j = data.feature_index(feature)
                for t in timesteps:
                    values[:, t, j] = values[:, t, j] + spec.severity
            out.append(data.with_values(values))
        return out

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["offset"] = float(spec.severity)
        return meta


class TextOnly(Perturbation):
    name = "text_only"
    modalities = frozenset({Modality.TEXT})

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        return [data]


class ClassificationOnly(Perturbation):
    name = "clf_only"
    task_types = frozenset({TaskType.CLASSIFICATION})

    def perturb(self, data, spec, rng):
        return [data]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def sequence():
    values = np.array([[[1.0, 100.0], [2.0, 100.0], [3.0, 100.0]]])
    return ArrayAdapter(values, ["speed", "vessel_type"], categorical_features=["vessel_type"])


# --------------------------------------------------------------------------
# subclass contract
# --------------------------------------------------------------------------


def test_a_concrete_subclass_must_declare_a_name_to_be_instantiated():
    class Nameless(Perturbation):
        def perturb(self, data, spec, rng):
            return [data]

    with pytest.raises(TypeError, match="must declare a class-level `name`"):
        Nameless()


def test_an_abstract_subclass_need_not_declare_a_name():
    """An intermediate base sharing helpers is legitimate and has no name of its own."""

    class SharedBase(Perturbation):
        def _helper(self):
            return 1

    assert SharedBase.name == ""
    with pytest.raises(TypeError, match="abstract"):
        SharedBase()  # still abstract: perturb is not implemented


def test_a_named_subclass_of_an_abstract_base_instantiates():
    class SharedBase(Perturbation):
        def _helper(self):
            return 1

    class Concrete(SharedBase):
        name = "concrete"

        def perturb(self, data, spec, rng):
            return [data]

    assert Concrete()._helper() == 1


def test_perturb_returns_n_variants_copies(sequence, rng):
    p = AddSeverity()
    out = p.perturb(sequence, p.spec(severity=1, n_variants=3), rng)
    assert len(out) == 3
    assert all(v.values().shape == sequence.values().shape for v in out)


def test_perturb_does_not_mutate_the_input(sequence, rng):
    p = AddSeverity()
    before = sequence.values().copy()
    p.perturb(sequence, p.spec(severity=4), rng)
    assert np.array_equal(sequence.values(), before)


# --------------------------------------------------------------------------
# scope resolution — repo B's sixteen methods, as configurations
# --------------------------------------------------------------------------


def test_numerical_scope_leaves_categorical_features_alone(sequence, rng):
    p = AddSeverity()
    out = p.perturb(sequence, p.spec(severity=5, scope=Scope.NUMERICAL), rng)[0]
    assert out.values()[0, :, 0].tolist() == [6.0, 7.0, 8.0]
    assert out.values()[0, :, 1].tolist() == [100.0, 100.0, 100.0]


def test_categorical_scope_leaves_numerical_features_alone(sequence, rng):
    p = AddSeverity()
    out = p.perturb(sequence, p.spec(severity=5, scope=Scope.CATEGORICAL), rng)[0]
    assert out.values()[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.values()[0, :, 1].tolist() == [105.0, 105.0, 105.0]


def test_all_features_scope_touches_everything(sequence, rng):
    p = AddSeverity()
    out = p.perturb(sequence, p.spec(severity=1, scope=Scope.ALL_FEATURES), rng)[0]
    assert out.values()[0, :, 0].tolist() == [2.0, 3.0, 4.0]
    assert out.values()[0, :, 1].tolist() == [101.0, 101.0, 101.0]


def test_named_scope_touches_only_the_listed_features(sequence, rng):
    p = AddSeverity()
    spec = p.spec(severity=2, scope=Scope.NAMED, features=("vessel_type",))
    out = p.perturb(sequence, spec, rng)[0]
    assert out.values()[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert out.values()[0, :, 1].tolist() == [102.0, 102.0, 102.0]


def test_at_indices_touches_only_the_listed_timesteps(sequence, rng):
    p = AddSeverity()
    spec = p.spec(
        severity=10 if False else 3,
        scope=Scope.NUMERICAL,
        time_scope=TimeScope.AT_INDICES,
        time_indices=(0, 2),
    )
    out = p.perturb(sequence, spec, rng)[0]
    assert out.values()[0, :, 0].tolist() == [4.0, 2.0, 6.0]


def test_resolve_features_matches_the_scope(sequence):
    p = AddSeverity()
    assert p.resolve_features(sequence, p.spec(scope=Scope.NUMERICAL)) == ["speed"]
    assert p.resolve_features(sequence, p.spec(scope=Scope.CATEGORICAL)) == ["vessel_type"]
    assert p.resolve_features(sequence, p.spec()) == ["speed", "vessel_type"]


def test_resolve_timesteps_matches_the_time_scope(sequence):
    p = AddSeverity()
    assert p.resolve_timesteps(sequence, p.spec()) == [0, 1, 2]
    spec = p.spec(time_scope=TimeScope.AT_INDICES, time_indices=(1,))
    assert p.resolve_timesteps(sequence, spec) == [1]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_validate_rejects_a_spec_naming_another_perturbation(sequence):
    with pytest.raises(ValueError, match="but this is 'add_severity'"):
        AddSeverity().validate(sequence, PerturbationSpec(name="something_else"))


def test_validate_rejects_a_severity_beyond_the_declared_levels(sequence):
    p = AddSeverity()
    with pytest.raises(ValueError, match="exposes 5 severity level"):
        p.validate(sequence, p.spec(severity=6))


def test_validate_rejects_an_unknown_feature(sequence):
    p = AddSeverity()
    spec = p.spec(scope=Scope.NAMED, features=("draught",))
    with pytest.raises(ValueError, match="feature\\(s\\) not in the data"):
        p.validate(sequence, spec)


def test_validate_rejects_out_of_range_timesteps(sequence):
    p = AddSeverity()
    spec = p.spec(time_scope=TimeScope.AT_INDICES, time_indices=(9,))
    with pytest.raises(ValueError, match="exceed the 3 timestep"):
        p.validate(sequence, spec)


def test_validate_rejects_a_time_scope_on_tabular_data():
    p = AddSeverity()
    tabular = ArrayAdapter(np.zeros((2, 2)), ["a", "b"])
    spec = p.spec(time_scope=TimeScope.PER_TIMESTEP)
    with pytest.raises(ValueError, match="needs sequential data"):
        p.validate(tabular, spec)


def test_validate_rejects_a_modality_the_perturbation_does_not_serve(sequence):
    p = TextOnly()
    with pytest.raises(ValueError, match="applies to \\['text'\\]"):
        p.validate(sequence, p.spec())


# --------------------------------------------------------------------------
# applicability
# --------------------------------------------------------------------------


def test_applies_to_checks_modality_and_task(sequence):
    text = TextAdapter(["a document"])
    assert AddSeverity().applies_to(sequence)
    assert not AddSeverity().applies_to(text)
    assert TextOnly().applies_to(text)
    assert ClassificationOnly().applies_to(sequence, TaskType.CLASSIFICATION)
    assert not ClassificationOnly().applies_to(sequence, TaskType.FORECASTING)


# --------------------------------------------------------------------------
# sequences — survey Eqs. 23-25
# --------------------------------------------------------------------------


def test_sequence_walks_severities_in_increasing_order(sequence, rng):
    p = AddSeverity()
    frames = p.sequence(sequence, p.spec(scope=Scope.NUMERICAL), rng)
    assert len(frames) == 5
    first_values = [f.values()[0, 0, 0] for f in frames]
    assert first_values == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert first_values == sorted(first_values)


def test_sequence_forces_one_variant_per_frame(sequence, rng):
    """Frame i must be the same input at severity i+1, not a fresh random draw."""
    p = AddSeverity()
    frames = p.sequence(sequence, p.spec(n_variants=9), rng)
    assert len(frames) == p.severity_levels


def test_sequence_accepts_explicit_levels(sequence, rng):
    p = AddSeverity()
    frames = p.sequence(sequence, p.spec(scope=Scope.NUMERICAL), rng, levels=[1, 3, 5])
    assert [f.values()[0, 0, 0] for f in frames] == [2.0, 4.0, 6.0]


def test_sequence_needs_at_least_two_levels(sequence, rng):
    p = AddSeverity()
    with pytest.raises(ValueError, match="at least 2 severity levels"):
        p.sequence(sequence, p.spec(), rng, levels=[3])


def test_sequence_rejects_non_monotonic_levels(sequence, rng):
    p = AddSeverity()
    with pytest.raises(ValueError, match="increase monotonically"):
        p.sequence(sequence, p.spec(), rng, levels=[3, 1])


def test_severity_curve_yields_every_level(sequence, rng):
    p = AddSeverity()
    got = list(p.severity_curve(sequence, p.spec(scope=Scope.NUMERICAL), rng))
    assert [level for level, _ in got] == [1, 2, 3, 4, 5]
    assert all(len(variants) == 1 for _, variants in got)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def test_metadata_describes_the_perturbation():
    meta = AddSeverity().metadata()
    assert meta["name"] == "add_severity"
    assert meta["severity_levels"] == 5
    assert meta["label_preserving"] is True
    assert meta["modalities"] == ["sequence", "tabular"]


def test_metadata_states_what_a_severity_meant_when_given_a_spec():
    """A report cannot say 'severity 3' and stop; it must say what that was."""
    p = AddSeverity()
    assert p.metadata(p.spec(severity=3))["offset"] == pytest.approx(3.0)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_holds_and_retrieves():
    registry = PerturbationRegistry([AddSeverity(), TextOnly()])
    assert len(registry) == 2
    assert registry.names() == ["add_severity", "text_only"]
    assert isinstance(registry["add_severity"], AddSeverity)
    assert "text_only" in registry


def test_registry_rejects_a_duplicate_without_overwrite():
    registry = PerturbationRegistry([AddSeverity()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(AddSeverity())
    registry.register(AddSeverity(), overwrite=True)
    assert len(registry) == 1


def test_registry_suggests_near_matches():
    registry = PerturbationRegistry([AddSeverity()])
    with pytest.raises(KeyError, match="Did you mean: add_severity"):
        registry["severity"]


def test_registry_names_what_is_available_when_empty():
    with pytest.raises(KeyError, match="none registered"):
        PerturbationRegistry()["anything"]


def test_registry_filters_by_what_can_act_on_the_data(sequence):
    registry = PerturbationRegistry([AddSeverity(), TextOnly(), ClassificationOnly()])
    text = TextAdapter(["doc"])
    assert registry.applicable(sequence) == ["add_severity", "clf_only"]
    assert registry.applicable(text) == ["text_only"]
    assert registry.applicable(sequence, TaskType.FORECASTING) == ["add_severity"]


def test_registry_does_not_decide_what_to_run(sequence):
    """Repo D's note: selection is the caller's responsibility. The registry only lists."""
    registry = PerturbationRegistry([AddSeverity(), ClassificationOnly()])
    assert not hasattr(registry, "run")
    assert not hasattr(registry, "perturb")


def test_registry_collects_metadata_for_reporting():
    registry = PerturbationRegistry([AddSeverity(), TextOnly()])
    meta = registry.metadata()
    assert set(meta) == {"add_severity", "text_only"}
    assert meta["text_only"]["modalities"] == ["text"]
