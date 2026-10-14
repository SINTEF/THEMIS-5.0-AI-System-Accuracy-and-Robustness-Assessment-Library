"""Tests for the data adapters — the boundary that makes perturbations format-agnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from themis_assess.core.adapters import (
    ArrayAdapter,
    DataAdapter,
    FrameAdapter,
    TextAdapter,
    infer_categorical_features,
)
from themis_assess.core.types import Alignment, Modality


@pytest.fixture
def tabular():
    return ArrayAdapter(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), ["age", "bmi"])


@pytest.fixture
def ragged_sequence():
    """Two entities of length 3 and 4, end-aligned; one feature constant per entity."""
    values = np.full((2, 4, 2), np.nan)
    values[0, 1:, :] = [[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]]
    values[1, :, :] = [[9.0, 7.0], [8.0, 7.0], [7.0, 7.0], [6.0, 7.0]]
    return ArrayAdapter(values, ["speed", "vessel_type"], entity_lengths=[3, 4])


# --------------------------------------------------------------------------
# the protocol
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: ArrayAdapter(np.zeros((2, 3)), ["a", "b", "c"]),
        lambda: FrameAdapter(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})),
        lambda: TextAdapter(["one", "two"]),
    ],
)
def test_every_adapter_satisfies_the_protocol(make):
    assert isinstance(make(), DataAdapter)


def test_tabular_has_a_singleton_time_axis_not_a_missing_one(tabular):
    """Deliberate deviation from the sketch: n_timesteps is always an int, never None."""
    assert tabular.n_timesteps == 1
    assert tabular.values().shape == (3, 1, 2)
    assert tabular.is_sequential is False
    assert tabular.modality is Modality.TABULAR


def test_sequence_is_detected_from_the_time_axis(ragged_sequence):
    assert ragged_sequence.is_sequential is True
    assert ragged_sequence.modality is Modality.SEQUENCE
    assert ragged_sequence.n_timesteps == 4


def test_sequential_can_be_forced_for_a_length_one_sequence():
    a = ArrayAdapter(np.zeros((2, 1, 3)), ["a", "b", "c"], sequential=True)
    assert a.is_sequential is True
    assert a.modality is Modality.SEQUENCE


# --------------------------------------------------------------------------
# values, masks, ragged entities
# --------------------------------------------------------------------------


def test_values_returns_a_copy_so_callers_cannot_mutate_the_adapter(tabular):
    v = tabular.values()
    v[0, 0, 0] = 999.0
    assert tabular.values()[0, 0, 0] == pytest.approx(1.0)


def test_valid_mask_is_all_true_for_equal_lengths(tabular):
    assert tabular.valid_mask().all()


def test_end_alignment_puts_padding_on_the_left(ragged_sequence):
    expected = np.array([[False, True, True, True], [True, True, True, True]])
    assert np.array_equal(ragged_sequence.valid_mask(), expected)


def test_start_alignment_puts_padding_on_the_right():
    values = np.full((2, 4, 1), np.nan)
    values[0, :3, 0] = [1.0, 2.0, 3.0]
    values[1, :, 0] = [1.0, 2.0, 3.0, 4.0]
    a = ArrayAdapter(values, ["x"], entity_lengths=[3, 4], alignment=Alignment.START)
    expected = np.array([[True, True, True, False], [True, True, True, True]])
    assert np.array_equal(a.valid_mask(), expected)


def test_entity_lengths_are_reported_and_copied(ragged_sequence):
    lengths = ragged_sequence.entity_lengths
    lengths[0] = 99
    assert ragged_sequence.entity_lengths[0] == 3


def test_entity_lengths_beyond_the_time_axis_are_rejected():
    with pytest.raises(ValueError, match="entity_lengths must lie in"):
        ArrayAdapter(np.zeros((2, 3, 1)), ["x"], entity_lengths=[3, 9])


# --------------------------------------------------------------------------
# feature types
# --------------------------------------------------------------------------


def test_declared_categorical_features_win_over_inference(ragged_sequence):
    declared = ArrayAdapter(
        ragged_sequence.values(),
        ["speed", "vessel_type"],
        entity_lengths=[3, 4],
        categorical_features=["speed"],
    )
    assert declared.categorical_features == ["speed"]
    assert declared.numerical_features == ["vessel_type"]


def test_declaring_an_unknown_feature_raises():
    with pytest.raises(ValueError, match="unknown feature"):
        ArrayAdapter(np.zeros((2, 2)), ["a", "b"], categorical_features=["c"])


def test_sequence_inference_uses_constant_within_entity(ragged_sequence):
    """Repo B's original heuristic: a per-entity attribute never varies over time."""
    assert ragged_sequence.categorical_features == ["vessel_type"]
    assert ragged_sequence.numerical_features == ["speed"]


def test_tabular_inference_does_not_guess_at_integer_columns():
    """An encoded category and a small count look identical; guessing wrong is worse."""
    a = ArrayAdapter(np.array([[0, 5], [1, 7], [2, 5]]), ["encoded", "count"])
    assert a.categorical_features == []


def test_tabular_inference_treats_object_dtype_as_categorical():
    values = np.array([["x", "p"], ["y", "q"]], dtype=object)
    a = ArrayAdapter(values, ["a", "b"])
    assert a.categorical_features == ["a", "b"]


def test_infer_categorical_features_is_callable_directly():
    values = np.array([[[1.0, 5.0], [2.0, 5.0]], [[3.0, 9.0], [4.0, 9.0]]])
    got = infer_categorical_features(values, ["varies", "constant"], is_sequential=True)
    assert got == ["constant"]


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        ArrayAdapter(np.zeros((2, 2)), ["a", "a"])


def test_wrong_number_of_feature_names_is_rejected():
    with pytest.raises(ValueError, match="feature name"):
        ArrayAdapter(np.zeros((2, 3)), ["a", "b"])


def test_feature_index_and_indices(tabular):
    assert tabular.feature_index("bmi") == 1
    assert tabular.feature_indices(["bmi", "age"]) == [1, 0]
    with pytest.raises(KeyError, match="Unknown feature"):
        tabular.feature_index("height")


# --------------------------------------------------------------------------
# category pools
# --------------------------------------------------------------------------


def test_category_pool_is_dataset_wide(ragged_sequence):
    """A categorical perturbation must swap in a category that occurs somewhere."""
    assert np.array_equal(ragged_sequence.category_pool("vessel_type"), [7.0, 10.0])


def test_category_pool_ignores_padding(ragged_sequence):
    assert not np.isnan(ragged_sequence.category_pool("speed")).any()


def test_category_pool_handles_strings():
    a = ArrayAdapter(np.array([["x"], ["y"], ["x"]], dtype=object), ["job"])
    assert sorted(a.category_pool("job")) == ["x", "y"]


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def test_with_values_preserves_names_types_lengths_and_ids(ragged_sequence):
    new = ragged_sequence.with_values(ragged_sequence.values() * 2)
    assert new.feature_names == ragged_sequence.feature_names
    assert new.categorical_features == ragged_sequence.categorical_features
    assert np.array_equal(new.entity_lengths, ragged_sequence.entity_lengths)
    assert new.alignment is ragged_sequence.alignment
    assert new.entity_ids == ragged_sequence.entity_ids


def test_with_values_rejects_a_shape_change(tabular):
    with pytest.raises(ValueError, match="with_values expects shape"):
        tabular.with_values(np.zeros((2, 1, 2)))


def test_with_values_accepts_the_2d_form_for_tabular(tabular):
    new = tabular.with_values(np.zeros((3, 2)))
    assert new.values().shape == (3, 1, 2)


def test_take_selects_entities_in_order(ragged_sequence):
    subset = ragged_sequence.take([1, 0])
    assert subset.n_entities == 2
    assert list(subset.entity_lengths) == [4, 3]
    assert subset.entity_ids == [1, 0]


def test_take_accepts_a_boolean_selector(ragged_sequence):
    subset = ragged_sequence.take([False, True])
    assert subset.n_entities == 1
    assert list(subset.entity_lengths) == [4]


def test_take_rejects_out_of_range_indices(tabular):
    with pytest.raises(ValueError, match="Entity indices must lie in"):
        tabular.take([0, 5])


def test_take_rejects_a_boolean_selector_of_the_wrong_length(tabular):
    with pytest.raises(ValueError, match="Boolean selector has length"):
        tabular.take([True, False])


# --------------------------------------------------------------------------
# FrameAdapter
# --------------------------------------------------------------------------


def test_frame_adapter_infers_categoricals_from_dtypes():
    frame = pd.DataFrame({"age": [30, 40], "job": ["nurse", "teacher"]})
    a = FrameAdapter(frame)
    assert a.categorical_features == ["job"]
    assert a.numerical_features == ["age"]


def test_frame_adapter_respects_a_categorical_dtype():
    frame = pd.DataFrame({"grade": pd.Categorical(["a", "b"]), "n": [1, 2]})
    assert FrameAdapter(frame).categorical_features == ["grade"]


def test_frame_adapter_round_trips_to_a_frame():
    frame = pd.DataFrame({"age": [30, 40], "bmi": [22.0, 25.0]}, index=["p1", "p2"])
    out = FrameAdapter(frame).to_frame()
    assert list(out.columns) == ["age", "bmi"]
    assert list(out.index) == ["p1", "p2"]


def test_frame_adapter_keeps_the_index_as_entity_ids():
    frame = pd.DataFrame({"x": [1, 2]}, index=["patient_a", "patient_b"])
    assert FrameAdapter(frame).entity_ids == ["patient_a", "patient_b"]


def test_frame_adapter_can_select_a_column_subset():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    a = FrameAdapter(frame, feature_names=["c", "a"])
    assert a.feature_names == ["c", "a"]


def test_frame_adapter_rejects_a_missing_column():
    with pytest.raises(ValueError, match="missing column"):
        FrameAdapter(pd.DataFrame({"a": [1]}), feature_names=["b"])


def test_frame_adapter_rejects_a_non_frame():
    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        FrameAdapter(np.zeros((2, 2)))


# --------------------------------------------------------------------------
# TextAdapter
# --------------------------------------------------------------------------


def test_text_adapter_exposes_texts_and_a_uniform_shape():
    a = TextAdapter(["first doc", "second doc"])
    assert a.texts() == ["first doc", "second doc"]
    assert a.values().shape == (2, 1, 1)
    assert a.modality is Modality.TEXT
    assert a.feature_names == ["text"]


def test_text_adapter_with_texts_replaces_documents():
    a = TextAdapter(["a", "b"], entity_ids=["d1", "d2"])
    new = a.with_texts(["A", "B"])
    assert new.texts() == ["A", "B"]
    assert new.entity_ids == ["d1", "d2"]


def test_text_adapter_with_texts_rejects_a_count_change():
    with pytest.raises(ValueError, match="expects 2 document"):
        TextAdapter(["a", "b"]).with_texts(["only one"])


def test_text_adapter_rejects_non_strings():
    with pytest.raises(TypeError, match="expects strings"):
        TextAdapter(["fine", 42])


def test_text_has_no_categorical_features():
    assert TextAdapter(["a"]).categorical_features == []


# --------------------------------------------------------------------------
# the point of the whole module
# --------------------------------------------------------------------------


def test_one_operation_serves_tabular_and_sequence_data():
    """A perturbation written once works on both, which is the reason adapters exist."""

    def scale_numerical(adapter, factor):
        values = adapter.values()
        for name in adapter.numerical_features:
            j = adapter.feature_index(name)
            values[:, :, j] = values[:, :, j] * factor
        return adapter.with_values(values)

    tab = ArrayAdapter(np.array([[2.0, 4.0]]), ["a", "b"], categorical_features=["b"])
    seq = ArrayAdapter(
        np.array([[[2.0, 4.0], [3.0, 4.0]]]), ["a", "b"], categorical_features=["b"]
    )

    assert scale_numerical(tab, 10.0).values()[0, 0, 0] == pytest.approx(20.0)
    assert scale_numerical(tab, 10.0).values()[0, 0, 1] == pytest.approx(4.0)
    assert scale_numerical(seq, 10.0).values()[0, :, 0].tolist() == [20.0, 30.0]
    assert scale_numerical(seq, 10.0).values()[0, :, 1].tolist() == [4.0, 4.0]
