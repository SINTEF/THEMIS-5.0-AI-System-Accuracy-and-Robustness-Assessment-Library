"""Tests for the darts adapter.

Skipped when darts is not installed, since it ships under the ``[timeseries]`` extra and
repo A's core must run without it (docs/DECISIONS.md, D7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

darts = pytest.importorskip("darts", reason="darts ships under the [timeseries] extra")
TimeSeries = darts.TimeSeries

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import Alignment, Modality
from themis_assess.integrations.darts_adapter import (
    DartsAdapter,
)


def make_series(n, start="2024-01-01", vessel_type=10.0, static=None):
    index = pd.date_range(start, periods=n, freq="h")
    values = np.column_stack(
        [np.arange(n, dtype=float), np.full(n, vessel_type, dtype=float)]
    )
    frame = pd.DataFrame(values, columns=["speed", "vessel_type"], index=index)
    return TimeSeries.from_dataframe(frame, static_covariates=static)


@pytest.fixture
def ragged():
    """Two vessels of length 3 and 5 — the shape repo B actually works with."""
    return [
        make_series(3, "2024-01-01", vessel_type=10.0),
        make_series(5, "2024-02-01", vessel_type=7.0),
    ]


def test_adapter_satisfies_the_protocol(ragged):
    assert isinstance(DartsAdapter(ragged), DataAdapter)


def test_components_become_feature_names(ragged):
    a = DartsAdapter(ragged)
    assert a.feature_names == ["speed", "vessel_type"]
    assert a.modality is Modality.SEQUENCE
    assert a.is_sequential is True


def test_ragged_series_are_padded_to_the_longest(ragged):
    a = DartsAdapter(ragged)
    assert a.values().shape == (2, 5, 2)
    assert list(a.entity_lengths) == [3, 5]


def test_end_alignment_is_the_default_and_pads_on_the_left(ragged):
    a = DartsAdapter(ragged)
    assert a.alignment is Alignment.END
    expected = np.array(
        [[False, False, True, True, True], [True, True, True, True, True]]
    )
    assert np.array_equal(a.valid_mask(), expected)
    assert np.isnan(a.values()[0, :2, :]).all()


def test_start_alignment_pads_on_the_right(ragged):
    a = DartsAdapter(ragged, alignment=Alignment.START)
    expected = np.array(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    assert np.array_equal(a.valid_mask(), expected)


def test_categorical_inference_finds_the_per_vessel_attribute(ragged):
    """vessel_type never varies within a vessel; speed does."""
    a = DartsAdapter(ragged)
    assert a.categorical_features == ["vessel_type"]
    assert a.numerical_features == ["speed"]


def test_declared_categorical_features_win(ragged):
    a = DartsAdapter(ragged, categorical_features=["speed"])
    assert a.categorical_features == ["speed"]


def test_category_pool_is_dataset_wide_and_ignores_padding(ragged):
    a = DartsAdapter(ragged)
    assert np.array_equal(a.category_pool("vessel_type"), [7.0, 10.0])


def test_round_trip_preserves_values_lengths_and_time_index(ragged):
    a = DartsAdapter(ragged)
    back = a.to_series()
    assert [len(s) for s in back] == [3, 5]
    for original, restored in zip(ragged, back, strict=True):
        assert np.allclose(original.values(), restored.values())
        assert original.time_index.equals(restored.time_index)
        assert list(original.components) == list(restored.components)


def test_round_trip_preserves_static_covariates():
    static = pd.DataFrame({"flag": ["NO"]})
    series = [make_series(3, static=static)]
    back = DartsAdapter(series).to_series()
    assert back[0].static_covariates is not None
    assert back[0].static_covariates["flag"].iloc[0] == "NO"


def test_start_alignment_also_round_trips(ragged):
    back = DartsAdapter(ragged, alignment=Alignment.START).to_series()
    assert [len(s) for s in back] == [3, 5]
    for original, restored in zip(ragged, back, strict=True):
        assert np.allclose(original.values(), restored.values())


def test_a_perturbation_written_without_darts_round_trips_through_the_adapter(ragged):
    """The whole point: generic code in, darts out.

    This function knows nothing about darts, vessels or forecasting — it is the same
    shape as the tabular helper in tests/core/test_adapters.py.
    """

    def zero_numerical(adapter):
        values = adapter.values()
        for name in adapter.numerical_features:
            j = adapter.feature_index(name)
            values[:, :, j] = np.where(np.isnan(values[:, :, j]), np.nan, 0.0)
        return adapter.with_values(values)

    a = DartsAdapter(ragged)
    perturbed = zero_numerical(a).to_series()

    assert [len(s) for s in perturbed] == [3, 5]
    for s in perturbed:
        assert np.allclose(s["speed"].values(), 0.0)
    # the categorical component is untouched
    assert np.allclose(perturbed[0]["vessel_type"].values(), 10.0)
    assert np.allclose(perturbed[1]["vessel_type"].values(), 7.0)


def test_take_selects_series_and_keeps_their_time_indices(ragged):
    a = DartsAdapter(ragged)
    subset = a.take([1])
    assert subset.n_entities == 1
    assert list(subset.entity_lengths) == [5]
    assert subset.to_series()[0].time_index.equals(ragged[1].time_index)


def test_take_accepts_a_boolean_selector(ragged):
    a = DartsAdapter(ragged)
    assert a.take([True, False]).n_entities == 1


def test_with_values_keeps_darts_metadata(ragged):
    a = DartsAdapter(ragged)
    new = a.with_values(a.values())
    assert isinstance(new, DartsAdapter)
    assert [len(s) for s in new.to_series()] == [3, 5]


def test_mismatched_components_are_rejected():
    a = make_series(3)
    b = TimeSeries.from_dataframe(
        pd.DataFrame(
            np.zeros((3, 2)),
            columns=["vessel_type", "speed"],  # same names, wrong order
            index=pd.date_range("2024-01-01", periods=3, freq="h"),
        )
    )
    with pytest.raises(ValueError, match="same components in the same order"):
        DartsAdapter([a, b])


def test_non_timeseries_elements_are_rejected(ragged):
    with pytest.raises(TypeError, match="not a darts TimeSeries"):
        DartsAdapter([ragged[0], np.zeros((3, 2))])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one TimeSeries"):
        DartsAdapter([])


def test_core_does_not_import_darts_on_its_own():
    """Importing the library must not pull in an optional extra."""
    from tests.conftest import run_child

    code = (
        "import sys, themis_assess, themis_assess.core, themis_assess.metrics.robustness; "
        "assert 'darts' not in sys.modules, sorted(m for m in sys.modules if 'darts' in m)"
    )
    completed = run_child(code)
    assert completed.returncode == 0, completed.stderr
