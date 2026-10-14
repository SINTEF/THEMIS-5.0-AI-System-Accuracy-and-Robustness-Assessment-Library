"""Tests for the time-axis perturbations (survey §4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.types import Alignment
from themis_assess.perturbations.timeseries.time_axis import (
    InputWindowShift,
    TimeWarp,
    WindowSlice,
)

ALL = (WindowSlice, TimeWarp, InputWindowShift)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def ramp():
    return ArrayAdapter(
        (np.arange(40, dtype=float)).reshape(1, 40, 1), ["x"], sequential=True
    )


@pytest.fixture
def ragged():
    """Lengths 10 and 25, end-aligned."""
    values = np.full((2, 25, 2), np.nan)
    values[0, 15:, 0] = np.arange(10, dtype=float)
    values[0, 15:, 1] = 3.0
    values[1, :, 0] = np.arange(25, dtype=float)
    values[1, :, 1] = 7.0
    return ArrayAdapter(
        values, ["x", "kind"], entity_lengths=[10, 25], alignment=Alignment.END
    )


# --------------------------------------------------------------------------
# shared properties
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL)
def test_all_are_sequence_only(cls, rng):
    tabular = ArrayAdapter(np.ones((3, 2)), ["a", "b"])
    p = cls()
    assert not p.applies_to(tabular)
    with pytest.raises(ValueError, match="applies to \\['sequence'\\]"):
        p.perturb(tabular, p.spec(severity=3), rng)


@pytest.mark.parametrize("cls", ALL)
def test_all_keep_the_array_shape(cls, ragged, rng):
    """A shortened series loses steps to padding; the array itself never changes shape."""
    p = cls()
    out = p.perturb(ragged, p.spec(severity=3), rng)[0]
    assert out.values().shape == ragged.values().shape


@pytest.mark.parametrize("cls", ALL)
def test_all_move_every_feature_together(cls, ragged, rng):
    """A shifted time axis for one feature and not another is a record that never existed."""
    p = cls()
    assert p.accepts_categorical and p.accepts_numerical
    out = p.perturb(ragged, p.spec(severity=3), rng)[0]
    for e in range(out.n_entities):
        valid = out.valid_mask()[e]
        # the categorical column stays valid exactly where the numeric one does
        assert np.array_equal(
            np.isnan(out.values()[e, :, 0]), np.isnan(out.values()[e, :, 1])
        )
        assert np.array_equal(valid, ~np.isnan(out.values()[e, :, 0]))


@pytest.mark.parametrize("cls", ALL)
def test_all_report_a_severity_mapping(cls):
    p = cls()
    meta = p.metadata(p.spec(severity=2))
    assert meta["severity"] == 2
    assert "severity_factor" in meta


# --------------------------------------------------------------------------
# WindowSlice
# --------------------------------------------------------------------------


def test_window_slice_shortens_the_series(ramp, rng):
    p = WindowSlice()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]  # 0.40 removed
    assert int(out.entity_lengths[0]) == 24  # round(0.6 * 40)
    assert int(np.sum(np.isfinite(out.values()))) == 24


def test_window_slice_keeps_a_contiguous_run_of_the_original(ramp, rng):
    p = WindowSlice()
    out = p.perturb(ramp, p.spec(severity=2), rng)[0]
    kept = out.values()[0][out.valid_mask()[0], 0]
    assert np.allclose(np.diff(kept), 1.0)  # consecutive values from the ramp


def test_window_slice_re_aligns_to_the_adapters_alignment(ramp, rng):
    """Under END alignment the kept window sits at the right-hand edge."""
    p = WindowSlice()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]
    mask = out.valid_mask()[0]
    assert mask[-1]
    assert not mask[0]


def test_window_slice_respects_start_alignment():
    values = np.arange(20, dtype=float).reshape(1, 20, 1)
    data = ArrayAdapter(values, ["x"], sequential=True, alignment=Alignment.START)
    p = WindowSlice()
    out = p.perturb(data, p.spec(severity=3), np.random.default_rng(1))[0]
    mask = out.valid_mask()[0]
    assert mask[0]
    assert not mask[-1]


def test_window_slice_severity_removes_more(ramp, rng):
    p = WindowSlice()
    lengths = [
        int(p.perturb(ramp, p.spec(severity=s), np.random.default_rng(2))[0].entity_lengths[0])
        for s in range(1, 6)
    ]
    assert lengths == sorted(lengths, reverse=True)


def test_window_slice_scales_to_each_entitys_own_length(ragged, rng):
    p = WindowSlice()
    out = p.perturb(ragged, p.spec(severity=3), rng)[0]  # keep 0.6
    assert int(out.entity_lengths[0]) == 6  # round(0.6 * 10)
    assert int(out.entity_lengths[1]) == 15  # round(0.6 * 25)


def test_window_slice_never_empties_a_series(ramp, rng):
    p = WindowSlice()
    out = p.perturb(ramp, p.spec(severity=5), rng)[0]
    assert int(out.entity_lengths[0]) >= 1


def test_window_slice_differs_from_a_missing_segment(ramp, rng):
    """MissingSegment leaves a hole with data either side; this hands over a shorter record."""
    from themis_assess.perturbations.timeseries.value_domain import MissingSegment

    sliced = WindowSlice().perturb(ramp, WindowSlice().spec(severity=3), rng)[0]
    gapped = MissingSegment().perturb(ramp, MissingSegment().spec(severity=3), rng)[0]
    assert int(sliced.entity_lengths[0]) < 40
    assert int(gapped.entity_lengths[0]) == 40  # length unchanged, values NaN inside


def test_window_slice_reports_what_it_kept():
    p = WindowSlice()
    meta = p.metadata(p.spec(severity=3))
    assert meta["fraction_removed"] == pytest.approx(0.40)
    assert meta["fraction_kept"] == pytest.approx(0.60)
    assert meta["changes_series_length"] is True


# --------------------------------------------------------------------------
# TimeWarp
# --------------------------------------------------------------------------


def test_time_warp_keeps_the_length(ramp, rng):
    p = TimeWarp()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]
    assert int(out.entity_lengths[0]) == 40
    assert np.isfinite(out.values()).all()


def test_time_warp_keeps_values_within_the_original_range(ramp, rng):
    """Interpolation, not extrapolation — no value the series never approached."""
    p = TimeWarp()
    out = p.perturb(ramp, p.spec(severity=5), rng)[0].values()[0, :, 0]
    original = ramp.values()[0, :, 0]
    assert out.min() >= original.min() - 1e-9
    assert out.max() <= original.max() + 1e-9


def test_time_warp_preserves_monotonicity_of_a_monotone_series(ramp, rng):
    """Time never runs backwards, so a rising series stays rising."""
    p = TimeWarp()
    out = p.perturb(ramp, p.spec(severity=5), rng)[0].values()[0, :, 0]
    assert np.all(np.diff(out) >= -1e-9)


def test_time_warp_pins_both_endpoints(ramp, rng):
    """The warp maps the first step to the first and the last to the last."""
    p = TimeWarp()
    out = p.perturb(ramp, p.spec(severity=5), rng)[0].values()[0, :, 0]
    original = ramp.values()[0, :, 0]
    assert out[0] == pytest.approx(original[0])
    assert out[-1] == pytest.approx(original[-1])


def test_time_warp_actually_changes_the_spacing(ramp, rng):
    p = TimeWarp()
    out = p.perturb(ramp, p.spec(severity=5), rng)[0].values()[0, :, 0]
    assert not np.allclose(np.diff(out), 1.0)


def test_time_warp_severity_increases_the_distortion(ramp):
    p = TimeWarp()
    original = ramp.values()[0, :, 0]
    distortions = []
    for s in (1, 5):
        out = p.perturb(ramp, p.spec(severity=s), np.random.default_rng(3))[0]
        distortions.append(float(np.mean(np.abs(out.values()[0, :, 0] - original))))
    assert distortions[1] > distortions[0]


def test_time_warp_resamples_categorical_features_by_nearest_neighbour():
    """The midpoint between category 1 and category 3 is not category 2."""
    values = np.zeros((1, 30, 2))
    values[0, :, 0] = np.arange(30, dtype=float)
    values[0, :15, 1] = 1.0
    values[0, 15:, 1] = 3.0
    data = ArrayAdapter(values, ["x", "kind"], categorical_features=["kind"])
    p = TimeWarp()
    out = p.perturb(data, p.spec(severity=5), np.random.default_rng(4))[0]
    assert set(np.unique(out.values()[0, :, 1])) <= {1.0, 3.0}


def test_time_warp_leaves_very_short_series_alone(rng):
    values = np.array([[[1.0], [2.0]]])
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = TimeWarp()
    out = p.perturb(data, p.spec(severity=5), rng)[0]
    assert np.allclose(out.values(), values)


def test_time_warp_reports_that_it_is_monotone():
    meta = TimeWarp().metadata(TimeWarp().spec(severity=3))
    assert meta["monotone"] is True
    assert meta["categorical_resampling"] == "nearest neighbour"


# --------------------------------------------------------------------------
# InputWindowShift
# --------------------------------------------------------------------------


def test_input_window_shift_drops_from_the_end(ramp, rng):
    """Repo B's truncation: vessel_series[:-prediction_end_offset]."""
    p = InputWindowShift()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]  # 0.10 dropped
    assert int(out.entity_lengths[0]) == 36
    kept = out.values()[0][out.valid_mask()[0], 0]
    assert kept[0] == pytest.approx(0.0)  # the *start* survives
    assert kept[-1] == pytest.approx(35.0)  # the end is gone


def test_input_window_shift_is_deterministic(ramp):
    """No random position to draw — the one perturbation independent of the generator."""
    a = InputWindowShift().perturb(ramp, InputWindowShift().spec(severity=3), np.random.default_rng(1))[0]
    b = InputWindowShift().perturb(ramp, InputWindowShift().spec(severity=3), np.random.default_rng(999))[0]
    assert np.array_equal(a.values(), b.values(), equal_nan=True)


def test_input_window_shift_costs_one_model_call_regardless_of_variants(ramp):
    p = InputWindowShift()
    assert p.n_model_calls(ramp, p.spec(severity=3, n_variants=50)) == 1
    assert p.metadata(p.spec(severity=1))["deterministic"] is True


def test_input_window_shift_severity_moves_the_origin_further_back(ramp, rng):
    p = InputWindowShift()
    lengths = [
        int(p.perturb(ramp, p.spec(severity=s), rng)[0].entity_lengths[0])
        for s in range(1, 6)
    ]
    assert lengths == sorted(lengths, reverse=True)
    assert lengths[0] > lengths[-1]


def test_input_window_shift_always_drops_at_least_one_step(rng):
    """Severity 1 is 2%, so a short series would otherwise be untouched."""
    values = np.arange(10, dtype=float).reshape(1, 10, 1)
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = InputWindowShift()
    out = p.perturb(data, p.spec(severity=1), rng)[0]
    assert int(out.entity_lengths[0]) == 9


def test_input_window_shift_never_empties_a_series(rng):
    values = np.arange(3, dtype=float).reshape(1, 3, 1)
    data = ArrayAdapter(values, ["x"], sequential=True, categorical_features=[])
    p = InputWindowShift()
    out = p.perturb(data, p.spec(severity=5), rng)[0]
    assert int(out.entity_lengths[0]) >= 1


def test_input_window_shift_keeps_the_window_right_aligned(ramp, rng):
    """So a model reading the last N steps sees an earlier window, not a shifted array."""
    p = InputWindowShift()
    out = p.perturb(ramp, p.spec(severity=3), rng)[0]
    mask = out.valid_mask()[0]
    assert mask[-1]
    assert not mask[0]


def test_input_window_shift_asks_the_same_question_earlier(ramp, rng):
    """Every value the model sees is real and correct — only the origin moved."""
    p = InputWindowShift()
    out = p.perturb(ramp, p.spec(severity=4), rng)[0]
    kept = out.values()[0][out.valid_mask()[0], 0]
    original = ramp.values()[0, :, 0]
    assert np.array_equal(kept, original[: kept.size])
