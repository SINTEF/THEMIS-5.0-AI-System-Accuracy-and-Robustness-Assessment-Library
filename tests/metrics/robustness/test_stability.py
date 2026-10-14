"""Golden-value tests for the prediction-stability metrics (survey Eqs. 23-25)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.errors import ZeroDenominatorError
from themis_assess.metrics.robustness.stability import (
    confidence_retention,
    flip_probability,
    flip_rate,
    mean_flip_rate,
    pairwise_flip_probability,
)


def test_flip_probability_eq23_counts_transitions_between_frames():
    # sample 0: 0 0 1 -> 1 transition of 2 ; sample 1: 1 1 1 -> 0 of 2
    # FP = mean(1/2, 0/2) = 0.25
    preds = np.array([[0, 0, 1], [1, 1, 1]])
    assert flip_probability(preds) == pytest.approx(0.25)


def test_flip_probability_is_zero_for_a_perfectly_stable_model():
    preds = np.tile(np.array([[2], [7], [3]]), (1, 5))
    assert flip_probability(preds) == pytest.approx(0.0)


def test_flip_probability_is_one_when_every_frame_differs():
    preds = np.array([[0, 1, 0, 1]])
    assert flip_probability(preds) == pytest.approx(1.0)


def test_flip_probability_needs_no_ground_truth():
    """It is a stability measure, so it works on unlabelled data by construction."""
    preds = np.array([[0, 1], [1, 1]])
    assert flip_probability(preds) == pytest.approx(0.5)


def test_flip_probability_handles_string_labels():
    preds = np.array([["cat", "cat", "dog"], ["dog", "dog", "dog"]])
    assert flip_probability(preds) == pytest.approx(0.25)


def test_flip_probability_rejects_a_1d_array_with_a_helpful_message():
    with pytest.raises(ValueError, match="pairwise_flip_probability"):
        flip_probability(np.array([0, 1, 0]))


def test_flip_probability_needs_at_least_two_frames():
    with pytest.raises(ValueError, match="at least 2 frames"):
        flip_probability(np.array([[0], [1]]))


def test_pairwise_flip_probability_is_the_two_frame_special_case():
    clean = [0, 1, 1, 0]
    perturbed = [0, 0, 1, 1]
    expected = 0.5
    assert pairwise_flip_probability(clean, perturbed) == pytest.approx(expected)
    stacked = np.column_stack([clean, perturbed])
    assert flip_probability(stacked) == pytest.approx(expected)


def test_pairwise_flip_probability_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        pairwise_flip_probability([0, 1], [0, 1, 1])


def test_flip_rate_eq24_normalises_by_the_baseline():
    assert flip_rate(fp_model=0.1, fp_baseline=0.4) == pytest.approx(0.25)


def test_flip_rate_below_one_means_more_stable_than_baseline():
    assert flip_rate(0.2, 0.5) < 1.0
    assert flip_rate(0.5, 0.2) > 1.0


def test_flip_rate_raises_when_the_baseline_never_flips():
    with pytest.raises(ZeroDenominatorError, match="baseline flip probability"):
        flip_rate(0.1, 0.0)


def test_mean_flip_rate_eq25():
    assert mean_flip_rate({"noise": 0.5, "blur": 1.0, "shift": 1.5}) == pytest.approx(1.0)


def test_mean_flip_rate_accepts_a_sequence():
    assert mean_flip_rate([0.5, 1.5]) == pytest.approx(1.0)


def test_confidence_retention_uses_the_originally_predicted_class():
    """Repo D's caution, encoded as a test.

    Confidence on the original class collapses from 0.9 to 0.3 even though the model is
    now confidently predicting something else. Retention must show the collapse.
    """
    clean_conf_on_original = [0.9, 0.9]
    perturbed_conf_on_original = [0.3, 0.3]
    assert confidence_retention(
        clean_conf_on_original, perturbed_conf_on_original
    ) == pytest.approx(1 / 3)


def test_confidence_retention_is_one_when_confidence_is_unchanged():
    conf = [0.6, 0.7, 0.8]
    assert confidence_retention(conf, conf) == pytest.approx(1.0)


def test_confidence_retention_can_exceed_one():
    assert confidence_retention([0.4], [0.8]) == pytest.approx(2.0)


def test_confidence_retention_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        confidence_retention([0.5], [0.5, 0.5])


def test_stability_and_accuracy_measure_different_things():
    """The survey's argument for the family existing at all.

    Two models with identical accuracy at every severity, one stable and one
    oscillating. Only the flip probability separates them.
    """
    truth = np.array([0, 0, 1, 1])

    # Stable: gets samples 0 and 2 right at every severity, never changes its mind.
    stable = np.tile(np.array([[0], [1], [1], [0]]), (1, 4))
    # Oscillating: same per-frame accuracy (2/4), but alternating which two.
    oscillating = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]])

    acc_stable = [(stable[:, t] == truth).mean() for t in range(4)]
    acc_osc = [(oscillating[:, t] == truth).mean() for t in range(4)]
    assert acc_stable == pytest.approx(acc_osc)

    assert flip_probability(stable) == pytest.approx(0.0)
    assert flip_probability(oscillating) == pytest.approx(1.0)
