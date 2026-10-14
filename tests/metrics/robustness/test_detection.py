"""Tests for the detector-evaluation metrics (survey §5.5)."""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.core.errors import ThemisAssessError
from themis_assess.metrics.robustness.detection import (
    auprc,
    auroc,
    autc,
    fpr_at_tpr,
    weighted_autc,
)

# Positive class = what the detector should catch.
PERFECT_SCORES = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
PERFECT_LABELS = [0, 0, 0, 1, 1, 1]


def test_auroc_is_one_for_perfect_separation():
    assert auroc(PERFECT_SCORES, PERFECT_LABELS) == pytest.approx(1.0)


def test_auroc_is_half_for_a_useless_detector():
    rng = np.random.default_rng(0)
    scores = rng.random(2000)
    labels = rng.integers(0, 2, 2000)
    assert auroc(scores, labels) == pytest.approx(0.5, abs=0.05)


def test_auroc_is_zero_when_the_score_is_inverted():
    assert auroc(PERFECT_SCORES, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.0)


def test_auroc_matches_its_probabilistic_definition():
    """AUROC = Pr(s(x+) > s(x-)), computed here by brute force over all pairs."""
    rng = np.random.default_rng(7)
    labels = np.array([0] * 40 + [1] * 25)
    scores = np.concatenate([rng.normal(0.0, 1.0, 40), rng.normal(1.0, 1.0, 25)])
    pos, neg = scores[labels == 1], scores[labels == 0]
    brute = np.mean(
        [(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]
    )
    assert auroc(scores, labels) == pytest.approx(brute)


def test_auprc_is_one_for_perfect_separation():
    assert auprc(PERFECT_SCORES, PERFECT_LABELS) == pytest.approx(1.0)


def test_auprc_is_the_more_honest_metric_when_positives_are_rare():
    """The survey's point: AUROC can stay high while precision on the rare class is poor."""
    rng = np.random.default_rng(3)
    n_neg, n_pos = 5000, 50
    labels = np.array([0] * n_neg + [1] * n_pos)
    scores = np.concatenate([rng.normal(0.0, 1.0, n_neg), rng.normal(1.6, 1.0, n_pos)])
    assert auroc(scores, labels) > 0.85
    assert auprc(scores, labels) < 0.30


def test_fpr_at_tpr_is_zero_for_perfect_separation():
    assert fpr_at_tpr(PERFECT_SCORES, PERFECT_LABELS) == pytest.approx(0.0)


def test_fpr_at_tpr_reaches_the_requested_recall():
    """One negative sits above the lowest positive, so catching all positives costs it."""
    scores = [0.1, 0.2, 0.85, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1, 1]
    assert fpr_at_tpr(scores, labels, tpr_target=1.0) == pytest.approx(1 / 3)


def test_fpr_at_tpr_rejects_an_out_of_range_target():
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="tpr_target"):
            fpr_at_tpr(PERFECT_SCORES, PERFECT_LABELS, tpr_target=bad)


def test_autc_is_zero_for_a_perfectly_separated_detector():
    """Best AUTC is 0, unlike AUROC where best is 1."""
    scores = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert autc(scores, PERFECT_LABELS) == pytest.approx(0.0, abs=1e-3)


def test_autc_distinguishes_detectors_that_auroc_cannot():
    """The survey's stated rationale for AUTC.

    Two detectors with identical ranking — hence identical AUROC — but one separates
    the score distributions by a much wider gap. Only AUTC sees it.
    """
    labels = [0, 0, 0, 1, 1, 1]
    narrow = [0.45, 0.46, 0.47, 0.53, 0.54, 0.55]
    wide = [0.00, 0.01, 0.02, 0.98, 0.99, 1.00]

    assert auroc(narrow, labels) == pytest.approx(auroc(wide, labels))
    assert autc(wide, labels) < autc(narrow, labels)


def test_autc_is_bounded_in_zero_one():
    rng = np.random.default_rng(11)
    scores = rng.random(500)
    labels = rng.integers(0, 2, 500)
    assert 0.0 <= autc(scores, labels) <= 1.0


def test_weighted_autc_with_equal_weights_matches_autc():
    rng = np.random.default_rng(5)
    labels = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    assert weighted_autc(scores, labels, w_fpr=0.5, w_fnr=0.5) == pytest.approx(
        autc(scores, labels)
    )


def test_weighted_autc_shifts_with_the_weights():
    """Weighting one error type changes the verdict, which is the whole point."""
    labels = [0, 0, 0, 0, 1, 1]
    # Detector biased toward flagging: few misses, many false alarms.
    scores = [0.4, 0.5, 0.6, 0.7, 0.95, 1.0]
    fpr_heavy = weighted_autc(scores, labels, w_fpr=0.9, w_fnr=0.1)
    fnr_heavy = weighted_autc(scores, labels, w_fpr=0.1, w_fnr=0.9)
    assert fpr_heavy > fnr_heavy


def test_weighted_autc_requires_weights_summing_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        weighted_autc(PERFECT_SCORES, PERFECT_LABELS, w_fpr=0.9, w_fnr=0.9)


def test_weighted_autc_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_autc(PERFECT_SCORES, PERFECT_LABELS, w_fpr=1.5, w_fnr=-0.5)


@pytest.mark.parametrize("fn", [auroc, auprc, fpr_at_tpr, autc])
def test_single_class_input_raises_a_clear_error(fn):
    with pytest.raises(ThemisAssessError, match="positive and negative"):
        fn([0.1, 0.2, 0.3], [1, 1, 1])


@pytest.mark.parametrize("fn", [auroc, auprc, fpr_at_tpr, autc])
def test_mismatched_lengths_raise(fn):
    with pytest.raises(ValueError, match="same length"):
        fn([0.1, 0.2], [0, 1, 1])


def test_non_finite_scores_are_rejected():
    with pytest.raises(ValueError, match="NaN or infinity"):
        auroc([0.1, float("nan"), 0.3], [0, 1, 1])


def test_autc_rejects_constant_scores():
    with pytest.raises(ThemisAssessError, match="identical"):
        autc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])


def test_these_metrics_are_not_ood_specific():
    """D20: they score any binary detector, e.g. misclassification flags."""
    is_misclassified = [0, 0, 1, 0, 1, 1]
    detector_score = [0.05, 0.10, 0.80, 0.15, 0.90, 0.95]
    assert auroc(detector_score, is_misclassified) == pytest.approx(1.0)
