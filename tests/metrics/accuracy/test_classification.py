"""Tests for the confusion matrix and the breakdowns derived from it.

The fixture is built so every expected number is computable by hand, and the interesting
case is deliberately included: a rare class that is confused most often *proportionally*
while contributing the fewest errors in absolute terms. That is the case where ranking by
raw count sends the reader to the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.metrics.accuracy.classification import (
    ConfusionMatrix,
    confusion_matrix,
    error_rate_by_class,
    per_class_metrics,
    top_confusions,
)


@pytest.fixture
def labelled():
    """Three classes with hand-countable confusions.

    class 0: 200 samples, 20 predicted as 1        -> 10% error
    class 1: 150 samples, 15 predicted as 0        -> 10% error
    class 2:  20 samples, 14 predicted as 1        -> 70% error
    """
    actual = np.array([0] * 200 + [1] * 150 + [2] * 20)
    predicted = actual.copy()
    predicted[:20] = 1
    predicted[200:215] = 0
    predicted[350:364] = 1
    return actual, predicted


# --------------------------------------------------------------------------
# the container
# --------------------------------------------------------------------------


def test_a_non_square_matrix_is_rejected():
    with pytest.raises(ValueError, match="must be square"):
        ConfusionMatrix(matrix=np.zeros((2, 3)), labels=(0, 1))


def test_mismatched_labels_are_rejected():
    with pytest.raises(ValueError, match="label"):
        ConfusionMatrix(matrix=np.zeros((3, 3)), labels=(0, 1))


def test_the_matrix_knows_its_own_labels(labelled):
    """A bare array's row order is np.unique order, not the caller's."""
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted)
    assert matrix.labels == (0, 1, 2)
    assert matrix.count(2, 1) == 14.0
    assert matrix.count(0, 1) == 20.0
    assert matrix.count(1, 2) == 0.0


def test_an_unknown_label_lookup_lists_the_classes(labelled):
    actual, predicted = labelled
    with pytest.raises(KeyError, match="not one of the classes"):
        confusion_matrix(actual, predicted).count(9, 0)


def test_support_is_the_row_sums(labelled):
    actual, predicted = labelled
    assert confusion_matrix(actual, predicted).support().tolist() == [200, 150, 20]


def test_support_refuses_a_row_normalised_matrix(labelled):
    """Every row sums to 1 there, so returning ones would be a lie dressed as data."""
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted, normalize="true")
    with pytest.raises(ValueError, match="row-normalised"):
        matrix.support()


def test_the_frame_has_labelled_axes(labelled):
    actual, predicted = labelled
    frame = confusion_matrix(actual, predicted).to_frame()
    assert frame.index.name == "true"
    assert frame.columns.name == "predicted"
    assert frame.loc[2, 1] == 14


# --------------------------------------------------------------------------
# building it
# --------------------------------------------------------------------------


def test_counts_are_the_hand_counted_ones(labelled):
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted).matrix
    assert matrix.tolist() == [[180, 20, 0], [15, 135, 0], [0, 14, 6]]


def test_row_normalisation_gives_recall_per_class(labelled):
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted, normalize="true").matrix
    assert np.diag(matrix) == pytest.approx([0.9, 0.9, 0.3])


def test_all_normalisation_sums_to_one(labelled):
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted, normalize="all").matrix
    assert matrix.sum() == pytest.approx(1.0)


def test_explicit_labels_fix_the_row_order():
    """Needed whenever two matrices are compared: a class absent from one would
    otherwise shift every subsequent row and the two would be silently incomparable."""
    actual = np.array(["b", "a", "a"])
    predicted = np.array(["b", "a", "b"])
    matrix = confusion_matrix(actual, predicted, labels=["b", "a"])
    assert matrix.labels == ("b", "a")
    assert matrix.count("a", "b") == 1.0


def test_labels_that_omit_a_present_class_are_rejected():
    """Silently dropping those samples would understate the error count."""
    actual = np.array([0, 1, 2])
    predicted = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="does not cover class"):
        confusion_matrix(actual, predicted, labels=[0, 1])


def test_misaligned_inputs_are_rejected():
    with pytest.raises(ValueError, match="must align"):
        confusion_matrix(np.zeros(5), np.zeros(4))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="at least one observation"):
        confusion_matrix(np.array([]), np.array([]))


# --------------------------------------------------------------------------
# per-class
# --------------------------------------------------------------------------


def test_per_class_recall_matches_the_hand_computation(labelled):
    actual, predicted = labelled
    per_class = per_class_metrics(actual, predicted)
    assert per_class[0]["recall"] == pytest.approx(0.9)
    assert per_class[1]["recall"] == pytest.approx(0.9)
    assert per_class[2]["recall"] == pytest.approx(0.3)


def test_per_class_error_rate_is_one_minus_recall(labelled):
    actual, predicted = labelled
    for label, metrics in per_class_metrics(actual, predicted).items():
        assert metrics["error_rate"] == pytest.approx(1.0 - metrics["recall"]), label


def test_per_class_support_is_the_true_class_count(labelled):
    actual, predicted = labelled
    per_class = per_class_metrics(actual, predicted)
    assert [per_class[c]["support"] for c in (0, 1, 2)] == [200.0, 150.0, 20.0]


def test_per_class_precision_for_a_class_never_predicted():
    """zero_division=0 rather than nan: a precision of nan propagates, a zero does not."""
    actual = np.array([0, 0, 1, 1])
    predicted = np.zeros(4, dtype=int)
    assert per_class_metrics(actual, predicted)[1]["precision"] == pytest.approx(0.0)


def test_error_rate_by_class_returns_what_repo_c_discarded(labelled):
    """Repo C computed these inside a loop whose only consumer was a commented-out
    print, so its caller never received them."""
    actual, predicted = labelled
    rates = error_rate_by_class(actual, predicted)
    assert rates == pytest.approx({0: 0.1, 1: 0.1, 2: 0.7})


def test_a_class_with_no_samples_is_absent_rather_than_nan():
    """'No evidence about this class' and 'error rate nan' read very differently."""
    actual = np.array([0, 0, 1])
    predicted = np.array([0, 0, 1])
    rates = error_rate_by_class(actual, predicted, labels=[0, 1, 2])
    assert set(rates) == {0, 1}


# --------------------------------------------------------------------------
# top confusions
# --------------------------------------------------------------------------


def test_ranking_by_fraction_surfaces_the_rare_class(labelled):
    """The case the default exists for: the 70%-of-class failure is only 14 samples, so
    a raw count ranks it last and sends the reader to the wrong place."""
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted)

    by_fraction = top_confusions(matrix, n=3)
    assert (by_fraction[0]["true"], by_fraction[0]["predicted"]) == (2, 1)
    assert by_fraction[0]["fraction_of_class"] == pytest.approx(0.7)

    by_count = top_confusions(matrix, n=3, normalize_by_support=False)
    assert (by_count[0]["true"], by_count[0]["predicted"]) == (0, 1)
    assert (by_count[-1]["true"], by_count[-1]["predicted"]) == (2, 1)


def test_the_diagonal_is_never_a_confusion(labelled):
    actual, predicted = labelled
    for entry in top_confusions(confusion_matrix(actual, predicted), n=10):
        assert entry["true"] != entry["predicted"]


def test_zero_count_confusions_are_omitted(labelled):
    actual, predicted = labelled
    entries = top_confusions(confusion_matrix(actual, predicted), n=10)
    assert all(entry["count"] > 0 for entry in entries)
    assert len(entries) == 3  # only three non-zero off-diagonal cells exist


def test_top_confusions_refuses_a_normalised_matrix(labelled):
    actual, predicted = labelled
    matrix = confusion_matrix(actual, predicted, normalize="true")
    with pytest.raises(ValueError, match="count matrix"):
        top_confusions(matrix)


def test_top_confusions_validates_n(labelled):
    actual, predicted = labelled
    with pytest.raises(ValueError, match="n must be at least 1"):
        top_confusions(confusion_matrix(actual, predicted), n=0)
