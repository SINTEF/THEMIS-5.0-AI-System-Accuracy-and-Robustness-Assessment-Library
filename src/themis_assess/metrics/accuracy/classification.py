"""Where a classifier's errors actually are: the confusion matrix and what follows from it.

An aggregate accuracy says how often a model is right. It never says *which* mistakes it
makes, and for most deployments the distinction is the whole decision — a fake-news
classifier that misses hate speech and one that over-flags benign posts can report the
same F1 and be differently unusable.

Repos A, C and D all had a ``compute_confusion_matrix`` and one of them a
``per_class_assessment``. Three things are different here.

*The data is separated from the plot.* Repo D had already adopted a
``compute_*_data`` / ``plot_*`` split; repos A and C had methods that computed and drew in
one call, so the numbers could not be tested or reported without a rendering step. Only
the ``compute`` half lives here; drawing is tranche 8's business.

*scikit-learn is not re-wrapped.* :func:`confusion_matrix` here is scikit-learn's, called
directly. What is added is what scikit-learn does not give: the matrix *labelled*, and the
per-class and per-confusion breakdowns derived from it.

*The per-class error rates are returned.* Repo C's ``error_analysis_by_true_class``
computed them in a loop whose only consumer was a commented-out ``print``, built a
margin-annotated crosstab and never returned it, and returned only the per-sample frame.
The numbers its name promises were computed and discarded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ConfusionMatrix",
    "confusion_matrix",
    "per_class_metrics",
    "top_confusions",
    "error_rate_by_class",
]

Normalize = Literal["true", "pred", "all"] | None


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """A confusion matrix that knows its own labels.

    Rows are true classes, columns predicted — scikit-learn's convention, and the one
    every repo used. Keeping the labels attached is not cosmetic: a bare array's row order
    is ``np.unique`` order, which is not the order a caller passed classes in, and reading
    a matrix against the wrong label order is a mistake that produces a confident and
    completely wrong story about which classes get confused.

    Parameters
    ----------
    matrix:
        ``(n_classes, n_classes)`` counts, or normalised values.
    labels:
        Class labels, in row/column order.
    normalize:
        Which normalisation produced ``matrix``, if any. Recorded because a normalised
        matrix and a count matrix look alike and mean different things.
    """

    matrix: NDArray
    labels: tuple[Any, ...]
    normalize: Normalize = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"A confusion matrix must be square; got {array.shape}.")
        if len(self.labels) != array.shape[0]:
            raise ValueError(
                f"Got {len(self.labels)} label(s) for a {array.shape[0]}x"
                f"{array.shape[1]} matrix."
            )
        object.__setattr__(self, "matrix", array)

    @property
    def n_classes(self) -> int:
        return int(self.matrix.shape[0])

    def label_index(self, label: Any) -> int:
        try:
            return self.labels.index(label)
        except ValueError:
            raise KeyError(
                f"{label!r} is not one of the classes: {list(self.labels)}."
            ) from None

    def count(self, true_label: Any, predicted_label: Any) -> float:
        """How often ``true_label`` was predicted as ``predicted_label``."""
        return float(
            self.matrix[self.label_index(true_label), self.label_index(predicted_label)]
        )

    def support(self) -> NDArray:
        """Samples per true class — the row sums.

        Meaningless for a ``"true"``-normalised matrix, where every row sums to 1 by
        construction, so that case raises rather than returning a row of ones.
        """
        if self.normalize == "true":
            raise ValueError(
                "Support cannot be recovered from a row-normalised matrix: every row "
                "sums to 1 by construction. Build the matrix with normalize=None."
            )
        return self.matrix.sum(axis=1)

    def to_frame(self) -> Any:
        """As a pandas DataFrame with labelled axes."""
        import pandas as pd

        return pd.DataFrame(
            self.matrix,
            index=pd.Index(list(self.labels), name="true"),
            columns=pd.Index(list(self.labels), name="predicted"),
        )


def confusion_matrix(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    normalize: Normalize = None,
) -> ConfusionMatrix:
    """Confusion matrix with its labels attached. Wraps scikit-learn's.

    Parameters
    ----------
    labels:
        Class labels and their order. Defaults to the sorted union of what appears in
        ``actual`` and ``predicted``. **Pass this explicitly when comparing matrices** —
        clean against perturbed, or one subgroup against another — because a class absent
        from one of them would otherwise shift every subsequent row, and the two matrices
        would be silently incomparable.
    normalize:
        ``"true"`` divides by row (recall per class), ``"pred"`` by column (precision per
        class), ``"all"`` by the total. ``None`` gives counts.
    """
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"actual and predicted must align; got {y_true.size} and {y_pred.size}."
        )
    if y_true.size == 0:
        raise ValueError("Need at least one observation.")

    if labels is None:
        resolved = np.unique(np.concatenate([y_true, y_pred]))
    else:
        resolved = np.asarray(list(labels))
        unseen = set(np.unique(np.concatenate([y_true, y_pred])).tolist()) - set(
            resolved.tolist()
        )
        if unseen:
            raise ValueError(
                f"labels does not cover class(es) present in the data: {sorted(unseen)}. "
                f"A confusion matrix missing a class silently drops those samples."
            )

    matrix = sklearn_confusion_matrix(
        y_true, y_pred, labels=resolved, normalize=normalize
    )
    return ConfusionMatrix(
        matrix=matrix,
        labels=tuple(resolved.tolist()),
        normalize=normalize,
        metadata={"n_samples": int(y_true.size)},
    )


def per_class_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
) -> dict[Any, dict[str, float]]:
    """Precision, recall, F1, support and error rate for each class.

    The per-class view survey §5.1.2 asks for whenever an aggregate is reported. Note that
    it is *not* the same thing as a subgroup breakdown, and the library offers both
    deliberately: a class is defined by the label, a subgroup by a property of the input.
    A model can be even-handed across classes and badly unequal across sites.

    ``error_rate`` is ``1 - recall``, included because it is what an operational
    conversation is usually about — "how often do we miss this?" — and because the
    subtraction is easy to get backwards.
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"actual and predicted must align; got {y_true.size} and {y_pred.size}."
        )
    if labels is None:
        resolved = np.unique(np.concatenate([y_true, y_pred]))
    else:
        resolved = np.asarray(list(labels))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=resolved, zero_division=0
    )
    out: dict[Any, dict[str, float]] = {}
    for i, label in enumerate(resolved.tolist()):
        out[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
            "error_rate": float(1.0 - recall[i]),
        }
    return out


def error_rate_by_class(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
) -> dict[Any, float]:
    """Fraction of each true class that was misclassified.

    The quantity repo C's ``error_analysis_by_true_class`` computed inside a loop whose
    only consumer was a commented-out ``print``, and which its caller therefore never
    received. Returned here, which is the entire fix.

    A class with no samples is absent from the result rather than present as ``nan``:
    "we have no evidence about this class" and "this class has an error rate of nan" read
    very differently in a report.
    """
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"actual and predicted must align; got {y_true.size} and {y_pred.size}."
        )
    resolved = (
        np.unique(y_true) if labels is None else np.asarray(list(labels))
    )
    out: dict[Any, float] = {}
    for label in resolved.tolist():
        mask = y_true == label
        n = int(mask.sum())
        if n == 0:
            continue
        out[label] = float(np.count_nonzero(y_pred[mask] != label) / n)
    return out


def top_confusions(
    matrix: ConfusionMatrix, *, n: int = 5, normalize_by_support: bool = True
) -> list[dict[str, Any]]:
    """The most common specific mistakes, worst first.

    The single most actionable output of a confusion matrix, and none of the four repos
    produced it — each drew the matrix and left the reader to scan it, which does not
    scale past a handful of classes.

    Parameters
    ----------
    normalize_by_support:
        Rank by the fraction of the true class that was confused, rather than by raw
        count. On by default, because a raw count ranks by class frequency: the largest
        class's mistakes dominate the list even when a small class is being confused far
        more often, proportionally. The raw count is reported either way.
    """
    if matrix.normalize is not None:
        raise ValueError(
            f"top_confusions needs a count matrix to rank by; this one is normalised by "
            f"{matrix.normalize!r}. Rebuild it with normalize=None."
        )
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}.")

    support = matrix.support()
    entries = []
    for i, true_label in enumerate(matrix.labels):
        for j, predicted_label in enumerate(matrix.labels):
            if i == j:
                continue
            count = float(matrix.matrix[i, j])
            if count == 0:
                continue
            fraction = float(count / support[i]) if support[i] else 0.0
            entries.append(
                {
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": count,
                    "fraction_of_class": fraction,
                    "class_support": float(support[i]),
                }
            )
    key = "fraction_of_class" if normalize_by_support else "count"
    entries.sort(key=lambda e: e[key], reverse=True)
    return entries[:n]
