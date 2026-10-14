"""Detector-evaluation metrics — survey §5.5.

The survey presents these as OOD-detection metrics, but nothing in them is specific to
out-of-distribution data: each is a pure function of ``(scores, labels)`` and evaluates
any binary detector that emits a score. They live here under a modality-neutral name so
they can also score a misclassification detector's flags — SafetyCage's, when that
integration lands — without pretending to be about OOD.

Detectors themselves are out of scope for now (``docs/DECISIONS.md``, D20). This module
is the evaluation half only.

Convention, following the survey: the **positive class is what the detector is meant to
catch** — the OOD input, the misclassification, the anomaly — and higher scores mean
"more likely positive".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from themis_assess.core.errors import ThemisAssessError

__all__ = [
    "auroc",
    "auprc",
    "fpr_at_tpr",
    "autc",
    "weighted_autc",
]


def _validate(scores: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    if s.shape != y.shape:
        raise ValueError(
            f"scores and labels must have the same length; got {s.size} and {y.size}."
        )
    if s.size == 0:
        raise ValueError("Need at least one sample.")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity.")
    y = y.astype(bool)
    if y.all() or not y.any():
        raise ThemisAssessError(
            "Both positive and negative samples are required. Got "
            f"{int(y.sum())} positive and {int((~y).sum())} negative. "
            "These metrics are all defined by comparing the two score distributions."
        )
    return s, y


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve. Survey §5.5, Eq. 46.

    ``AUROC = Pr(s(x⁺) > s(x⁻))`` — the probability a random positive scores above a
    random negative. 0.5 is chance, 1.0 is perfect separation.

    Threshold-free, which is its strength and its limitation: a deployed detector must
    commit to one threshold, so report :func:`fpr_at_tpr` alongside it.
    """
    s, y = _validate(scores, labels)
    return float(roc_auc_score(y, s))


def auprc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the precision-recall curve. Survey §5.5, Eq. 47.

    The more informative companion to AUROC when positives are rare — as flagged inputs
    typically are in deployment — because AUROC can stay high while precision on the
    rare class is poor.

    Computed as average precision, the step-wise estimate that does not interpolate
    between operating points.
    """
    s, y = _validate(scores, labels)
    return float(average_precision_score(y, s))


def fpr_at_tpr(scores: ArrayLike, labels: ArrayLike, *, tpr_target: float = 0.95) -> float:
    """False-positive rate at the threshold reaching ``tpr_target``. Survey §5.5.

    In words: how much normal traffic gets falsely flagged when the detector is tuned to
    catch at least 95% of what it is looking for. The one single-threshold metric in the
    family, and with AUROC the standard reporting pair.

    Parameters
    ----------
    tpr_target:
        The true-positive rate to achieve, in ``(0, 1]``. Default 0.95.
    """
    if not 0.0 < tpr_target <= 1.0:
        raise ValueError(f"tpr_target must be in (0, 1]; got {tpr_target}.")
    s, y = _validate(scores, labels)
    fpr, tpr, _ = roc_curve(y, s)
    reached = np.searchsorted(tpr, tpr_target, side="left")
    if reached >= len(tpr):  # pragma: no cover - roc_curve always ends at tpr == 1
        return 1.0
    return float(fpr[reached])


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x``.

    Implemented here rather than via numpy because the function was renamed
    ``trapz`` -> ``trapezoid`` in numpy 2.0, and this library supports numpy 1.26 so
    that the use-case repos can adopt it without a numpy upgrade.
    """
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def _threshold_curves(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FPR and FNR as functions of a threshold sweep over min-max scaled scores."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi == lo:
        raise ThemisAssessError(
            "All scores are identical, so no threshold separates the classes and the "
            "threshold curves are degenerate."
        )
    scaled = (scores - lo) / (hi - lo)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    pos = scaled[labels]
    neg = scaled[~labels]
    # Flagged positive when score >= threshold.
    fnr = np.array([(pos < t).mean() for t in thresholds])
    fpr = np.array([(neg >= t).mean() for t in thresholds])
    return thresholds, fpr, fnr


def autc(scores: ArrayLike, labels: ArrayLike, *, n_thresholds: int = 1000) -> float:
    """Area under the threshold curve. Survey §5.5, Eq. 48.

    ``AUTC = (AUFPR + AUFNR) / 2``, with scores min-max scaled to ``[0, 1]`` and the
    threshold swept across that range. **Lower is better; 0 is the best score.**

    Why it exists alongside AUROC: AUROC rewards ranking but not *separation*, and
    separation is what keeps a detector stable as the threshold moves. Two detectors can
    have identical AUROC and different AUTC when one pushes the two score distributions
    further apart. The survey labels this a recent proposal, so report it as a
    supplement to AUROC rather than a replacement.
    """
    s, y = _validate(scores, labels)
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be at least 2; got {n_thresholds}.")
    thresholds, fpr, fnr = _threshold_curves(s, y, n_thresholds)
    au_fpr = _trapezoid(fpr, thresholds)
    au_fnr = _trapezoid(fnr, thresholds)
    return (au_fpr + au_fnr) / 2.0


def weighted_autc(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    w_fpr: float = 0.5,
    w_fnr: float = 0.5,
    n_thresholds: int = 1000,
) -> float:
    """AUTC with the two error types weighted. Survey §5.5, Eq. 49.

    ``wAUTC = w₁·AUFPR + w₂·AUFNR``, with ``w₁ + w₂ = 1``.

    Use it when the two failures cost differently — missing a genuine anomaly usually
    costs more than a false alarm, but in a high-volume pipeline the reverse can hold.
    The weights are a deployment decision, not a property of the detector, so state them
    wherever the number is reported.
    """
    total = w_fpr + w_fnr
    if not np.isclose(total, 1.0):
        raise ValueError(f"w_fpr and w_fnr must sum to 1; got {w_fpr} + {w_fnr} = {total}.")
    if w_fpr < 0 or w_fnr < 0:
        raise ValueError(f"Weights must be non-negative; got w_fpr={w_fpr}, w_fnr={w_fnr}.")
    s, y = _validate(scores, labels)
    if n_thresholds < 2:
        raise ValueError(f"n_thresholds must be at least 2; got {n_thresholds}.")
    thresholds, fpr, fnr = _threshold_curves(s, y, n_thresholds)
    return w_fpr * _trapezoid(fpr, thresholds) + w_fnr * _trapezoid(fnr, thresholds)
