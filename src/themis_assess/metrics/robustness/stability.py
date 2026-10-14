"""Prediction-stability metrics — survey §5.1.3, Equations 23-25.

These measure something the accuracy-based metrics cannot: whether individual
predictions stay put. A model can hold a respectable average accuracy while its
per-sample predictions oscillate under mild perturbation, and only this family sees it.

**A naming correction.** Both repo A's
``ClassificationRobustnessAssessment.flip_rate`` and repo D's
``robustness_utils.flip_rate`` compute the survey's *flip probability* (Eq. 23) over a
single clean/perturbed pair. In the survey, ``FR`` (Eq. 24) is the baseline-normalised
quantity. The names here follow the survey (``docs/DECISIONS.md``, D19), so
``flip_rate`` means something different from what those two functions returned.

**Sequences, not pairs.** Eq. 23 is defined over a *perturbation sequence* — a run of
inputs at monotonically increasing severity — and counts transitions between
consecutive frames. :func:`flip_probability` takes that sequence.
:func:`pairwise_flip_probability` covers the weaker clean-versus-perturbed comparison
the two existing implementations do, and is kept because it is genuinely useful when no
sequence is available; it is not the same number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.core.errors import safe_divide

__all__ = [
    "flip_probability",
    "pairwise_flip_probability",
    "flip_rate",
    "mean_flip_rate",
    "confidence_retention",
]


def flip_probability(prediction_sequences: ArrayLike) -> float:
    """Rate of prediction change between consecutive frames. Survey §5.1.3, Eq. 23.

    ``FP_φ = (1/M) Σ_i (1/(T-1)) Σ_t 1[f(x_{i,t}) ≠ f(x_{i,t+1})]``

    Parameters
    ----------
    prediction_sequences:
        Array of shape ``(M, T)`` of predicted class labels: ``M`` samples, each with
        ``T`` frames of increasing perturbation severity. Labels may be ints or
        strings.

    Returns
    -------
    float
        In ``[0, 1]``; 0 means no prediction ever changed.

    Notes
    -----
    No ground-truth labels are needed — this is a stability measure, not an accuracy
    measure, which makes it usable on unlabelled data.
    """
    preds = np.asarray(prediction_sequences)
    if preds.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array of shape (n_samples, n_frames), got shape {preds.shape}. "
            f"For a single clean/perturbed pair use pairwise_flip_probability instead."
        )
    n_samples, n_frames = preds.shape
    if n_samples == 0:
        raise ValueError("flip_probability needs at least one sample.")
    if n_frames < 2:
        raise ValueError(
            f"A perturbation sequence needs at least 2 frames to have a transition; "
            f"got {n_frames}."
        )
    flips = preds[:, :-1] != preds[:, 1:]
    return float(flips.mean())


def pairwise_flip_probability(
    clean_predictions: ArrayLike, perturbed_predictions: ArrayLike
) -> float:
    """Fraction of predictions that changed between clean and perturbed input.

    Not a survey equation. This is the quantity repo A's and repo D's ``flip_rate``
    functions computed, preserved here under an honest name. It is a special case of
    :func:`flip_probability` with ``T = 2``, and is *not* interchangeable with it:
    Eq. 23 averages transitions along a graded sequence, which is a different and more
    informative measurement.
    """
    clean = np.asarray(clean_predictions).ravel()
    perturbed = np.asarray(perturbed_predictions).ravel()
    if clean.shape != perturbed.shape:
        raise ValueError(
            f"clean_predictions and perturbed_predictions must have the same length; "
            f"got {clean.size} and {perturbed.size}."
        )
    if clean.size == 0:
        raise ValueError("pairwise_flip_probability needs at least one prediction.")
    return float(np.mean(clean != perturbed))


def flip_rate(fp_model: float, fp_baseline: float) -> float:
    """Flip probability relative to a baseline model's. Survey §5.1.3, Eq. 24.

    ``FR_φ = FP^f_φ / FP^base_φ``

    Below 1 means the model's predictions are more stable than the baseline's under the
    same perturbation. Normalising is what makes flip probabilities comparable across
    perturbations of different difficulty.
    """
    return safe_divide(float(fp_model), float(fp_baseline), what="baseline flip probability")


def mean_flip_rate(flip_rates: Mapping[str, float] | Sequence[float]) -> float:
    """Mean flip rate across perturbation types. Survey §5.1.3, Eq. 25.

    ``mFR = (1/|Φ|) Σ_φ FR_φ``

    A low value means predictions stay consistent as severity increases — a stability
    that accuracy-based scores do not capture.
    """
    values = list(flip_rates.values()) if isinstance(flip_rates, Mapping) else list(flip_rates)
    if not values:
        raise ValueError("mean_flip_rate needs at least one perturbation.")
    return float(sum(values) / len(values))


def confidence_retention(
    clean_confidence: ArrayLike, perturbed_confidence: ArrayLike
) -> float:
    """Fraction of confidence retained on the originally predicted class.

    Repo-A extension, not in the survey. Ported from repo D's
    ``robustness_utils.confidence_retention``.

    **Both inputs must be probabilities assigned to the same class — the one predicted
    on the clean input.** Using the perturbed prediction's own top-class probability
    instead would hide flips: a confident switch to a different class would read as
    perfectly retained confidence. This caution is repo D's, and it is the whole reason
    the function is not simply "mean top-1 probability after perturbation".

    A soft counterpart to :func:`flip_probability`: it sees a prediction sliding toward
    the boundary before it actually crosses, and unlike the decision-margin metrics of
    survey §5.2.1 it needs no gradients — only class probabilities.
    """
    clean = np.asarray(clean_confidence, dtype=float).ravel()
    perturbed = np.asarray(perturbed_confidence, dtype=float).ravel()
    if clean.shape != perturbed.shape:
        raise ValueError(
            f"clean_confidence and perturbed_confidence must have the same length; "
            f"got {clean.size} and {perturbed.size}."
        )
    if clean.size == 0:
        raise ValueError("confidence_retention needs at least one sample.")
    return safe_divide(
        float(perturbed.mean()), float(clean.mean()), what="mean clean confidence"
    )
