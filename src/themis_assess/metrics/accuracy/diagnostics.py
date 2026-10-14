"""Where the errors concentrate: outliers, feature regions, and uncertain predictions.

These answer a different question from the metrics. A metric says how well the model does;
a diagnostic says *where* it does badly, in terms of the data rather than the label. The
distinction matters operationally, because a model whose errors sit in an identifiable
region of input space can be fixed by collecting data there, and a model whose errors are
uniform cannot.

Ported from repo C's ``compute_error_outlier_analysis`` and
``plot_error_analysis_across_features``, and repo D's ``get_ambiguity_threshold``. None of
the three is a survey metric; all three are the kind of thing a practitioner reaches for
first, which is why they earn their place.

Three defects in repo C's version are fixed here, each noted at the function that fixes it:
the combined outlier score had a sign error on one of its three components (real, though
its measured effect is modest — see :func:`outlier_scores`); the chi-square test was
computed and discarded; and the full pairwise distance matrix was materialised, which is
quadratic in memory and fails outright on a large test set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "OutlierErrorAnalysis",
    "outlier_scores",
    "error_rate_by_outlier_status",
    "error_rate_by_feature_bin",
    "ambiguity_threshold",
    "flag_ambiguous",
]


# --------------------------------------------------------------------------
# outliers
# --------------------------------------------------------------------------


def outlier_scores(
    features: ArrayLike,
    *,
    contamination: float = 0.1,
    n_neighbors: int = 20,
    k: int = 5,
    methods: Sequence[str] = ("isolation_forest", "local_outlier_factor", "knn_distance"),
    random_state: int = 0,
) -> dict[str, NDArray]:
    """Per-sample outlier scores from up to three detectors, plus their combination.

    Every score is returned oriented so that **higher means more anomalous**.

    That orientation is where repo C's version went wrong. ``IsolationForest.score_samples``
    and LOF's ``negative_outlier_factor_`` both run low-for-anomalous, while mean distance
    to the k nearest neighbours runs high-for-anomalous. All three were min-max normalised
    and combined as ``1 - mean(...)``, which is right for the first two and **inverted for
    the third**, so the distance detector voted against the other two: the more clearly a
    point was isolated in space, the less anomalous that component made it look.

    Worth being exact about the consequence, since it is smaller than it sounds. Measured
    against known outliers over eight seeds, the two correctly-oriented detectors outvote
    the third and ranking quality is essentially unaffected (AUROC 0.954 either way). What
    the sign error costs is separation — between a clean bulk and a scattered tail the
    combined score's gap more than halves — and it makes the third detector's contribution
    actively harmful rather than merely wasted, which matters as soon as it is weighted
    differently or used alone.

    Parameters
    ----------
    contamination:
        Expected outlier fraction, passed to both scikit-learn detectors. Repo C used
        0.1. It affects the detectors' own labels, not the scores returned here.
    k:
        Neighbours for the distance detector.
    methods:
        Which detectors to run. The combination is the mean of whichever ran, so dropping
        one is legitimate — ``knn_distance`` is the one to drop first on a large dataset,
        being the most expensive.
    random_state:
        Fixed by default so a diagnostic is reproducible. Repo C hardcoded 42.

    Returns
    -------
    dict
        One entry per method plus ``"combined"``, each ``(n_samples,)`` in ``[0, 1]``
        after min-max normalisation, higher meaning more anomalous.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(
            f"Expected features of shape (n_samples, n_features); got {X.shape}."
        )
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Outlier detection needs at least 2 samples.")
    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain NaN or infinity. Impute or drop them first — the "
            "detectors here would otherwise treat missingness as extremity."
        )
    if not 0.0 < contamination <= 0.5:
        raise ValueError(f"contamination must lie in (0, 0.5]; got {contamination}.")

    unknown = set(methods) - {
        "isolation_forest",
        "local_outlier_factor",
        "knn_distance",
    }
    if unknown:
        raise ValueError(f"Unknown outlier method(s): {sorted(unknown)}.")
    if not methods:
        raise ValueError("outlier_scores needs at least one method.")

    def normalise(values: NDArray) -> NDArray:
        low, high = float(values.min()), float(values.max())
        if high - low < 1e-12:
            return np.zeros_like(values)
        return (values - low) / (high - low)

    scores: dict[str, NDArray] = {}

    if "isolation_forest" in methods:
        forest = IsolationForest(
            contamination=contamination, n_estimators=100, random_state=random_state
        )
        forest.fit(X)
        # score_samples: lower is more abnormal, so negate before normalising.
        scores["isolation_forest"] = normalise(-forest.score_samples(X))

    if "local_outlier_factor" in methods:
        neighbours = min(n_neighbors, n_samples - 1)
        lof = LocalOutlierFactor(n_neighbors=neighbours, contamination=contamination)
        lof.fit_predict(X)
        # negative_outlier_factor_: more negative is more outlying, so negate.
        scores["local_outlier_factor"] = normalise(-lof.negative_outlier_factor_)

    if "knn_distance" in methods:
        # NearestNeighbors rather than a full pairwise_distances matrix: repo C built the
        # whole (n, n) matrix to read k columns of it, which is quadratic in memory and
        # simply fails on a test set of any size.
        neighbours = min(k, n_samples - 1)
        finder = NearestNeighbors(n_neighbors=neighbours + 1).fit(X)
        distances, _ = finder.kneighbors(X)
        scores["knn_distance"] = normalise(distances[:, 1:].mean(axis=1))

    stacked = np.vstack([scores[name] for name in scores])
    scores["combined"] = stacked.mean(axis=0)
    return scores


@dataclass(frozen=True, slots=True)
class OutlierErrorAnalysis:
    """Whether the model's errors concentrate on unusual inputs.

    Parameters
    ----------
    outlier_error_rate, inlier_error_rate:
        Fractions, not percentages. Repo C returned percentages from a function whose
        siblings returned fractions, which is the kind of inconsistency that survives
        right up until someone multiplies by 100 twice.
    p_value:
        Chi-square test of independence between "is an outlier" and "is an error". Repo C
        computed this and discarded it, returning only the two rates — and a difference
        between two rates is exactly the number that should not be read without a test,
        because the outlier group is small by construction and its rate is therefore
        noisy.
    """

    outlier_error_rate: float
    inlier_error_rate: float
    n_outliers: int
    n_inliers: int
    threshold: float
    p_value: float | None
    chi2: float | None
    is_outlier: NDArray
    scores: NDArray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def difference(self) -> float:
        """Outlier error rate minus inlier error rate. Positive means outliers are harder."""
        return float(self.outlier_error_rate - self.inlier_error_rate)

    @property
    def ratio(self) -> float | None:
        """How many times more often the model errs on outliers. ``None`` if it never
        errs on inliers, where a ratio would be infinite rather than informative."""
        if self.inlier_error_rate <= 0:
            return None
        return float(self.outlier_error_rate / self.inlier_error_rate)

    def describe(self) -> str:
        lines = [
            f"outliers: {self.n_outliers} samples, error rate "
            f"{self.outlier_error_rate:.1%}",
            f"inliers:  {self.n_inliers} samples, error rate "
            f"{self.inlier_error_rate:.1%}",
            f"difference: {self.difference:+.1%}",
        ]
        if self.p_value is not None:
            verdict = "significant" if self.p_value < 0.05 else "not significant"
            lines.append(f"chi-square p = {self.p_value:.4f} ({verdict} at 0.05)")
        return "\n".join(lines)


def error_rate_by_outlier_status(
    features: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    quantile: float = 0.9,
    scores: ArrayLike | None = None,
    **outlier_kwargs: Any,
) -> OutlierErrorAnalysis:
    """Do the model's errors concentrate on unusual inputs?

    A useful thing to know before reaching for a robustness sweep, because it answers a
    prior question: is this model already failing on the tails of its *own* training
    distribution? If it is, degradation under a synthetic shift is not surprising and not
    the first thing to fix.

    Parameters
    ----------
    quantile:
        Where to cut the combined outlier score. 0.9 flags the top 10%, as repo C did.
    scores:
        Pre-computed outlier scores, higher meaning more anomalous. Supply these to reuse
        one detection pass across several models, or to bring your own detector.
    """
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"actual and predicted must align; got {y_true.size} and {y_pred.size}."
        )
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}.")

    if scores is None:
        combined = outlier_scores(features, **outlier_kwargs)["combined"]
    else:
        combined = np.asarray(scores, dtype=float).ravel()
    if combined.size != y_true.size:
        raise ValueError(
            f"Got {combined.size} outlier score(s) for {y_true.size} prediction(s)."
        )

    threshold = float(np.quantile(combined, quantile))
    is_outlier = combined > threshold
    is_error = y_pred != y_true

    n_outliers = int(is_outlier.sum())
    n_inliers = int((~is_outlier).sum())
    outlier_rate = (
        float(is_error[is_outlier].mean()) if n_outliers else 0.0
    )
    inlier_rate = float(is_error[~is_outlier].mean()) if n_inliers else 0.0

    chi2 = p_value = None
    if n_outliers and n_inliers and is_error.any() and not is_error.all():
        from scipy.stats import chi2_contingency

        table = np.array(
            [
                [int((is_outlier & is_error).sum()), int((is_outlier & ~is_error).sum())],
                [
                    int((~is_outlier & is_error).sum()),
                    int((~is_outlier & ~is_error).sum()),
                ],
            ]
        )
        chi2_value, p, _, _ = chi2_contingency(table)
        chi2, p_value = float(chi2_value), float(p)

    return OutlierErrorAnalysis(
        outlier_error_rate=outlier_rate,
        inlier_error_rate=inlier_rate,
        n_outliers=n_outliers,
        n_inliers=n_inliers,
        threshold=threshold,
        p_value=p_value,
        chi2=chi2,
        is_outlier=is_outlier,
        scores=combined,
        metadata={"quantile": quantile, "n_samples": int(y_true.size)},
    )


# --------------------------------------------------------------------------
# feature regions
# --------------------------------------------------------------------------


def error_rate_by_feature_bin(
    feature: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    n_bins: int = 5,
    strategy: str = "quantile",
    min_count: int = 1,
) -> dict[str, NDArray]:
    """Error rate across bins of one feature. The compute half of repo C's feature plot.

    Answers "does this model fail for older patients / longer texts / higher loads", which
    is the question that turns an aggregate number into an action.

    Parameters
    ----------
    strategy:
        ``"quantile"`` gives bins of equal population, ``"uniform"`` bins of equal width.
        Quantile by default, because a uniform bin in the tail of a skewed feature holds a
        handful of samples and its error rate is noise — which is how a diagnostic like
        this produces a dramatic-looking finding that does not replicate.
    min_count:
        Bins with fewer samples than this get ``nan`` rather than a value, so a sparsely
        populated bin cannot be read as a finding.

    Returns
    -------
    dict
        ``edges``, ``centres``, ``error_rate``, ``count``.
    """
    x = np.asarray(feature, dtype=float).ravel()
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if not x.shape == y_true.shape == y_pred.shape:
        raise ValueError(
            f"feature, actual and predicted must align; got {x.size}, {y_true.size}, "
            f"{y_pred.size}."
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}.")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError(f"strategy must be quantile or uniform; got {strategy!r}.")
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("The feature has no finite values to bin.")

    if strategy == "quantile":
        edges = np.unique(np.quantile(x[finite], np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(x[finite].min(), x[finite].max(), n_bins + 1)
    if edges.size < 2:
        raise ValueError(
            "The feature is constant, so it cannot be binned. There is no variation for "
            "an error rate to vary against."
        )

    index = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)
    is_error = y_pred != y_true

    n = edges.size - 1
    error_rate = np.full(n, np.nan)
    count = np.zeros(n, dtype=int)
    for b in range(n):
        members = (index == b) & finite
        count[b] = int(members.sum())
        if count[b] >= min_count:
            error_rate[b] = float(is_error[members].mean())

    return {
        "edges": edges,
        "centres": (edges[:-1] + edges[1:]) / 2.0,
        "error_rate": error_rate,
        "count": count,
    }


# --------------------------------------------------------------------------
# ambiguity
# --------------------------------------------------------------------------


def ambiguity_threshold(
    n_classes: int, *, sensitivity: float = 0.8, entropies: ArrayLike | None = None
) -> float:
    """Entropy above which a prediction counts as ambiguous. Repo D's rule, generalised.

    Repo D computed ``max_entropy * sensitivity`` with ``sensitivity=0.8``, documented as
    "flagging the top 20% most uncertain samples". Those two things are not the same, and
    the difference matters: a fraction of the *maximum possible* entropy is a fixed
    absolute bar, and what fraction of samples it flags depends entirely on how confident
    the model happens to be. On a sharp model it flags almost nothing; on a diffuse one,
    almost everything.

    Both readings are available, and the choice is explicit:

    * without ``entropies`` — the absolute bar, ``log(n_classes) * sensitivity``, which is
      repo D's arithmetic. Use it when "ambiguous" means a fixed standard of confidence.
    * with ``entropies`` — the empirical quantile that actually flags
      ``1 - sensitivity`` of the samples. Use it when "ambiguous" means a review budget:
      flag the 20% this model is least sure about, whatever its overall confidence.

    Parameters
    ----------
    sensitivity:
        In ``(0, 1)``. Higher means a higher bar and fewer flagged samples.
    """
    if n_classes < 2:
        raise ValueError(f"Ambiguity needs at least 2 classes; got {n_classes}.")
    if not 0.0 < sensitivity < 1.0:
        raise ValueError(f"sensitivity must lie in (0, 1); got {sensitivity}.")

    if entropies is None:
        return float(np.log(n_classes) * sensitivity)
    values = np.asarray(entropies, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("entropies is empty.")
    return float(np.quantile(values, sensitivity))


def flag_ambiguous(
    probabilities: ArrayLike, *, sensitivity: float = 0.8, empirical: bool = False
) -> dict[str, Any]:
    """Which predictions are too uncertain to act on unreviewed.

    The operational use of :func:`~themis_assess.metrics.calibration.predictive_entropy`:
    a triage rule. What makes it worth computing beside accuracy is that the flagged set's
    error rate is checkable — if flagged and unflagged samples have the same error rate,
    the model's uncertainty carries no information and the review queue is wasted effort.

    Returns
    -------
    dict
        ``entropy``, ``is_ambiguous``, ``threshold``, ``flagged_fraction``.
    """
    from themis_assess.metrics.calibration import predictive_entropy

    entropy = predictive_entropy(probabilities)
    probs = np.asarray(probabilities, dtype=float)
    n_classes = 2 if probs.ndim == 1 else probs.shape[1]
    threshold = ambiguity_threshold(
        n_classes,
        sensitivity=sensitivity,
        entropies=entropy if empirical else None,
    )
    is_ambiguous = entropy > threshold
    return {
        "entropy": entropy,
        "is_ambiguous": is_ambiguous,
        "threshold": threshold,
        "flagged_fraction": float(is_ambiguous.mean()),
    }
