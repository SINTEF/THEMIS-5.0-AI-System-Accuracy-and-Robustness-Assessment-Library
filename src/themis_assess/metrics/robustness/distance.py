"""Distribution distance — how far the data moved. Survey §5.3.

These are **not robustness scores**, and the survey is emphatic:

    They are not robustness scores in themselves, but provide essential context for
    interpreting observed performance degradation: a large drop in accuracy is more
    expected (and less diagnostic of a fragile model) when the shift distance is large.

A model that loses ten points of accuracy under a tiny shift is fragile. One that loses
ten points under an enormous shift may be doing well. Nothing in §5.1 can tell those
apart; this is what supplies the denominator. The report layer refuses to present a value
from this module as if it were a robustness score.

All three are model-free — functions of two samples, no predictions involved. They can be
computed in raw input space or in a model's feature space, and the survey notes the
latter "is often the only tractable option in high dimension and reflects the model as
well as the data". Which space was used changes the number, so record it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "wasserstein_distance",
    "maximum_mean_discrepancy",
    "kl_divergence",
    "median_heuristic_bandwidth",
]


def _two_samples(p: ArrayLike, q: ArrayLike) -> tuple[NDArray, NDArray]:
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected samples of shape (n_samples, n_features); got {a.shape} and "
            f"{b.shape}."
        )
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Both samples need the same number of features; got {a.shape[1]} and "
            f"{b.shape[1]}."
        )
    if a.shape[0] < 1 or b.shape[0] < 1:
        raise ValueError("Both samples need at least one observation.")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Samples contain NaN or infinity. Drop or impute them first.")
    return a, b


def wasserstein_distance(
    p: ArrayLike,
    q: ArrayLike,
    *,
    order: int = 1,
    aggregate: str = "mean",
) -> float:
    """Earth-mover distance between two samples. Survey §5.3, Eq. 38.

    The most interpretable of the three, because it comes out **in the units of the
    data**. The survey gives the case that makes this concrete: "if every test reading is
    offset from training by a constant delta, then W_p equals exactly delta, as under
    sensor drift". A Wasserstein distance of 3.2 on a temperature sensor means the
    distribution moved by about 3.2 degrees, which is a sentence an engineer can act on.

    Multivariate optimal transport needs a solver this library does not depend on, so
    for more than one feature the per-feature one-dimensional distances are computed
    exactly and combined. That is a lower bound on the true joint distance and ignores
    changes in dependence structure — a shift that only rotates the correlation between
    two features registers as zero here. Use
    :func:`maximum_mean_discrepancy` when joint structure is the question.

    Parameters
    ----------
    order:
        ``p`` in ``W_p``. 1 is the earth-mover distance; 2 weights large displacements
        more heavily.
    aggregate:
        How to combine per-feature distances: ``"mean"``, ``"max"`` (the worst-shifted
        feature) or ``"sum"``.
    """
    from scipy import stats  # scipy is a core dependency

    a, b = _two_samples(p, q)
    if order < 1:
        raise ValueError(f"order must be at least 1; got {order}.")
    if aggregate not in {"mean", "max", "sum"}:
        raise ValueError(f"aggregate must be mean, max or sum; got {aggregate!r}.")

    per_feature = []
    for j in range(a.shape[1]):
        if order == 1:
            per_feature.append(float(stats.wasserstein_distance(a[:, j], b[:, j])))
        else:
            # W_p between empirical measures is the L_p norm of the difference between
            # their quantile functions, evaluated on a common grid.
            grid = np.linspace(0.0, 1.0, max(a.shape[0], b.shape[0], 100), endpoint=False)
            grid = grid + 0.5 / grid.size
            gap = np.abs(np.quantile(a[:, j], grid) - np.quantile(b[:, j], grid))
            per_feature.append(float(np.mean(gap**order) ** (1.0 / order)))

    values = np.asarray(per_feature)
    if aggregate == "mean":
        return float(values.mean())
    if aggregate == "max":
        return float(values.max())
    return float(values.sum())


def median_heuristic_bandwidth(p: ArrayLike, q: ArrayLike, *, max_samples: int = 500) -> float:
    """Kernel bandwidth from the median pairwise distance — the standard default.

    MMD's value depends entirely on the kernel, and picking a bandwidth by hand makes
    numbers incomparable between runs. The median heuristic sets it from the data: half
    the median squared distance among the pooled samples, which puts the kernel on the
    scale the data actually occupies.

    Subsamples above ``max_samples`` so the pairwise computation stays quadratic in a
    bounded number.
    """
    a, b = _two_samples(p, q)
    pooled = np.vstack([a, b])
    if pooled.shape[0] > max_samples:
        rng = np.random.default_rng(0)  # deterministic: a bandwidth must be reproducible
        pooled = pooled[rng.choice(pooled.shape[0], max_samples, replace=False)]
    diffs = pooled[:, None, :] - pooled[None, :, :]
    squared = np.sum(diffs**2, axis=2)
    upper = squared[np.triu_indices_from(squared, k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return float(np.sqrt(max(median, 1e-12) / 2.0))


def maximum_mean_discrepancy(
    p: ArrayLike,
    q: ArrayLike,
    *,
    bandwidth: float | None = None,
    unbiased: bool = True,
) -> float:
    """Kernel two-sample distance. Survey §5.3, Eqs. 40-41.

    ``MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]``

    The one of the three that sees *joint* structure. With a characteristic kernel — the
    Gaussian used here is one — MMD is zero if and only if the two distributions are
    identical, so it detects a change in dependence between features that a per-feature
    comparison cannot. It is the right tool when the question is "did the data change",
    and the wrong one when the question is "by how much", since its scale is set by the
    kernel rather than by the data's units.

    Parameters
    ----------
    bandwidth:
        Gaussian kernel bandwidth. Defaults to the median heuristic, computed from the
        data. The survey: "Sensitivity and scale are set by the kernel and bandwidth" —
        so whichever is used must be reported.
    unbiased:
        Use the unbiased estimator of Eq. 41, which excludes the diagonal terms. It can
        return a small negative value when the true MMD is zero; that is a property of
        the estimator, not an error, and is left visible rather than clipped.

    Returns
    -------
    float
        ``MMD^2``, not ``MMD``. The squared form is what the estimator computes and what
        the survey's equations give; taking a square root of a possibly-negative
        unbiased estimate would force a clip.
    """
    a, b = _two_samples(p, q)
    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(a, b)
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth}.")

    def gaussian(x: NDArray, y: NDArray) -> NDArray:
        squared = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        return np.exp(-squared / (2.0 * bandwidth**2))

    m, n = a.shape[0], b.shape[0]
    k_aa, k_bb, k_ab = gaussian(a, a), gaussian(b, b), gaussian(a, b)

    if unbiased:
        if m < 2 or n < 2:
            raise ValueError(
                f"The unbiased estimator needs at least 2 samples per side; got {m} "
                f"and {n}. Pass unbiased=False for the biased estimator."
            )
        term_a = (k_aa.sum() - np.trace(k_aa)) / (m * (m - 1))
        term_b = (k_bb.sum() - np.trace(k_bb)) / (n * (n - 1))
    else:
        term_a = k_aa.mean()
        term_b = k_bb.mean()
    return float(term_a + term_b - 2.0 * k_ab.mean())


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    n_bins: int = 30,
    eps: float = 1e-10,
) -> float:
    """Kullback-Leibler divergence, from binned histograms. Survey §5.3, Eq. 39.

    ``D_KL(P||Q) = E_P[log(p(x)/q(x))]``

    Included for completeness, and the survey's own caveats are the reason to reach for
    one of the other two first:

    * **Asymmetric.** ``D_KL(P||Q) != D_KL(Q||P)``, and it fails the triangle
      inequality, so it is not a distance despite being used as one.
    * **Blind to geometry.** It "registers only mismatches in probability mass and is
      blind to how far apart the mismatched regions are" — two distributions shifted by
      1 unit and by 100 units can score identically, where Wasserstein separates them.
    * **Hard to estimate.** "bias and variance are substantial in high dimension".

    Estimated here by histogram, per feature, summed. That keeps it usable without a
    density model, at the cost of depending on ``n_bins`` and treating features as
    independent. For anything load-bearing, prefer Wasserstein for magnitude and MMD for
    detection.

    ``eps`` floors the reference density so a region ``P`` occupies and ``Q`` does not
    yields a large finite value rather than infinity.
    """
    a, b = _two_samples(p, q)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2; got {n_bins}.")

    total = 0.0
    for j in range(a.shape[1]):
        lo = min(a[:, j].min(), b[:, j].min())
        hi = max(a[:, j].max(), b[:, j].max())
        if hi <= lo:
            continue  # a feature constant across both samples contributes nothing
        edges = np.linspace(lo, hi, n_bins + 1)
        hist_p, _ = np.histogram(a[:, j], bins=edges, density=False)
        hist_q, _ = np.histogram(b[:, j], bins=edges, density=False)
        dist_p = hist_p / hist_p.sum()
        dist_q = hist_q / hist_q.sum()
        mask = dist_p > 0
        total += float(
            np.sum(dist_p[mask] * np.log(dist_p[mask] / np.maximum(dist_q[mask], eps)))
        )
    return total
