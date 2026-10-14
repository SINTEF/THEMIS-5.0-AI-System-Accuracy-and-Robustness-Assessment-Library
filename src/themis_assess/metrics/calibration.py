"""Calibration under shift. Survey §5.4.

A distinct axis of robustness, and the survey is blunt about why it needs its own
section: "a model can degrade moderately in accuracy yet become severely overconfident
under shift". Accuracy metrics cannot see that, and a downstream system that acts on
confidence will.

The prescription that governs every function here:

    calibration established on clean data need not survive a shift... a robustness
    report that cites only clean-data calibration can be misleading

So these are computed on the *shifted* data. Nothing in the signatures enforces that —
they are pure functions of predictions and outcomes — but the report layer checks it.

Repo D had ECE, predictive entropy, temperature scaling and a reliability plot; repo C
had Brier and log loss. Neither had anything for forecasting, which is where PICP, MPIW
and CRPS come in.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    # classification
    "expected_calibration_error",
    "reliability_bins",
    "brier_score",
    "negative_log_likelihood",
    "predictive_entropy",
    "fit_temperature",
    "apply_temperature",
    # forecasting
    "prediction_interval_coverage",
    "mean_prediction_interval_width",
    "interval_score",
    "crps_ensemble",
    "crps_gaussian",
    "energy_score",
]


def _probabilities(probs: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(probs, dtype=float)
    if array.ndim == 1:
        array = np.column_stack([1.0 - array, array])
    if array.ndim != 2:
        raise ValueError(
            f"Expected probabilities of shape (n_samples, n_classes) or (n_samples,) "
            f"for the binary case; got shape {array.shape}."
        )
    if array.size == 0:
        raise ValueError("Need at least one sample.")
    if not np.isfinite(array).all():
        raise ValueError("Probabilities contain NaN or infinity.")
    if (array < -1e-9).any() or (array > 1 + 1e-9).any():
        raise ValueError("Probabilities must lie in [0, 1]. Apply a softmax first.")
    totals = array.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-4):
        raise ValueError(
            f"Each row of probabilities must sum to 1; row sums range from "
            f"{totals.min():.4f} to {totals.max():.4f}."
        )
    return array


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def reliability_bins(
    probs: ArrayLike, labels: ArrayLike, *, n_bins: int = 15
) -> dict[str, NDArray]:
    """Per-bin confidence, accuracy and count — the reliability diagram's data.

    Separated from the plot so the numbers can be tested and reported without a
    rendering step, following repo D's ``compute_*_data`` / ``plot_*`` convention.

    Returns
    -------
    dict
        ``bin_lower``, ``bin_upper``, ``confidence``, ``accuracy``, ``count``. Empty
        bins carry ``NaN`` confidence and accuracy with a count of zero, rather than
        being dropped — a gap in the diagram is informative.
    """
    p = _probabilities(probs)
    y = np.asarray(labels).ravel()
    if y.size != p.shape[0]:
        raise ValueError(f"Got {y.size} label(s) for {p.shape[0]} prediction(s).")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}.")

    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so a confidence of exactly 1.0 lands in the last one.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    out = {
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "confidence": np.full(n_bins, np.nan),
        "accuracy": np.full(n_bins, np.nan),
        "count": np.zeros(n_bins, dtype=int),
    }
    for b in range(n_bins):
        members = index == b
        count = int(members.sum())
        out["count"][b] = count
        if count:
            out["confidence"][b] = float(confidence[members].mean())
            out["accuracy"][b] = float(correct[members].mean())
    return out


def expected_calibration_error(
    probs: ArrayLike, labels: ArrayLike, *, n_bins: int = 15
) -> float:
    """Weighted gap between confidence and accuracy. Survey §5.4, Eq. 42.

    ``ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|``

    Zero is perfect. A model that says 80% and is right 80% of the time scores 0 whatever
    its accuracy — this measures honesty, not skill.

    Bin count matters and is not neutral: too few bins hide miscalibration by averaging
    over-and under-confident regions together, too many leave bins with a handful of
    samples and a noisy accuracy estimate. Fifteen is the common default; report it.
    """
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    total = int(bins["count"].sum())
    if total == 0:  # pragma: no cover - _probabilities rejects empty input
        raise ValueError("No samples to calibrate.")
    occupied = bins["count"] > 0
    gaps = np.abs(bins["accuracy"][occupied] - bins["confidence"][occupied])
    return float(np.sum(bins["count"][occupied] / total * gaps))


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    """Mean squared error of the predicted probability vector. Survey §5.4.

    A *proper* scoring rule: minimised in expectation only when the predicted
    distribution equals the true one, so it cannot be gamed by hedging. Unlike ECE it
    penalises being wrong as well as being dishonest, which makes the two complementary
    rather than interchangeable — ECE isolates calibration, Brier folds in sharpness.

    Multi-class form, averaged over samples and classes.
    """
    p = _probabilities(probs)
    y = np.asarray(labels).ravel()
    if y.size != p.shape[0]:
        raise ValueError(f"Got {y.size} label(s) for {p.shape[0]} prediction(s).")
    onehot = np.zeros_like(p)
    onehot[np.arange(y.size), y.astype(int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def negative_log_likelihood(
    probs: ArrayLike, labels: ArrayLike, *, eps: float = 1e-12
) -> float:
    """Mean negative log probability of the true class. Survey §5.4.

    The other standard proper scoring rule. Far harsher than Brier on confident
    mistakes — a probability of 0.001 on the truth costs ~6.9 nats, where Brier caps a
    single sample's contribution at 2. That makes it the more sensitive detector of
    overconfidence under shift, and the more volatile number.

    ``eps`` floors the probability so a confident error is expensive rather than
    infinite.
    """
    p = _probabilities(probs)
    y = np.asarray(labels).ravel().astype(int)
    if y.size != p.shape[0]:
        raise ValueError(f"Got {y.size} label(s) for {p.shape[0]} prediction(s).")
    truth = np.clip(p[np.arange(y.size), y], eps, 1.0)
    return float(-np.mean(np.log(truth)))


def predictive_entropy(probs: ArrayLike) -> NDArray[np.float64]:
    """Per-sample Shannon entropy of the predicted distribution.

    Ported from repo D. Not a calibration metric — it needs no labels — but the natural
    companion: entropy says how uncertain the model *claims* to be, and calibration says
    whether that claim is honest. Under shift a well-behaved model's entropy rises; one
    that stays confident while its accuracy falls is the failure mode §5.4 is about.

    In classification, entropy is what plays the role that interval width plays in
    forecasting — the sharpness half of the sharpness-subject-to-calibration principle.
    """
    p = _probabilities(probs)
    safe = np.clip(p, 1e-12, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def fit_temperature(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
    tolerance: float = 1e-4,
) -> float:
    """Find the temperature minimising negative log likelihood. Repo-A port of repo D.

    Temperature scaling divides the logits by a single learned scalar before the
    softmax. It cannot change which class is predicted — dividing every logit by the
    same positive number preserves their order — so accuracy is exactly unchanged and
    only the confidence moves. ``T > 1`` softens an overconfident model; ``T < 1``
    sharpens an underconfident one.

    Repo D fitted it with a torch optimiser. Here it is a bounded scalar minimisation
    via golden section, which needs no torch, is deterministic, and is more than
    adequate for a one-parameter convex-in-practice problem.

    **Fit on a held-out set, not on the shifted data being evaluated.** A temperature
    fitted on the same data it is then judged on will look perfectly calibrated and tell
    you nothing.
    """
    z = np.asarray(logits, dtype=float)
    if z.ndim == 1:
        z = np.column_stack([np.zeros_like(z), z])
    y = np.asarray(labels).ravel().astype(int)
    if y.size != z.shape[0]:
        raise ValueError(f"Got {y.size} label(s) for {z.shape[0]} prediction(s).")
    low, high = bounds
    if not 0 < low < high:
        raise ValueError(f"bounds must satisfy 0 < low < high; got {bounds}.")

    def loss(temperature: float) -> float:
        return negative_log_likelihood(apply_temperature(z, temperature), y)

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c, d = b - golden * (b - a), a + golden * (b - a)
    while b - a > tolerance:
        if loss(c) < loss(d):
            b, d = d, c
            c = b - golden * (b - a)
        else:
            a, c = c, d
            d = a + golden * (b - a)
    return float((a + b) / 2.0)


def apply_temperature(logits: ArrayLike, temperature: float) -> NDArray[np.float64]:
    """Softmax of ``logits / temperature``, computed stably."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}.")
    z = np.asarray(logits, dtype=float)
    if z.ndim == 1:
        z = np.column_stack([np.zeros_like(z), z])
    scaled = z / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# forecasting
# --------------------------------------------------------------------------


def prediction_interval_coverage(
    lower: ArrayLike, upper: ArrayLike, actual: ArrayLike
) -> float:
    """Fraction of observations inside their prediction interval. Survey §5.4, Eq. 43.

    ``PICP = (1/n) sum 1[L_i <= y_i <= U_i]``

    Compare against the nominal level: a 90% interval covering 62% of observations is
    badly miscalibrated, and under shift that is the usual direction.

    **Never report this without** :func:`mean_prediction_interval_width`. The survey:
    "calibration on its own can be achieved trivially: arbitrarily wide intervals reach
    any coverage level". The report layer enforces the pairing.
    """
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    y = np.asarray(actual, dtype=float).ravel()
    if not lo.shape == hi.shape == y.shape:
        raise ValueError(
            f"lower, upper and actual must align; got {lo.size}, {hi.size}, {y.size}."
        )
    if lo.size == 0:
        raise ValueError("Need at least one observation.")
    if np.any(hi < lo):
        raise ValueError("Every upper bound must be at or above its lower bound.")
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_prediction_interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean interval width — sharpness. Survey §5.4.

    ``MPIW = (1/n) sum (U_i - L_i)``

    Read with coverage, and read as a pair the two diagnose both failure directions
    under shift: coverage collapsing while widths stay at their training values, or
    coverage surviving only because the intervals widened into uselessness. The survey's
    principle is "maximize sharpness subject to calibration".
    """
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    if lo.shape != hi.shape:
        raise ValueError(f"lower and upper must align; got {lo.size} and {hi.size}.")
    if lo.size == 0:
        raise ValueError("Need at least one interval.")
    if np.any(hi < lo):
        raise ValueError("Every upper bound must be at or above its lower bound.")
    return float(np.mean(hi - lo))


def interval_score(
    lower: ArrayLike, upper: ArrayLike, actual: ArrayLike, *, alpha: float = 0.1
) -> float:
    """Coverage and sharpness in one proper score. Not a survey equation.

    ``(U - L) + (2/alpha)(L - y) if y < L + (2/alpha)(y - U) if y > U``

    Included because the survey insists PICP and MPIW be read together but offers no
    single number that does it. The interval score is the standard proper scoring rule
    for intervals: width is paid for directly, and a miss is penalised in proportion to
    how far outside it fell, scaled so that at nominal level ``alpha`` the two pressures
    balance. Lower is better.

    It does not replace reporting the pair — a single number cannot say *which* of the
    two went wrong — but it ranks models correctly where PICP alone does not.
    """
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    y = np.asarray(actual, dtype=float).ravel()
    if not lo.shape == hi.shape == y.shape:
        raise ValueError(
            f"lower, upper and actual must align; got {lo.size}, {hi.size}, {y.size}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
    if np.any(hi < lo):
        raise ValueError("Every upper bound must be at or above its lower bound.")
    width = hi - lo
    below = (2.0 / alpha) * np.clip(lo - y, 0.0, None)
    above = (2.0 / alpha) * np.clip(y - hi, 0.0, None)
    return float(np.mean(width + below + above))


def crps_ensemble(samples: ArrayLike, actual: ArrayLike, *, fair: bool = False) -> float:
    """Continuous ranked probability score from an ensemble. Survey §5.4, Eq. 44.

    ``CRPS(F, y) = E_F|X - y| - 0.5 * E_F|X - X'|``

    The forecasting counterpart of the log score, and the survey notes the property that
    makes it easy to interpret: "For a point forecast, CRPS reduces to the absolute
    error, which makes it directly comparable with MAE." A single-member ensemble scores
    exactly its MAE, so CRPS and MAE sit on the same axis and a probabilistic forecast
    can be compared directly against a point one.

    Parameters
    ----------
    samples:
        ``(n_observations, n_ensemble_members)`` draws from each predictive distribution.
    actual:
        ``(n_observations,)`` outcomes.
    fair:
        Which estimator of the spread term ``E|X - X'|`` to use, and the choice matters
        for a small ensemble. The default plug-in estimator averages over all ``m^2``
        pairs including the ``m`` zero self-pairs, which underestimates the spread and so
        *rewards under-dispersion*: an ensemble of ten members is scored as if it were
        slightly tighter than it is. ``fair=True`` averages over the ``m(m-1)`` distinct
        pairs instead, which is unbiased for the spread of the underlying distribution
        and is the right choice below roughly fifty members. Both agree in the limit and
        both reduce to MAE at ``m = 1``. Whichever is used, use it consistently — the
        two are not comparable, and :func:`energy_score` takes the same flag.
    """
    ensemble = np.asarray(samples, dtype=float)
    if ensemble.ndim == 1:
        ensemble = ensemble[:, None]
    y = np.asarray(actual, dtype=float).ravel()
    if ensemble.shape[0] != y.size:
        raise ValueError(
            f"Got {ensemble.shape[0]} ensemble row(s) for {y.size} observation(s)."
        )
    if ensemble.shape[1] == 0:
        raise ValueError("Each observation needs at least one ensemble member.")

    m = ensemble.shape[1]
    accuracy_term = np.mean(np.abs(ensemble - y[:, None]), axis=1)
    if m == 1:
        spread_term = np.zeros_like(accuracy_term)
    else:
        # Sorted form: sum over all m^2 ordered pairs of |x_i - x_j| equals
        # 2 * sum_i (2i - m + 1) * x_(i) on the sorted ensemble.
        ordered = np.sort(ensemble, axis=1)
        weights = 2 * np.arange(m) - m + 1
        pair_total = 2.0 * np.sum(ordered * weights, axis=1)
        divisor = m * (m - 1) if fair else m * m
        spread_term = pair_total / divisor
    return float(np.mean(accuracy_term - 0.5 * spread_term))


def crps_gaussian(mean: ArrayLike, std: ArrayLike, actual: ArrayLike) -> float:
    """CRPS with a Gaussian predictive distribution, in closed form. Survey §5.4.

    ``sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]`` with ``z = (y - mu)/sigma``.

    Exact and fast where a model reports a mean and a standard deviation, which is the
    common case for a forecaster with a Gaussian head — no need to sample an ensemble
    just to score it.
    """
    from scipy import stats  # scipy is a core dependency

    mu = np.asarray(mean, dtype=float).ravel()
    sigma = np.asarray(std, dtype=float).ravel()
    y = np.asarray(actual, dtype=float).ravel()
    if not mu.shape == sigma.shape == y.shape:
        raise ValueError(
            f"mean, std and actual must align; got {mu.size}, {sigma.size}, {y.size}."
        )
    if np.any(sigma <= 0):
        raise ValueError("Every standard deviation must be positive.")
    z = (y - mu) / sigma
    score = sigma * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(score))


def energy_score(
    samples: ArrayLike, actual: ArrayLike, *, beta: float = 1.0, fair: bool = False
) -> float:
    """Multivariate generalisation of CRPS. Survey §5.4, Eq. 45.

    ``ES(F, y) = E_F||X - y|| - 0.5 * E_F||X - X'||``

    The survey names when it is the right choice: "the natural choice when calibrated
    joint behavior across several outputs matters, for example in multi-sensor or
    **multi-horizon forecasting**". Scoring each horizon separately with CRPS and
    averaging misses whether the *joint* forecast is coherent — a model can be calibrated
    at every horizon individually while its trajectories are implausible.

    Parameters
    ----------
    samples:
        ``(n_observations, n_members, n_dimensions)``.
    actual:
        ``(n_observations, n_dimensions)``.
    beta:
        Exponent on the norm, in ``(0, 2)``. 1.0 is the standard energy score.
    fair:
        The spread-term estimator, exactly as in :func:`crps_ensemble`, and kept in step
        with it deliberately: at one dimension the energy score *is* CRPS, and the two
        functions must return the same number on the same data. They did not in an
        earlier revision — this function used the unbiased divisor while
        :func:`crps_ensemble` used the plug-in one — which a test comparing them caught.
    """
    ensemble = np.asarray(samples, dtype=float)
    y = np.asarray(actual, dtype=float)
    if ensemble.ndim != 3:
        raise ValueError(
            f"Expected samples of shape (n_observations, n_members, n_dimensions); "
            f"got {ensemble.shape}."
        )
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != ensemble.shape[0] or y.shape[1] != ensemble.shape[2]:
        raise ValueError(
            f"actual must be (n_observations, n_dimensions) matching samples; got "
            f"{y.shape} against {ensemble.shape}."
        )
    if not 0.0 < beta < 2.0:
        raise ValueError(f"beta must lie in (0, 2); got {beta}.")

    m = ensemble.shape[1]
    accuracy_term = np.mean(
        np.linalg.norm(ensemble - y[:, None, :], axis=2) ** beta, axis=1
    )
    if m == 1:
        spread_term = np.zeros_like(accuracy_term)
    else:
        diffs = ensemble[:, :, None, :] - ensemble[:, None, :, :]
        pairwise = np.linalg.norm(diffs, axis=3) ** beta
        divisor = m * (m - 1) if fair else m * m
        spread_term = pairwise.sum(axis=(1, 2)) / divisor
    return float(np.mean(accuracy_term - 0.5 * spread_term))
