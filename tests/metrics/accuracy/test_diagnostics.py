"""Tests for the error-location diagnostics.

Two properties matter most. Every outlier score must run **higher for more anomalous** —
that is the fix for repo C's sign error, and it is only checkable by constructing data with
known outliers. And every diagnostic must be capable of reporting *no* finding: a
diagnostic that always finds something is worthless, so there are tests that assert
absence on data with nothing to find.
"""

from __future__ import annotations

import numpy as np
import pytest

from themis_assess.metrics.accuracy.diagnostics import (
    ambiguity_threshold,
    error_rate_by_feature_bin,
    error_rate_by_outlier_status,
    flag_ambiguous,
    outlier_scores,
)


@pytest.fixture
def with_outliers():
    """400 tight points plus 20 scattered far out. Truth marks which are which."""
    rng = np.random.default_rng(0)
    bulk = rng.normal(size=(400, 2))
    tail = rng.normal(scale=6.0, size=(20, 2))
    features = np.vstack([bulk, tail])
    is_true_outlier = np.r_[np.zeros(400, dtype=bool), np.ones(20, dtype=bool)]
    return features, is_true_outlier


# --------------------------------------------------------------------------
# outlier scores
# --------------------------------------------------------------------------


def test_every_detector_scores_higher_for_more_anomalous(with_outliers):
    """Repo C's central defect: one of its three components ran the other way, so the
    more clearly a point was isolated the less anomalous that component made it look."""
    features, is_outlier = with_outliers
    for name, scores in outlier_scores(features).items():
        assert scores[is_outlier].mean() > scores[~is_outlier].mean(), name


def test_scores_are_normalised_to_the_unit_interval(with_outliers):
    features, _ = with_outliers
    for name, scores in outlier_scores(features).items():
        assert scores.min() >= 0.0, name
        assert scores.max() <= 1.0, name


def test_the_combination_is_the_mean_of_whichever_detectors_ran(with_outliers):
    features, _ = with_outliers
    scores = outlier_scores(features, methods=("isolation_forest", "knn_distance"))
    assert set(scores) == {"isolation_forest", "knn_distance", "combined"}
    expected = (scores["isolation_forest"] + scores["knn_distance"]) / 2
    assert scores["combined"] == pytest.approx(expected)


def test_the_scores_are_reproducible(with_outliers):
    """A diagnostic that moves between runs cannot be cited. Repo C hardcoded seed 42."""
    features, _ = with_outliers
    a = outlier_scores(features)["combined"]
    b = outlier_scores(features)["combined"]
    assert a == pytest.approx(b)


def test_a_single_detector_is_allowed(with_outliers):
    features, is_outlier = with_outliers
    scores = outlier_scores(features, methods=("knn_distance",))
    assert set(scores) == {"knn_distance", "combined"}
    assert scores["combined"][is_outlier].mean() > scores["combined"][~is_outlier].mean()


def test_a_constant_feature_scores_zero_rather_than_dividing_by_zero():
    scores = outlier_scores(np.zeros((20, 2)), methods=("knn_distance",))
    assert scores["knn_distance"] == pytest.approx(np.zeros(20))


def test_non_finite_features_are_rejected_with_a_reason():
    """The detectors would otherwise treat missingness as extremity."""
    features = np.zeros((10, 2))
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="Impute or drop"):
        outlier_scores(features)


def test_outlier_scores_validate_their_arguments():
    with pytest.raises(ValueError, match="at least 2 samples"):
        outlier_scores(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="contamination must lie"):
        outlier_scores(np.zeros((10, 2)), contamination=0.8)
    with pytest.raises(ValueError, match="Unknown outlier method"):
        outlier_scores(np.zeros((10, 2)), methods=("magic",))
    with pytest.raises(ValueError, match="at least one method"):
        outlier_scores(np.zeros((10, 2)), methods=())


def test_a_large_sample_does_not_need_a_full_pairwise_matrix():
    """Repo C built the whole (n, n) matrix to read k columns of it. At 20000 samples
    that is 3.2 GB; this must simply work."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(20_000, 3))
    scores = outlier_scores(features, methods=("knn_distance",))
    assert scores["knn_distance"].shape == (20_000,)


# --------------------------------------------------------------------------
# errors against outlier status
# --------------------------------------------------------------------------


def test_errors_concentrated_on_outliers_are_detected(with_outliers):
    features, is_outlier = with_outliers
    actual = np.zeros(features.shape[0], dtype=int)
    predicted = actual.copy()
    predicted[is_outlier] = 1  # every outlier is misclassified, no inlier is

    analysis = error_rate_by_outlier_status(features, actual, predicted, quantile=0.9)
    assert analysis.outlier_error_rate > analysis.inlier_error_rate
    assert analysis.difference > 0
    assert analysis.p_value < 0.01


def test_the_chi_square_test_is_returned(with_outliers):
    """Repo C computed it and discarded it, returning two rates with no test — and the
    outlier group is small by construction, so its rate is exactly the noisy one."""
    features, is_outlier = with_outliers
    actual = np.zeros(features.shape[0], dtype=int)
    predicted = actual.copy()
    predicted[is_outlier] = 1
    analysis = error_rate_by_outlier_status(features, actual, predicted)
    assert analysis.p_value is not None
    assert analysis.chi2 is not None
    assert "chi-square" in analysis.describe()


def test_no_association_is_reported_as_no_association(with_outliers):
    """A diagnostic that always finds something is worthless."""
    rng = np.random.default_rng(2)
    features, _ = with_outliers
    actual = rng.integers(0, 2, features.shape[0])
    predicted = rng.integers(0, 2, features.shape[0])  # errors independent of position
    analysis = error_rate_by_outlier_status(features, actual, predicted)
    assert analysis.p_value > 0.05


def test_rates_are_fractions_not_percentages(with_outliers):
    """Repo C returned percentages from a function whose siblings returned fractions."""
    features, is_outlier = with_outliers
    actual = np.zeros(features.shape[0], dtype=int)
    predicted = actual.copy()
    predicted[is_outlier] = 1
    analysis = error_rate_by_outlier_status(features, actual, predicted)
    assert 0.0 <= analysis.outlier_error_rate <= 1.0
    assert 0.0 <= analysis.inlier_error_rate <= 1.0
    # Well short of 1.0 even though every true outlier is misclassified: the detected
    # set is the top 10% of the combined score, which does not coincide with the 20 true
    # outliers. That gap is the detector's recall, not a scaling error.
    assert analysis.outlier_error_rate > 5 * analysis.inlier_error_rate


def test_a_perfect_model_gives_no_ratio(with_outliers):
    """An infinite ratio is not informative, so it is None."""
    features, _ = with_outliers
    actual = np.zeros(features.shape[0], dtype=int)
    analysis = error_rate_by_outlier_status(features, actual, actual)
    assert analysis.ratio is None
    assert analysis.p_value is None  # no errors at all: nothing to test


def test_precomputed_scores_can_be_supplied(with_outliers):
    """So one detection pass can be reused across several models."""
    features, is_outlier = with_outliers
    scores = outlier_scores(features)["combined"]
    actual = np.zeros(features.shape[0], dtype=int)
    predicted = actual.copy()
    predicted[is_outlier] = 1
    a = error_rate_by_outlier_status(features, actual, predicted, scores=scores)
    b = error_rate_by_outlier_status(features, actual, predicted)
    assert a.outlier_error_rate == pytest.approx(b.outlier_error_rate)


def test_mismatched_scores_are_rejected(with_outliers):
    features, _ = with_outliers
    actual = np.zeros(features.shape[0], dtype=int)
    with pytest.raises(ValueError, match="outlier score"):
        error_rate_by_outlier_status(features, actual, actual, scores=np.zeros(5))


# --------------------------------------------------------------------------
# feature bins
# --------------------------------------------------------------------------


def test_errors_concentrated_in_a_feature_region_are_found():
    rng = np.random.default_rng(3)
    feature = rng.uniform(0, 100, 1000)
    actual = np.zeros(1000, dtype=int)
    predicted = actual.copy()
    predicted[feature > 80] = 1  # only the top of the range is wrong

    bins = error_rate_by_feature_bin(feature, actual, predicted, n_bins=5)
    assert bins["error_rate"][-1] > 0.5
    assert bins["error_rate"][0] == pytest.approx(0.0)


def test_quantile_bins_hold_equal_populations():
    """The default, because a uniform bin in the tail of a skewed feature holds a
    handful of samples and its error rate is noise."""
    rng = np.random.default_rng(4)
    feature = rng.exponential(size=1000)  # heavily skewed
    actual = rng.integers(0, 2, 1000)
    quantile = error_rate_by_feature_bin(feature, actual, actual, n_bins=5)
    uniform = error_rate_by_feature_bin(
        feature, actual, actual, n_bins=5, strategy="uniform"
    )
    assert quantile["count"].std() < uniform["count"].std()


def test_a_sparse_bin_is_nan_rather_than_a_finding():
    feature = np.r_[np.zeros(99), [100.0]]
    actual = np.zeros(100, dtype=int)
    predicted = actual.copy()
    predicted[-1] = 1
    bins = error_rate_by_feature_bin(
        feature, actual, predicted, n_bins=2, strategy="uniform", min_count=5
    )
    assert np.isnan(bins["error_rate"][-1])
    assert bins["count"][-1] == 1


def test_a_constant_feature_cannot_be_binned():
    actual = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="constant"):
        error_rate_by_feature_bin(np.ones(10), actual, actual)


def test_feature_bins_validate_their_arguments():
    actual = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="must align"):
        error_rate_by_feature_bin(np.arange(5), actual, actual)
    with pytest.raises(ValueError, match="strategy must be"):
        error_rate_by_feature_bin(np.arange(10), actual, actual, strategy="magic")


# --------------------------------------------------------------------------
# ambiguity
# --------------------------------------------------------------------------


def test_the_absolute_threshold_is_a_fraction_of_maximum_entropy():
    """Repo D's arithmetic: max_entropy * sensitivity."""
    assert ambiguity_threshold(2, sensitivity=0.8) == pytest.approx(np.log(2) * 0.8)
    assert ambiguity_threshold(4, sensitivity=0.5) == pytest.approx(np.log(4) * 0.5)


def test_the_absolute_rule_does_not_flag_the_fraction_it_claims_to():
    """Repo D documented sensitivity=0.8 as 'flagging the top 20% most uncertain
    samples'. It is a fixed bar, and what fraction it flags depends entirely on how
    confident the model happens to be — which is why both readings are offered."""
    rng = np.random.default_rng(5)
    confident = rng.uniform(0.9, 1.0, 500)
    probabilities = np.column_stack([1 - confident, confident])

    absolute = flag_ambiguous(probabilities, sensitivity=0.8)
    empirical = flag_ambiguous(probabilities, sensitivity=0.8, empirical=True)

    assert absolute["flagged_fraction"] == pytest.approx(0.0, abs=0.02)  # not 20%
    assert empirical["flagged_fraction"] == pytest.approx(0.2, abs=0.02)


def test_the_empirical_rule_flags_the_requested_fraction():
    rng = np.random.default_rng(6)
    probabilities = rng.dirichlet(np.ones(3), size=600)
    for sensitivity in (0.5, 0.8, 0.9):
        flags = flag_ambiguous(probabilities, sensitivity=sensitivity, empirical=True)
        assert flags["flagged_fraction"] == pytest.approx(1 - sensitivity, abs=0.02)


def test_a_diffuse_model_is_flagged_wholesale_by_the_absolute_rule():
    """The other direction of the same discrepancy."""
    probabilities = np.tile([0.34, 0.33, 0.33], (200, 1))
    flags = flag_ambiguous(probabilities, sensitivity=0.8)
    assert flags["flagged_fraction"] == pytest.approx(1.0)


def test_the_empirical_threshold_uses_the_observed_entropies():
    entropies = np.linspace(0.0, 1.0, 101)
    assert ambiguity_threshold(3, sensitivity=0.9, entropies=entropies) == pytest.approx(
        0.9
    )


def test_ambiguity_validates_its_arguments():
    with pytest.raises(ValueError, match="at least 2 classes"):
        ambiguity_threshold(1)
    with pytest.raises(ValueError, match="sensitivity must lie"):
        ambiguity_threshold(2, sensitivity=1.5)
    with pytest.raises(ValueError, match="empty"):
        ambiguity_threshold(2, entropies=np.array([]))
