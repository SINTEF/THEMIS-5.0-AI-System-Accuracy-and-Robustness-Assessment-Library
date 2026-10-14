"""Tests for group significance testing.

The Welch's ANOVA implementation is validated against an exact mathematical identity
rather than against a captured value: for two groups, Welch's ANOVA F must equal the
square of Welch's t statistic and the p-values must match to machine precision. That is a
stronger check than any golden number, and it is the reason the formula can be written out
here rather than pulling in ``pingouin`` for it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from themis_assess.metrics.accuracy.significance import (
    compare_groups,
    epsilon_squared,
    group_values,
    kruskal_wallis,
    pairwise_welch,
    welch_anova,
)


@pytest.fixture
def three_sites():
    """Two indistinguishable groups and one that is both worse and much noisier."""
    rng = np.random.default_rng(0)
    return {
        "oslo": rng.normal(0.90, 0.02, 40),
        "plovdiv": rng.normal(0.90, 0.02, 200),
        "valencia": rng.normal(0.82, 0.09, 30),
    }


# --------------------------------------------------------------------------
# Welch's ANOVA, validated against an identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_two_groups_reduce_to_welchs_t_test_squared(seed):
    """The exact identity: at k=2, Welch's ANOVA is Welch's t-test squared."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.5, rng.uniform(0.05, 0.5), rng.integers(10, 80))
    b = rng.normal(0.6, rng.uniform(0.05, 0.5), rng.integers(10, 80))

    result = welch_anova({"a": a, "b": b})
    statistic, p_value = stats.ttest_ind(a, b, equal_var=False)

    assert result.statistic == pytest.approx(statistic**2)
    assert result.p_value == pytest.approx(p_value)


def test_the_denominator_degrees_of_freedom_match_welch_satterthwaite():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, 25), rng.normal(0, 3, 60)
    result = welch_anova({"a": a, "b": b})

    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    expected = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    assert result.df_within == pytest.approx(expected)


def test_welch_differs_from_the_equal_variance_anova(three_sites):
    """Why the Welch correction is needed rather than scipy's f_oneway.

    A tight 200-sample group pooled with a noisy 30-sample one makes the standard ANOVA
    wildly overconfident: measured here it reports p = 3e-21 where Welch reports 3e-4.
    Both reject, but only one of them is honest about how much of the evidence rests on a
    small noisy group.
    """
    standard_f, standard_p = stats.f_oneway(*three_sites.values())
    welch = welch_anova(three_sites)

    assert standard_f > 5 * welch.statistic
    assert standard_p < welch.p_value / 1000
    assert welch.df_within < sum(len(v) for v in three_sites.values()) - 3


def test_welch_rejects_a_real_difference(three_sites):
    assert welch_anova(three_sites).rejected


def test_welch_does_not_reject_identical_groups():
    """A test that always rejects is worthless."""
    rng = np.random.default_rng(5)
    groups = {name: rng.normal(0.8, 0.05, 60) for name in ("a", "b", "c")}
    assert not welch_anova(groups).rejected


def test_welch_needs_two_groups():
    with pytest.raises(ValueError, match="at least 2 groups"):
        welch_anova({"only": np.arange(10.0)})


def test_welch_needs_two_samples_per_group():
    with pytest.raises(ValueError, match="at least 2"):
        welch_anova({"a": np.array([1.0]), "b": np.arange(10.0)})


def test_welch_refuses_a_zero_variance_group():
    """Its weight is n / s^2, which is undefined. The message points at the alternative."""
    with pytest.raises(ValueError, match="kruskal_wallis"):
        welch_anova({"a": np.full(10, 0.5), "b": np.random.default_rng(0).normal(size=10)})


# --------------------------------------------------------------------------
# Kruskal-Wallis
# --------------------------------------------------------------------------


def test_kruskal_matches_scipy(three_sites):
    result = kruskal_wallis(three_sites)
    statistic, p_value = stats.kruskal(*three_sites.values())
    assert result.statistic == pytest.approx(statistic)
    assert result.p_value == pytest.approx(p_value)


def test_kruskal_handles_a_zero_variance_group():
    """Where Welch cannot: it ranks rather than weighting by variance."""
    groups = {"a": np.full(12, 0.5), "b": np.random.default_rng(0).normal(0.9, 0.1, 12)}
    assert kruskal_wallis(groups).rejected


def test_kruskal_reports_an_effect_size(three_sites):
    result = kruskal_wallis(three_sites)
    assert result.effect_size_name == "epsilon squared"
    assert 0.0 <= result.effect_size <= 1.0


def test_epsilon_squared_is_bounded_below_at_zero():
    """The formula can go negative when H is below its expectation; a negative
    proportion of explained variability is not a quantity."""
    assert epsilon_squared(0.1, 100, 3) == 0.0


def test_epsilon_squared_needs_more_samples_than_groups():
    with pytest.raises(ValueError, match="more samples than groups"):
        epsilon_squared(5.0, 3, 3)


# --------------------------------------------------------------------------
# pairwise
# --------------------------------------------------------------------------


def test_the_pairs_identify_the_odd_group_out(three_sites):
    """An omnibus test says the groups are not all alike; this says which one differs."""
    pairs = pairwise_welch(three_sites)
    significant = {frozenset((p.left, p.right)) for p in pairs if p.rejected}
    assert significant == {
        frozenset(("oslo", "valencia")),
        frozenset(("plovdiv", "valencia")),
    }


def test_holm_is_never_less_powerful_than_bonferroni(three_sites):
    holm = {(p.left, p.right): p.p_value_adjusted for p in pairwise_welch(three_sites)}
    bonferroni = {
        (p.left, p.right): p.p_value_adjusted
        for p in pairwise_welch(three_sites, correction="bonferroni")
    }
    assert all(holm[key] <= bonferroni[key] + 1e-12 for key in holm)


def test_holm_adjusted_values_are_monotone_in_the_raw_values():
    """Without the running maximum, a later comparison can end up with a smaller
    adjusted p than an earlier one, which is not a valid ordering."""
    rng = np.random.default_rng(6)
    groups = {f"g{i}": rng.normal(i * 0.05, 0.1, 30) for i in range(5)}
    pairs = sorted(pairwise_welch(groups), key=lambda p: p.p_value)
    adjusted = [p.p_value_adjusted for p in pairs]
    assert adjusted == sorted(adjusted)


def test_no_correction_leaves_the_raw_values(three_sites):
    for pair in pairwise_welch(three_sites, correction="none"):
        assert pair.p_value_adjusted == pytest.approx(pair.p_value)


def test_an_unknown_correction_is_rejected(three_sites):
    with pytest.raises(ValueError, match="correction must be"):
        pairwise_welch(three_sites, correction="magic")


def test_the_pair_difference_is_signed(three_sites):
    pairs = {(p.left, p.right): p.difference for p in pairwise_welch(three_sites)}
    assert pairs[("oslo", "valencia")] > 0  # oslo is better
    assert "above" in next(
        p.describe() for p in pairwise_welch(three_sites)
        if (p.left, p.right) == ("oslo", "valencia")
    )


# --------------------------------------------------------------------------
# splitting and the whole comparison
# --------------------------------------------------------------------------


def test_group_values_splits_by_label():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    groups = np.array(["a", "a", "b", "b"])
    split = group_values(values, groups)
    assert split["a"].tolist() == [1.0, 2.0]
    assert split["b"].tolist() == [3.0, 4.0]


def test_groups_too_small_to_test_are_dropped_not_silently_kept():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    groups = np.array(["a", "a", "a", "b", "c"])  # b and c have one sample each
    assert set(group_values(values, groups)) == {"a"}


def test_misaligned_values_and_groups_are_rejected():
    with pytest.raises(ValueError, match="must align"):
        group_values(np.zeros(5), np.zeros(4))


def test_compare_groups_runs_both_omnibus_tests(three_sites):
    """Neither is chosen for the caller: they compare means against distributions, and a
    disagreement is itself the finding."""
    values = np.concatenate(list(three_sites.values()))
    labels = np.concatenate(
        [[name] * len(v) for name, v in three_sites.items()]
    )
    comparison = compare_groups(values, labels, metric="accuracy")
    assert comparison.welch is not None
    assert comparison.kruskal is not None
    assert comparison.tests_agree is True
    assert comparison.metric == "accuracy"


def test_compare_groups_reports_per_group_statistics(three_sites):
    values = np.concatenate(list(three_sites.values()))
    labels = np.concatenate([[n] * len(v) for n, v in three_sites.items()])
    comparison = compare_groups(values, labels)
    assert set(comparison.per_group) == set(three_sites)
    assert comparison.per_group["plovdiv"]["n"] == 200
    assert comparison.per_group["valencia"]["mean"] < comparison.per_group["oslo"]["mean"]


def test_compare_groups_records_dropped_groups():
    values = np.r_[np.random.default_rng(0).normal(size=20), [0.5], [0.6]]
    labels = np.r_[["a"] * 10, ["b"] * 10, ["tiny1"], ["tiny2"]]
    comparison = compare_groups(values, labels)
    assert set(comparison.dropped_groups) == {"tiny1", "tiny2"}
    assert "dropped as too small" in comparison.summary()


def test_welch_is_skipped_rather_than_fatal_when_a_group_is_constant():
    """A bounded metric where one group is uniformly perfect. Kruskal stands."""
    values = np.r_[np.full(15, 1.0), np.random.default_rng(0).normal(0.8, 0.05, 15)]
    labels = np.r_[["perfect"] * 15, ["mixed"] * 15]
    comparison = compare_groups(values, labels)
    assert comparison.welch is None
    assert comparison.kruskal is not None
    assert comparison.metadata["welch_skipped"]


def test_a_single_testable_group_is_refused():
    values = np.array([1.0, 2.0, 3.0, 9.0])
    labels = np.array(["a", "a", "a", "solo"])
    with pytest.raises(ValueError, match="nothing to compare"):
        compare_groups(values, labels)


def test_the_conclusion_warns_that_not_rejecting_is_not_equivalence():
    """Repo B's wording was already careful about this and it is worth keeping."""
    rng = np.random.default_rng(7)
    values = np.r_[rng.normal(0.8, 0.1, 8), rng.normal(0.8, 0.1, 8)]
    labels = np.r_[["a"] * 8, ["b"] * 8]
    comparison = compare_groups(values, labels)
    text = comparison.kruskal.describe()
    assert "not evidence that the groups are equal" in text
    assert "power" in text


def test_the_summary_reads_as_a_report(three_sites):
    values = np.concatenate(list(three_sites.values()))
    labels = np.concatenate([[n] * len(v) for n, v in three_sites.items()])
    summary = compare_groups(values, labels, metric="accuracy").summary()
    assert "accuracy across 3 group(s)" in summary
    assert "Welch's ANOVA" in summary
    assert "Kruskal-Wallis" in summary
    assert "pairs that differ" in summary


def test_the_frame_carries_the_per_group_statistics(three_sites):
    values = np.concatenate(list(three_sites.values()))
    labels = np.concatenate([[n] * len(v) for n, v in three_sites.items()])
    frame = compare_groups(values, labels).to_frame()
    assert set(frame.index) == set(three_sites)
    assert "median" in frame.columns
