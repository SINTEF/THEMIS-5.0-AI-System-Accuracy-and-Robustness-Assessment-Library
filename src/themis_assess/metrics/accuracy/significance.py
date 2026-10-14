"""Is the gap between groups more than sampling noise?

The inferential counterpart to survey §5.1.2's disparity metrics, which are purely
descriptive. ``accuracy_gap`` of 0.085 across three sites is a number; whether 0.085 is
distinguishable from what random splits of the same data would produce is a different
question, and the two get confused constantly. A report that states a gap without it
invites the reader to act on noise.

Ported from repo B, where this analysis existed **only inside a plotting function**.
``compute_aggregated_accuracy_per_category`` computed accuracy per series, grouped it by a
category, drew a box plot, ran Welch's ANOVA, built a plain-language conclusion — and then
assembled ``{"figure", "summary", "hypothesis_test_result", "conclusion"}`` into
``self.results[title]`` and executed ``return fig``. The caller received the picture and
never the statistics. Same pattern at ``compute_accuracy_per_lead_time``.

Three changes on the way across.

*Welch's ANOVA is implemented here rather than imported.* Repo B used ``pingouin``, which
is a heavy dependency for one formula, and SciPy has no Welch variant — only
``f_oneway``, which assumes equal variances across groups. That assumption is wrong for
exactly this use: a small site has both fewer samples and a noisier accuracy estimate, so
group variances differ by construction. Welch's correction is a dozen lines and it is the
right test, so it is written out.

*A non-parametric alternative is offered beside it.* Both ANOVA variants compare *means*
and assume approximate normality within groups. A bounded accuracy near its ceiling is
skewed, and a per-series error metric is usually right-skewed, so :func:`kruskal_wallis`
is often the more defensible test on this data. Neither is the default; the caller picks,
and :func:`compare_groups` runs both so a disagreement between them is visible.

*Which groups differ is answered, not just whether any do.* An omnibus test that rejects
says only that the groups are not all alike. Repo B stopped there. :func:`pairwise_welch`
identifies the pairs, with a multiplicity correction, because "site C is worse than the
other two" is the actionable form of the finding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "GroupTest",
    "PairwiseComparison",
    "GroupComparison",
    "welch_anova",
    "kruskal_wallis",
    "pairwise_welch",
    "epsilon_squared",
    "compare_groups",
    "group_values",
]


def group_values(
    values: ArrayLike, groups: ArrayLike, *, minimum: int = 2
) -> dict[Any, np.ndarray]:
    """Split per-sample values by group label, dropping groups too small to test.

    ``minimum`` is 2 because a group of one has no variance, and every test here needs
    one. Dropped groups are not silently forgotten — :func:`compare_groups` records them.
    """
    v = np.asarray(values, dtype=float).ravel()
    g = np.asarray(groups).ravel()
    if v.shape != g.shape:
        raise ValueError(
            f"values and groups must align; got {v.size} value(s) and {g.size} label(s)."
        )
    if v.size == 0:
        raise ValueError("No values to compare.")

    out: dict[Any, np.ndarray] = {}
    for label in dict.fromkeys(g.tolist()):
        members = v[g == label]
        members = members[np.isfinite(members)]
        if members.size >= minimum:
            out[label] = members
    return out


@dataclass(frozen=True, slots=True)
class GroupTest:
    """One omnibus test of whether a metric differs across groups."""

    test: str
    statistic: float
    p_value: float
    df_between: float | None
    df_within: float | None
    n_groups: int
    n_samples: int
    significance_level: float = 0.05
    effect_size: float | None = None
    effect_size_name: str | None = None

    @property
    def rejected(self) -> bool:
        """Whether the null hypothesis of equal groups is rejected."""
        return self.p_value < self.significance_level

    def describe(self) -> str:
        """The plain-language conclusion repo B built and then did not return.

        Deliberately says what a *failure* to reject does and does not mean. Repo B's
        wording — "there is no clear sign that the group's means differ" — was already
        careful about this, and it is worth keeping: absence of a significant difference
        is not evidence that the groups are equal, particularly with small groups where
        the test has little power to detect anything.
        """
        verdict = (
            "the groups differ"
            if self.rejected
            else "there is no clear sign that the groups differ"
        )
        detail = (
            f"{self.test}: statistic {self.statistic:.4g}, p = {self.p_value:.4g}, "
            f"significance level {self.significance_level:g} -> {verdict}"
        )
        if self.effect_size is not None:
            detail += f"; {self.effect_size_name} = {self.effect_size:.4g}"
        if not self.rejected:
            detail += (
                ". Not rejecting is not evidence that the groups are equal — with "
                f"{self.n_groups} group(s) over {self.n_samples} sample(s) the test may "
                "simply lack the power to detect a difference that matters."
            )
        return detail


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """One pair of groups compared, with the multiplicity correction applied."""

    left: Any
    right: Any
    difference: float
    statistic: float
    p_value: float
    p_value_adjusted: float
    significance_level: float = 0.05

    @property
    def rejected(self) -> bool:
        return self.p_value_adjusted < self.significance_level

    def describe(self) -> str:
        direction = "above" if self.difference > 0 else "below"
        verdict = "differ" if self.rejected else "not distinguishable"
        return (
            f"{self.left} vs {self.right}: {abs(self.difference):.4g} {direction}, "
            f"adjusted p = {self.p_value_adjusted:.4g} -> {verdict}"
        )


@dataclass(frozen=True, slots=True)
class GroupComparison:
    """Everything one group comparison produced: both omnibus tests and the pairs.

    Both parametric and non-parametric tests are run rather than one being chosen for
    the caller, because they answer slightly different questions — means against
    distributions — and where they disagree that is itself the finding. A single p-value
    would hide it.
    """

    per_group: Mapping[Any, Mapping[str, float]]
    welch: GroupTest | None = None
    kruskal: GroupTest | None = None
    pairwise: tuple[PairwiseComparison, ...] = ()
    dropped_groups: tuple[Any, ...] = ()
    metric: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def group_names(self) -> list[Any]:
        return list(self.per_group)

    @property
    def tests_agree(self) -> bool | None:
        """Whether the parametric and non-parametric tests reach the same verdict."""
        if self.welch is None or self.kruskal is None:
            return None
        return self.welch.rejected == self.kruskal.rejected

    @property
    def significant_pairs(self) -> list[PairwiseComparison]:
        return [pair for pair in self.pairwise if pair.rejected]

    def summary(self) -> str:
        """A readable digest, the thing a report quotes."""
        lines = []
        if self.metric:
            lines.append(f"{self.metric} across {len(self.per_group)} group(s)")
        for name, stats in self.per_group.items():
            lines.append(
                f"  {name}: mean {stats['mean']:.4g}, median {stats['median']:.4g}, "
                f"sd {stats['std']:.4g}, n = {int(stats['n'])}"
            )
        if self.welch is not None:
            lines.append(f"  {self.welch.describe()}")
        if self.kruskal is not None:
            lines.append(f"  {self.kruskal.describe()}")
        if self.tests_agree is False:
            lines.append(
                "  the two tests disagree: the group *means* and the group "
                "*distributions* are telling different stories, which usually means a "
                "skewed metric or one group with an outlier"
            )
        if self.significant_pairs:
            lines.append("  pairs that differ:")
            lines.extend(f"    {pair.describe()}" for pair in self.significant_pairs)
        elif self.pairwise:
            lines.append("  no pair survives the multiplicity correction")
        if self.dropped_groups:
            lines.append(
                f"  dropped as too small to test: "
                f"{', '.join(map(str, self.dropped_groups))}"
            )
        return "\n".join(lines)

    def to_frame(self) -> Any:
        """Per-group descriptive statistics, as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.per_group).T


# --------------------------------------------------------------------------
# omnibus tests
# --------------------------------------------------------------------------


def welch_anova(
    grouped: Mapping[Any, ArrayLike], *, significance_level: float = 0.05
) -> GroupTest:
    """Welch's one-way ANOVA: do the group means differ, without assuming equal variances?

    Repo B's test, via ``pingouin.welch_anova``. Written out here because SciPy has no
    Welch variant and ``pingouin`` is a large dependency for one formula.

    Equal variances is the wrong assumption for this data. A site contributing 40 samples
    and one contributing 400 have accuracy estimates of different precision by
    construction, so pooling their variances — which the standard ANOVA does — inflates
    or deflates the statistic depending on which group is the small one. Welch weights
    each group by ``n_i / s_i^2`` instead and adjusts the denominator degrees of freedom.

    ``F = [Σ w_i (m_i - m*)^2 / (k - 1)] / [1 + 2(k - 2)/(k^2 - 1) · Λ]``

    where ``w_i = n_i / s_i^2``, ``m*`` is the weighted grand mean, and ``Λ`` accumulates
    ``(1 - w_i/Σw)^2 / (n_i - 1)``.
    """
    from scipy import stats

    groups = {
        label: np.asarray(values, dtype=float).ravel()
        for label, values in grouped.items()
    }
    if len(groups) < 2:
        raise ValueError(
            f"Welch's ANOVA needs at least 2 groups; got {len(groups)}. With one group "
            f"there is nothing to compare."
        )
    for label, values in groups.items():
        if values.size < 2:
            raise ValueError(
                f"Group {label!r} has {values.size} sample(s). Every group needs at "
                f"least 2, since the test weights each group by its own variance."
            )

    k = len(groups)
    weights, means, sizes = [], [], []
    for values in groups.values():
        variance = float(np.var(values, ddof=1))
        if variance <= 0:
            raise ValueError(
                "At least one group has zero variance, so its Welch weight "
                "(n / s^2) is undefined. Use kruskal_wallis, which ranks rather than "
                "weighting by variance."
            )
        weights.append(values.size / variance)
        means.append(float(np.mean(values)))
        sizes.append(values.size)

    w = np.asarray(weights)
    m = np.asarray(means)
    n = np.asarray(sizes, dtype=float)
    total_weight = w.sum()
    grand_mean = float(np.sum(w * m) / total_weight)

    numerator = float(np.sum(w * (m - grand_mean) ** 2) / (k - 1))
    lambda_term = float(np.sum((1.0 - w / total_weight) ** 2 / (n - 1.0)))
    denominator = 1.0 + (2.0 * (k - 2) / (k * k - 1)) * lambda_term
    statistic = numerator / denominator

    df_between = float(k - 1)
    df_within = float(1.0 / (3.0 / (k * k - 1) * lambda_term)) if lambda_term > 0 else math.inf
    p_value = float(stats.f.sf(statistic, df_between, df_within))

    return GroupTest(
        test="Welch's ANOVA",
        statistic=statistic,
        p_value=p_value,
        df_between=df_between,
        df_within=df_within,
        n_groups=k,
        n_samples=int(n.sum()),
        significance_level=significance_level,
    )


def kruskal_wallis(
    grouped: Mapping[Any, ArrayLike], *, significance_level: float = 0.05
) -> GroupTest:
    """Kruskal-Wallis: do the group *distributions* differ? Non-parametric.

    Compares mean ranks rather than means, so it assumes nothing about the shape of the
    distributions. Often the more defensible test on this data than either ANOVA: a
    bounded accuracy near its ceiling is left-skewed, and a per-series error metric is
    usually right-skewed, which is precisely where a mean is a poor summary.

    Reported with ``epsilon_squared`` as the effect size, because a significant result on
    a large dataset says a difference exists and nothing about whether it is large enough
    to matter.
    """
    from scipy import stats

    groups = {
        label: np.asarray(values, dtype=float).ravel()
        for label, values in grouped.items()
    }
    if len(groups) < 2:
        raise ValueError(f"Kruskal-Wallis needs at least 2 groups; got {len(groups)}.")
    if any(values.size < 1 for values in groups.values()):
        raise ValueError("Every group needs at least one sample.")

    statistic, p_value = stats.kruskal(*groups.values())
    n_samples = int(sum(values.size for values in groups.values()))
    return GroupTest(
        test="Kruskal-Wallis",
        statistic=float(statistic),
        p_value=float(p_value),
        df_between=float(len(groups) - 1),
        df_within=None,
        n_groups=len(groups),
        n_samples=n_samples,
        significance_level=significance_level,
        effect_size=epsilon_squared(float(statistic), n_samples, len(groups)),
        effect_size_name="epsilon squared",
    )


def epsilon_squared(statistic: float, n_samples: int, n_groups: int) -> float:
    """Effect size for Kruskal-Wallis: ``(H - k + 1) / (n - k)``.

    In ``[0, 1]``, interpretable as the proportion of variability in the ranks explained
    by the grouping. Clipped at zero, since the formula can go slightly negative when
    ``H`` is below its expectation and a negative proportion is not a quantity.
    """
    if n_samples <= n_groups:
        raise ValueError(
            f"epsilon squared needs more samples than groups; got {n_samples} and "
            f"{n_groups}."
        )
    return max(0.0, float((statistic - n_groups + 1) / (n_samples - n_groups)))


# --------------------------------------------------------------------------
# post hoc
# --------------------------------------------------------------------------


def pairwise_welch(
    grouped: Mapping[Any, ArrayLike],
    *,
    significance_level: float = 0.05,
    correction: str = "holm",
) -> list[PairwiseComparison]:
    """Which pairs of groups differ. Welch's t-test on each pair, corrected.

    An omnibus test that rejects says only that the groups are not all alike; repo B
    stopped there, and "site C is worse than the other two" is the form of the finding
    someone can act on.

    Parameters
    ----------
    correction:
        ``"holm"`` (the default) is uniformly more powerful than Bonferroni at the same
        family-wise error rate, so there is no reason to prefer Bonferroni; it is offered
        because it is what a reader may expect to see. ``"none"`` leaves the raw p-values,
        which with more than a couple of groups will find differences that are not there —
        six groups means fifteen pairs and roughly a 1-in-2 chance of at least one false
        positive at 0.05.
    """
    if correction not in {"holm", "bonferroni", "none"}:
        raise ValueError(
            f"correction must be holm, bonferroni or none; got {correction!r}."
        )
    from scipy import stats

    groups = {
        label: np.asarray(values, dtype=float).ravel()
        for label, values in grouped.items()
    }
    if len(groups) < 2:
        raise ValueError("Pairwise comparison needs at least 2 groups.")

    raw = []
    for left, right in combinations(groups, 2):
        a, b = groups[left], groups[right]
        statistic, p_value = stats.ttest_ind(a, b, equal_var=False)
        raw.append(
            (left, right, float(np.mean(a) - np.mean(b)), float(statistic), float(p_value))
        )

    p_values = np.array([entry[4] for entry in raw])
    m = p_values.size
    if correction == "none":
        adjusted = p_values
    elif correction == "bonferroni":
        adjusted = np.minimum(1.0, p_values * m)
    else:
        # Holm-Bonferroni: sort ascending, scale by (m - i), then enforce monotonicity so
        # a later comparison can never end up with a smaller adjusted p than an earlier
        # one — without that step the adjusted values are not a valid ordering.
        order = np.argsort(p_values)
        scaled = p_values[order] * (m - np.arange(m))
        running = np.maximum.accumulate(np.minimum(1.0, scaled))
        adjusted = np.empty(m)
        adjusted[order] = running

    return [
        PairwiseComparison(
            left=entry[0],
            right=entry[1],
            difference=entry[2],
            statistic=entry[3],
            p_value=entry[4],
            p_value_adjusted=float(adjusted[i]),
            significance_level=significance_level,
        )
        for i, entry in enumerate(raw)
    ]


# --------------------------------------------------------------------------
# the whole comparison
# --------------------------------------------------------------------------


def compare_groups(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    metric: str | None = None,
    significance_level: float = 0.05,
    correction: str = "holm",
    pairwise: bool = True,
) -> GroupComparison:
    """Per-group statistics, both omnibus tests, and the pairwise comparisons.

    The analysis half of repo B's ``compute_aggregated_accuracy_per_category``, returned
    in full rather than stashed on the assessment object while the plot was returned.

    Parameters
    ----------
    values:
        One value per sample — a per-series accuracy, a per-sample error, a per-origin
        metric. Not group aggregates: a test needs the within-group spread, and that is
        exactly what an aggregate discards.
    groups:
        The group label for each value.
    pairwise:
        Run the post-hoc comparisons. On by default when there are more than two groups.

    Notes
    -----
    Welch's ANOVA is skipped, rather than raising, when any group has zero variance —
    which happens with a bounded metric where one group is uniformly perfect. The
    Kruskal-Wallis result stands in that case, and the reason is recorded in
    ``metadata["welch_skipped"]``.
    """
    grouped = group_values(values, groups)
    all_labels = list(dict.fromkeys(np.asarray(groups).ravel().tolist()))
    dropped = tuple(label for label in all_labels if label not in grouped)

    if len(grouped) < 2:
        raise ValueError(
            f"Only {len(grouped)} group(s) have the 2+ samples a test needs "
            f"(dropped: {dropped or 'none'}). There is nothing to compare."
        )

    per_group = {
        label: {
            "mean": float(np.mean(members)),
            "median": float(np.median(members)),
            "std": float(np.std(members, ddof=1)) if members.size > 1 else 0.0,
            "min": float(np.min(members)),
            "max": float(np.max(members)),
            "n": float(members.size),
        }
        for label, members in grouped.items()
    }

    welch: GroupTest | None = None
    welch_skipped: str | None = None
    try:
        welch = welch_anova(grouped, significance_level=significance_level)
    except ValueError as exc:
        welch_skipped = str(exc)

    kruskal = kruskal_wallis(grouped, significance_level=significance_level)

    pairs: tuple[PairwiseComparison, ...] = ()
    if pairwise and len(grouped) >= 2:
        pairs = tuple(
            pairwise_welch(
                grouped, significance_level=significance_level, correction=correction
            )
        )

    return GroupComparison(
        per_group=per_group,
        welch=welch,
        kruskal=kruskal,
        pairwise=pairs,
        dropped_groups=dropped,
        metric=metric,
        metadata={
            "significance_level": significance_level,
            "correction": correction,
            "welch_skipped": welch_skipped,
            "n_samples": int(sum(m.size for m in grouped.values())),
        },
    )
