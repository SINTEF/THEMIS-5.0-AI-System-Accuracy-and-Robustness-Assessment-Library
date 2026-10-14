"""One call, analysis and figure together — the workflow the use-case repos actually had.

The library separates analysis from plotting, and that separation is right: it is what
lets a number be tested, reported in a table, cited in a document, or drawn, without
running a rendering step to get it. But the four repos' functions were *bundled* — compute
and draw in one call — and a bundled call is genuinely the more convenient thing to have
in a notebook, which is where this work happens. Losing it would be a regression in
usability dressed up as an improvement in architecture.

So both exist. :mod:`themis_assess.viz.figures` takes results and draws them;
this module computes and draws in one call, and **returns both halves**.

That last part is the fix rather than a restatement. Repo B's
``compute_aggregated_accuracy_per_category`` assembled exactly the right bundle —

.. code-block:: python

    results = {"figure": fig, "summary": df.describe(),
               "hypothesis_test_result": res, "conclusion": conclusion}
    self.results[title] = results
    return fig

— and then returned only the figure. The statistics were reachable solely through a
side-effect dictionary keyed by the plot's title string. Every bundle here returns a
frozen result object carrying the figure *and* the analysis, so a caller who wants the
p-value does not have to know that ``self.results`` exists or what the title was.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.metrics.accuracy.significance import GroupComparison, compare_groups
from themis_assess.viz import figures
from themis_assess.viz.theme import Palette

__all__ = [
    "Bundle",
    "GroupBundle",
    "assess_groups",
    "assess_feature_bins",
    "assess_outliers",
    "assess_calibration",
    "assess_severity",
    "assess_accuracy_over_time",
    "assess_models",
]


@dataclass(frozen=True, slots=True)
class Bundle:
    """A figure and the analysis behind it.

    ``analysis`` holds whatever the underlying analysis function returned, unchanged — an
    :class:`~themis_assess.metrics.accuracy.diagnostics.OutlierErrorAnalysis`, a bins
    mapping, a curve. Nothing is reshaped for the plot's benefit, so the same object can
    go into a table or a report.
    """

    figure: Any
    analysis: Any
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def show(self) -> None:  # pragma: no cover - interactive convenience
        """Display the figure. For a notebook."""
        self.figure.show()

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook protocol
        """Render the figure when the bundle is the last expression in a notebook cell.

        So that ``assess_groups(...)`` in a cell shows the plot, as repo B's function
        did by returning the figure — while ``.analysis`` still holds the statistics.
        """
        from IPython.display import display

        display(self.figure)
        if self.summary:
            print(self.summary)


@dataclass(frozen=True, slots=True)
class GroupBundle(Bundle):
    """A group comparison: the box plot, the tests, and the pairwise results.

    ``analysis`` is a
    :class:`~themis_assess.metrics.accuracy.significance.GroupComparison`, so
    ``bundle.analysis.welch.p_value`` reaches the number repo B computed and dropped.
    """

    @property
    def comparison(self) -> GroupComparison:
        """The group comparison, named for what it is."""
        return self.analysis

    @property
    def p_value(self) -> float | None:
        """The non-parametric test's p-value — the headline number.

        Kruskal-Wallis rather than Welch, because it makes no normality assumption and
        an accuracy metric is rarely normal within a group. Use
        ``.comparison.welch.p_value`` for the parametric one; where the two disagree,
        ``.comparison.tests_agree`` says so and the summary explains it.
        """
        test = self.analysis.kruskal
        return None if test is None else test.p_value

    @property
    def groups_differ(self) -> bool | None:
        test = self.analysis.kruskal
        return None if test is None else test.rejected


# --------------------------------------------------------------------------
# groups
# --------------------------------------------------------------------------


def assess_groups(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    metric: str = "metric",
    significance_level: float = 0.05,
    correction: str = "holm",
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> GroupBundle:
    """Compare a metric across groups: box plot, both omnibus tests, pairwise results.

    Repo B's ``compute_aggregated_accuracy_per_category``, returning everything it
    computed. The figure carries the test verdict in its subtitle, so the gap and whether
    the gap is distinguishable from noise cannot be read apart from each other.

    Parameters
    ----------
    values:
        One value per sample — a per-series accuracy, a per-origin error. Not group
        aggregates: the test needs the within-group spread, which an aggregate discards.
    groups:
        The group label per value.

    Examples
    --------
    >>> bundle = assess_groups(accuracies, sites, metric="accuracy")  # doctest: +SKIP
    >>> bundle.figure                     # the box plot            # doctest: +SKIP
    >>> bundle.p_value                    # what repo B dropped     # doctest: +SKIP
    >>> bundle.comparison.significant_pairs                          # doctest: +SKIP
    """
    comparison = compare_groups(
        values,
        groups,
        metric=metric,
        significance_level=significance_level,
        correction=correction,
    )
    grouped = {
        label: np.asarray(values, dtype=float).ravel()[np.asarray(groups).ravel() == label]
        for label in comparison.group_names
    }
    figure = figures.plot_group_boxes(
        grouped,
        comparison=comparison,
        metric=metric,
        palette=palette,
        title=title,
        **layout,
    )
    return GroupBundle(
        figure=figure,
        analysis=comparison,
        summary=comparison.summary(),
        metadata={"metric": metric, "n_groups": len(grouped)},
    )


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def assess_feature_bins(
    feature_values: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    feature: str = "feature",
    n_bins: int = 5,
    strategy: str = "quantile",
    min_count: int = 1,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Bundle:
    """Error rate across bins of one feature, computed and drawn.

    Repo C's ``plot_error_analysis_across_features``, which computed its bins inside the
    plotting call. The overall error rate is drawn as a reference line, because a bar
    chart of per-bin rates with no aggregate invites the reader to compare bins against
    each other and not against the model's own average.
    """
    from themis_assess.metrics.accuracy.diagnostics import error_rate_by_feature_bin

    bins = error_rate_by_feature_bin(
        feature_values,
        actual,
        predicted,
        n_bins=n_bins,
        strategy=strategy,
        min_count=min_count,
    )
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    overall = float(np.mean(y_true != y_pred))
    figure = figures.plot_feature_bins(
        bins,
        feature=feature,
        overall_error_rate=overall,
        palette=palette,
        title=title,
        **layout,
    )
    populated = int(np.sum(np.asarray(bins["count"]) >= min_count))
    return Bundle(
        figure=figure,
        analysis=bins,
        summary=(
            f"{feature}: overall error rate {overall:.3f} across {populated} populated "
            f"bin(s); worst bin "
            f"{np.nanmax(np.asarray(bins['error_rate'], dtype=float)):.3f}"
        ),
        metadata={"feature": feature, "overall_error_rate": overall},
    )


def assess_outliers(
    features: ArrayLike,
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    quantile: float = 0.9,
    metric: str = "error rate",
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Bundle:
    """Do the errors concentrate on unusual inputs? Analysis, chi-square test and figure.

    Repo C's ``compute_error_outlier_analysis`` plus its plot. Drawn as two boxes —
    outliers against inliers — over the per-sample error indicator, with the chi-square
    p-value in the subtitle. Repo C computed that p-value and discarded it, returning two
    bare rates whose difference the reader had no way to calibrate.
    """
    from themis_assess.metrics.accuracy.diagnostics import error_rate_by_outlier_status

    analysis = error_rate_by_outlier_status(
        features, actual, predicted, quantile=quantile
    )
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    is_error = (y_true != y_pred).astype(float)

    grouped = {
        "inlier": is_error[~analysis.is_outlier],
        "outlier": is_error[analysis.is_outlier],
    }
    subtitle = title or "Error rate by outlier status"
    if analysis.p_value is not None:
        verdict = "significant" if analysis.p_value < 0.05 else "not significant"
        subtitle += (
            f"<br><sup>chi-square p = {analysis.p_value:.3g} ({verdict} at 0.05)</sup>"
        )
    figure = figures.plot_group_boxes(
        grouped, metric=metric, palette=palette, title=subtitle, **layout
    )
    return Bundle(
        figure=figure,
        analysis=analysis,
        summary=analysis.describe(),
        metadata={"quantile": quantile},
    )


def assess_calibration(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = 15,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Bundle:
    """Reliability bins, ECE, and the diagram. Repo D's calibration assessment.

    Repo D already separated ``compute_*_data`` from ``plot_*``, which is the convention
    the rest of the library follows — this is the bundled form for the notebook case.
    """
    from themis_assess.metrics.calibration import (
        expected_calibration_error,
        reliability_bins,
    )

    bins = reliability_bins(probabilities, labels, n_bins=n_bins)
    ece = expected_calibration_error(probabilities, labels, n_bins=n_bins)
    figure = figures.plot_reliability_diagram(
        bins, ece=ece, palette=palette, title=title, **layout
    )
    occupied = int(np.sum(np.asarray(bins["count"]) > 0))
    return Bundle(
        figure=figure,
        analysis={"bins": bins, "ece": ece},
        summary=(
            f"ECE {ece:.4f} over {occupied} occupied bin(s) of {n_bins}. "
            f"Report the bin count: too few hide miscalibration by averaging over- and "
            f"under-confident regions together, too many leave bins with a handful of "
            f"samples and a noisy accuracy estimate."
        ),
        metadata={"n_bins": n_bins, "ece": ece},
    )


# --------------------------------------------------------------------------
# sweeps and time
# --------------------------------------------------------------------------


def assess_severity(
    sweep: Any,
    metric: str,
    *,
    palette: Palette | None = None,
    title: str | None = None,
    **layout: Any,
) -> Bundle:
    """Severity curves plus the worst cell, from a sweep that already ran.

    No model calls: the sweep's cells are the analysis. The summary names the worst
    (perturbation, severity) cell, which survey §8 asks for beside the average and which
    a curve alone leaves the reader to find by eye.
    """
    from themis_assess.metrics.registry import default_registry

    spec = default_registry().get(metric)
    if spec.higher_is_better is None:
        raise ValueError(
            f"{metric!r} has no direction, so 'worst' is undefined for it. Use an "
            f"accuracy or an error metric."
        )
    perturbation, severity, value = sweep.worst(
        metric, higher_is_better=bool(spec.higher_is_better)
    )
    figure = figures.plot_severity_curves(
        sweep, metric, palette=palette, title=title, **layout
    )
    clean = sweep.clean[metric]
    return Bundle(
        figure=figure,
        analysis=sweep.curves(metric),
        summary=(
            f"clean {metric} {clean:.4f}; worst cell {perturbation}@{severity} "
            f"= {value:.4f}"
        ),
        metadata={
            "metric": metric,
            "clean": clean,
            "worst": {"perturbation": perturbation, "severity": severity, "value": value},
        },
    )


def assess_accuracy_over_time(
    forecasts: ArrayLike,
    actual: ArrayLike,
    metric_fn: Callable[..., float],
    *,
    origins: Sequence[Any] | None = None,
    metric: str = "metric",
    aggregation: str = "mean",
    highlight_worst: int = 3,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Bundle:
    """A metric per forecast origin, aggregated, with the worst origins marked.

    Repo B's ``compute_accuracy_over_time`` and ``plot_accuracy_over_time`` in one call —
    the pair that could not run in the general repo at all, since its compute half passed
    ``covariates=`` to a handler taking ``past_covariates`` and its aggregate half was an
    abstract method with no body.

    ``metric_fn`` is called as ``metric_fn(actual_window, forecast_window)``, actual
    first, which is this library's convention throughout and the opposite of darts'.
    """
    from themis_assess.metrics.accuracy.temporal import (
        accuracy_over_time,
        aggregate_over_time,
        worst_windows,
    )

    curve = accuracy_over_time(forecasts, actual, metric_fn)
    aggregate = aggregate_over_time(curve, method=aggregation)
    worst = worst_windows(
        curve, n=highlight_worst, origins=origins if origins is not None else None
    )
    figure = figures.plot_accuracy_over_time(
        curve,
        origins=origins,
        metric=metric,
        highlight_worst=highlight_worst,
        palette=palette,
        title=title,
        **layout,
    )
    return Bundle(
        figure=figure,
        analysis={"curve": curve, "worst": worst},
        summary=(
            f"{metric}: {aggregation} {aggregate:.4f} over {curve.size} origin(s); "
            f"max {aggregate_over_time(curve, method='max'):.4f} at origin "
            f"{worst[0]['origin']}"
        ),
        metadata={"metric": metric, "aggregation": aggregation, "aggregate": aggregate},
    )


def assess_models(
    reports: Any,
    *,
    metrics: Sequence[str] | None = None,
    axes: Sequence[str] | None = None,
    include_robustness: bool = True,
    table: bool = False,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Bundle:
    """Compare several models and draw the polar chart, in one call.

    The comparison itself is in ``.analysis`` — the raw and normalised tables, the
    ranking, the metrics that could not be compared and why, and the pairs of metrics
    that disagree about which model is best. That last is the part worth reading: the
    chart shows which loop is bigger, and the analysis says whether the metrics agree
    about it.

    ``metrics`` restricts what is *compared*; ``axes`` restricts what is *drawn*, which
    are different choices — the analysis wants every shared metric and a polar chart
    wants the handful the decision turns on. ``table=True`` draws the numbers instead of
    the shape, which is what a comparison on two axes needs, and the fallback whenever
    there are more metrics than a polar chart can carry.
    """
    from themis_assess.comparison.compare import compare_models

    comparison = compare_models(
        reports, metrics=metrics, include_robustness=include_robustness
    )
    if table:
        figure = figures.plot_comparison_table(
            comparison, palette=palette, title=title, **layout
        )
    else:
        figure = figures.plot_model_comparison(
            comparison, metrics=axes, palette=palette, title=title, **layout
        )
    return Bundle(
        figure=figure,
        analysis=comparison,
        summary=comparison.summary(),
        metadata={
            "models": list(comparison.models),
            "metrics": list(comparison.metrics),
            "dropped": dict(comparison.dropped),
            "is_unanimous": comparison.is_unanimous,
        },
    )
