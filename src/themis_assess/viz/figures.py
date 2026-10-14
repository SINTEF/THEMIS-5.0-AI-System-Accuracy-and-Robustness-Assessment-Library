"""Figure builders. Each takes a result object and returns a plotly figure.

Every function here is a *pure* function of a result: give it the same
:class:`~themis_assess.assessment.robustness.SweepResult` twice and you get the same
figure, and none of them calls a model, fits anything, or runs a statistical test. That
is the separation the library maintains — and :mod:`themis_assess.viz.bundles` is where
the one-call convenience lives for callers who want analysis and plot together.

The forms are chosen by what the data has to do, not by variety:

===============================  ==================================  ===================
result                           form                                colour encoding
===============================  ==================================  ===================
severity curves                  line per perturbation               categorical
:class:`SensitivityMap`          heatmap                             sequential
:class:`ConfusionMatrix`         heatmap                             sequential
reliability bins                 line against the diagonal           one series
accuracy over time               line                                one series
:class:`GroupComparison`         box per group                       categorical
subgroup metrics                 grouped bars                        categorical
feature-bin error rates          bars along an ordered axis          one series
signed attribution               horizontal bars                     diverging
unsigned attribution             horizontal bars                     one series
verdict tables                   table with status cells             status
:class:`ModelComparison`         closed polar loop per model          categorical
===============================  ==================================  ===================

Plotly throughout (docs/DECISIONS.md, D10), so the figures are interactive in a notebook
and serialisable to HTML for a report without a rendering backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.viz.theme import (
    DEFAULT_PALETTE,
    Palette,
    apply_layout,
    status_colour,
    status_symbol,
)

__all__ = [
    "plot_severity_curves",
    "plot_sensitivity_map",
    "plot_confusion_matrix",
    "plot_reliability_diagram",
    "plot_accuracy_over_time",
    "plot_group_boxes",
    "plot_subgroup_metrics",
    "plot_feature_bins",
    "plot_attribution",
    "plot_metrics_table",
    "plot_tolerance_table",
    "plot_model_comparison",
    "plot_comparison_table",
]


def _go() -> Any:
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------


def plot_severity_curves(
    sweep: Any,
    metric: str,
    *,
    show_clean: bool = True,
    palette: Palette | None = None,
    title: str | None = None,
    **layout: Any,
) -> Any:
    """One line per perturbation, across the severity axis.

    The figure the survey's §8 reporting standard effectively requires: it asks for the
    severity *curve* rather than a single severity, and for the clean value beside it.
    ``show_clean`` draws the clean value as a reference line for that reason — a curve
    without it shows how much the metric moved and not whether there was anything to
    lose.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    curves = sweep.curves(metric)
    if not curves:
        raise ValueError("This sweep has no perturbed cells to draw.")

    figure = go.Figure()
    for index, name in enumerate(sorted(curves)):
        curve = curves[name]
        severities = sorted(curve)
        figure.add_trace(
            go.Scatter(
                x=severities,
                y=[curve[s] for s in severities],
                mode="lines+markers",
                name=name,
                line=dict(color=p.series(index), width=2),
                marker=dict(size=8),
                hovertemplate=(
                    f"<b>{name}</b><br>severity %{{x}}<br>{metric} %{{y:.4f}}"
                    "<extra></extra>"
                ),
            )
        )

    if show_clean:
        clean = sweep.clean[metric]
        every_severity = sorted({s for curve in curves.values() for s in curve})
        figure.add_trace(
            go.Scatter(
                x=every_severity,
                y=[clean] * len(every_severity),
                mode="lines",
                name=f"clean ({clean:.4f})",
                line=dict(color=p.reference, width=1.5, dash="dash"),
                hovertemplate=f"clean {metric} {clean:.4f}<extra></extra>",
            )
        )

    return apply_layout(
        figure,
        title=title if title is not None else f"{metric} under perturbation",
        x_title="severity",
        y_title=metric,
        palette=p,
        **layout,
    )


def plot_accuracy_over_time(
    values: ArrayLike,
    *,
    origins: Sequence[Any] | None = None,
    metric: str = "metric",
    highlight_worst: int = 0,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """A metric per forecast origin. Repo B's ``plot_accuracy_over_time``, single-axis.

    ``highlight_worst`` marks the worst origins, which is what turns the shape into
    something actionable: the curve says the model has bad moments and the markers say
    which ones to go and look at.

    One y-axis only. A second scale for the series being forecast — the obvious
    temptation here, and a common one in forecasting dashboards — makes two unrelated
    magnitudes look comparable and their crossings look meaningful. Draw the series in
    its own figure below.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    curve = np.asarray(values, dtype=float).ravel()
    if curve.size == 0:
        raise ValueError("Nothing to plot.")
    x = list(origins) if origins is not None else list(range(curve.size))
    if len(x) != curve.size:
        raise ValueError(f"Got {len(x)} origin label(s) for {curve.size} value(s).")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=curve,
            mode="lines",
            name=metric,
            line=dict(color=p.series(0), width=2),
            hovertemplate=f"origin %{{x}}<br>{metric} %{{y:.4f}}<extra></extra>",
        )
    )
    if highlight_worst > 0:
        worst = np.argsort(curve)[::-1][:highlight_worst]
        figure.add_trace(
            go.Scatter(
                x=[x[i] for i in worst],
                y=curve[worst],
                mode="markers",
                name=f"worst {len(worst)}",
                marker=dict(color=p.status["critical"], size=11, symbol="circle-open",
                            line=dict(width=2)),
                hovertemplate=f"worst<br>origin %{{x}}<br>{metric} %{{y:.4f}}<extra></extra>",
            )
        )
    return apply_layout(
        figure,
        title=title,
        x_title="forecast origin",
        y_title=metric,
        palette=p,
        showlegend=highlight_worst > 0,
        **layout,
    )


def plot_reliability_diagram(
    bins: Mapping[str, ArrayLike],
    *,
    ece: float | None = None,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """Confidence against accuracy, with the diagonal a perfectly calibrated model sits on.

    Takes the output of
    :func:`~themis_assess.metrics.calibration.reliability_bins`. Empty bins are drawn as
    gaps rather than joined across, because a diagram that interpolates through a region
    with no data invents calibration evidence that does not exist — repo D's version
    dropped empty bins, which closes the gap and does exactly that.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    confidence = np.asarray(bins["confidence"], dtype=float)
    accuracy = np.asarray(bins["accuracy"], dtype=float)
    count = np.asarray(bins["count"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="perfectly calibrated",
            line=dict(color=p.reference, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=confidence,
            y=accuracy,
            mode="lines+markers",
            name="model",
            connectgaps=False,
            line=dict(color=p.series(0), width=2),
            marker=dict(size=9),
            customdata=count,
            hovertemplate=(
                "confidence %{x:.3f}<br>accuracy %{y:.3f}"
                "<br>%{customdata} sample(s)<extra></extra>"
            ),
        )
    )
    heading = title or "Reliability"
    if ece is not None:
        heading = f"{heading} (ECE {ece:.4f})"
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1])
    return apply_layout(
        figure,
        title=heading,
        x_title="predicted confidence",
        y_title="observed accuracy",
        palette=p,
        **layout,
    )


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------


def plot_sensitivity_map(
    sensitivity_map: Any,
    *,
    palette: Palette | None = None,
    title: str = "",
    show_values: bool | None = None,
    **layout: Any,
) -> Any:
    """A sensitivity grid as a heatmap. Repo B's heatmap family, with the colour fixed.

    Single-hue sequential, light to dark. Repo B offered ``scale="log"`` or ``"linear"``
    against a matplotlib colormap; the switch is dropped because a log colour scale on a
    grid with no colourbar legend is unreadable, and the values are on the cells instead.

    ``show_values`` writes each cell's number on it. Defaults to on for grids small
    enough to read — which is also what satisfies the contrast relief rule, since the
    lightest steps of a sequential ramp sit close to the surface by design.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    grid = np.asarray(sensitivity_map.grid, dtype=float)
    if show_values is None:
        show_values = grid.size <= 120

    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=list(sensitivity_map.column_labels),
            y=list(sensitivity_map.row_labels),
            colorscale=p.colorscale(),
            colorbar=dict(
                title=dict(text=sensitivity_map.measure.replace("_", " "), side="right"),
                thickness=14,
            ),
            texttemplate="%{z:.3g}" if show_values else None,
            textfont=dict(size=10),
            hovertemplate=(
                f"{sensitivity_map.row_axis} %{{y}}<br>"
                f"{sensitivity_map.column_axis} %{{x}}<br>"
                "%{z:.4g}<extra></extra>"
            ),
        )
    )
    heading = title or (
        f"Sensitivity ({sensitivity_map.measure.replace('_', ' ')}"
        + (f", {sensitivity_map.metric}" if sensitivity_map.metric else "")
        + ")"
    )
    return apply_layout(
        figure,
        title=heading,
        x_title=sensitivity_map.column_axis,
        y_title=sensitivity_map.row_axis,
        palette=p,
        **layout,
    )


def plot_confusion_matrix(
    matrix: Any,
    *,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """A confusion matrix as a labelled heatmap.

    Rows are true classes, columns predicted — and both axes carry their labels, which is
    the whole reason
    :class:`~themis_assess.metrics.accuracy.classification.ConfusionMatrix` keeps them.
    The y axis is reversed so the diagonal runs top-left to bottom-right, matching how
    the matrix is written down.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    grid = np.asarray(matrix.matrix, dtype=float)
    labels = [str(label) for label in matrix.labels]
    is_normalised = matrix.normalize is not None

    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=labels,
            y=labels,
            colorscale=p.colorscale(),
            colorbar=dict(
                title=dict(
                    text="fraction" if is_normalised else "count", side="right"
                ),
                thickness=14,
            ),
            texttemplate="%{z:.2f}" if is_normalised else "%{z:.0f}",
            textfont=dict(size=11),
            hovertemplate=(
                "true %{y}<br>predicted %{x}<br>"
                + ("%{z:.3f}" if is_normalised else "%{z:.0f}")
                + "<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(autorange="reversed")
    suffix = f" (normalised by {matrix.normalize})" if is_normalised else ""
    return apply_layout(
        figure,
        title=title or f"Confusion matrix{suffix}",
        x_title="predicted",
        y_title="true",
        palette=p,
        **layout,
    )


# --------------------------------------------------------------------------
# groups and bars
# --------------------------------------------------------------------------


def plot_group_boxes(
    grouped: Mapping[Any, ArrayLike],
    *,
    comparison: Any = None,
    metric: str = "metric",
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """A box per group, with the sample count on each tick and the test in the subtitle.

    Repo B's box plot, and the reason this whole tranche needed an audit: repo B ran
    Welch's ANOVA beside this figure, wrote a plain-language conclusion, and then returned
    only the figure. Here the test travels with the picture — pass the
    :class:`~themis_assess.metrics.accuracy.significance.GroupComparison` as
    ``comparison`` and its verdict is annotated on the chart, so a reader cannot see the
    gap without also seeing whether it is distinguishable from noise.

    Repo B's sample-count-under-the-label trick is kept. It matters more than it looks: a
    box drawn from six series and one drawn from six hundred look identical, and the
    reader has no other cue.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    if not grouped:
        raise ValueError("No groups to draw.")
    if len(grouped) > len(p.categorical):
        raise ValueError(
            f"{len(grouped)} groups exceeds the {len(p.categorical)} categorical slots. "
            f"Fold the smallest into an 'other' group, or draw small multiples."
        )

    figure = go.Figure()
    tick_text = []
    for index, (label, values) in enumerate(grouped.items()):
        members = np.asarray(values, dtype=float).ravel()
        members = members[np.isfinite(members)]
        figure.add_trace(
            go.Box(
                y=members,
                name=str(label),
                marker=dict(color=p.series(index)),
                boxmean=True,
                hovertemplate=f"<b>{label}</b><br>{metric} %{{y:.4f}}<extra></extra>",
            )
        )
        tick_text.append(f"{label}<br>n = {members.size}")

    figure.update_xaxes(
        tickvals=[str(label) for label in grouped],
        ticktext=tick_text,
    )

    heading = title or f"{metric} by group"
    if comparison is not None:
        test = comparison.kruskal or comparison.welch
        if test is not None:
            verdict = "differ" if test.rejected else "not distinguishable"
            heading = (
                f"{heading}<br><sup>{test.test}: p = {test.p_value:.3g} "
                f"-> groups {verdict}</sup>"
            )
    return apply_layout(
        figure,
        title=heading,
        x_title="group",
        y_title=metric,
        palette=p,
        showlegend=False,
        **layout,
    )


def plot_subgroup_metrics(
    subgroups: Mapping[str, Any],
    metrics: Sequence[str],
    *,
    overall: Any = None,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """Grouped bars: one cluster per metric, one bar per subgroup.

    Takes the ``subgroups`` mapping from an
    :class:`~themis_assess.assessment.accuracy.AccuracyResult`. ``overall`` draws the
    whole-dataset value as a marker on each cluster, which is what makes the disparity
    visible rather than merely present: a subgroup bar means little without the aggregate
    it is being compared against.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    names = list(subgroups)
    if len(names) > len(p.categorical):
        raise ValueError(
            f"{len(names)} subgroups exceeds the {len(p.categorical)} categorical slots."
        )
    chosen = [m for m in metrics if any(m in r.metrics for r in subgroups.values())]
    if not chosen:
        raise ValueError(
            f"None of {list(metrics)} was computed for any subgroup. Available: "
            f"{sorted({m for r in subgroups.values() for m in r.metrics})}."
        )

    figure = go.Figure()
    for index, name in enumerate(names):
        result = subgroups[name]
        figure.add_trace(
            go.Bar(
                x=chosen,
                y=[result.metrics.get(m, np.nan) for m in chosen],
                name=f"{name} (n = {result.n_samples})",
                marker=dict(color=p.series(index)),
                hovertemplate=f"<b>{name}</b><br>%{{x}} %{{y:.4f}}<extra></extra>",
            )
        )
    if overall is not None:
        figure.add_trace(
            go.Scatter(
                x=chosen,
                y=[overall.metrics.get(m, np.nan) for m in chosen],
                mode="markers",
                name="overall",
                marker=dict(
                    color=p.reference, size=13, symbol="line-ew",
                    line=dict(width=3, color=p.reference),
                ),
                hovertemplate="overall %{x} %{y:.4f}<extra></extra>",
            )
        )
    return apply_layout(
        figure,
        title=title or "Metrics by subgroup",
        x_title="metric",
        y_title="value",
        palette=p,
        **layout,
    )


def plot_feature_bins(
    bins: Mapping[str, ArrayLike],
    *,
    feature: str = "feature",
    overall_error_rate: float | None = None,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """Error rate across bins of one feature, with the sample count per bin.

    Takes the output of
    :func:`~themis_assess.metrics.accuracy.diagnostics.error_rate_by_feature_bin`. Bins
    below the minimum count come back as ``nan`` from that function and are drawn as
    gaps, which is the point: a bin holding four samples has an error rate that is noise,
    and a bar there would be read as a finding.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    centres = np.asarray(bins["centres"], dtype=float)
    rate = np.asarray(bins["error_rate"], dtype=float)
    count = np.asarray(bins["count"])
    edges = np.asarray(bins["edges"], dtype=float)
    labels = [f"{edges[i]:.3g} – {edges[i + 1]:.3g}" for i in range(len(centres))]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=labels,
            y=rate,
            name="error rate",
            marker=dict(color=p.series(0)),
            customdata=count,
            hovertemplate=(
                f"{feature} %{{x}}<br>error rate %{{y:.3f}}"
                "<br>%{customdata} sample(s)<extra></extra>"
            ),
        )
    )
    if overall_error_rate is not None:
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=[overall_error_rate] * len(labels),
                mode="lines",
                name=f"overall ({overall_error_rate:.3f})",
                line=dict(color=p.reference, width=1.5, dash="dash"),
                hoverinfo="skip",
            )
        )
    return apply_layout(
        figure,
        title=title or f"Error rate across {feature}",
        x_title=feature,
        y_title="error rate",
        palette=p,
        showlegend=overall_error_rate is not None,
        **layout,
    )


def plot_attribution(
    attribution: Any,
    *,
    n: int = 12,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """Feature attribution as horizontal bars, most important at the top.

    The colour encoding follows what the attribution method can actually say. A signed
    method — SHAP — gets the diverging ramp, so a bar's side and hue both carry whether
    the feature pushes error up or down. An unsigned method — permutation importance —
    gets one hue, because colouring an unsigned quantity as though it had a direction
    would invent information the method does not produce.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    top = attribution.top(n)
    if not top:
        raise ValueError("No attributions to draw.")
    features = [entry["feature"] for entry in top][::-1]

    figure = go.Figure()
    if attribution.signed is not None:
        values = [entry["signed"] for entry in top][::-1]
        limit = max(abs(v) for v in values) or 1.0
        figure.add_trace(
            go.Bar(
                x=values,
                y=features,
                orientation="h",
                marker=dict(
                    color=values,
                    colorscale=p.diverging_colorscale(),
                    cmin=-limit,
                    cmax=limit,
                    colorbar=dict(
                        title=dict(text="pushes error", side="right"), thickness=14
                    ),
                ),
                hovertemplate="%{y}<br>%{x:+.4g}<extra></extra>",
            )
        )
        x_title = "attribution (positive pushes error up)"
    else:
        figure.add_trace(
            go.Bar(
                x=[entry["importance"] for entry in top][::-1],
                y=features,
                orientation="h",
                marker=dict(color=p.series(0)),
                hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
            )
        )
        x_title = "importance (drop in surrogate R² when shuffled)"

    heading = title or f"{attribution.method} attribution ({attribution.scope})"
    gate = attribution.metadata.get("test_r2")
    if gate is not None:
        heading = f"{heading}<br><sup>surrogate test R² {gate:.3f}</sup>"
    return apply_layout(
        figure,
        title=heading,
        x_title=x_title,
        y_title="",
        palette=p,
        showlegend=False,
        height=layout.pop("height", max(320, 28 * len(features) + 120)),
        **layout,
    )


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def plot_metrics_table(
    report: Any,
    *,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """The metrics table with its verdicts. Repo A's and repo D's ``get_metrics_table``.

    Their version had columns for value, baseline and range, and the reader compared each
    row against a dictionary of acceptable values kept somewhere else. Here the verdict is
    a column, and it carries a symbol as well as a colour, so a pass is legible to a
    colour-blind reader and in a black-and-white print.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    verdicts = list(report.verdicts)
    if not verdicts:
        raise ValueError("This report has no verdicts to draw.")

    def cell(verdict: Any) -> str:
        if verdict.requirement is None:
            return "not checked"
        return f"{status_symbol(verdict.passed)} {'pass' if verdict.passed else 'fail'}"

    def verdict_fill(verdict: Any) -> str:
        # A metric with no requirement is not a warning — it simply was not part of the
        # criteria. Painting it with the warning colour, as an earlier revision did,
        # made half the table glow orange and buried the rows that had actually failed.
        if verdict.requirement is None:
            return p.surface
        return status_colour(verdict.passed, palette=p)

    figure = go.Figure(
        go.Table(
            header=dict(
                values=["metric", "value", "acceptable", "verdict", "margin"],
                fill_color=p.grid,
                font=dict(color=p.text_primary, size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    [v.metric for v in verdicts],
                    [f"{v.value:.4f}" for v in verdicts],
                    [
                        "—" if v.requirement is None
                        else f"{v.requirement.acceptable:g}"
                        if v.requirement.acceptable is not None
                        else f"±{v.requirement.tolerance:g}"
                        for v in verdicts
                    ],
                    [cell(v) for v in verdicts],
                    ["—" if v.margin is None else f"{v.margin:+.4f}" for v in verdicts],
                ],
                fill_color=[
                    [p.surface] * len(verdicts),
                    [p.surface] * len(verdicts),
                    [p.surface] * len(verdicts),
                    [verdict_fill(v) for v in verdicts],
                    [p.surface] * len(verdicts),
                ],
                font=dict(color=p.text_primary, size=11),
                align="left",
                height=26,
            ),
            # The metric column carries the longest strings by far
            # (expected_calibration_error); without this it truncates.
            columnwidth=[2.6, 1.0, 1.1, 1.2, 1.0],
        )
    )
    outcome = {True: "PASS", False: "FAIL", None: "NO CRITERIA"}[report.passed]
    return apply_layout(
        figure,
        title=title or f"Acceptance: {outcome}",
        palette=p,
        height=layout.pop("height", 120 + 28 * len(verdicts)),
        **layout,
    )


def plot_tolerance_table(
    report: Any,
    *,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """The robustness tolerance table, with the basis and the culprit in it.

    ``basis`` and ``worst_at`` are columns rather than a footnote. A tolerance verdict
    read without its basis is ambiguous — the same sweep passes on the aggregate and
    fails on the worst perturbation — and without the culprit it is not actionable.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    verdicts = list(report.verdicts)
    if not verdicts:
        raise ValueError("This report has no verdicts to draw.")

    figure = go.Figure(
        go.Table(
            header=dict(
                values=["metric", "over", "basis", "value", "acceptable", "verdict",
                        "worst at"],
                fill_color=p.grid,
                font=dict(color=p.text_primary, size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    [v.metric for v in verdicts],
                    [v.accuracy_metric for v in verdicts],
                    [v.basis for v in verdicts],
                    [f"{v.value:.4f}" for v in verdicts],
                    [f"{v.requirement.acceptable:g}" for v in verdicts],
                    [
                        f"{status_symbol(v.passed)} {'pass' if v.passed else 'fail'}"
                        for v in verdicts
                    ],
                    [v.worst_at or "—" for v in verdicts],
                ],
                fill_color=[
                    *[[p.surface] * len(verdicts) for _ in range(5)],
                    [status_colour(v.passed, palette=p) for v in verdicts],
                    [p.surface] * len(verdicts),
                ],
                font=dict(color=p.text_primary, size=11),
                align="left",
                height=26,
            ),
            columnwidth=[2.2, 1.8, 1.4, 1.0, 1.0, 1.0, 1.4],
        )
    )
    outcome = {True: "PASS", False: "FAIL", None: "NO TOLERANCE"}[report.passed]
    heading = title or f"Robustness tolerance: {outcome}"
    if report.unchecked:
        heading += f"<br><sup>not checked: {', '.join(report.unchecked)}</sup>"
    return apply_layout(
        figure,
        title=heading,
        palette=p,
        height=layout.pop("height", 140 + 28 * len(verdicts)),
        **layout,
    )


# --------------------------------------------------------------------------
# model comparison
# --------------------------------------------------------------------------


def _rgba(hex_colour: str, alpha: float) -> str:
    """A hex colour as ``rgba(...)``, for a translucent polar fill."""
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def plot_model_comparison(
    comparison: Any,
    *,
    metrics: Sequence[str] | None = None,
    max_axes: int = 8,
    palette: Palette | None = None,
    title: str = "",
    fill_alpha: float = 0.12,
    **layout: Any,
) -> Any:
    """The polar chart: one closed loop per model over normalised metrics.

    Repo D's ``plot_polar_chart_metric`` is the ancestor, and two things change. That
    version plotted raw values on a radial axis fixed to ``[0, 1]``, which is correct
    only while every metric happens to be a bounded accuracy — an error metric drawn
    there would put the better model *closer to the centre*, and an unbounded one would
    leave the chart entirely. Here the values come from
    :attr:`~themis_assess.comparison.compare.ModelComparison.normalised`, so every axis
    genuinely runs 0 to 1 with 1 outermost and better; the raw value is on the hover.

    The second change is that the second series is another *model* rather than a random
    baseline. Repo D compared one model against random guessing, which answers "is this
    model doing anything" and not "which of these models should we deploy".

    Parameters
    ----------
    metrics:
        Which axes to draw, in order. A full assessment shares a dozen or more metrics
        and a polar chart of a dozen axes is a circle — so choosing the handful the
        decision turns on is part of making the figure, not a detail. Defaults to every
        compared metric, which is right for a small comparison and too many for a large
        one.

    Raises
    ------
    ValueError
        Below three axes, where the shape carries no more than its own edge, or above
        ``max_axes``, where the labels collide and the loops stop being distinguishable.
        Both are cases where the honest answer is a different chart, and the message says
        which.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    chosen = list(comparison.metrics if metrics is None else metrics)
    unknown = [m for m in chosen if m not in comparison.metrics]
    if unknown:
        raise KeyError(
            f"{unknown} are not axes of this comparison. "
            f"Available: {list(comparison.metrics)}. "
            f"Not compared: {sorted(comparison.dropped)}."
        )
    if len(chosen) < 3:
        raise ValueError(
            f"A polar chart needs at least three axes; this one has "
            f"{len(chosen)} ({chosen}). With two metrics the shape carries no more "
            f"information than its edge — use plot_comparison_table, or a grouped bar "
            f"chart via plot_subgroup_metrics."
        )
    if len(chosen) > max_axes:
        raise ValueError(
            f"{len(chosen)} axes is too many for one polar chart (limit {max_axes}): the "
            f"angular labels collide and the loops stop being separable, which is the "
            f"radar chart's standard failure. Pass metrics=[...] with the handful the "
            f"decision turns on — plot_comparison_table shows all "
            f"{len(comparison.metrics)} of them, and ModelComparison.disagreements() "
            f"says which ones actually distinguish these models."
        )
    metrics = chosen

    labels = [metric.replace("_", " ") for metric in metrics]
    # Three translucent fills overlap into one muddy region and the individual loops stop
    # being readable — the thing that makes multi-series radar charts notorious. Two fills
    # still read, so the fill is kept for one or two models and dropped beyond that, where
    # the outlines carry it.
    filled = len(comparison.models) <= 2

    figure = go.Figure()
    for index, model in enumerate(comparison.models):
        normalised = [comparison.normalised[model][m] for m in metrics]
        raw = [comparison.raw[model][m] for m in metrics]
        colour = p.series(index)
        figure.add_trace(
            go.Scatterpolar(
                # Closed loop: the first point repeated at the end, or the outline has a
                # gap between the last axis and the first.
                r=normalised + normalised[:1],
                theta=labels + labels[:1],
                name=model,
                fill="toself" if filled else "none",
                fillcolor=_rgba(colour, fill_alpha) if filled else None,
                line=dict(color=colour, width=2),
                marker=dict(size=8, color=colour),
                customdata=[f"{v:.4g}" for v in raw + raw[:1]],
                hovertemplate=(
                    f"<b>{model}</b><br>%{{theta}}<br>"
                    "normalised %{r:.3f}<br>measured %{customdata}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        polar=dict(
            bgcolor=p.surface,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showgrid=True,
                gridcolor=p.grid,
                griddash="dot",
                # No axis line: plotly draws it as a solid spoke from the centre, which
                # on a chart whose content is lines reads as one of them.
                showline=False,
                # The radial scale is kept horizontal at 0°, and the metrics are then
                # rotated by half a sector so that no metric sits at 0° — which puts the
                # tick numbers in the gap between two spokes instead of on top of a
                # model's vertex. Plotly rotates radial labels to follow the axis angle
                # and ``tickangle`` cannot override that, so the chart has to move rather
                # than the text.
                angle=0,
                tickangle=0,
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=10, color=p.text_secondary),
            ),
            angularaxis=dict(
                showgrid=True,
                gridcolor=p.grid,
                linecolor=p.grid,
                rotation=180.0 / len(metrics),
                tickfont=dict(size=11, color=p.text_primary),
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
    )
    return apply_layout(
        figure,
        title=title or "Normalised metrics (1 = best on every axis)",
        palette=p,
        height=layout.pop("height", 520),
        width=layout.pop("width", 560),
        showlegend=True,
        **layout,
    )


def plot_comparison_table(
    comparison: Any,
    *,
    normalised: bool = False,
    palette: Palette | None = None,
    title: str = "",
    **layout: Any,
) -> Any:
    """The comparison as numbers: one row per model, the best cell per metric marked.

    The companion to the polar chart and the chart to prefer when there are two axes, or
    when the reader needs the values rather than the shape. Best-in-column is marked with
    a symbol as well as a fill, so the marking survives a black-and-white print.
    """
    go = _go()
    p = palette or DEFAULT_PALETTE
    metrics = list(comparison.metrics)
    table = comparison.normalised if normalised else comparison.raw
    winners = {metric: comparison.best(metric) for metric in metrics}

    columns = [[model for model in comparison.models]]
    fills = [[p.surface] * len(comparison.models)]
    for metric in metrics:
        columns.append(
            [
                f"{table[model][metric]:.4g}"
                + (" ✓" if winners[metric] == model else "")
                for model in comparison.models
            ]
        )
        fills.append(
            [
                p.status["good"] if winners[metric] == model else p.surface
                for model in comparison.models
            ]
        )

    figure = go.Figure(
        go.Table(
            header=dict(
                values=["model"] + [m.replace("_", " ") for m in metrics],
                fill_color=p.grid,
                font=dict(color=p.text_primary, size=12),
                align="left",
            ),
            cells=dict(
                values=columns,
                fill_color=fills,
                font=dict(color=p.text_primary, size=11),
                align="left",
                height=26,
            ),
            columnwidth=[1.6] + [1.0] * len(metrics),
        )
    )
    heading = title or (
        "Normalised comparison (1 = best)" if normalised else "Measured values"
    )
    if comparison.dropped:
        heading += (
            f"<br><sup>not compared: {', '.join(sorted(comparison.dropped))}</sup>"
        )
    return apply_layout(
        figure,
        title=heading,
        palette=p,
        height=layout.pop("height", 150 + 28 * len(comparison.models)),
        **layout,
    )
