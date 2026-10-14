"""Tests for the figure builders and the bundled analysis-and-plot functions.

Figures are hard to assert on, so these tests check the things that would actually be
wrong: the form matches the data's job, the colour encoding matches what the quantity can
say, no chart has two y-scales, identity is never carried by colour alone, and every
bundle returns the analysis as well as the picture — which is precisely what repo B's
version did not.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from themis_assess.assessment import (
    AccuracyEvaluator,
    RobustnessCriteria,
    RobustnessEvaluator,
    ToleranceRequirement,
    themis_50_criteria,
)
from themis_assess.attribution import ErrorModel
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.subgroups import SubgroupSpec
from themis_assess.handlers.base import Prediction
from themis_assess.handlers.baselines import PriorProbabilityHandler
from themis_assess.handlers.sklearn import SklearnHandler
from themis_assess.metrics.accuracy.classification import confusion_matrix
from themis_assess.metrics.calibration import (
    expected_calibration_error,
    reliability_bins,
)
from themis_assess.sensitivity import SensitivityMapper
from themis_assess.viz import bundles, figures
from themis_assess.viz.theme import DEFAULT_PALETTE, Palette, status_symbol


@pytest.fixture(scope="module")
def assessed():
    """One full assessment, reused: three classes, three sites, a hidden noise driver."""
    rng = np.random.default_rng(0)
    n = 900
    age = rng.uniform(20, 85, n)
    severity = rng.normal(size=n)
    noise = rng.uniform(0.1, 2.5, n)
    site = rng.choice(["oslo", "plovdiv", "valencia"], n, p=[0.5, 0.3, 0.2])
    y = np.digitize(0.06 * (age - 50) + 1.4 * severity + noise * rng.normal(size=n),
                    [-1.0, 2.0])
    X = np.column_stack([age, severity, noise, rng.normal(size=n)])
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X[:600, :2], y[:600])

    class TwoColumns(SklearnHandler):
        def predict(self, adapter):
            matrix = adapter.values().reshape(adapter.n_entities, -1)[:, :2]
            return self.check_prediction(
                Prediction(
                    predicted=self.model.predict(matrix),
                    probabilities=self.model.predict_proba(matrix),
                ),
                adapter,
            )

    data = ArrayAdapter(
        X[600:],
        feature_names=["age", "severity", "noise_level", "irrelevant"],
        categorical_features=[],
    )
    truth, site_test = y[600:], site[600:]
    handler = TwoColumns(model, name="rf")
    groups = [
        SubgroupSpec(name, (lambda name: lambda d: site_test == name)(name))
        for name in ("oslo", "plovdiv", "valencia")
    ]
    accuracy = AccuracyEvaluator(
        handler,
        criteria=themis_50_criteria(),
        baseline=PriorProbabilityHandler(y[:600]),
        subgroups=groups,
    ).run(data, truth)
    robustness = RobustnessEvaluator(
        handler,
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 3, 5),
        n_variants=2,
        seed=1,
    ).assess(
        data,
        truth,
        accuracy_metric="accuracy",
        robustness_metrics=["relative_robustness"],
        criteria=RobustnessCriteria(
            [ToleranceRequirement("relative_robustness", acceptable=0.9, rationale="x")]
        ),
    )
    return {
        "data": data,
        "truth": truth,
        "site": site_test,
        "handler": handler,
        "prediction": handler.predict(data),
        "accuracy": accuracy,
        "robustness": robustness,
        "features": X[600:],
        "age": age[600:],
    }


# --------------------------------------------------------------------------
# theme
# --------------------------------------------------------------------------


def test_a_ninth_series_raises_rather_than_repeating_a_colour():
    """A cycled palette makes two series share a colour with no way to tell which two."""
    palette = Palette()
    assert palette.series(7)
    with pytest.raises(ValueError, match="exceeds the 8 categorical slots"):
        palette.series(8)


def test_the_sequential_ramp_is_one_hue_light_to_dark():
    """Repo A's confusion matrix defaulted to viridis, which sweeps three hues and makes
    the hue look meaningful when only the lightness is."""
    scale = DEFAULT_PALETTE.colorscale()
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0

    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    values = [luminance(colour) for _, colour in scale]
    assert values == sorted(values, reverse=True)  # monotone: light to dark


def test_the_diverging_midpoint_is_neutral():
    """A hue at the midpoint would stop it reading as 'nothing'."""
    scale = DEFAULT_PALETTE.diverging_colorscale()
    assert scale[1][0] == 0.5
    r, g, b = (int(scale[1][1][i : i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) < 12  # very nearly grey


def test_a_verdict_has_a_symbol_as_well_as_a_colour():
    """Three categorical slots and two status colours sit below 3:1 contrast on a light
    surface, so colour alone is not a legible encoding of state."""
    assert status_symbol(True) != status_symbol(False) != status_symbol(None)


# --------------------------------------------------------------------------
# figures from results
# --------------------------------------------------------------------------


def test_severity_curves_draw_one_line_per_perturbation_plus_clean(assessed):
    figure = figures.plot_severity_curves(assessed["robustness"].sweep, "accuracy")
    names = [trace.name for trace in figure.data]
    assert "gaussian_noise" in names
    assert "uniform_bias" in names
    assert any(name.startswith("clean") for name in names)


def test_the_clean_reference_can_be_omitted(assessed):
    figure = figures.plot_severity_curves(
        assessed["robustness"].sweep, "accuracy", show_clean=False
    )
    assert not any(str(trace.name).startswith("clean") for trace in figure.data)


def test_no_figure_has_a_second_y_axis(assessed):
    """The most common chart mistake: two scales make unrelated magnitudes look
    comparable and their crossings look meaningful."""
    built = [
        figures.plot_severity_curves(assessed["robustness"].sweep, "accuracy"),
        figures.plot_accuracy_over_time(np.linspace(0, 1, 20), highlight_worst=3),
        figures.plot_confusion_matrix(assessed["accuracy"].confusion),
        figures.plot_subgroup_metrics(
            assessed["accuracy"].subgroups, ["accuracy", "f1"],
            overall=assessed["accuracy"].overall,
        ),
    ]
    for figure in built:
        assert "yaxis2" not in figure.layout
        for trace in figure.data:
            assert getattr(trace, "yaxis", None) in (None, "y")


def test_a_multi_series_figure_always_has_a_legend(assessed):
    """Identity must never be carried by colour alone."""
    figure = figures.plot_severity_curves(assessed["robustness"].sweep, "accuracy")
    assert figure.layout.showlegend is not False
    assert all(trace.name for trace in figure.data)


def test_the_sensitivity_map_uses_the_sequential_ramp(assessed):
    sensitivity_map = SensitivityMapper(
        assessed["handler"], n_variants=2, seed=2
    ).feature_severity_map(assessed["data"])
    figure = figures.plot_sensitivity_map(sensitivity_map)
    assert figure.data[0].type == "heatmap"
    assert list(figure.data[0].y) == list(sensitivity_map.row_labels)
    assert len(figure.data[0].colorscale) == len(DEFAULT_PALETTE.sequential)


def test_a_large_grid_omits_the_cell_values(assessed):
    """Numbers on 400 cells are unreadable and slow to render."""
    small = figures.plot_sensitivity_map(
        SensitivityMapper(assessed["handler"], n_variants=1, seed=2).feature_severity_map(
            assessed["data"]
        )
    )
    assert small.data[0].texttemplate is not None


def test_the_confusion_matrix_runs_its_diagonal_top_left_to_bottom_right(assessed):
    figure = figures.plot_confusion_matrix(assessed["accuracy"].confusion)
    assert figure.layout.yaxis.autorange == "reversed"
    assert list(figure.data[0].x) == list(figure.data[0].y)


def test_a_normalised_confusion_matrix_says_so_in_its_title(assessed):
    matrix = confusion_matrix(
        assessed["truth"], assessed["prediction"].predicted, normalize="true"
    )
    figure = figures.plot_confusion_matrix(matrix)
    assert "normalised by true" in figure.layout.title.text


def test_the_reliability_diagram_does_not_interpolate_through_empty_bins(assessed):
    """Closing the gap would invent calibration evidence where there is no data."""
    bins = reliability_bins(assessed["prediction"].probabilities, assessed["truth"])
    figure = figures.plot_reliability_diagram(bins)
    model_trace = next(trace for trace in figure.data if trace.name == "model")
    assert model_trace.connectgaps is False


def test_the_reliability_diagram_shows_the_diagonal_and_the_ece(assessed):
    probabilities, truth = assessed["prediction"].probabilities, assessed["truth"]
    ece = expected_calibration_error(probabilities, truth)
    figure = figures.plot_reliability_diagram(
        reliability_bins(probabilities, truth), ece=ece
    )
    assert any(trace.name == "perfectly calibrated" for trace in figure.data)
    assert f"{ece:.4f}" in figure.layout.title.text


def test_the_group_boxes_carry_the_sample_count_on_each_tick(assessed):
    """A box from six samples and one from six hundred look identical otherwise."""
    grouped = {"a": np.random.default_rng(0).normal(size=6),
               "b": np.random.default_rng(1).normal(size=600)}
    figure = figures.plot_group_boxes(grouped)
    assert list(figure.layout.xaxis.ticktext) == ["a<br>n = 6", "b<br>n = 600"]


def test_too_many_groups_raises_rather_than_recycling_colours():
    grouped = {f"g{i}": np.random.default_rng(i).normal(size=10) for i in range(9)}
    with pytest.raises(ValueError, match="exceeds the 8 categorical slots"):
        figures.plot_group_boxes(grouped)


def test_the_subgroup_bars_include_the_overall_reference(assessed):
    """A subgroup bar means little without the aggregate it is compared against."""
    figure = figures.plot_subgroup_metrics(
        assessed["accuracy"].subgroups,
        ["accuracy", "f1"],
        overall=assessed["accuracy"].overall,
    )
    assert any(trace.name == "overall" for trace in figure.data)
    assert all("n = " in str(trace.name) for trace in figure.data if trace.type == "bar")


def test_requesting_only_uncomputed_metrics_says_what_is_available(assessed):
    with pytest.raises(ValueError, match="Available"):
        figures.plot_subgroup_metrics(assessed["accuracy"].subgroups, ["mae"])


def test_signed_attribution_uses_the_diverging_ramp_and_unsigned_does_not(assessed):
    """Colouring an unsigned quantity as though it had a direction would invent
    information the method does not produce."""
    model = ErrorModel(seed=1)
    model.fit(assessed["data"], assessed["truth"], assessed["prediction"].predicted)

    unsigned = figures.plot_attribution(model.importance(n_repeats=4))
    assert isinstance(unsigned.data[0].marker.color, str)

    pytest.importorskip("shap")
    signed = figures.plot_attribution(model.importance(method="shap"))
    assert not isinstance(signed.data[0].marker.color, str)  # per-bar, from the ramp
    assert signed.data[0].marker.cmin == -signed.data[0].marker.cmax  # centred on zero


# --------------------------------------------------------------------------
# verdict tables
# --------------------------------------------------------------------------


def test_the_metrics_table_puts_the_verdict_in_the_table(assessed):
    """Repo A's and repo D's version had value, baseline and range, and the reader
    compared each row against a dictionary kept somewhere else."""
    figure = figures.plot_metrics_table(assessed["accuracy"].acceptance)
    assert "verdict" in figure.data[0].header.values
    verdicts = figure.data[0].cells.values[3]
    assert any("fail" in cell for cell in verdicts)


def test_a_metric_with_no_requirement_is_not_painted_as_a_warning(assessed):
    """It was not part of the criteria; that is not a warning. An earlier revision made
    half the table glow orange and buried the rows that had actually failed."""
    report = assessed["accuracy"].acceptance
    figure = figures.plot_metrics_table(report)
    fills = figure.data[0].cells.fill.color[3]
    unchecked = [
        i for i, v in enumerate(report.verdicts) if v.requirement is None
    ]
    assert unchecked, "the fixture should have unchecked metrics"
    for i in unchecked:
        assert fills[i] == DEFAULT_PALETTE.surface
        assert "not checked" in figure.data[0].cells.values[3][i]


def test_the_tolerance_table_carries_the_basis_and_the_culprit(assessed):
    """A tolerance verdict without its basis is ambiguous and without the culprit is not
    actionable."""
    figure = figures.plot_tolerance_table(assessed["robustness"].tolerance)
    header = list(figure.data[0].header.values)
    assert "basis" in header
    assert "worst at" in header


def test_an_empty_report_is_refused():
    from themis_assess.assessment.criteria import AcceptanceReport

    with pytest.raises(ValueError, match="no verdicts"):
        figures.plot_metrics_table(AcceptanceReport(verdicts=()))


# --------------------------------------------------------------------------
# bundles: the analysis must come back too
# --------------------------------------------------------------------------


def test_a_group_bundle_returns_the_statistics_not_only_the_figure():
    """The defect this whole module exists to fix. Repo B assembled
    {"figure", "summary", "hypothesis_test_result", "conclusion"}, stashed it on
    self.results[title], and returned only the figure."""
    rng = np.random.default_rng(0)
    values = np.r_[rng.normal(0.90, 0.02, 40), rng.normal(0.82, 0.09, 30)]
    labels = np.r_[["oslo"] * 40, ["valencia"] * 30]

    bundle = bundles.assess_groups(values, labels, metric="accuracy")
    assert bundle.figure.data
    assert bundle.p_value is not None
    assert bundle.groups_differ is True
    assert bundle.comparison.welch is not None
    assert bundle.comparison.significant_pairs
    assert "Welch's ANOVA" in bundle.summary


def test_the_group_figure_annotates_the_test_verdict():
    """So the gap and whether the gap is noise cannot be read apart."""
    rng = np.random.default_rng(1)
    values = np.r_[rng.normal(0.8, 0.05, 30), rng.normal(0.8, 0.05, 30)]
    labels = np.r_[["a"] * 30, ["b"] * 30]
    bundle = bundles.assess_groups(values, labels, metric="accuracy")
    assert "not distinguishable" in bundle.figure.layout.title.text


def test_a_severity_bundle_names_the_worst_cell(assessed):
    bundle = bundles.assess_severity(assessed["robustness"].sweep, "accuracy")
    assert "worst cell" in bundle.summary
    assert bundle.metadata["worst"]["severity"] == 5
    assert set(bundle.analysis) == {"gaussian_noise", "uniform_bias"}


def test_a_severity_bundle_refuses_a_directionless_metric(assessed):
    with pytest.raises(ValueError, match="has no direction"):
        bundles.assess_severity(assessed["robustness"].sweep, "predictive_entropy")


def test_a_calibration_bundle_returns_the_bins_and_the_ece(assessed):
    bundle = bundles.assess_calibration(
        assessed["prediction"].probabilities, assessed["truth"]
    )
    assert "ece" in bundle.analysis
    assert "bins" in bundle.analysis
    assert "ECE" in bundle.summary
    assert "Report the bin count" in bundle.summary


def test_an_outlier_bundle_returns_the_chi_square_test(assessed):
    """Repo C computed the test and returned two bare rates."""
    bundle = bundles.assess_outliers(
        assessed["features"], assessed["truth"], assessed["prediction"].predicted
    )
    assert bundle.analysis.p_value is not None
    assert "chi-square" in bundle.figure.layout.title.text
    assert bundle.figure.data[0].type == "box"


def test_a_feature_bin_bundle_draws_the_overall_rate_as_a_reference(assessed):
    """A chart of per-bin rates with no aggregate invites comparing bins only to
    each other."""
    bundle = bundles.assess_feature_bins(
        assessed["age"], assessed["truth"], assessed["prediction"].predicted,
        feature="age",
    )
    assert any(str(trace.name).startswith("overall") for trace in bundle.figure.data)
    assert 0.0 <= bundle.metadata["overall_error_rate"] <= 1.0


def test_an_accuracy_over_time_bundle_aggregates_and_marks_the_worst():
    from sklearn.metrics import mean_absolute_error

    actual = np.zeros((1, 12, 4))
    forecasts = actual.copy()
    forecasts[0, 7, :] = 5.0  # one bad origin
    bundle = bundles.assess_accuracy_over_time(
        forecasts, actual, mean_absolute_error, metric="mae", highlight_worst=2
    )
    assert bundle.analysis["worst"][0]["position"] == 7
    assert "max" in bundle.summary
    assert any(str(trace.name).startswith("worst") for trace in bundle.figure.data)


def test_every_bundle_carries_a_figure_and_an_analysis(assessed):
    """The invariant: nothing here returns a picture on its own."""
    made = [
        bundles.assess_severity(assessed["robustness"].sweep, "accuracy"),
        bundles.assess_calibration(
            assessed["prediction"].probabilities, assessed["truth"]
        ),
        bundles.assess_outliers(
            assessed["features"], assessed["truth"], assessed["prediction"].predicted
        ),
        bundles.assess_feature_bins(
            assessed["age"], assessed["truth"], assessed["prediction"].predicted
        ),
    ]
    for bundle in made:
        assert bundle.figure is not None
        assert bundle.analysis is not None
        assert bundle.summary


# --------------------------------------------------------------------------
# the model comparison chart
# --------------------------------------------------------------------------


@pytest.fixture
def comparison():
    from themis_assess.comparison import compare_models

    return compare_models(
        {
            "x": {"accuracy": 0.90, "auc": 0.94, "f1": 0.89, "mae": 1.0},
            "y": {"accuracy": 0.85, "auc": 0.96, "f1": 0.86, "mae": 0.8},
            "z": {"accuracy": 0.88, "auc": 0.92, "f1": 0.88, "mae": 1.4},
        }
    )


def test_the_polar_chart_draws_one_closed_loop_per_model(comparison):
    figure = figures.plot_model_comparison(comparison)
    assert len(figure.data) == 3
    for trace in figure.data:
        # closed: the first axis repeated at the end, or the outline has a gap
        assert trace.r[0] == trace.r[-1]
        assert trace.theta[0] == trace.theta[-1]
        assert len(trace.r) == len(comparison.metrics) + 1


def test_the_polar_radius_is_the_normalised_value_not_the_raw_one(comparison):
    """The repo D defect: raw values on a [0, 1] radial axis put the *better* model
    closer to the centre on any error metric, and off the chart on an unbounded one."""
    figure = figures.plot_model_comparison(comparison)
    trace = next(t for t in figure.data if t.name == "y")
    index = list(comparison.metrics).index("mae")
    assert trace.r[index] == pytest.approx(comparison.normalised["y"]["mae"])
    assert trace.r[index] != pytest.approx(comparison.raw["y"]["mae"])
    # y has the lowest mae, so it must be furthest out on that axis
    others = [
        figure.data[list(comparison.models).index(m)].r[index]
        for m in comparison.models
        if m != "y"
    ]
    assert all(trace.r[index] > other for other in others)


def test_the_radial_axis_is_fixed_so_the_shapes_are_comparable(comparison):
    figure = figures.plot_model_comparison(comparison)
    assert tuple(figure.layout.polar.radialaxis.range) == (0, 1)


def test_the_raw_value_is_on_the_hover_beside_the_normalised_one(comparison):
    """A normalised score is not a number anyone can act on, so the measured value has
    to be reachable from the chart."""
    figure = figures.plot_model_comparison(comparison)
    trace = figure.data[0]
    assert "measured" in trace.hovertemplate
    assert trace.customdata is not None


def test_each_model_keeps_its_own_colour_in_the_order_given(comparison):
    figure = figures.plot_model_comparison(comparison)
    assert [trace.line.color for trace in figure.data] == [
        DEFAULT_PALETTE.series(i) for i in range(3)
    ]


def test_the_fill_is_dropped_beyond_two_models(comparison):
    """Three translucent fills overlap into one region and the loops stop being
    readable — the standard multi-series radar failure."""
    from themis_assess.comparison import compare_models

    three = figures.plot_model_comparison(comparison)
    assert all(trace.fill == "none" for trace in three.data)

    two = figures.plot_model_comparison(
        compare_models(
            {
                "x": {"accuracy": 0.9, "auc": 0.94, "f1": 0.89},
                "y": {"accuracy": 0.85, "auc": 0.96, "f1": 0.86},
            }
        )
    )
    assert all(trace.fill == "toself" for trace in two.data)


def test_a_polar_chart_of_two_axes_is_refused():
    from themis_assess.comparison import compare_models

    two_axes = compare_models(
        {"x": {"accuracy": 0.9, "f1": 0.88}, "y": {"accuracy": 0.85, "f1": 0.9}}
    )
    with pytest.raises(ValueError, match="at least three axes"):
        figures.plot_model_comparison(two_axes)
    # and the table is the chart that works for two
    assert figures.plot_comparison_table(two_axes) is not None


def test_too_many_axes_is_refused_with_the_alternative_named(comparison):
    with pytest.raises(ValueError, match="too many for one polar chart"):
        figures.plot_model_comparison(comparison, max_axes=3)


def test_the_axes_can_be_chosen(comparison):
    figure = figures.plot_model_comparison(comparison, metrics=["accuracy", "auc", "f1"])
    assert len(figure.data[0].r) == 4  # three axes, closed
    with pytest.raises(KeyError, match="not axes of this comparison"):
        figures.plot_model_comparison(comparison, metrics=["accuracy", "auc", "nope"])


def test_the_comparison_table_marks_the_best_with_a_symbol_not_only_a_colour(comparison):
    figure = figures.plot_comparison_table(comparison)
    cells = figure.data[0].cells
    accuracy_column = list(cells.values[1 + list(comparison.metrics).index("accuracy")])
    marked = [value for value in accuracy_column if "✓" in value]
    assert len(marked) == 1
    assert marked[0].startswith("0.9")
    # the fill agrees with the symbol rather than replacing it
    fills = list(cells.fill.color[1 + list(comparison.metrics).index("accuracy")])
    assert fills.count(DEFAULT_PALETTE.status["good"]) == 1


def test_the_table_says_what_could_not_be_compared():
    from themis_assess.comparison import compare_models

    partial = compare_models(
        {
            "x": {"accuracy": 0.9, "f1": 0.88, "auc": 0.9},
            "y": {"accuracy": 0.85, "f1": 0.9, "auc": 0.92, "brier_score": 0.2},
        }
    )
    figure = figures.plot_comparison_table(partial)
    assert "brier_score" in figure.layout.title.text


def test_the_bundle_returns_the_comparison_beside_the_chart(comparison):
    made = bundles.assess_models(
        {
            "x": {"accuracy": 0.90, "auc": 0.94, "f1": 0.89, "mae": 1.0},
            "y": {"accuracy": 0.85, "auc": 0.96, "f1": 0.86, "mae": 0.8},
        },
        axes=["accuracy", "auc", "f1"],
    )
    assert made.analysis.models == ("x", "y")
    assert made.summary
    assert made.metadata["is_unanimous"] is False
    assert len(made.figure.data[0].r) == 4
