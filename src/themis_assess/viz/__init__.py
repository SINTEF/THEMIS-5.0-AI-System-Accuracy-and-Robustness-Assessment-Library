"""Figures, in two shapes.

:mod:`themis_assess.viz.figures` holds pure functions from a result object to a plotly
figure — no model calls, no fitting, no statistical tests. That separation is what lets a
number be tested, tabulated and cited without a rendering step.

:mod:`themis_assess.viz.bundles` holds the *bundled* form the four use-case repos actually
used: compute and draw in one call, which is the more convenient thing in a notebook.
Every bundle returns the figure **and** the analysis, which is where repo B's version fell
short — it assembled the right bundle and then returned only the figure, leaving the
statistics reachable solely through a side-effect dictionary keyed on the plot's title.

:mod:`themis_assess.viz.theme` is the one place colour and layout are decided.
"""

from themis_assess.viz.bundles import (
    Bundle,
    GroupBundle,
    assess_accuracy_over_time,
    assess_calibration,
    assess_feature_bins,
    assess_groups,
    assess_models,
    assess_outliers,
    assess_severity,
)
from themis_assess.viz.figures import (
    plot_accuracy_over_time,
    plot_attribution,
    plot_comparison_table,
    plot_confusion_matrix,
    plot_feature_bins,
    plot_group_boxes,
    plot_metrics_table,
    plot_reliability_diagram,
    plot_sensitivity_map,
    plot_model_comparison,
    plot_severity_curves,
    plot_subgroup_metrics,
    plot_tolerance_table,
)
from themis_assess.viz.theme import DEFAULT_PALETTE, Palette, apply_layout

__all__ = [
    # theme
    "Palette",
    "DEFAULT_PALETTE",
    "apply_layout",
    # figures from results
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
    # bundled analysis + figure
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
