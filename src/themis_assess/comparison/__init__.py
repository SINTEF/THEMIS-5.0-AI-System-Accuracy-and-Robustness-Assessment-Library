"""Comparing several models — the question a project actually asks.

:func:`~themis_assess.comparison.compare.compare_models` takes the assessment reports of
model x, model y and model z and puts them on axes that are comparable: normalised to
[0, 1] with 1 always better, restricted to the metrics every model carries, and refusing
to share an axis between robustness scores computed over different underlying metrics.

The figures live with the rest of the figures, in :mod:`themis_assess.viz.figures`
(``plot_model_comparison`` for the polar chart, ``plot_comparison_table`` for the
numbers), and the one-call analysis-plus-plot form is
:func:`themis_assess.viz.bundles.assess_models`.
"""

from themis_assess.comparison.compare import (
    MetricDisagreement,
    ModelComparison,
    compare_models,
)

__all__ = ["ModelComparison", "MetricDisagreement", "compare_models"]
