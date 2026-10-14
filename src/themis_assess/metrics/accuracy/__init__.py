"""Accuracy metrics: how well the model does, and where it does badly.

The distinction from :mod:`themis_assess.metrics.robustness` is what each measures. A
metric here scores one model on one dataset. A metric there compares scores — clean
against shifted, group against group, model against baseline — and needs at least two of
the numbers this package produces.

Four modules:

``catalog``         name -> callable, for everything the library can compute
``classification``  the confusion matrix and what follows from it
``diagnostics``     where the errors concentrate: outliers, feature regions, uncertainty
``significance``    whether a gap between groups is more than sampling noise
``temporal``        accuracy along the time axis, for forecasting
``weighted``        event-weighted errors, for series where the rare moment is the point

Calibration lives in :mod:`themis_assess.metrics.calibration`, one level up, because it
belongs to both halves: whether a model's confidence is honest is an accuracy property of
its uncertainty, and only its degradation under shift is a robustness question.
"""

from themis_assess.metrics.accuracy.catalog import (
    MetricCatalog,
    MetricEntry,
    default_catalog,
)
from themis_assess.metrics.accuracy.classification import (
    ConfusionMatrix,
    confusion_matrix,
    error_rate_by_class,
    per_class_metrics,
    top_confusions,
)
from themis_assess.metrics.accuracy.diagnostics import (
    OutlierErrorAnalysis,
    ambiguity_threshold,
    error_rate_by_feature_bin,
    error_rate_by_outlier_status,
    flag_ambiguous,
    outlier_scores,
)
from themis_assess.metrics.accuracy.significance import (
    GroupComparison,
    GroupTest,
    PairwiseComparison,
    compare_groups,
    kruskal_wallis,
    pairwise_welch,
    welch_anova,
)
from themis_assess.metrics.accuracy.temporal import (
    accuracy_by_horizon,
    accuracy_over_time,
    aggregate_over_time,
    worst_windows,
)
from themis_assess.metrics.accuracy.weighted import (
    event_weights,
    mean_signed_deviation,
    step_onsets,
    weighted_mae,
    weighted_mse,
)

__all__ = [
    # catalog
    "MetricCatalog",
    "MetricEntry",
    "default_catalog",
    # classification structure
    "ConfusionMatrix",
    "confusion_matrix",
    "per_class_metrics",
    "error_rate_by_class",
    "top_confusions",
    # diagnostics
    "OutlierErrorAnalysis",
    "outlier_scores",
    "error_rate_by_outlier_status",
    "error_rate_by_feature_bin",
    "ambiguity_threshold",
    "flag_ambiguous",
    # group significance
    "compare_groups",
    "GroupComparison",
    "GroupTest",
    "PairwiseComparison",
    "welch_anova",
    "kruskal_wallis",
    "pairwise_welch",
    # temporal
    "accuracy_over_time",
    "accuracy_by_horizon",
    "aggregate_over_time",
    "worst_windows",
    # event-weighted errors
    "event_weights",
    "step_onsets",
    "weighted_mae",
    "weighted_mse",
    "mean_signed_deviation",
]
