"""Learned error attribution: a surrogate model of the assessed model's error.

Where a sensitivity map perturbs the input and watches the model move, this fits a second
small model that predicts the first model's error from a description of its input, and
reads that surrogate's attributions. It answers a question the survey raises but does not
instrument: not "what happens if the input changes" but "what do the cases this model
already gets wrong have in common".

The whole capability costs one prediction pass. Its constraint is sample size — there is
one row per entity — and its gate is the surrogate's own held-out R²: an error model that
cannot predict the error has attributions indistinguishable from noise, so
:attr:`~themis_assess.attribution.error_model.ErrorModel.is_trustworthy` guards the
attribution methods.
"""

from themis_assess.attribution.error_model import (
    ErrorModel,
    ErrorModelFit,
    FeatureAttribution,
    per_sample_error,
)
from themis_assess.attribution.features import (
    FeatureExtractor,
    TabularFeatures,
    TSFreshFeatures,
    WindowSummaryFeatures,
)

__all__ = [
    "ErrorModel",
    "ErrorModelFit",
    "FeatureAttribution",
    "per_sample_error",
    "FeatureExtractor",
    "TabularFeatures",
    "WindowSummaryFeatures",
    "TSFreshFeatures",
]
