"""Perturbations: label-preserving transformations of model input.

The contract is in :mod:`themis_assess.perturbations.base`; implementations are grouped
by what they act on:

``tabular``                  feature-space perturbations, tabular and sequence alike
``correlated``               noise respecting the features' joint structure (survey §7)
``timeseries.value_domain``  what the sensor reads (survey §4.2)
``timeseries.time_axis``     when it was read (survey §4.2)
``timeseries.structural``    STL component-selective perturbation (survey §4.2)
``text``                     character noise, masking, word dropout (survey §4.3)

:func:`themis_assess.perturbations.registry.default_registry` assembles the eighteen
that need no configuration; ``STLComponentPerturbation`` is constructed by the caller
because it needs a seasonal period.

Severity is an ordered level, never a raw parameter. Each perturbation maps ``1..5`` onto
its own natural scale and reports the mapping in ``metadata()``, which is what makes the
severity-aggregating metrics (``mCE``, ``mFR``, ``R_Phi``) comparable across
perturbations. Numeric magnitudes are relative to each feature's own standard deviation
by default — see the module docstring of :mod:`themis_assess.perturbations.tabular` for
why that matters.
"""

from themis_assess.perturbations.base import Perturbation, PerturbationRegistry
from themis_assess.perturbations.correlated import CorrelatedNoise, estimate_covariance
from themis_assess.perturbations.registry import (
    BUILTIN_PERTURBATIONS,
    default_registry,
    estimate_cost,
)
from themis_assess.perturbations.tabular import (
    AmplitudeScaling,
    CategorySwap,
    GaussianNoise,
    MissingValues,
    UniformBias,
    feature_reference_scale,
)
from themis_assess.perturbations.text import (
    CharacterNoise,
    CharacterOperation,
    TokenMasking,
    WordDropout,
)
from themis_assess.perturbations.timeseries.structural import (
    Component,
    STLComponentPerturbation,
    decompose,
)
from themis_assess.perturbations.timeseries.time_axis import (
    InputWindowShift,
    TimeWarp,
    WindowSlice,
)
from themis_assess.perturbations.timeseries.value_domain import (
    AlphaStableNoise,
    ExponentialDamping,
    FlatSensor,
    MissingSegment,
    OutlierSpikes,
    SensorDrift,
    sample_alpha_stable,
)

__all__ = [
    # contract and registry
    "Perturbation",
    "PerturbationRegistry",
    "default_registry",
    "BUILTIN_PERTURBATIONS",
    "estimate_cost",
    # feature space
    "GaussianNoise",
    "UniformBias",
    "AmplitudeScaling",
    "MissingValues",
    "CategorySwap",
    "CorrelatedNoise",
    # time-series value domain
    "SensorDrift",
    "ExponentialDamping",
    "FlatSensor",
    "OutlierSpikes",
    "MissingSegment",
    "AlphaStableNoise",
    # time-series time axis
    "WindowSlice",
    "TimeWarp",
    "InputWindowShift",
    # time-series structural
    "STLComponentPerturbation",
    "Component",
    "decompose",
    # text
    "CharacterNoise",
    "CharacterOperation",
    "TokenMasking",
    "WordDropout",
    # helpers
    "feature_reference_scale",
    "estimate_covariance",
    "sample_alpha_stable",
]
