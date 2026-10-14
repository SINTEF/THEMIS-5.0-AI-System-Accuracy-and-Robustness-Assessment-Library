"""Time-series perturbations, organised by the domain in which they act (survey §4.2).

``value_domain``  what the sensor reads — noise, drift, damping, freezes, spikes, gaps
``time_axis``     when it was read — slicing, warping, a shifted forecast origin
``structural``    STL decomposition, perturbing one component and preserving the rest
"""

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
)

__all__ = [
    # value domain
    "SensorDrift",
    "ExponentialDamping",
    "FlatSensor",
    "OutlierSpikes",
    "MissingSegment",
    "AlphaStableNoise",
    # time axis
    "WindowSlice",
    "TimeWarp",
    "InputWindowShift",
    # structural
    "STLComponentPerturbation",
    "Component",
    "decompose",
]
