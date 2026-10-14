"""Sensitivity maps: which input, perturbed where, moves the model most.

A robustness score says how much a model degrades; a sensitivity map says what it is
sensitive to. The two are not the same measurement, and only the second is actionable.
"""

from themis_assess.sensitivity.maps import (
    SensitivityMap,
    SensitivityMapper,
    map_from_sweep,
)

__all__ = ["SensitivityMap", "SensitivityMapper", "map_from_sweep"]
