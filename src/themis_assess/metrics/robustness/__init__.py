"""Robustness metrics, as pure functions of numbers.

Grouped by the survey section that defines them:

``degradation``  §5.1.1 and §5.1.3 — performance loss under shift and perturbation
``stability``    §5.1.3 Eqs. 23-25 — whether individual predictions stay put
``subgroup``     §5.1.2 — disparity between groups
``detection``    §5.5 — evaluating any binary detector's scores
``composite``    §5.1.4 — single-number summaries, with their components
``calibration``  §5.4 — whether the model's confidence survives the shift
``distance``     §5.3 — how far the data moved, the context a degradation is read against
"""

from themis_assess.metrics.robustness import (
    calibration,
    composite,
    degradation,
    detection,
    distance,
    stability,
    subgroup,
)

__all__ = [
    "degradation",
    "stability",
    "subgroup",
    "detection",
    "composite",
    "calibration",
    "distance",
]
