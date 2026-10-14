"""The built-in perturbation registry.

Assembles every perturbation the library ships with. Following repo D's design note on
``PerturbationHandler``, the registry holds and filters but never decides:

    Selection of which method(s) to run is the caller's responsibility.

:func:`default_registry` returns a fresh instance each call, so a caller registering a
use-case-specific perturbation does not mutate anyone else's registry.
"""

from __future__ import annotations

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import PerturbationSpec, TaskType
from themis_assess.perturbations.base import Perturbation, PerturbationRegistry
from themis_assess.perturbations.correlated import CorrelatedNoise
from themis_assess.perturbations.tabular import (
    AmplitudeScaling,
    CategorySwap,
    GaussianNoise,
    MissingValues,
    UniformBias,
)
from themis_assess.perturbations.text import CharacterNoise, TokenMasking, WordDropout
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

__all__ = ["default_registry", "BUILTIN_PERTURBATIONS", "estimate_cost"]

#: Every perturbation the library ships, in the order they are registered.
#:
#: ``STLComponentPerturbation`` is deliberately absent: it needs a seasonal ``period``,
#: which cannot be inferred from an unlabelled array, so there is no sensible default
#: instance. Construct it and register it yourself.
BUILTIN_PERTURBATIONS: tuple[type[Perturbation], ...] = (
    # feature space — tabular and sequence alike
    GaussianNoise,
    UniformBias,
    AmplitudeScaling,
    MissingValues,
    CategorySwap,
    CorrelatedNoise,
    # time-series value domain
    SensorDrift,
    ExponentialDamping,
    FlatSensor,
    OutlierSpikes,
    MissingSegment,
    AlphaStableNoise,
    # time-series time axis
    WindowSlice,
    TimeWarp,
    InputWindowShift,
    # text
    CharacterNoise,
    TokenMasking,
    WordDropout,
)


def default_registry() -> PerturbationRegistry:
    """A fresh registry holding every built-in perturbation."""
    return PerturbationRegistry([cls() for cls in BUILTIN_PERTURBATIONS])


def estimate_cost(
    registry: PerturbationRegistry,
    data: DataAdapter,
    specs: list[PerturbationSpec],
    *,
    task: TaskType | None = None,
) -> dict[str, int]:
    """How many model calls a set of specifications will need, per perturbation.

    Both repo B and repo C left cost warnings in their notebooks rather than in code —
    repo C's missing-value sweep "will take about 15mins on a 12 thread laptop", and repo
    B's heatmap "calls the model ``n_features x n_timesteps x 10`` times". Neither number
    was available before the run started.

    Returns
    -------
    dict
        ``{spec.key: n_model_calls}`` plus a ``"total"`` entry. A perturbation that
        cannot act on this data is omitted rather than counted as zero, so the total is
        what will actually be spent.
    """
    out: dict[str, int] = {}
    total = 0
    for spec in specs:
        perturbation = registry[spec.name]
        if not perturbation.applies_to(data, task):
            continue
        cost = perturbation.n_model_calls(data, spec)
        out[spec.key] = cost
        total += cost
    out["total"] = total
    return out
