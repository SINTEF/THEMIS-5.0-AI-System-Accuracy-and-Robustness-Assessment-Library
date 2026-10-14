"""Shared enumerations, specifications and type aliases.

Deliberately dependency-free: nothing here imports anything outside the standard
library, so every other module can depend on it without pulling anything in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

__all__ = [
    "TaskType",
    "Modality",
    "AccessLevel",
    "Scope",
    "TimeScope",
    "Alignment",
    "PerturbationSpec",
    "DEFAULT_SEVERITY_LEVELS",
    "SeverityCurve",
    "PerturbationCurves",
    "GroupScores",
]

#: Number of discrete severity levels a perturbation exposes unless it declares
#: otherwise. Five follows ImageNet-C's convention (docs/DECISIONS.md, D16). A fixed
#: convention is what makes cross-perturbation aggregation — mCE, mFR, R_Phi — mean
#: anything at all.
DEFAULT_SEVERITY_LEVELS = 5


class TaskType(str, Enum):
    """The kind of prediction problem a metric or perturbation applies to."""

    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"


class Modality(str, Enum):
    """The shape of the input data.

    ``IMAGE`` is declared so adapters and perturbations can accommodate it, but no
    image corruptions are implemented (docs/DECISIONS.md, D17).
    """

    TABULAR = "tabular"
    SEQUENCE = "sequence"
    TEXT = "text"
    IMAGE = "image"


class AccessLevel(int, Enum):
    """What a metric needs from the model, ordered from least to most demanding.

    A handler declares the highest level it can serve; the runner compares that against
    every metric's requirement and reports what it cannot compute *before* running
    anything, rather than failing part-way through.

    ``BLACK_BOX``      predictions only
    ``PROBABILISTIC``  class probabilities, or predictive intervals for forecasting
    ``QUERY``          many repeated forward passes (e.g. boundary search)
    ``EMBEDDING``      penultimate-layer representations
    ``ACTIVATION``     intermediate-layer activations
    ``GRADIENT``       gradients with respect to the input

    This is a **total order**, and declaring a level asserts everything below it as
    well: a handler declaring ``QUERY`` is promising probabilities too, and will be
    offered the calibration metrics. That reading is a simplification — being callable
    many times and exposing probabilities are in principle independent — so a model that
    can be queried freely but returns bare labels should declare ``BLACK_BOX``. Nothing
    is lost by doing so while the query-based metrics of survey §5.2 remain deferred
    (docs/DECISIONS.md, D21).
    """

    BLACK_BOX = 0
    PROBABILISTIC = 1
    QUERY = 2
    EMBEDDING = 3
    ACTIVATION = 4
    GRADIENT = 5

    def satisfies(self, required: AccessLevel) -> bool:
        """Whether a handler at this level can serve a metric requiring ``required``."""
        return self.value >= required.value


class Scope(str, Enum):
    """Which features a perturbation touches.

    With :class:`TimeScope`, this is what collapses repo B's sixteen ``perturb_*``
    methods into a handful of implementations plus a specification.
    """

    ALL_FEATURES = "all"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    NAMED = "named"


class TimeScope(str, Enum):
    """When along the time axis a perturbation applies.

    ``CONSTANT``     one draw per entity, applied at every timestep
    ``PER_TIMESTEP`` an independent draw at each timestep
    ``AT_INDICES``   one draw, applied only at the listed timesteps

    For tabular data there is a single timestep and all three coincide.
    """

    CONSTANT = "constant"
    PER_TIMESTEP = "per_timestep"
    AT_INDICES = "at_indices"


class Alignment(str, Enum):
    """How entities of unequal length line up on a shared time axis.

    ``END``   last timestep of every entity in the rightmost column
    ``START`` first timestep of every entity in the leftmost column

    ``END`` is the right default for event forecasting, where what matters is time
    remaining until the event rather than time since the record began. Repo B's
    ``all_vessels_heatmap`` end-aligns for exactly that reason.
    """

    END = "end"
    START = "start"


@dataclass(frozen=True, slots=True)
class PerturbationSpec:
    """A perturbation request: which method, how strong, where applied.

    Separating the request from the implementation is what lets one ``GaussianNoise``
    class serve what were previously distinct methods —
    ``perturb_only_numerical_over_time``, ``perturb_all_data``,
    ``perturb_one_feature_at_k_times`` are all specifications over the same code.

    Parameters
    ----------
    name:
        Registered name of the perturbation method.
    severity:
        Ordered level, ``1..S``. Not a raw parameter value: each perturbation maps
        severity onto its own natural scale, and reports that mapping in
        ``metadata()`` so a report can say what "severity 3" meant.
    params:
        Method-specific overrides. Prefer ``severity`` where a method offers it, so
        results stay comparable across perturbations.
    scope:
        Which features to touch. ``NAMED`` requires ``features``.
    features:
        Feature names, when ``scope`` is ``NAMED``.
    time_scope:
        When along the time axis to apply it. ``AT_INDICES`` requires ``time_indices``.
    time_indices:
        Positional timestep indices, when ``time_scope`` is ``AT_INDICES``.
    n_variants:
        How many independent perturbed copies to draw. Repo D called this
        ``transformations_per_example``, repo B called it ``n_copies``.
    """

    name: str
    severity: int = 1
    params: Mapping[str, Any] = field(default_factory=dict)
    scope: Scope = Scope.ALL_FEATURES
    features: tuple[str, ...] = ()
    time_scope: TimeScope = TimeScope.CONSTANT
    time_indices: tuple[int, ...] = ()
    n_variants: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PerturbationSpec needs a non-empty name.")
        if self.severity < 1:
            raise ValueError(
                f"severity is 1-based and must be at least 1; got {self.severity}."
            )
        if self.n_variants < 1:
            raise ValueError(f"n_variants must be at least 1; got {self.n_variants}.")
        if self.scope is Scope.NAMED and not self.features:
            raise ValueError(
                "scope=NAMED requires at least one feature name in `features`."
            )
        if self.scope is not Scope.NAMED and self.features:
            raise ValueError(
                f"`features` is only meaningful with scope=NAMED; got scope={self.scope.value} "
                f"and features={self.features}."
            )
        if self.time_scope is TimeScope.AT_INDICES and not self.time_indices:
            raise ValueError(
                "time_scope=AT_INDICES requires at least one index in `time_indices`."
            )
        if self.time_scope is not TimeScope.AT_INDICES and self.time_indices:
            raise ValueError(
                f"`time_indices` is only meaningful with time_scope=AT_INDICES; got "
                f"time_scope={self.time_scope.value}."
            )
        if any(i < 0 for i in self.time_indices):
            raise ValueError(f"time_indices must be non-negative; got {self.time_indices}.")
        # Freeze `params` so a spec shared between runs cannot be mutated under them.
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        """A hashable summary of the whole specification, for use as a cache key.

        The spec itself is not hashable — ``params`` is a mapping, and flattening it
        into the dataclass fields purely to satisfy ``hash()`` would make the natural
        ``params={"distribution": "alpha_stable"}`` construction awkward. Perturbing is
        expensive enough to be worth memoising, so this gives callers a key without
        distorting the API. Specs that are equal have equal fingerprints.
        """
        return (
            self.name,
            self.severity,
            tuple(sorted((str(k), repr(v)) for k, v in self.params.items())),
            self.scope.value,
            self.features,
            self.time_scope.value,
            self.time_indices,
            self.n_variants,
        )

    @property
    def key(self) -> str:
        """Stable identifier combining method and severity, e.g. ``'gaussian_noise@3'``.

        Replaces the ``"normal_0_01"`` string keys the old ``compute_robustness``
        parsed with ``split("_", 1)`` — which mis-parsed any method name containing an
        underscore and raised on any name without one. Here method and severity are
        never recovered by parsing; they are kept as fields and this is only a label.
        """
        return f"{self.name}@{self.severity}"

    def at_severity(self, severity: int) -> PerturbationSpec:
        """A copy of this spec at a different severity — for building severity curves."""
        return PerturbationSpec(
            name=self.name,
            severity=severity,
            params=self.params,
            scope=self.scope,
            features=self.features,
            time_scope=self.time_scope,
            time_indices=self.time_indices,
            n_variants=self.n_variants,
        )


# A metric value at each severity level, indexed s = 1..S.
# Position 0 holds severity 1: ``curve[s - 1]`` is the value at severity ``s``.
SeverityCurve = Sequence[float]

# Perturbation name -> its severity curve.
PerturbationCurves = Mapping[str, SeverityCurve]

# Subgroup name -> a scalar score for that group.
GroupScores = Mapping[str, float]
