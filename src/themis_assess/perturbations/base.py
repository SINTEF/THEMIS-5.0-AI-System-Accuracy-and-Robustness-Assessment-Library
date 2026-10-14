"""The perturbation contract.

Built on repo D's ``BasePerturbationMethod`` / ``PerturbationHandler``, which had already
got the essential move right: a perturbation is an object with its own parameters and a
``perturb`` method, registered by name, and *which* perturbations to run is the caller's
decision rather than the perturbation's. Repos A, B and C instead carried perturbations
as methods on the assessment class, where they could not be shared between task types or
tested without a model.

Two things are added to repo D's design:

* **Severity as an ordered axis.** Repo D parameterised its methods directly
  (``pct_words_to_swap=0.2``). Every aggregate metric in survey §5.1.3 — ``mCE``,
  ``mFR``, ``mREI``, ``R_Phi`` — sums over severity levels ``s = 1..S``, and a raw
  parameter is not a level. Each method maps severity onto its own natural scale and
  reports that mapping in :meth:`Perturbation.metadata`, so a report can say what
  "severity 3" actually was.

* **Sequences.** Survey Eq. 23 is defined over a run of inputs at monotonically
  increasing severity, not over a clean/perturbed pair. :meth:`Perturbation.sequence`
  produces that run.

No implementations live here — this module is the contract. The perturbations are in
:mod:`themis_assess.perturbations.tabular`,
:mod:`themis_assess.perturbations.correlated` and
:mod:`themis_assess.perturbations.timeseries`, and
:func:`themis_assess.perturbations.registry.default_registry` assembles them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Self

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import (
    DEFAULT_SEVERITY_LEVELS,
    Modality,
    PerturbationSpec,
    Scope,
    TaskType,
    TimeScope,
)

__all__ = ["Perturbation", "PerturbationRegistry"]


class Perturbation(ABC):
    """A label-preserving transformation of model input.

    Subclasses declare what they apply to and implement :meth:`perturb`. Everything
    else — resolving which features a :class:`~themis_assess.core.types.Scope` selects,
    building severity sequences, validating a specification against the data — is
    provided here, so a new perturbation is usually one method.

    Class attributes
    ----------------
    name:
        Registry key. Required.
    task_types:
        Which task types this applies to. Both by default.
    modalities:
        Which data modalities this applies to. A text perturbation declares ``{TEXT}``
        and will never be handed a tabular adapter.
    label_preserving:
        Whether the transformation is assumed to leave the true label unchanged. Survey
        §6.3 treats this as a *modelling assumption*, not a fact: it weakens as severity
        grows, and if it breaks the bookkeeping inverts — a correct prediction change
        gets scored as brittleness. Set ``False`` for a transformation intended to
        change the label, so a report can say which results rest on the assumption.
    severity_levels:
        How many discrete levels this exposes. Five by default (docs/DECISIONS.md, D16).
    """

    name: ClassVar[str] = ""
    task_types: ClassVar[frozenset[TaskType]] = frozenset(TaskType)
    modalities: ClassVar[frozenset[Modality]] = frozenset(
        {Modality.TABULAR, Modality.SEQUENCE}
    )
    label_preserving: ClassVar[bool] = True
    severity_levels: ClassVar[int] = DEFAULT_SEVERITY_LEVELS
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = True
    severity_scale: ClassVar[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # The name is checked at instantiation, not class creation: an abstract
        # intermediate subclass legitimately has no name, and __init_subclass__ cannot
        # tell the difference — `__abstractmethods__` is a descriptor on `type` that
        # ABCMeta populates only after __init_subclass__ has already run.
        #
        # super() goes first so ABCMeta's "still abstract" error takes precedence: an
        # abstract base is missing both a name and an implementation, and the missing
        # implementation is the more useful thing to be told about.
        instance = super().__new__(cls)
        if not cls.name:
            raise TypeError(
                f"{cls.__name__} must declare a class-level `name` before it can be "
                f"instantiated or registered."
            )
        return instance

    # -- the one method a subclass must write ----------------------------------

    @abstractmethod
    def perturb(
        self,
        data: DataAdapter,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> list[DataAdapter]:
        """Return ``spec.n_variants`` independently perturbed copies of ``data``.

        Implementations must draw every random quantity from ``rng`` and never from
        global numpy state. Repo D's ``MaskingPerturbation`` used ``np.random.rand()``
        and repo B's methods reseeded globally from a ``seed`` argument; neither is
        reproducible when anything else in the process also draws.

        Returning a list rather than one copy generalises repo D's
        ``transformations_per_example`` and repo B's ``n_copies`` — the sensitivity
        heatmaps need many draws per cell, and a single-copy signature would force the
        caller to re-implement the loop.
        """

    # -- provided ---------------------------------------------------------------

    def sequence(
        self,
        data: DataAdapter,
        spec: PerturbationSpec,
        rng: np.random.Generator,
        *,
        levels: Sequence[int] | None = None,
    ) -> list[DataAdapter]:
        """One perturbed copy per severity level, in increasing order.

        This is what the flip-rate family (survey Eqs. 23-25) requires: a run of inputs
        at graded severity, so transitions between consecutive frames can be counted.
        Independent draws at one severity are a different and weaker measurement.

        The default walks ``1..severity_levels`` with ``n_variants`` forced to 1, so
        frame ``i`` is the same input at severity ``i + 1``.
        """
        wanted = list(levels) if levels is not None else list(range(1, self.severity_levels + 1))
        if len(wanted) < 2:
            raise ValueError(
                f"A perturbation sequence needs at least 2 severity levels to have a "
                f"transition; got {wanted}."
            )
        if sorted(wanted) != wanted:
            raise ValueError(f"Severity levels must increase monotonically; got {wanted}.")
        out: list[DataAdapter] = []
        for level in wanted:
            step = spec.at_severity(level)
            step = PerturbationSpec(
                name=step.name,
                severity=step.severity,
                params=step.params,
                scope=step.scope,
                features=step.features,
                time_scope=step.time_scope,
                time_indices=step.time_indices,
                n_variants=1,
            )
            out.append(self.perturb(data, step, rng)[0])
        return out

    def severity_curve(
        self,
        data: DataAdapter,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> Iterator[tuple[int, list[DataAdapter]]]:
        """Yield ``(severity, variants)`` for each level, for building a severity curve."""
        for level in range(1, self.severity_levels + 1):
            yield level, self.perturb(data, spec.at_severity(level), rng)

    def applies_to(self, data: DataAdapter, task: TaskType | None = None) -> bool:
        """Whether this perturbation can act on ``data`` for ``task``.

        Checks three things: the modality, the task type, and whether the data actually
        contains a feature of a kind this perturbation accepts. The last matters —
        without it a registry query would offer ``category_swap`` on a dataset with no
        categorical features, and the caller would only find out when it raised.
        """
        if data.modality not in self.modalities:
            return False
        if task is not None and task not in self.task_types:
            return False
        has_numerical = bool(data.numerical_features)
        has_categorical = bool(data.categorical_features)
        return (self.accepts_numerical and has_numerical) or (
            self.accepts_categorical and has_categorical
        )

    def validate(self, data: DataAdapter, spec: PerturbationSpec) -> None:
        """Check a specification against the data, raising with a specific message.

        Called at the top of a well-behaved :meth:`perturb`. Catching a mismatch here
        beats discovering it as an index error several thousand model calls later.
        """
        if spec.name != self.name:
            raise ValueError(
                f"Specification names {spec.name!r} but this is {self.name!r}."
            )
        if data.modality not in self.modalities:
            raise ValueError(
                f"{self.name!r} applies to "
                f"{sorted(m.value for m in self.modalities)}, but the data is "
                f"{data.modality.value}."
            )
        if spec.severity > self.severity_levels:
            raise ValueError(
                f"{self.name!r} exposes {self.severity_levels} severity level(s); "
                f"got severity {spec.severity}."
            )
        if spec.scope is Scope.NAMED:
            unknown = [f for f in spec.features if f not in set(data.feature_names)]
            if unknown:
                raise ValueError(
                    f"Specification names feature(s) not in the data: {unknown}. "
                    f"Available: {data.feature_names}."
                )
        if spec.time_scope is TimeScope.AT_INDICES:
            out_of_range = [i for i in spec.time_indices if i >= data.n_timesteps]
            if out_of_range:
                raise ValueError(
                    f"time_indices {out_of_range} exceed the {data.n_timesteps} "
                    f"timestep(s) available."
                )
        if not data.is_sequential and spec.time_scope is not TimeScope.CONSTANT:
            raise ValueError(
                f"time_scope={spec.time_scope.value} needs sequential data, but this "
                f"adapter is {data.modality.value} with a single timestep."
            )

    def resolve_features(self, data: DataAdapter, spec: PerturbationSpec) -> list[str]:
        """Which feature names the specification's scope selects.

        This is what collapses repo B's sixteen methods: ``perturb_only_numerical``,
        ``perturb_only_categorical`` and ``perturb_all_data`` differ only in what this
        returns.
        """
        if spec.scope is Scope.ALL_FEATURES:
            return data.feature_names
        if spec.scope is Scope.NUMERICAL:
            return data.numerical_features
        if spec.scope is Scope.CATEGORICAL:
            return data.categorical_features
        return list(spec.features)

    def eligible_features(self, data: DataAdapter, spec: PerturbationSpec) -> list[str]:
        """The features this perturbation may actually touch, given the scope.

        :meth:`resolve_features` answers "what did the caller ask for"; this answers
        "what can this perturbation legitimately do to it". Gaussian noise on an encoded
        category is not a milder version of a valid perturbation, it is nonsense.

        A ``NAMED`` scope that names an ineligible feature **raises** — the caller was
        explicit and got it wrong. A broader scope silently narrows, and
        :meth:`metadata` records which features were skipped, so the narrowing appears
        in the report rather than being invisible.
        """
        requested = self.resolve_features(data, spec)
        categorical = set(data.categorical_features)
        eligible = [
            f
            for f in requested
            if (self.accepts_categorical if f in categorical else self.accepts_numerical)
        ]
        if spec.scope is Scope.NAMED and len(eligible) != len(requested):
            rejected = [f for f in requested if f not in set(eligible)]
            kind = "categorical" if not self.accepts_categorical else "numerical"
            raise ValueError(
                f"{self.name!r} does not apply to {kind} features, but the "
                f"specification names {rejected}. Either drop them or use a "
                f"perturbation that accepts them."
            )
        if not eligible:
            accepted = []
            if self.accepts_numerical:
                accepted.append("numerical")
            if self.accepts_categorical:
                accepted.append("categorical")
            raise ValueError(
                f"{self.name!r} has nothing to perturb: scope={spec.scope.value} "
                f"selected {requested or 'no features'}, and it accepts only "
                f"{' and '.join(accepted)} features. The data declares "
                f"numerical={data.numerical_features} and "
                f"categorical={data.categorical_features} — check that the feature "
                f"types are declared as you intend, since a constant feature is "
                f"inferred categorical by default. Returning the data unchanged would "
                f"score as perfect robustness."
            )
        return eligible

    def severity_factor(self, spec: PerturbationSpec) -> float:
        """The multiplier this severity corresponds to, from :attr:`severity_scale`.

        Severity is an ordinal level; this is where it becomes a number. Keeping the
        mapping on the class — rather than letting callers pass raw parameters — is what
        makes ``mCE``, ``mFR`` and the other severity-aggregating metrics comparable
        across perturbations.
        """
        if len(self.severity_scale) != self.severity_levels:
            raise ValueError(
                f"{type(self).__name__} declares {self.severity_levels} severity "
                f"level(s) but a severity_scale of length {len(self.severity_scale)}."
            )
        if not 1 <= spec.severity <= self.severity_levels:
            raise ValueError(
                f"{self.name!r} exposes severities 1..{self.severity_levels}; "
                f"got {spec.severity}."
            )
        return float(self.severity_scale[spec.severity - 1])

    def n_model_calls(self, data: DataAdapter, spec: PerturbationSpec) -> int:
        """How many model evaluations running this specification will cost.

        Both repo B and repo C left cost warnings in their notebooks — "100 repeats will
        take about 15mins on a 12 thread laptop", and a heatmap that "calls the model
        n_features x n_timesteps x 10 times". Making the estimate available before the
        run beats discovering it afterwards.
        """
        return int(spec.n_variants)

    def resolve_timesteps(self, data: DataAdapter, spec: PerturbationSpec) -> list[int]:
        """Which timestep indices the specification's time scope selects."""
        if spec.time_scope is TimeScope.AT_INDICES:
            return list(spec.time_indices)
        return list(range(data.n_timesteps))

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        """What this perturbation did, for the report.

        Repo D's reporting hook, with severity added. When ``spec`` is given, the
        returned dictionary **must** state what its severity meant in the method's own
        units — the survey requires a report to say what was varied, and "severity 3" on
        its own does not.
        """
        meta: dict[str, Any] = {
            "name": self.name,
            "type": type(self).__name__,
            "label_preserving": self.label_preserving,
            "severity_levels": self.severity_levels,
            "severity_scale": list(self.severity_scale),
            "modalities": sorted(m.value for m in self.modalities),
            "task_types": sorted(t.value for t in self.task_types),
            "accepts_numerical": self.accepts_numerical,
            "accepts_categorical": self.accepts_categorical,
        }
        if spec is not None:
            meta["severity"] = spec.severity
            meta["severity_factor"] = self.severity_factor(spec)
            meta["scope"] = spec.scope.value
            meta["time_scope"] = spec.time_scope.value
            meta["n_variants"] = spec.n_variants
            if spec.params:
                meta["params"] = dict(spec.params)
        return meta

    def spec(self, severity: int = 1, **kwargs: Any) -> PerturbationSpec:
        """Convenience constructor for a specification naming this perturbation."""
        return PerturbationSpec(name=self.name, severity=severity, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{type(self).__name__}(name={self.name!r}, S={self.severity_levels})"


class PerturbationRegistry:
    """Name to :class:`Perturbation`.

    Deliberately a pure registry, following repo D's note on ``PerturbationHandler``:

        "Selection of which method(s) to run is the caller's responsibility."

    It holds and filters; it does not decide. :meth:`applicable` is the query an
    robustness evaluator uses to find what can act on the data it has.
    """

    def __init__(self, perturbations: Sequence[Perturbation] | None = None) -> None:
        self._items: dict[str, Perturbation] = {}
        for perturbation in perturbations or ():
            self.register(perturbation)

    def register(self, perturbation: Perturbation, *, overwrite: bool = False) -> None:
        if not perturbation.name:
            raise ValueError(
                f"{type(perturbation).__name__} has no name and cannot be registered."
            )
        if not overwrite and perturbation.name in self._items:
            raise ValueError(
                f"Perturbation {perturbation.name!r} is already registered. "
                f"Pass overwrite=True to replace it."
            )
        self._items[perturbation.name] = perturbation

    def __getitem__(self, name: str) -> Perturbation:
        try:
            return self._items[name]
        except KeyError:
            near = [k for k in self._items if name.lower() in k.lower()]
            hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
            available = ", ".join(sorted(self._items)) or "none registered"
            raise KeyError(
                f"Unknown perturbation {name!r}.{hint} Available: {available}."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self) -> list[tuple[str, Perturbation]]:
        """``(name, perturbation)`` pairs, sorted by name."""
        return sorted(self._items.items())

    def perturbations(self) -> list[Perturbation]:
        """The registered perturbation objects, sorted by name."""
        return [p for _, p in self.items()]

    def applicable(
        self,
        data: DataAdapter,
        task: TaskType | None = None,
    ) -> list[str]:
        """Names of registered perturbations that can act on ``data`` for ``task``."""
        return sorted(
            name for name, p in self._items.items() if p.applies_to(data, task)
        )

    def metadata(self) -> dict[str, dict[str, Any]]:
        """Reporting metadata for every registered perturbation."""
        return {name: p.metadata() for name, p in self._items.items()}
