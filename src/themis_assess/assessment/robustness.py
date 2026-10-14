"""The robustness evaluator: the perturbation sweep, and the one place that calls a model.

Everything before this tranche was a piece — a metric, a perturbation, an adapter, a
handler. This is what puts them in the right order, and it is deliberately the only
component that knows the order, because that ordering is where all four repos went
wrong in the same way.

What it does, in sequence
-------------------------

1. **Plan.** Ask the handler what it can produce, ask the catalog what that supports, and
   report the metrics that will be skipped *with the reason*, before any model call. Also
   count the model calls the sweep will make, since the answer is often larger than the
   user expects and finding out afterwards is expensive.
2. **Evaluate clean.** One prediction, one set of metric values. This is the denominator
   for everything else, so it is computed once and reused rather than recomputed per
   perturbation — repo D recomputed it inside the loop.
3. **Sweep.** For each perturbation, for each severity, for each variant: perturb,
   predict, score. The random generator is derived per cell so that a cell's draw does
   not depend on how many cells ran before it.
4. **Baseline.** If given, the same sweep with the same seeds, which is what makes ``CE``
   and ``mCE`` a comparison rather than a coincidence.
5. **Subgroups.** Metrics per group, on the same predictions, via masks.

The one rule that is not obvious
--------------------------------

**The ground truth is taken from the clean data and never recomputed from perturbed
data.** In forecasting the target is a column of the input, so a perturbation that
touches it would move the truth along with the prediction, and a model could score
perfectly on data where it is entirely wrong. Survey §6.3's label-preservation assumption
is what makes an assessment meaningful; here it is enforced structurally, by capturing
``actual`` once, rather than trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.results import EvaluationResult, RobustnessResult
from themis_assess.core.seeding import derive_generator
from themis_assess.core.subgroups import SubgroupSpec, resolve_masks
from themis_assess.core.types import AccessLevel, PerturbationSpec, TaskType
from themis_assess.handlers.base import ModelHandler, Prediction
from themis_assess.metrics.accuracy.catalog import MetricCatalog, default_catalog
from themis_assess.metrics.registry import MetricRegistry, default_registry
from themis_assess.perturbations.base import Perturbation, PerturbationRegistry
from themis_assess.perturbations.registry import default_registry as default_perturbations

__all__ = [
    "AssessmentPlan",
    "RobustnessEvaluator",
    "RobustnessAssessment",
    "SweepResult",
]


@dataclass(frozen=True, slots=True)
class AssessmentPlan:
    """What a run will do, available before it does it.

    Parameters
    ----------
    metrics:
        What will be computed.
    skipped:
        ``{metric: reason}`` for everything requested that cannot be computed. This is
        the field the whole planning step exists for: none of the four repos could tell
        a user that their handler returns labels only and so nine of their twelve
        requested metrics would be missing, until the run had already happened.
    perturbations, severities:
        The sweep's extent.
    n_model_calls:
        Predictions the sweep will make, including the clean one and the baseline's.
    """

    metrics: tuple[str, ...]
    skipped: Mapping[str, str]
    perturbations: tuple[str, ...]
    severities: tuple[int, ...]
    n_model_calls: int
    task: TaskType | None = None
    access: AccessLevel = AccessLevel.BLACK_BOX

    def summary(self) -> str:
        """A short human-readable description, for logging before a long run."""
        lines = [
            f"{len(self.metrics)} metric(s): {', '.join(self.metrics)}",
            f"{len(self.perturbations)} perturbation(s) x "
            f"{len(self.severities)} severit(y/ies)",
            f"{self.n_model_calls} model call(s)",
        ]
        if self.skipped:
            lines.append("skipped:")
            lines.extend(f"  {name}: {why}" for name, why in sorted(self.skipped.items()))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Everything one assessment produced.

    Deliberately not collapsed to scores: the survey's reporting rules need the clean
    value, the whole severity curve and the baseline, so nothing here throws them away.
    Converting to a :class:`~themis_assess.core.results.RobustnessResult` for a
    particular metric is
    :meth:`~themis_assess.assessment.scoring.score` in the scoring module.
    """

    clean: EvaluationResult
    perturbed: Mapping[str, Mapping[int, EvaluationResult]]
    plan: AssessmentPlan
    baseline_clean: EvaluationResult | None = None
    baseline_perturbed: Mapping[str, Mapping[int, EvaluationResult]] | None = None
    subgroups: Mapping[str, EvaluationResult] = field(default_factory=dict)
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def perturbation_names(self) -> list[str]:
        return sorted(self.perturbed)

    def curve(self, perturbation: str, metric: str) -> dict[int, float]:
        """``{severity: value}`` of one metric under one perturbation."""
        try:
            per_severity = self.perturbed[perturbation]
        except KeyError:
            raise KeyError(
                f"No perturbation {perturbation!r} in this sweep. "
                f"Available: {self.perturbation_names}."
            ) from None
        return {s: result[metric] for s, result in sorted(per_severity.items())}

    def curves(self, metric: str) -> dict[str, dict[int, float]]:
        """``{perturbation: {severity: value}}`` for one metric — the sweep's shape."""
        return {name: self.curve(name, metric) for name in self.perturbation_names}

    def baseline_curves(self, metric: str) -> dict[str, dict[int, float]] | None:
        if self.baseline_perturbed is None:
            return None
        return {
            name: {s: r[metric] for s, r in sorted(per_severity.items())}
            for name, per_severity in self.baseline_perturbed.items()
        }

    def worst(self, metric: str, *, higher_is_better: bool) -> tuple[str, int, float]:
        """The (perturbation, severity, value) where the metric was worst.

        The single most useful thing to look at first, and the survey's §8 reporting
        standard asks for the worst case beside the average.
        """
        cells = [
            (name, severity, value)
            for name, curve in self.curves(metric).items()
            for severity, value in curve.items()
        ]
        if not cells:
            raise ValueError("This sweep has no perturbed cells.")
        return min(cells, key=lambda c: c[2]) if higher_is_better else max(
            cells, key=lambda c: c[2]
        )


@dataclass(frozen=True, slots=True)
class RobustnessAssessment:
    """A scored robustness assessment: the sweep, the §5.1.3 metrics, and the verdict.

    The robustness counterpart of
    :class:`~themis_assess.assessment.accuracy.AccuracyResult`. Both are what one call to
    an evaluator returns, and both carry a three-valued verdict whose ``None`` state means
    "nothing was checked" rather than "everything passed".

    The sweep stays attached rather than being summarised away, because the survey's
    reporting rules need it: ``mCE`` may not be stated without ``mRCE``, an aggregate not
    without the per-perturbation breakdown, one severity not without the curve. A result
    that kept only the scores would make those rules unenforceable.
    """

    sweep: SweepResult
    scored: Mapping[str, RobustnessResult]
    accuracy_metric: str
    tolerance: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, robustness_metric: str) -> RobustnessResult:
        try:
            return self.scored[robustness_metric]
        except KeyError:
            raise KeyError(
                f"{robustness_metric!r} was not scored. Available: "
                f"{sorted(self.scored)}."
            ) from None

    @property
    def passed(self) -> bool | None:
        """Whether every critical tolerance was met. ``None`` when none were supplied."""
        return None if self.tolerance is None else self.tolerance.passed

    @property
    def clean(self) -> float:
        """The clean value of the underlying metric — the denominator for everything."""
        return self.sweep.clean[self.accuracy_metric]

    @property
    def metric_names(self) -> list[str]:
        return sorted(self.scored)

    def curves(self) -> dict[str, dict[int, float]]:
        """``{perturbation: {severity: value}}`` for the underlying metric."""
        return self.sweep.curves(self.accuracy_metric)

    def worst_cell(self) -> tuple[str, int, float]:
        """The (perturbation, severity, value) at which the underlying metric was worst.

        The first thing to look at, and what survey §8 asks for beside the average.
        """
        from themis_assess.metrics.registry import default_registry

        spec = default_registry().get(self.accuracy_metric)
        return self.sweep.worst(
            self.accuracy_metric, higher_is_better=bool(spec.higher_is_better)
        )

    def most_damaging_perturbation(self, robustness_metric: str | None = None) -> str:
        """Which perturbation hurt most, by one of the scored metrics."""
        name = robustness_metric or next(iter(sorted(self.scored)))
        result = self[name]
        if not result.scores:
            raise ValueError(f"{name!r} has no per-perturbation scores.")
        from themis_assess.metrics.registry import default_registry

        spec = default_registry().get(name)
        higher_is_better = (
            spec.higher_is_better if spec.higher_is_better is not None else True
        )
        picker = min if higher_is_better else max
        return picker(result.scores, key=result.scores.get)

    def summary(self) -> str:
        """A readable digest — the thing to print or paste into a ticket."""
        model = self.metadata.get("model", "model")
        lines = [
            f"{model}: robustness over {self.accuracy_metric} "
            f"(clean {self.clean:.4f}, {self.sweep.clean.n_samples} samples)"
        ]
        if self.metadata.get("accuracy_metric_chosen_by") == "default":
            lines.append(
                f"  note: {self.accuracy_metric} was chosen by default; name one "
                f"explicitly for a comparable claim"
            )
        perturbation, severity, value = self.worst_cell()
        lines.append(f"  worst cell: {perturbation}@{severity} = {value:.4f}")
        for name in self.metric_names:
            result = self.scored[name]
            if result.aggregate is not None:
                lines.append(f"  {name}: {result.aggregate:.4f}")
        if self.tolerance is not None:
            outcome = {True: "PASS", False: "FAIL", None: "NO TOLERANCE"}[
                self.tolerance.passed
            ]
            lines.append(f"tolerance: {outcome}")
            for verdict in self.tolerance.failures:
                lines.append(f"  FAILED {verdict.describe()}")
            for name in self.tolerance.unchecked:
                lines.append(
                    f"  NOT CHECKED {name}: not scored over {self.accuracy_metric}, "
                    f"so the outcome above does not cover it"
                )
        else:
            lines.append(
                "tolerance: none supplied — an incomplete claim under survey §8"
            )
        return "\n".join(lines)


class RobustnessEvaluator:
    """Runs a model, a perturbation set and a metric set against one dataset.

    Parameters
    ----------
    handler:
        The model. Anything satisfying
        :class:`~themis_assess.handlers.base.ModelHandler`.
    metrics:
        Names to compute. Defaults to everything in the catalog that suits the handler's
        task *and* its access level — which is usually what is wanted, and is only
        knowable because both are declared.
    baseline:
        A second handler, for the metrics of survey §5.1.3 that are defined against one
        (docs/DECISIONS.md, D14). Evaluated under identical perturbations and seeds.
    perturbations:
        A registry, or a list of names. Defaults to every registered perturbation that
        applies to the data. A named perturbation that does not apply raises rather than
        being skipped: a silently skipped perturbation overstates robustness.
    perturbation_registry:
        Where names are looked up. Only needed for a perturbation of your own —
        ``default_perturbations()`` builds a fresh registry on each call, so registering
        into one instance is invisible to the next, and this is how a custom
        perturbation becomes selectable by name.
    severities:
        Which levels to sweep. Defaults to 1..5 (D16).
    n_variants:
        Independent draws per cell. Above 1 the metric is averaged over draws, which
        matters for a stochastic perturbation on a small dataset, where one draw is
        mostly noise.
    subgroups:
        Evaluated on the clean predictions, for the §5.1.2 disparity metrics.
    seed:
        Every draw in the run derives from this. Recorded in the result.
    """

    def __init__(
        self,
        handler: ModelHandler,
        *,
        metrics: Sequence[str] | None = None,
        baseline: ModelHandler | None = None,
        perturbations: PerturbationRegistry | Sequence[str] | None = None,
        perturbation_registry: PerturbationRegistry | None = None,
        severities: Sequence[int] | None = None,
        n_variants: int = 1,
        subgroups: Sequence[SubgroupSpec] = (),
        catalog: MetricCatalog | None = None,
        registry: MetricRegistry | None = None,
        seed: int = 0,
    ) -> None:
        self.handler = handler
        self.baseline = baseline
        self.catalog = catalog if catalog is not None else default_catalog()
        self.registry = registry if registry is not None else default_registry()
        self.requested_metrics = list(metrics) if metrics is not None else None
        self.severities = tuple(severities) if severities is not None else (1, 2, 3, 4, 5)
        if any(s < 1 for s in self.severities):
            raise ValueError(f"Severity levels are 1-based; got {self.severities}.")
        if n_variants < 1:
            raise ValueError(f"n_variants must be at least 1; got {n_variants}.")
        self.n_variants = n_variants
        self.subgroups = list(subgroups)
        self.seed = seed

        if baseline is not None and baseline.task is not handler.task:
            raise ValueError(
                f"The baseline is a {baseline.task.value} model but the model under "
                f"assessment is {handler.task.value}. A corruption error between two "
                f"task types is not a comparison."
            )

        # ``perturbations`` is either a registry (sweep everything in it that applies) or
        # a list of names. Names are looked up in ``perturbation_registry`` when one is
        # given, which is what makes a *custom* perturbation selectable by name:
        # ``default_perturbations()`` builds a fresh registry on every call, so
        # registering into one instance is invisible to the next.
        if perturbations is None or isinstance(perturbations, PerturbationRegistry):
            if perturbation_registry is not None and perturbations is not None:
                raise ValueError(
                    "Pass either a registry as `perturbations` or a name list plus "
                    "`perturbation_registry`, not two registries."
                )
            self._registry = (
                perturbations
                if perturbations is not None
                else (
                    perturbation_registry
                    if perturbation_registry is not None
                    else default_perturbations()
                )
            )
            self._names: list[str] | None = None
        else:
            self._registry = (
                perturbation_registry
                if perturbation_registry is not None
                else default_perturbations()
            )
            self._names = list(perturbations)

    # -- planning ---------------------------------------------------------------

    def _perturbations_for(self, data: DataAdapter) -> list[Perturbation]:
        if self._names is not None:
            chosen = [self._registry[name] for name in self._names]
            unusable = [
                p.name for p in chosen if not p.applies_to(data, self.handler.task)
            ]
            if unusable:
                raise ValueError(
                    f"Perturbation(s) {unusable} do not apply to this data "
                    f"(modality={data.modality.value}, task={self.handler.task.value}, "
                    f"numerical={data.numerical_features}, "
                    f"categorical={data.categorical_features}). Naming a perturbation "
                    f"explicitly and having it silently skipped would overstate "
                    f"robustness, so this raises."
                )
            return chosen
        return self._registry.applicable(data, self.handler.task)

    def _metric_names(self, prediction: Prediction) -> tuple[list[str], dict[str, str]]:
        """Which metrics to compute, and why the others were dropped."""
        available = self._bag(prediction, actual=np.zeros(len(prediction)))
        computable = set(self.catalog.computable(available, task=self.handler.task))

        if self.requested_metrics is None:
            requested = self.catalog.for_task(self.handler.task)
        else:
            requested = list(self.requested_metrics)

        keep, skipped = [], {}
        for name in requested:
            if name not in self.catalog:
                skipped[name] = "not in the metric catalog"
                continue
            entry = self.catalog.get(name)
            if self.handler.task not in entry.tasks:
                skipped[name] = (
                    f"defined for {sorted(t.value for t in entry.tasks)}, "
                    f"not {self.handler.task.value}"
                )
                continue
            if not self.handler.access.satisfies(entry.access):
                skipped[name] = (
                    f"needs {entry.access.name} access; {self.handler.name} declares "
                    f"{self.handler.access.name}"
                )
                continue
            if name not in computable:
                missing = ", ".join(entry.missing(available))
                skipped[name] = f"{self.handler.name} did not return {missing}"
                continue
            keep.append(name)

        if not keep:
            raise ValueError(
                f"No metric can be computed. {self.handler.name} declares "
                f"access={self.handler.access.name} for a {self.handler.task.value} "
                f"task and returned {sorted(prediction.available())}. Skipped: "
                + "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items()))
            )
        return keep, skipped

    def plan(self, data: DataAdapter, actual: Any = None) -> AssessmentPlan:
        """What the run will compute and what it will cost, without running it.

        Costs one model call, on the clean data — there is no way to know what a handler
        returns without asking it. :meth:`run` reuses that prediction rather than
        repeating it, which is why the two are wired through :meth:`_plan_from`.
        """
        del actual  # the plan does not score; it only asks what could be scored
        return self._plan_from(self.handler.predict(data), data)

    def _plan_from(self, prediction: Prediction, data: DataAdapter) -> AssessmentPlan:
        keep, skipped = self._metric_names(prediction)
        perturbations = self._perturbations_for(data)

        per_model = 1 + len(perturbations) * len(self.severities) * self.n_variants
        calls = per_model * (2 if self.baseline is not None else 1)
        return AssessmentPlan(
            metrics=tuple(keep),
            skipped=skipped,
            perturbations=tuple(p.name for p in perturbations),
            severities=tuple(self.severities),
            n_model_calls=calls,
            task=self.handler.task,
            access=self.handler.access,
        )

    # -- evaluation -------------------------------------------------------------

    def _bag(self, prediction: Prediction, actual: Any) -> dict[str, Any]:
        bag: dict[str, Any] = dict(prediction.available())
        bag["actual"] = actual
        return bag

    def evaluate(
        self,
        data: DataAdapter,
        actual: Any,
        *,
        metrics: Sequence[str] | None = None,
        label: str = "clean",
        handler: ModelHandler | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Predict once and score, for one model on one dataset."""
        model = handler if handler is not None else self.handler
        prediction = model.predict(data)
        return self.score(
            prediction, actual, metrics=metrics, label=label, metadata=metadata
        )

    def score(
        self,
        prediction: Prediction,
        actual: Any,
        *,
        metrics: Sequence[str] | None = None,
        label: str = "clean",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Score an existing prediction. Separated from :meth:`evaluate` so a subgroup
        or a split can be scored from predictions already made, rather than by
        re-predicting — which for a stochastic perturbation would draw different noise
        and quietly compare two different datasets."""
        names = list(metrics) if metrics is not None else self._metric_names(prediction)[0]
        bag = self._bag(prediction, actual)

        values: dict[str, float] = {}
        undefined: dict[str, str] = {}
        for name in names:
            entry = self.catalog.get(name)
            if entry.missing(bag):
                continue
            # A metric can be undefined on *particular* data while being perfectly
            # computable in general: AUC on a subgroup containing one class is the
            # standard case, and a single degenerate subgroup must not take down a
            # sweep of several thousand model calls. Recorded by name so the report can
            # distinguish "not computed" from "computed and fine".
            #
            # Both failure shapes are caught, because scikit-learn uses both: some
            # metrics raise on degenerate input and others warn and return NaN. A NaN
            # is the more dangerous of the two — it propagates silently through every
            # mean and ratio downstream — so it is treated as undefined, not as a value.
            try:
                value = entry.compute(bag)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                undefined[name] = f"{type(exc).__name__}: {exc}"
                continue
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                undefined[name] = "undefined on this data (returned NaN or infinity)"
                continue
            values[name] = value

        if not values:
            raise ValueError(
                f"No metric could be computed for {label!r}. Undefined: "
                + "; ".join(f"{k} ({v})" for k, v in sorted(undefined.items()))
            )
        return EvaluationResult(
            metrics=values,
            n_samples=len(prediction),
            label=label,
            task=self.handler.task,
            metadata={**dict(metadata or {}), **({"undefined": undefined} if undefined else {})},
        )

    # -- the sweep ---------------------------------------------------------------

    def _cell_rng(self, perturbation: str, severity: int, variant: int) -> np.random.Generator:
        """A generator whose stream depends only on the cell, not on execution order.

        Repo B seeded numpy globally once per run, so re-running a single severity gave
        different noise than that severity had inside a full sweep, and results could
        not be reproduced piecewise. Deriving from the cell's identity means cell
        (gaussian_noise, 3, 0) draws the same numbers however it is reached — and, because
        the derivation is a stable digest rather than ``hash()``, on the next run as well.
        See :mod:`themis_assess.core.seeding` for why that distinction cost a defect.
        """
        return derive_generator(self.seed, perturbation, severity, variant)

    def _predict_cell(
        self,
        model: ModelHandler,
        data: DataAdapter,
        perturbation: str,
        severity: int,
        variant: int,
    ) -> Prediction:
        """Predict for one sweep cell, naming the cell if the model refuses the input.

        A model that cannot consume what a perturbation produces — the usual case is
        ``missing_values``, since most scikit-learn estimators reject NaN — otherwise
        fails somewhere inside the model, dozens of cells into a run, with no indication
        of which cell or which perturbation caused it.

        This is not caught and skipped. Dropping the cell would leave a severity curve
        with a hole in it and a robustness score computed over the perturbations the
        model happened to tolerate, which reads as robustness and is not.
        """
        try:
            return model.predict(data)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                f"{model.name} failed on {perturbation}@{severity} (variant {variant}): "
                f"{type(exc).__name__}: {exc}\n"
                f"The model must accept whatever this perturbation produces. For "
                f"'missing_values' that means NaN — wrap the model in a pipeline with "
                f"an imputer, or drop that perturbation from the set. Excluding it "
                f"silently is not an option here: a robustness score over only the "
                f"perturbations a model tolerates overstates its robustness."
            ) from exc

    def _sweep_one(
        self,
        model: ModelHandler,
        data: DataAdapter,
        actual: Any,
        perturbations: Sequence[Perturbation],
        metrics: Sequence[str],
        *,
        specs: Mapping[str, PerturbationSpec] | None = None,
    ) -> dict[str, dict[int, EvaluationResult]]:
        out: dict[str, dict[int, EvaluationResult]] = {}
        for perturbation in perturbations:
            per_severity: dict[int, EvaluationResult] = {}
            for severity in self.severities:
                if severity > perturbation.severity_levels:
                    continue
                base_spec = (
                    specs.get(perturbation.name) if specs else None
                ) or perturbation.spec()
                spec = base_spec.at_severity(severity)
                accumulated: list[Mapping[str, float]] = []
                n_samples = 0
                for variant in range(self.n_variants):
                    rng = self._cell_rng(perturbation.name, severity, variant)
                    perturbed = perturbation.perturb(data, spec, rng)[0]
                    prediction = self._predict_cell(
                        model, perturbed, perturbation.name, severity, variant
                    )
                    result = self.score(
                        prediction,
                        actual,
                        metrics=metrics,
                        label=f"{perturbation.name}@{severity}",
                    )
                    accumulated.append(result.metrics)
                    n_samples = result.n_samples
                averaged = {
                    name: float(np.mean([m[name] for m in accumulated if name in m]))
                    for name in metrics
                    if any(name in m for m in accumulated)
                }
                per_severity[severity] = EvaluationResult(
                    metrics=averaged,
                    n_samples=n_samples,
                    label=f"{perturbation.name}@{severity}",
                    task=model.task,
                    metadata={
                        **perturbation.metadata(spec),
                        "n_variants": self.n_variants,
                        "model": model.name,
                    },
                )
            if per_severity:
                out[perturbation.name] = per_severity
        return out

    def run(
        self,
        data: DataAdapter,
        actual: Any,
        *,
        specs: Mapping[str, PerturbationSpec] | None = None,
        subgroup_dataset: Any = None,
    ) -> SweepResult:
        """The whole assessment: clean, sweep, baseline, subgroups.

        Parameters
        ----------
        actual:
            Ground truth, taken **from the clean data and held fixed**. For forecasting
            this is the target slice — :meth:`ForecastingHandler.actual` produces it —
            and it must not be recomputed from perturbed data: a perturbation touching
            the target column would move the truth along with the prediction and score a
            broken model as perfect.
        specs:
            Per-perturbation specification overrides, for anything needing a parameter
            the default cannot guess — ``stl_component`` needs a ``period``.
        subgroup_dataset:
            What the subgroup masks are evaluated against. Defaults to ``data``.
        """
        clean_prediction = self.handler.predict(data)
        plan = self._plan_from(clean_prediction, data)
        metrics = list(plan.metrics)
        perturbations = self._perturbations_for(data)

        clean = self.score(clean_prediction, actual, metrics=metrics, label="clean")

        perturbed = self._sweep_one(
            self.handler, data, actual, perturbations, metrics, specs=specs
        )

        baseline_clean = baseline_perturbed = None
        if self.baseline is not None:
            baseline_prediction = self.baseline.predict(data)
            baseline_metrics = [
                name
                for name in metrics
                if self.baseline.access.satisfies(self.catalog.get(name).access)
            ]
            baseline_clean = self.score(
                baseline_prediction,
                actual,
                metrics=baseline_metrics,
                label="clean",
                metadata={"model": self.baseline.name},
            )
            baseline_perturbed = self._sweep_one(
                self.baseline, data, actual, perturbations, baseline_metrics, specs=specs
            )

        subgroups: dict[str, EvaluationResult] = {}
        if self.subgroups:
            dataset = subgroup_dataset if subgroup_dataset is not None else data
            masks = resolve_masks(
                self.subgroups, dataset, n_samples=len(clean_prediction)
            )
            actual_array = np.asarray(actual)
            for name, mask in masks.items():
                subgroups[name] = self.score(
                    clean_prediction.take(mask),
                    actual_array[mask],
                    metrics=metrics,
                    label=f"group={name}",
                    metadata={"group": name, "n_selected": int(mask.sum())},
                )

        return SweepResult(
            clean=clean,
            perturbed=perturbed,
            plan=plan,
            baseline_clean=baseline_clean,
            baseline_perturbed=baseline_perturbed,
            subgroups=subgroups,
            seed=self.seed,
            metadata={
                "model": self.handler.name,
                "baseline": None if self.baseline is None else self.baseline.name,
                "n_variants": self.n_variants,
            },
        )

    # -- the assessment ----------------------------------------------------------

    def assess(
        self,
        data: DataAdapter,
        actual: Any,
        *,
        accuracy_metric: str | None = None,
        robustness_metrics: Sequence[str] | None = None,
        criteria: Any = None,
        specs: Mapping[str, PerturbationSpec] | None = None,
        subgroup_dataset: Any = None,
    ) -> RobustnessAssessment:
        """Sweep, score, and check against the deployment's tolerances, in one call.

        The counterpart of
        :meth:`~themis_assess.assessment.accuracy.AccuracyEvaluator.run`, and the reason
        it exists: without it the two halves of the library were not symmetric. An
        accuracy assessment returned a rich object carrying a verdict, while a robustness
        assessment returned raw severity curves that the caller had to feed back into
        :func:`~themis_assess.assessment.scoring.score` themselves — so the cheap half of
        the library was the more finished one.

        :meth:`run` is still here and still returns the unscored
        :class:`SweepResult`. Nothing is duplicated: this method calls it and scores what
        comes back. Use :meth:`run` when the sweep is wanted for something the scoring
        layer does not do — a sensitivity map, an attribution study — and this when the
        question is "how robust is this model".

        Parameters
        ----------
        accuracy_metric:
            Which underlying metric to score robustness over. Defaults to the first
            computable metric whose orientation suits the requested robustness metrics,
            preferring an error metric when the set needs one — but **naming it
            explicitly is better**, since two robustness scores over different underlying
            metrics are not comparable and the default is a convenience rather than a
            choice anyone made.
        robustness_metrics:
            Which §5.1.3 metrics to compute. Defaults to every one the sweep supports.
        criteria:
            A :class:`~themis_assess.assessment.tolerance.RobustnessCriteria`. Optional,
            and its absence is visible: :attr:`RobustnessAssessment.passed` is ``None``,
            never ``True``, because a robustness claim without a tolerance is incomplete
            under survey §8.
        """
        from themis_assess.assessment import scoring

        sweep = self.run(
            data, actual, specs=specs, subgroup_dataset=subgroup_dataset
        )

        chosen = (
            accuracy_metric
            if accuracy_metric is not None
            else self._default_accuracy_metric(sweep, robustness_metrics)
        )

        if robustness_metrics is None:
            scored = scoring.score_all(sweep, chosen, registry=self.registry)
        else:
            scored = {
                name: scoring.score(sweep, name, chosen, registry=self.registry)
                for name in robustness_metrics
            }

        report = None if criteria is None else criteria.check(scored)
        return RobustnessAssessment(
            sweep=sweep,
            scored=scored,
            accuracy_metric=chosen,
            tolerance=report,
            metadata={
                **dict(sweep.metadata),
                "accuracy_metric": chosen,
                "accuracy_metric_chosen_by": (
                    "caller" if accuracy_metric is not None else "default"
                ),
            },
        )

    def _default_accuracy_metric(
        self, sweep: SweepResult, robustness_metrics: Sequence[str] | None
    ) -> str:
        """Pick an underlying metric when the caller did not name one.

        Prefers an error metric, because the majority of the §5.1.3 family — the
        corruption-error and inflation equations — is written over an error and refuses an
        accuracy. Deterministic, and recorded in the result's metadata as a default rather
        than a choice.
        """
        available = sweep.clean.metric_names
        preferred_errors = ("negative_log_likelihood", "brier_score", "mae", "rmse", "mse")
        preferred_accuracies = ("accuracy", "balanced_accuracy", "f1", "r2")

        wants_error = robustness_metrics is None or any(
            name in scoring_error_metrics() for name in robustness_metrics
        )
        order = (
            preferred_errors + preferred_accuracies
            if wants_error
            else preferred_accuracies + preferred_errors
        )
        for name in order:
            if name in available:
                return name
        raise ValueError(
            f"No suitable underlying metric was computed to score robustness over. "
            f"Available: {available}. Name one explicitly with accuracy_metric=."
        )


def scoring_error_metrics() -> frozenset[str]:
    """The §5.1.3 equations written over an error. Indirection avoids a circular import."""
    from themis_assess.assessment.scoring import _ERROR_METRICS

    return _ERROR_METRICS
