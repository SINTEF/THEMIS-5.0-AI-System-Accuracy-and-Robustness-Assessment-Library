"""The accuracy assessment: one model, one dataset, everything worth knowing about it.

This is the counterpart to :mod:`themis_assess.assessment.robustness`, and it exists because
"assess this model's accuracy properly" is a first-class task and not a degenerate
robustness sweep with zero perturbations. It is also, judging by the four repos, the task
people actually start with: they carry 3,423 lines of accuracy assessment between them
against roughly a third of that for robustness.

What it produces, in one call: the metrics with a verdict against the deployment's own
acceptance criteria; the confusion matrix and the per-class breakdown; the subgroup
breakdown and its disparity metrics; a baseline comparison; and the diagnostics that say
*where* the errors are — on unusual inputs, in particular feature regions, among the
predictions the model was least sure of.

The division of labour with the robustness evaluator
---------------------------------------------------

``AccuracyEvaluator`` never perturbs anything. It calls the model once (twice with a
baseline), and everything else is arithmetic on those predictions. That makes it cheap
enough to run on every commit, which is the point — a robustness sweep is a periodic
exercise, and an accuracy check should not be.

Where the two overlap they share code rather than reimplementing it:
:class:`~themis_assess.assessment.robustness.RobustnessEvaluator` handles the metric selection
and the undefined-metric bookkeeping, and this class delegates to it for both. So a metric
skipped here and a metric skipped there are skipped for the same reason, reported the same
way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.assessment.criteria import AcceptanceCriteria, AcceptanceReport
from themis_assess.assessment.robustness import RobustnessEvaluator
from themis_assess.core.adapters import DataAdapter
from themis_assess.core.results import EvaluationResult
from themis_assess.core.subgroups import SubgroupSpec, resolve_masks
from themis_assess.core.types import TaskType
from themis_assess.handlers.base import ModelHandler, Prediction
from themis_assess.metrics.accuracy.classification import (
    ConfusionMatrix,
    confusion_matrix,
    per_class_metrics,
    top_confusions,
)
from themis_assess.metrics.accuracy.diagnostics import (
    OutlierErrorAnalysis,
    error_rate_by_feature_bin,
    error_rate_by_outlier_status,
    flag_ambiguous,
)
from themis_assess.metrics.robustness import subgroup as subgroup_metrics

__all__ = ["AccuracyResult", "AccuracyEvaluator"]


@dataclass(frozen=True, slots=True)
class AccuracyResult:
    """Everything one accuracy assessment produced.

    Parameters
    ----------
    overall:
        The metric values on the whole dataset.
    acceptance:
        The verdict against the deployment's criteria, when any were supplied. ``None``
        means none were — which the report layer must not render as a pass.
    per_class, confusion:
        Classification only.
    subgroups, disparity:
        Per-group metric values, and the §5.1.2 metrics computed across them (worst-group
        accuracy, accuracy gap, variance). The disparity metrics are computed here rather
        than left to the caller because the survey's rule is that an aggregate must be
        accompanied by them, and a result object that made that optional would make the
        rule unenforceable.
    baseline:
        The same metrics for a baseline model, where one was supplied.
    diagnostics:
        Named diagnostic outputs — outlier analysis, feature-bin error rates, the
        ambiguity flag. Populated only for what was requested.
    """

    overall: EvaluationResult
    acceptance: AcceptanceReport | None = None
    per_class: Mapping[Any, Mapping[str, float]] = field(default_factory=dict)
    confusion: ConfusionMatrix | None = None
    subgroups: Mapping[str, EvaluationResult] = field(default_factory=dict)
    disparity: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    baseline: EvaluationResult | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> float:
        return self.overall[metric]

    @property
    def passed(self) -> bool | None:
        """Whether the acceptance criteria were met. ``None`` when none were supplied."""
        return None if self.acceptance is None else self.acceptance.passed

    def improvement_over_baseline(self, metric: str) -> float | None:
        """How much better than the baseline, signed so positive is always better.

        The sign correction is the reason this is a method rather than a subtraction the
        caller writes: for an error metric the improvement is the baseline *minus* the
        model, and getting that backwards produces a plausible number with the wrong sign.
        """
        if self.baseline is None or metric not in self.baseline.metrics:
            return None
        registry_spec = self.metadata.get("registry_orientation", {}).get(metric)
        model_value, baseline_value = self.overall[metric], self.baseline[metric]
        if registry_spec is False:  # lower is better
            return float(baseline_value - model_value)
        return float(model_value - baseline_value)

    def worst_subgroup(self, metric: str, *, higher_is_better: bool = True) -> str | None:
        """Which subgroup the model does worst on. ``None`` if no subgroups were given."""
        candidates = {
            name: result.metrics[metric]
            for name, result in self.subgroups.items()
            if metric in result.metrics
        }
        if not candidates:
            return None
        return (min if higher_is_better else max)(candidates, key=candidates.get)

    def summary(self) -> str:
        """A readable digest — the thing to print or paste into a ticket."""
        lines = [f"{self.metadata.get('model', 'model')} on {self.overall.n_samples} samples"]
        for name in self.overall.metric_names:
            lines.append(f"  {name}: {self.overall[name]:.4f}")
        if self.acceptance is not None:
            outcome = {True: "PASS", False: "FAIL", None: "NO CRITERIA"}[
                self.acceptance.passed
            ]
            lines.append(f"acceptance: {outcome}")
            for verdict in self.acceptance.failures:
                lines.append(f"  FAILED {verdict.describe()}")
        if self.disparity:
            for metric, values in self.disparity.items():
                gap = values.get("accuracy_gap")
                if gap is not None:
                    lines.append(f"subgroup gap ({metric}): {gap:.4f}")
        undefined = self.overall.metadata.get("undefined")
        if undefined:
            lines.append(f"undefined: {', '.join(sorted(undefined))}")
        return "\n".join(lines)


class AccuracyEvaluator:
    """Assess one model's accuracy on one dataset, with no perturbation involved.

    Parameters
    ----------
    handler:
        The model.
    metrics:
        Names to compute. Defaults to everything in the catalog suiting the handler's task
        and access level.
    criteria:
        The deployment's acceptance thresholds. Optional, and its absence is visible in
        the result rather than defaulted away —
        :attr:`AccuracyResult.passed` is ``None``, not ``True``.
    baseline:
        A second handler to compare against. Useful independently of the §5.1.3 robustness
        metrics: "better than predicting the majority class" is the first question about
        any classifier, and three of the four repos had a ``metrics_baseline`` column for
        exactly this.
    subgroups:
        Groups to break the metrics down by, as masks.
    per_class:
        Compute the confusion matrix and per-class breakdown. Classification only;
        ignored for forecasting.
    """

    def __init__(
        self,
        handler: ModelHandler,
        *,
        metrics: Sequence[str] | None = None,
        criteria: AcceptanceCriteria | None = None,
        baseline: ModelHandler | None = None,
        subgroups: Sequence[SubgroupSpec] = (),
        per_class: bool = True,
        **runner_kwargs: Any,
    ) -> None:
        self.handler = handler
        self.criteria = criteria
        self.baseline = baseline
        self.subgroups = list(subgroups)
        self.per_class = per_class
        # Delegate metric selection, scoring and the undefined-metric bookkeeping, so a
        # metric skipped here is skipped for the same reason and reported the same way as
        # in a sweep. Perturbations are irrelevant here; the robustness evaluator only
        # consults them when a sweep is actually requested.
        self._engine = RobustnessEvaluator(
            handler, metrics=metrics, baseline=baseline, **runner_kwargs
        )

        if criteria is not None and baseline is not None and handler.task is not baseline.task:
            raise ValueError(
                f"The baseline is a {baseline.task.value} model but the model under "
                f"assessment is {handler.task.value}."
            )

    # -- planning ----------------------------------------------------------------

    def plan(self, data: DataAdapter) -> dict[str, Any]:
        """What will be computed and what will be skipped, before computing it.

        Cheaper than the sweep's plan — one model call and no perturbation arithmetic —
        but the same discipline: a metric that cannot be computed is named, with a reason,
        before the run rather than missing from the result afterwards.
        """
        prediction = self.handler.predict(data)
        keep, skipped = self._engine._metric_names(prediction)
        checked = [] if self.criteria is None else self.criteria.metrics()
        return {
            "metrics": tuple(keep),
            "skipped": skipped,
            "criteria": tuple(checked),
            "criteria_without_a_metric": tuple(sorted(set(checked) - set(keep))),
            "n_model_calls": 1 if self.baseline is None else 2,
            "task": self.handler.task,
            "access": self.handler.access,
        }

    # -- running -----------------------------------------------------------------

    def run(
        self,
        data: DataAdapter,
        actual: ArrayLike,
        *,
        subgroup_dataset: Any = None,
        diagnose_outliers: bool = False,
        diagnose_features: Sequence[str] = (),
        diagnose_ambiguity: bool = False,
        outlier_kwargs: Mapping[str, Any] | None = None,
    ) -> AccuracyResult:
        """Run the assessment.

        Parameters
        ----------
        diagnose_outliers:
            Whether errors concentrate on unusual inputs. Off by default because it fits
            three detectors and that cost should be opted into, not discovered.
        diagnose_features:
            Feature names to compute binned error rates for.
        diagnose_ambiguity:
            Flag the least confident predictions and report whether the flagged set really
            is harder. Needs probabilities.
        """
        prediction = self.handler.predict(data)
        metric_names, _ = self._engine._metric_names(prediction)
        truth = np.asarray(actual)

        overall = self._engine.score(
            prediction, truth, metrics=metric_names, label="clean"
        )

        acceptance = None
        if self.criteria is not None:
            acceptance = self.criteria.check(overall)

        baseline_result = None
        if self.baseline is not None:
            baseline_prediction = self.baseline.predict(data)
            baseline_metrics = [
                name
                for name in metric_names
                if self.baseline.access.satisfies(self._engine.catalog.get(name).access)
            ]
            baseline_result = self._engine.score(
                baseline_prediction,
                truth,
                metrics=baseline_metrics,
                label="baseline",
                metadata={"model": self.baseline.name},
            )

        per_class: dict[Any, Mapping[str, float]] = {}
        matrix: ConfusionMatrix | None = None
        if self.per_class and self.handler.task is TaskType.CLASSIFICATION:
            matrix = confusion_matrix(truth, prediction.predicted)
            per_class = per_class_metrics(truth, prediction.predicted)

        subgroups, disparity = self._assess_subgroups(
            prediction, truth, data, subgroup_dataset, metric_names
        )

        diagnostics = self._diagnose(
            prediction,
            truth,
            data,
            diagnose_outliers=diagnose_outliers,
            diagnose_features=diagnose_features,
            diagnose_ambiguity=diagnose_ambiguity,
            outlier_kwargs=dict(outlier_kwargs or {}),
        )
        if matrix is not None and matrix.n_classes > 2:
            diagnostics["top_confusions"] = top_confusions(matrix, n=5)

        orientation = {
            name: self._engine.registry.get(name).higher_is_better
            for name in metric_names
        }
        return AccuracyResult(
            overall=overall,
            acceptance=acceptance,
            per_class=per_class,
            confusion=matrix,
            subgroups=subgroups,
            disparity=disparity,
            baseline=baseline_result,
            diagnostics=diagnostics,
            metadata={
                "model": self.handler.name,
                "task": self.handler.task.value,
                "access": self.handler.access.name,
                "baseline": None if self.baseline is None else self.baseline.name,
                "registry_orientation": orientation,
            },
        )

    # -- pieces ------------------------------------------------------------------

    def _assess_subgroups(
        self,
        prediction: Prediction,
        truth: np.ndarray,
        data: DataAdapter,
        subgroup_dataset: Any,
        metric_names: Sequence[str],
    ) -> tuple[dict[str, EvaluationResult], dict[str, dict[str, float]]]:
        if not self.subgroups:
            return {}, {}
        dataset = subgroup_dataset if subgroup_dataset is not None else data
        masks = resolve_masks(self.subgroups, dataset, n_samples=len(prediction))

        results: dict[str, EvaluationResult] = {}
        for name, mask in masks.items():
            results[name] = self._engine.score(
                prediction.take(mask),
                truth[mask],
                metrics=metric_names,
                label=f"group={name}",
                metadata={"group": name, "n_selected": int(mask.sum())},
            )

        # The §5.1.2 disparity metrics, per underlying metric. Computed for every metric
        # that every group could produce: a gap over a metric one group is missing would
        # be a comparison against nothing.
        disparity: dict[str, dict[str, float]] = {}
        for metric in metric_names:
            values = {
                name: result.metrics[metric]
                for name, result in results.items()
                if metric in result.metrics
            }
            if len(values) < 2:
                continue
            higher_is_better = self._engine.registry.get(metric).higher_is_better
            entry: dict[str, float] = {
                "accuracy_gap": subgroup_metrics.accuracy_gap(values),
                "accuracy_variance": subgroup_metrics.accuracy_variance(values),
            }
            if higher_is_better:
                entry["worst_group_accuracy"] = subgroup_metrics.worst_group_accuracy(
                    values
                )
            else:
                entry["group_disparity"] = subgroup_metrics.group_disparity(values)
                entry["equal_weight_risk"] = subgroup_metrics.equal_weight_risk(values)
            disparity[metric] = entry
        return results, disparity

    def _diagnose(
        self,
        prediction: Prediction,
        truth: np.ndarray,
        data: DataAdapter,
        *,
        diagnose_outliers: bool,
        diagnose_features: Sequence[str],
        diagnose_ambiguity: bool,
        outlier_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.handler.task is not TaskType.CLASSIFICATION and (
            diagnose_outliers or diagnose_features or diagnose_ambiguity
        ):
            raise ValueError(
                "The diagnostics compare predicted labels against true labels, so they "
                "are classification-only. For a forecasting model use "
                "themis_assess.metrics.accuracy.temporal, which answers the same "
                "question along the time axis."
            )

        flat = data.values().reshape(data.n_entities, -1)

        if diagnose_outliers:
            analysis: OutlierErrorAnalysis = error_rate_by_outlier_status(
                flat, truth, prediction.predicted, **outlier_kwargs
            )
            out["outliers"] = analysis

        if diagnose_features:
            unknown = [f for f in diagnose_features if f not in set(data.feature_names)]
            if unknown:
                raise ValueError(
                    f"diagnose_features names feature(s) not in the data: {unknown}. "
                    f"Available: {data.feature_names}."
                )
            bins = {}
            for name in diagnose_features:
                column = data.values()[:, :, data.feature_index(name)]
                bins[name] = error_rate_by_feature_bin(
                    column.mean(axis=1) if column.shape[1] > 1 else column.ravel(),
                    truth,
                    prediction.predicted,
                )
            out["feature_bins"] = bins

        if diagnose_ambiguity:
            if prediction.probabilities is None:
                raise ValueError(
                    f"Ambiguity needs probabilities, and {self.handler.name} returned "
                    f"only {sorted(prediction.available())}. Use a handler declaring "
                    f"PROBABILISTIC access."
                )
            flags = flag_ambiguous(prediction.probabilities, empirical=True)
            is_error = np.asarray(prediction.predicted) != truth
            flagged = flags["is_ambiguous"]
            # The number that says whether the review queue is worth staffing: if the
            # flagged and unflagged error rates match, the model's uncertainty carries no
            # information and the flag is sorting at random.
            flags["error_rate_flagged"] = (
                float(is_error[flagged].mean()) if flagged.any() else None
            )
            flags["error_rate_unflagged"] = (
                float(is_error[~flagged].mean()) if (~flagged).any() else None
            )
            out["ambiguity"] = flags
        return out
