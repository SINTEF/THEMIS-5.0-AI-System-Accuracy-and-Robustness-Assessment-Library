"""Comparing several models on the same footing.

The question this answers is the one a project actually asks — model x, model y or model
z? — and it is harder than it looks, because the obvious implementation is wrong in three
ways that all produce a confident and meaningless answer.

**Metrics have different directions and different ranges.** Putting accuracy (0 to 1,
higher better) beside a mean absolute error (0 upwards, lower better) on the same axis
makes the worse model look better. So every value is normalised through
:func:`~themis_assess.metrics.normalize.normalize_metric` first, which uses the registry
to orient and bound each metric: after it, every axis runs 0 to 1 and 1 is always better.
The raw values are kept beside the normalised ones, because a normalised score is not a
number anyone can act on.

**A metric one model has and another does not is not a comparison.** A black-box model
has no calibration error; averaging over "the metrics each model happens to report"
silently rewards the model that reported fewer of the hard ones. Only metrics every model
in the set carries are compared, and the ones dropped are recorded with the models that
were missing them.

**Two robustness scores over different underlying metrics are not comparable.** A
``relative_robustness`` computed over accuracy and one computed over mean absolute error
are different quantities under one name, and this is the mistake the four repos were most
exposed to, since the underlying metric was chosen by a default nobody looked at. A
comparison refuses to put them on the same axis and says why.

What comes out is a :class:`ModelComparison`: the raw table, the normalised table, a
ranking, and the disagreements between metrics — the cases where ranking by one metric
contradicts ranking by another, which is the most useful output here and the one a single
composite score destroys.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from themis_assess.metrics.glossary import glossary_for
from themis_assess.metrics.normalize import normalize_metric
from themis_assess.metrics.registry import MetricRegistry, default_registry

__all__ = ["ModelComparison", "MetricDisagreement", "compare_models"]


@dataclass(frozen=True, slots=True)
class MetricDisagreement:
    """Two metrics that rank the models differently.

    The output worth reading. A set of models where every metric agrees needs no
    comparison layer — the ranking is obvious. Where they disagree there is a trade-off,
    and a trade-off is a decision for the deployment rather than for a scoring rule.
    """

    left_metric: str
    right_metric: str
    left_best: str
    right_best: str

    def describe(self) -> str:
        return (
            f"{self.left_metric} prefers {self.left_best}, "
            f"{self.right_metric} prefers {self.right_best}"
        )


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Several models on comparable axes, with what could not be compared recorded.

    Parameters
    ----------
    models:
        Model names, in the order given. That order fixes the colours in every figure, so
        adding a model to the end of the list never repaints the others.
    metrics:
        The metrics every model carries, sorted. These are the comparison's axes.
    raw:
        ``{model: {metric: value}}`` as measured.
    normalised:
        ``{model: {metric: value}}`` on [0, 1] with 1 best, which is what a polar chart
        or any cross-metric average may use.
    dropped:
        ``{metric: reason}`` for every metric that could not be compared. Never empty
        silently: a metric missing from one model appears here rather than vanishing.
    """

    models: tuple[str, ...]
    metrics: tuple[str, ...]
    raw: Mapping[str, Mapping[str, float]]
    normalised: Mapping[str, Mapping[str, float]]
    dropped: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.models) < 2:
            raise ValueError(
                f"A comparison needs at least two models; got {list(self.models)}. "
                f"For one model, the assessment report is the output."
            )
        if not self.metrics:
            raise ValueError(
                "No metric is present for every model, so there is nothing to compare "
                "on. The metrics each model carries individually are in metadata["
                "'per_model_metrics']; dropped says why each was excluded."
            )

    # -- reading ------------------------------------------------------------

    def value(self, model: str, metric: str) -> float:
        return self.raw[model][metric]

    def best(self, metric: str) -> str:
        """The model that scores best on ``metric``, by the metric's own direction."""
        if metric not in self.metrics:
            raise KeyError(
                f"{metric!r} is not a comparison axis. "
                f"Available: {list(self.metrics)}. Dropped: {dict(self.dropped)}."
            )
        return max(self.models, key=lambda m: self.normalised[m][metric])

    def mean_normalised(self) -> dict[str, float]:
        """The mean normalised score per model — a ranking device, not a verdict.

        This is a composite, and survey §8 asks that a composite never appear without its
        components. So it is a method rather than a field, :meth:`ranking` returns the
        components beside it, and nothing in this package reports it alone. Equal weights
        are used because no weighting the library could invent would be better justified
        than the one nobody chose; a deployment with a view should score its own.
        """
        return {
            model: float(np.mean([self.normalised[model][m] for m in self.metrics]))
            for model in self.models
        }

    def ranking(self) -> list[dict[str, Any]]:
        """Models best first, each with the components behind its score."""
        means = self.mean_normalised()
        rows = [
            {
                "model": model,
                "mean_normalised": means[model],
                "wins": sum(1 for m in self.metrics if self.best(m) == model),
                "per_metric_normalised": dict(self.normalised[model]),
                "per_metric_raw": dict(self.raw[model]),
            }
            for model in self.models
        ]
        return sorted(rows, key=lambda row: -row["mean_normalised"])

    def disagreements(self) -> list[MetricDisagreement]:
        """Every pair of metrics that prefers a different model."""
        out: list[MetricDisagreement] = []
        for i, left in enumerate(self.metrics):
            for right in self.metrics[i + 1 :]:
                left_best, right_best = self.best(left), self.best(right)
                if left_best != right_best:
                    out.append(
                        MetricDisagreement(left, right, left_best, right_best)
                    )
        return out

    @property
    def is_unanimous(self) -> bool:
        """Whether every compared metric prefers the same model."""
        return not self.disagreements()

    def to_frame(self, *, normalised: bool = False) -> Any:
        """The comparison table as a pandas DataFrame, models as rows."""
        import pandas as pd

        table = self.normalised if normalised else self.raw
        return pd.DataFrame(
            [[table[m][metric] for metric in self.metrics] for m in self.models],
            index=list(self.models),
            columns=list(self.metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        """The comparison as a plain dictionary, for the same handover as a report."""
        from themis_assess.report.bundle import to_plain

        return {
            "schema": "themis-assess/comparison@1",
            "models": list(self.models),
            "metrics": list(self.metrics),
            "raw": to_plain(dict(self.raw)),
            "normalised": to_plain(dict(self.normalised)),
            "dropped": dict(self.dropped),
            "ranking": to_plain(self.ranking()),
            "disagreements": [
                {
                    "left_metric": d.left_metric,
                    "right_metric": d.right_metric,
                    "left_best": d.left_best,
                    "right_best": d.right_best,
                }
                for d in self.disagreements()
            ],
            "is_unanimous": self.is_unanimous,
            "glossary": glossary_for(self.metrics),
            "metadata": to_plain(dict(self.metadata)),
        }

    def summary(self) -> str:
        lines = [f"comparing {len(self.models)} models on {len(self.metrics)} metric(s)"]
        for row in self.ranking():
            lines.append(
                f"  {row['model']:<24} {row['mean_normalised']:.4f} normalised "
                f"({row['wins']}/{len(self.metrics)} metric wins)"
            )
        if self.dropped:
            lines.append("  not compared:")
            for metric, reason in sorted(self.dropped.items()):
                lines.append(f"    {metric}: {reason}")
        disagreements = self.disagreements()
        if disagreements:
            lines.append("  the metrics disagree:")
            for d in disagreements[:5]:
                lines.append(f"    {d.describe()}")
            if len(disagreements) > 5:
                lines.append(f"    ... and {len(disagreements) - 5} more pairs")
        else:
            lines.append("  every compared metric prefers the same model")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def compare_models(
    reports: Mapping[str, Any] | Iterable[Any],
    *,
    metrics: Sequence[str] | None = None,
    include_robustness: bool = True,
    registry: MetricRegistry | None = None,
) -> ModelComparison:
    """Compare assessment reports — or plain metric dictionaries — on shared axes.

    Parameters
    ----------
    reports:
        Either ``{name: report}``, or a sequence of
        :class:`~themis_assess.report.bundle.AssessmentReport` objects (named by their
        own ``model_name``). A "report" may also be a plain ``{metric: value}`` mapping,
        which is how a comparison is built from numbers that came from somewhere else.
    metrics:
        Restrict the axes to these. A named metric that is not available for every model
        still appears in ``dropped`` with the reason, rather than being silently ignored
        — asking for an axis and not getting it should be visible.
    include_robustness:
        Whether to include robustness scores as axes. They are included only where every
        model computed them over the *same* underlying accuracy metric; a mismatch drops
        the axis with that as the reason.

    Raises
    ------
    ValueError
        If fewer than two models are given, or if no metric is shared by all of them.
        Both are cases where a comparison figure would show something misleading, and
        refusing is better than drawing it.
    """
    reg = registry or default_registry()
    named = _named_reports(reports)
    if len(named) < 2:
        raise ValueError(
            f"A comparison needs at least two models; got {list(named)}. "
            f"For a single model, the assessment report is the output."
        )

    per_model: dict[str, dict[str, float]] = {}
    robustness_basis: dict[str, dict[str, str]] = {}
    perturbation_sets: dict[str, dict[str, tuple[str, ...]]] = {}
    for name, report in named.items():
        values, basis, sets = _metrics_of(report, include_robustness=include_robustness)
        per_model[name] = values
        robustness_basis[name] = basis
        perturbation_sets[name] = sets

    models = tuple(named)
    dropped: dict[str, str] = {}

    requested = set(metrics) if metrics is not None else set().union(
        *(set(v) for v in per_model.values())
    )

    shared: list[str] = []
    for metric in sorted(requested):
        absent = [m for m in models if metric not in per_model[m]]
        if absent:
            dropped[metric] = (
                f"not measured for {', '.join(absent)}; a metric one model lacks is not "
                f"a comparison"
            )
            continue
        if metric not in reg:
            dropped[metric] = (
                "not in the metric registry, so it has no orientation or bounds and "
                "cannot be normalised onto a shared axis"
            )
            continue
        bases = {robustness_basis[m].get(metric) for m in models}
        if len(bases) > 1:
            described = ", ".join(
                f"{m}: {robustness_basis[m].get(metric)}" for m in models
            )
            dropped[metric] = (
                f"computed over different underlying metrics ({described}); two "
                f"robustness scores over different accuracy metrics are different "
                f"quantities"
            )
            continue
        sets = {perturbation_sets[m].get(metric) for m in models}
        if len(sets) > 1:
            described = " vs ".join(
                ", ".join(perturbation_sets[m].get(metric) or ()) or "none"
                for m in models
            )
            dropped[metric] = (
                f"aggregated over different perturbation sets ({described}); survey §8 "
                f"notes that a sum-based aggregate scales with the size of the set, so "
                f"these are not the same quantity"
            )
            continue
        if not all(np.isfinite(per_model[m][metric]) for m in models):
            dropped[metric] = "not finite for every model"
            continue
        shared.append(metric)

    if not shared:
        raise ValueError(
            "No metric is shared by every model, so there is nothing to compare on.\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(dropped.items()))
        )

    raw = {m: {metric: per_model[m][metric] for metric in shared} for m in models}
    normalised = {
        m: {
            metric: normalize_metric(raw[m][metric], metric, registry=reg)
            for metric in shared
        }
        for m in models
    }
    return ModelComparison(
        models=models,
        metrics=tuple(shared),
        raw=raw,
        normalised=normalised,
        dropped=dropped,
        metadata={
            "per_model_metrics": {m: sorted(per_model[m]) for m in models},
            "robustness_basis": {
                m: dict(robustness_basis[m]) for m in models if robustness_basis[m]
            },
            "perturbation_sets": {
                m: {k: list(v) for k, v in perturbation_sets[m].items()}
                for m in models
                if perturbation_sets[m]
            },
        },
    )


def _named_reports(reports: Mapping[str, Any] | Iterable[Any]) -> dict[str, Any]:
    if isinstance(reports, Mapping):
        # A {metric: value} mapping is itself a report, so a mapping whose values are
        # numbers is one model and not a set of models.
        if all(isinstance(v, (int, float)) for v in reports.values()) and reports:
            raise ValueError(
                "This looks like one model's metrics rather than several models. "
                "Pass {model_name: {metric: value}, ...}."
            )
        return dict(reports)
    out: dict[str, Any] = {}
    for index, report in enumerate(reports):
        name = getattr(report, "model_name", None) or f"model_{index}"
        if name in out:
            raise ValueError(
                f"Two reports are both named {name!r}. Pass a mapping with distinct "
                f"names, or set model_name on each report."
            )
        out[name] = report
    return out


def _metrics_of(
    report: Any, *, include_robustness: bool
) -> tuple[dict[str, float], dict[str, str], dict[str, tuple[str, ...]]]:
    """Every comparable metric value from one report, with what each one depends on.

    Two dependencies come back beside the values, and both exist to let the comparison
    refuse an axis rather than draw a misleading one. The *basis* is the underlying
    accuracy metric a robustness score was computed over — a robustness over accuracy
    and one over mean absolute error are different quantities under one name. The
    *perturbation set* is what a sweep-aggregated score was aggregated over, which
    survey §8 requires because a sum-based aggregate scales with the size of that set.
    """
    if isinstance(report, Mapping):
        return {str(k): float(v) for k, v in report.items()}, {}, {}

    values: dict[str, float] = {}
    basis: dict[str, str] = {}
    sets: dict[str, tuple[str, ...]] = {}

    accuracy = getattr(report, "accuracy", None)
    if accuracy is not None:
        values.update({k: float(v) for k, v in accuracy.overall.metrics.items()})
    elif hasattr(report, "overall"):  # an AccuracyResult passed directly
        values.update({k: float(v) for k, v in report.overall.metrics.items()})

    robustness = getattr(report, "robustness", None)
    if robustness is None and hasattr(report, "scored") and hasattr(report, "sweep"):
        robustness = report  # a RobustnessAssessment passed directly
    if include_robustness and robustness is not None:
        for name, result in robustness.scored.items():
            if result.aggregate is None or not np.isfinite(result.aggregate):
                continue
            values[name] = float(result.aggregate)
            basis[name] = result.accuracy_metric
            sets[name] = tuple(result.perturbation_set)
        values.setdefault(robustness.accuracy_metric, float(robustness.clean))

    return values, basis, sets
