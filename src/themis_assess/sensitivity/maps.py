"""Sensitivity maps: which input, perturbed where, moves the model most.

A robustness score says how much a model degrades. A sensitivity map says *what* it is
sensitive to — which feature, at which point in the input window, under which
perturbation. That is the difference between a number a report can quote and a finding an
engineer can act on, and repo B's heatmap family is the strongest original instrument in
any of the four repositories.

Ported from ``ValenciaRobustnessAssessment.one_vessel_heatmap`` and
``all_vessels_heatmap``, with four changes.

*The distinction repo B left implicit is made explicit.* Its heatmap compared each
perturbed prediction against the **clean prediction**, not against the truth. That is a
*prediction-shift* map, and it is the more broadly useful of the two: it needs no labels,
so it can be computed on production data. A *degradation* map compares against the truth
and answers a different question — sensitivity is not the same as harm, since a model can
move a lot and still be right, or barely move and be confidently wrong. Both are here,
selected by ``measure``, and the result records which was used.

*Data and plot are separated.* Repo B's function imported matplotlib, suppressed
PyTorch-Lightning loggers, and took a ``scale="log"`` colormap argument — inside the
computation. Only the grid is computed here.

*The window is a parameter.* Repo B hardcoded "only analyse the last 120 time steps".

*The cost is stated before it is incurred.* A feature-by-severity map is
``n_features x n_severities x n_variants`` model calls, which for repo B's data was
thousands; :meth:`SensitivityMapper.plan` says how many before starting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.seeding import derive_generator
from themis_assess.core.types import PerturbationSpec, Scope, TimeScope
from themis_assess.handlers.base import ModelHandler, Prediction
from themis_assess.perturbations.base import Perturbation, PerturbationRegistry
from themis_assess.perturbations.registry import default_registry as default_perturbations

__all__ = [
    "SensitivityMap",
    "SensitivityMapper",
    "map_from_sweep",
]

Measure = Literal["prediction_shift", "degradation"]


@dataclass(frozen=True, slots=True)
class SensitivityMap:
    """A 2-D grid of sensitivity values, with labelled axes.

    Parameters
    ----------
    grid:
        ``(n_rows, n_columns)``. Higher always means *more* sensitive, whichever
        ``measure`` produced it — a degradation map over an accuracy metric is negated on
        the way in, so a map can be read, sorted and compared without checking its
        orientation first.
    row_labels, column_labels:
        What the axes are. Kept attached for the same reason as on a confusion matrix:
        a bare grid read against the wrong axis order tells a confident, wrong story.
    measure:
        ``"prediction_shift"`` — how far the prediction moved from the clean one, no
        labels needed. ``"degradation"`` — how much worse the metric got, labels needed.
    """

    grid: NDArray
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    measure: Measure
    row_axis: str = "feature"
    column_axis: str = "severity"
    metric: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.grid, dtype=float)
        if array.ndim != 2:
            raise ValueError(f"A sensitivity map must be 2-D; got shape {array.shape}.")
        if array.shape != (len(self.row_labels), len(self.column_labels)):
            raise ValueError(
                f"Grid shape {array.shape} does not match "
                f"{len(self.row_labels)} row label(s) x "
                f"{len(self.column_labels)} column label(s)."
            )
        object.__setattr__(self, "grid", array)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def row(self, label: str) -> NDArray:
        try:
            return self.grid[self.row_labels.index(label)]
        except ValueError:
            raise KeyError(
                f"No row {label!r}. Available: {list(self.row_labels)}."
            ) from None

    def most_sensitive_rows(self, *, n: int = 5, aggregate: str = "max") -> list[
        dict[str, Any]
    ]:
        """The rows the model is most sensitive to, worst first.

        Parameters
        ----------
        aggregate:
            How to reduce each row to one number. ``"max"`` (the default) ranks by the
            worst cell, ``"mean"`` by the average across columns. They disagree, and the
            disagreement is informative: a feature with one catastrophic severity and a
            feature that is uniformly mediocre are different problems.
        """
        if aggregate not in {"max", "mean"}:
            raise ValueError(f"aggregate must be max or mean; got {aggregate!r}.")
        reducer = np.nanmax if aggregate == "max" else np.nanmean
        scores = [
            (label, float(reducer(self.grid[i])))
            for i, label in enumerate(self.row_labels)
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {"label": label, "score": score, "aggregate": aggregate}
            for label, score in scores[:n]
        ]

    def hottest_cell(self) -> dict[str, Any]:
        """The single most sensitive (row, column) cell."""
        flat = int(np.nanargmax(self.grid))
        i, j = np.unravel_index(flat, self.grid.shape)
        return {
            self.row_axis: self.row_labels[i],
            self.column_axis: self.column_labels[j],
            "value": float(self.grid[i, j]),
        }

    def tipping_point(self, label: str, threshold: float) -> str | None:
        """The first column at which a row's sensitivity exceeds ``threshold``.

        The most decision-relevant thing a severity-axis map contains: not "how bad does
        it get" but "how much perturbation can this feature take before it matters".
        ``None`` when the row never crosses.
        """
        values = self.row(label)
        crossed = np.flatnonzero(values > threshold)
        if crossed.size == 0:
            return None
        return self.column_labels[int(crossed[0])]

    def to_frame(self) -> Any:
        """As a pandas DataFrame with labelled axes."""
        import pandas as pd

        return pd.DataFrame(
            self.grid,
            index=pd.Index(list(self.row_labels), name=self.row_axis),
            columns=pd.Index(list(self.column_labels), name=self.column_axis),
        )


def _prediction_shift(clean: Prediction, perturbed: Prediction, *, order: int = 1) -> float:
    """Mean distance between two predictions. No labels involved.

    Defined over the point predictions, which every handler produces, so a shift map is
    available at black-box access. Repo B used MAE (``order=1``) and MSE (``order=2``)
    against the clean prediction; both are here.
    """
    a = np.asarray(clean.predicted, dtype=float).ravel()
    b = np.asarray(perturbed.predicted, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"The clean and perturbed predictions differ in shape: {a.shape} vs "
            f"{b.shape}. A length-changing perturbation cannot be measured as a shift; "
            f"use measure='degradation'."
        )
    difference = np.abs(a - b)
    return float(np.mean(difference**order) ** (1.0 / order))


class SensitivityMapper:
    """Builds sensitivity grids by perturbing one thing at a time.

    Parameters
    ----------
    handler:
        The model.
    perturbation:
        Which perturbation to use, by name or as an object. One perturbation per map:
        the axes are already two-dimensional, and a third would not be a grid.
    severities:
        The column axis for :meth:`feature_severity_map`.
    n_variants:
        Draws per cell, averaged. A single draw of a stochastic perturbation on one
        feature is mostly noise, so this matters more here than in a whole-dataset
        sweep — every cell perturbs *less* of the input and therefore has a smaller
        effect to detect.
    measure:
        ``"prediction_shift"`` or ``"degradation"``. Shift by default, because it needs
        no labels and answers the sensitivity question directly.
    metric:
        For ``measure="degradation"``, which metric to track.
    seed:
        Cell-derived, as in the sweep: a cell's draw does not depend on execution order.
    """

    def __init__(
        self,
        handler: ModelHandler,
        *,
        perturbation: str | Perturbation = "gaussian_noise",
        severities: Sequence[int] = (1, 2, 3, 4, 5),
        n_variants: int = 3,
        measure: Measure = "prediction_shift",
        metric: str | None = None,
        shift_order: int = 1,
        registry: PerturbationRegistry | None = None,
        seed: int = 0,
    ) -> None:
        self.handler = handler
        self._registry = registry if registry is not None else default_perturbations()
        self.perturbation = (
            self._registry[perturbation]
            if isinstance(perturbation, str)
            else perturbation
        )
        self.severities = tuple(severities)
        if any(s < 1 for s in self.severities):
            raise ValueError(f"Severity levels are 1-based; got {self.severities}.")
        if n_variants < 1:
            raise ValueError(f"n_variants must be at least 1; got {n_variants}.")
        self.n_variants = n_variants
        if measure not in ("prediction_shift", "degradation"):
            raise ValueError(
                f"measure must be 'prediction_shift' or 'degradation'; got {measure!r}."
            )
        self.measure = measure
        self.shift_order = shift_order
        self.seed = seed

        if measure == "degradation" and metric is None:
            raise ValueError(
                "measure='degradation' needs a metric to track — it compares the metric "
                "under perturbation against the metric on clean data. "
                "measure='prediction_shift' needs no metric and no labels."
            )
        self.metric = metric

    # -- planning ----------------------------------------------------------------

    def plan(self, data: DataAdapter, *, axis: str = "feature_severity") -> dict[str, Any]:
        """How many model calls a map will cost, before making them."""
        eligible = self._eligible_features(data)
        if axis == "feature_severity":
            cells = len(eligible) * len(self.severities)
        elif axis == "feature_time":
            cells = len(eligible) * self._n_time_groups(data, 8)
        else:
            raise ValueError(f"Unknown axis: {axis!r}.")
        return {
            "features": tuple(eligible),
            "cells": cells,
            "n_model_calls": 1 + cells * self.n_variants,
            "measure": self.measure,
            "perturbation": self.perturbation.name,
        }

    def _eligible_features(self, data: DataAdapter) -> list[str]:
        """Features this perturbation can legitimately act on, one at a time."""
        categorical = set(data.categorical_features)
        eligible = [
            name
            for name in data.feature_names
            if (
                self.perturbation.accepts_categorical
                if name in categorical
                else self.perturbation.accepts_numerical
            )
        ]
        if not eligible:
            raise ValueError(
                f"{self.perturbation.name!r} cannot act on any feature of this data: it "
                f"accepts "
                f"{'categorical' if self.perturbation.accepts_categorical else 'numerical'}"
                f" features, and the data declares "
                f"numerical={data.numerical_features}, "
                f"categorical={data.categorical_features}."
            )
        return eligible

    def _n_time_groups(self, data: DataAdapter, granularity: int) -> int:
        if not data.is_sequential:
            raise ValueError(
                "A feature-by-time map needs sequential data; this adapter is "
                f"{data.modality.value} with a single timestep."
            )
        return int(np.ceil(data.n_timesteps / granularity))

    # -- the maps ----------------------------------------------------------------

    def _cell_rng(self, row: str, column: object, variant: int) -> np.random.Generator:
        """A generator per cell, stable across processes.

        Same contract as the sweep's, and via the same helper — see
        :mod:`themis_assess.core.seeding`.
        """
        return derive_generator(
            self.seed, self.perturbation.name, row, column, variant
        )

    def _measure_cell(
        self,
        clean: Prediction,
        perturbed_data: DataAdapter,
        actual: Any,
        clean_metric: float | None,
    ) -> float:
        prediction = self.handler.predict(perturbed_data)
        if self.measure == "prediction_shift":
            return _prediction_shift(clean, prediction, order=self.shift_order)

        from themis_assess.metrics.accuracy.catalog import default_catalog
        from themis_assess.metrics.registry import default_registry

        catalog, registry = default_catalog(), default_registry()
        bag = {**prediction.available(), "actual": actual}
        value = float(catalog.compute(self.metric, bag))
        spec = registry.get(self.metric)
        # Higher always means more sensitive, so an accuracy metric's *drop* is the
        # sensitivity and an error metric's *rise* is. Without this a map over accuracy
        # and a map over error would be read in opposite directions.
        return (
            clean_metric - value if spec.higher_is_better else value - clean_metric
        )

    def _clean_reference(
        self, data: DataAdapter, actual: Any
    ) -> tuple[Prediction, float | None]:
        clean = self.handler.predict(data)
        if self.measure == "prediction_shift":
            return clean, None
        from themis_assess.metrics.accuracy.catalog import default_catalog

        bag = {**clean.available(), "actual": actual}
        return clean, float(default_catalog().compute(self.metric, bag))

    def feature_severity_map(
        self, data: DataAdapter, actual: Any = None
    ) -> SensitivityMap:
        """One row per feature, one column per severity.

        The map to build first. It answers "which input does this model lean on, and how
        hard can that input be pushed before the model notices" — and because each cell
        perturbs exactly one feature, the answer is attributable in a way a whole-dataset
        sweep is not.
        """
        if self.measure == "degradation" and actual is None:
            raise ValueError("measure='degradation' needs the ground truth.")

        features = self._eligible_features(data)
        clean, clean_metric = self._clean_reference(data, actual)

        grid = np.zeros((len(features), len(self.severities)))
        for i, feature in enumerate(features):
            for j, severity in enumerate(self.severities):
                spec = PerturbationSpec(
                    name=self.perturbation.name,
                    severity=severity,
                    scope=Scope.NAMED,
                    features=(feature,),
                )
                draws = []
                for variant in range(self.n_variants):
                    rng = self._cell_rng(feature, severity, variant)
                    perturbed = self.perturbation.perturb(data, spec, rng)[0]
                    draws.append(
                        self._measure_cell(clean, perturbed, actual, clean_metric)
                    )
                grid[i, j] = float(np.mean(draws))

        return SensitivityMap(
            grid=grid,
            row_labels=tuple(features),
            column_labels=tuple(str(s) for s in self.severities),
            measure=self.measure,
            row_axis="feature",
            column_axis="severity",
            metric=self.metric,
            metadata={
                "perturbation": self.perturbation.name,
                "n_variants": self.n_variants,
                "seed": self.seed,
                "model": self.handler.name,
                "clean_metric": clean_metric,
                **self.perturbation.metadata(),
            },
        )

    def feature_time_map(
        self,
        data: DataAdapter,
        actual: Any = None,
        *,
        severity: int = 3,
        time_granularity: int = 8,
        window: int | None = None,
    ) -> SensitivityMap:
        """One row per feature, one column per group of timesteps. Repo B's heatmap.

        Answers *when* in the input window the model is sensitive — whether it leans on
        the most recent observations or on something further back. For a windowed
        forecaster that is often the more surprising axis: a model that turns out to
        depend on a value 60 steps back is a model whose behaviour under a data outage is
        not what anyone assumed.

        Parameters
        ----------
        window:
            How many timesteps back to analyse, ending at the last. Repo B hardcoded 120.
            ``None`` analyses the whole series.
        time_granularity:
            Timesteps per column, perturbed together. Repo B's default was 8.
        """
        if self.measure == "degradation" and actual is None:
            raise ValueError("measure='degradation' needs the ground truth.")
        if not data.is_sequential:
            raise ValueError(
                "A feature-by-time map needs sequential data; this adapter is "
                f"{data.modality.value} with a single timestep. Use "
                f"feature_severity_map instead."
            )
        if time_granularity < 1:
            raise ValueError(
                f"time_granularity must be at least 1; got {time_granularity}."
            )

        n_timesteps = data.n_timesteps
        span = n_timesteps if window is None else min(window, n_timesteps)
        start = n_timesteps - span
        groups = [
            tuple(range(t, min(t + time_granularity, n_timesteps)))
            for t in range(start, n_timesteps, time_granularity)
        ]

        features = self._eligible_features(data)
        clean, clean_metric = self._clean_reference(data, actual)

        grid = np.zeros((len(features), len(groups)))
        for i, feature in enumerate(features):
            for j, indices in enumerate(groups):
                spec = PerturbationSpec(
                    name=self.perturbation.name,
                    severity=severity,
                    scope=Scope.NAMED,
                    features=(feature,),
                    time_scope=TimeScope.AT_INDICES,
                    time_indices=indices,
                )
                draws = []
                for variant in range(self.n_variants):
                    rng = self._cell_rng(feature, indices[0], variant)
                    perturbed = self.perturbation.perturb(data, spec, rng)[0]
                    draws.append(
                        self._measure_cell(clean, perturbed, actual, clean_metric)
                    )
                grid[i, j] = float(np.mean(draws))

        return SensitivityMap(
            grid=grid,
            row_labels=tuple(features),
            column_labels=tuple(f"t{indices[0]}-{indices[-1]}" for indices in groups),
            measure=self.measure,
            row_axis="feature",
            column_axis="time",
            metric=self.metric,
            metadata={
                "perturbation": self.perturbation.name,
                "severity": severity,
                "time_granularity": time_granularity,
                "window": span,
                "n_variants": self.n_variants,
                "seed": self.seed,
                "model": self.handler.name,
                "clean_metric": clean_metric,
            },
        )


def map_from_sweep(
    sweep: Any, metric: str, *, higher_is_better: bool | None = None
) -> SensitivityMap:
    """A perturbation-by-severity map from a sweep that already ran. Free.

    No model calls: the sweep's cells *are* the grid. Worth reaching for first, since a
    robustness assessment produces this as a by-product and it answers the coarse version
    of the question — which *kind* of corruption hurts — before spending thousands of
    calls on which feature.

    Oriented like every other map, higher meaning more sensitive: an accuracy metric's
    drop from clean, an error metric's rise.
    """
    if higher_is_better is None:
        from themis_assess.metrics.registry import default_registry

        spec = default_registry().get(metric)
        if spec.higher_is_better is None:
            raise ValueError(
                f"{metric!r} has no direction, so a sensitivity map over it cannot be "
                f"oriented. Pass higher_is_better= explicitly if you know what you mean."
            )
        higher_is_better = spec.higher_is_better

    curves = sweep.curves(metric)
    if not curves:
        raise ValueError("This sweep has no perturbed cells.")
    clean = sweep.clean[metric]

    names = sorted(curves)
    severities = sorted({s for curve in curves.values() for s in curve})
    grid = np.full((len(names), len(severities)), np.nan)
    for i, name in enumerate(names):
        for j, severity in enumerate(severities):
            if severity not in curves[name]:
                continue
            value = curves[name][severity]
            grid[i, j] = clean - value if higher_is_better else value - clean

    return SensitivityMap(
        grid=grid,
        row_labels=tuple(names),
        column_labels=tuple(str(s) for s in severities),
        measure="degradation",
        row_axis="perturbation",
        column_axis="severity",
        metric=metric,
        metadata={"clean_metric": clean, "from_sweep": True, **dict(sweep.metadata)},
    )
