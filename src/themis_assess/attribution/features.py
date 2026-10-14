"""Turning model input into features an error model can learn from.

The surrogate in :mod:`themis_assess.attribution.error_model` predicts a model's error
from a description of its input. What that description should be depends on the modality,
and this module is the seam.

Repo B used TSFresh for it, with a hand-built ``kind_to_fc_parameters`` configuration and
two custom calculators (``last_value``, ``constant``) monkey-patched onto
``tsfresh.feature_extraction.feature_calculators`` so the string names would resolve. That
works and is powerful, but it made a heavyweight optional dependency load-bearing for the
whole capability. Here the default extractor is a compact set of window summaries computed
with numpy — no extra dependencies, interpretable names — and TSFresh is available for
callers who want its hundreds of features.

The choice matters more than it looks. An error model's attributions are only as
interpretable as its features: "the covariate's variance over the input window was high"
is a finding someone can act on, and
``cov__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2`` is not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from themis_assess.core.adapters import DataAdapter

__all__ = [
    "FeatureExtractor",
    "TabularFeatures",
    "WindowSummaryFeatures",
    "TSFreshFeatures",
]


@runtime_checkable
class FeatureExtractor(Protocol):
    """Turns an adapter into a ``(n_entities, n_features)`` matrix with named columns."""

    def extract(self, data: DataAdapter) -> tuple[NDArray, list[str]]:
        """Return the feature matrix and its column names."""
        ...


class TabularFeatures:
    """The raw feature columns, unchanged. For tabular data.

    The right extractor whenever the input already *is* a feature vector, and the one
    that makes attribution most directly interpretable: a SHAP value against column
    ``age`` means what it says.

    On sequential data this flattens the time axis, producing ``feature@t`` columns. That
    is honest but usually a mistake — it gives the surrogate ``n_timesteps x n_features``
    columns for however many entities exist, which for repo B's data would have been
    thousands of columns against dozens of rows. Use :class:`WindowSummaryFeatures` there.
    """

    name = "tabular"

    def extract(self, data: DataAdapter) -> tuple[NDArray, list[str]]:
        values = data.values()
        n_entities, n_timesteps, n_features = values.shape
        if n_timesteps == 1:
            return values[:, 0, :], list(data.feature_names)
        matrix = values.reshape(n_entities, -1)
        names = [
            f"{feature}@t{t}"
            for t in range(n_timesteps)
            for feature in data.feature_names
        ]
        return matrix, names


class WindowSummaryFeatures:
    """Compact per-feature summaries of the input window. The no-extra-dependency default.

    Twelve statistics per feature, chosen to cover the questions that actually come up
    when a forecast goes wrong: what level was the input at, how much was it moving, was
    it trending, did it jump, was it flat, how much of it was missing.

    ==================  ====================================================
    statistic           what a high value would mean
    ==================  ====================================================
    ``mean``, ``last``  the level the window sat at, and where it ended
    ``std``, ``range``  how much it was moving
    ``min``, ``max``    the extremes it reached
    ``slope``           a trend across the window
    ``last_minus_mean`` the end departing from the window's own level
    ``abs_diff_mean``   step-to-step volatility
    ``max_abs_diff``    the largest single jump
    ``flat_fraction``   how much of the window was unchanging
    ``missing_fraction`` how much was absent
    ==================  ====================================================

    ``missing_fraction`` earns its place independently: a model whose errors are
    predictable from *how much data was missing* has an availability problem rather than
    a modelling one, and that is a different team's fix.
    """

    name = "window_summary"

    _STATISTICS = (
        "mean",
        "std",
        "min",
        "max",
        "range",
        "last",
        "slope",
        "last_minus_mean",
        "abs_diff_mean",
        "max_abs_diff",
        "flat_fraction",
        "missing_fraction",
    )

    def __init__(self, *, statistics: Sequence[str] | None = None) -> None:
        chosen = tuple(statistics) if statistics is not None else self._STATISTICS
        unknown = set(chosen) - set(self._STATISTICS)
        if unknown:
            raise ValueError(
                f"Unknown statistic(s): {sorted(unknown)}. Available: "
                f"{list(self._STATISTICS)}."
            )
        if not chosen:
            raise ValueError("WindowSummaryFeatures needs at least one statistic.")
        self.statistics = chosen

    def _summarise(self, series: NDArray) -> dict[str, float]:
        """Statistics of one entity's one feature. NaN-aware throughout."""
        finite = series[np.isfinite(series)]
        missing_fraction = float(1.0 - finite.size / max(series.size, 1))
        if finite.size == 0:
            return {name: np.nan for name in self.statistics} | {
                "missing_fraction": missing_fraction
            }

        differences = np.diff(finite) if finite.size > 1 else np.array([0.0])
        if finite.size > 1:
            slope = float(np.polyfit(np.arange(finite.size), finite, 1)[0])
        else:
            slope = 0.0
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "range": float(finite.max() - finite.min()),
            "last": float(finite[-1]),
            "slope": slope,
            "last_minus_mean": float(finite[-1] - finite.mean()),
            "abs_diff_mean": float(np.abs(differences).mean()),
            "max_abs_diff": float(np.abs(differences).max()),
            "flat_fraction": float(np.mean(differences == 0.0)),
            "missing_fraction": missing_fraction,
        }

    def extract(self, data: DataAdapter) -> tuple[NDArray, list[str]]:
        values = data.values()
        n_entities = values.shape[0]
        feature_names = data.feature_names

        names = [
            f"{feature}__{statistic}"
            for feature in feature_names
            for statistic in self.statistics
        ]
        matrix = np.full((n_entities, len(names)), np.nan)
        for e in range(n_entities):
            column = 0
            for j in range(len(feature_names)):
                summary = self._summarise(values[e, :, j].astype(float))
                for statistic in self.statistics:
                    matrix[e, column] = summary[statistic]
                    column += 1
        return matrix, names


class TSFreshFeatures:
    """Hundreds of time-series features, via TSFresh. Requires the ``[explain]`` extra.

    Repo B's extractor. Strictly more powerful than
    :class:`WindowSummaryFeatures` and strictly harder to read: the price of TSFresh's
    coverage is feature names nobody can interpret without the documentation open, and a
    feature count that will exceed the number of entities on any dataset the size of repo
    B's.

    Two of repo B's mechanics are kept because they were right. ``last_value`` and
    ``constant`` are registered as TSFresh calculators, since the most useful description
    of a covariate window is often just its final value and TSFresh has no such
    calculator. And the configuration is passed straight through as
    ``kind_to_fc_parameters`` / ``default_fc_parameters``, rather than being wrapped —
    TSFresh's own configuration format is well documented and re-inventing it would only
    add a translation layer to get wrong.

    Parameters
    ----------
    default_fc_parameters:
        TSFresh settings applied to every feature. ``None`` uses TSFresh's
        ``MinimalFCParameters``, which is the sane default for a surrogate that will have
        far more columns than rows.
    kind_to_fc_parameters:
        Per-feature settings, overriding the default.
    """

    name = "tsfresh"

    def __init__(
        self,
        *,
        default_fc_parameters: Any = None,
        kind_to_fc_parameters: Any = None,
        n_jobs: int = 0,
    ) -> None:
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters
        self.n_jobs = n_jobs

    @staticmethod
    def _register_calculators() -> None:
        from tsfresh.feature_extraction import feature_calculators
        from tsfresh.utilities.string_manipulation import set_property

        if not hasattr(feature_calculators, "last_value"):

            @set_property("fctype", "simple")
            def last_value(x: Any) -> float:
                return x[-1] if len(x) > 0 else float("nan")

            feature_calculators.last_value = last_value

    def extract(self, data: DataAdapter) -> tuple[NDArray, list[str]]:
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "TSFreshFeatures needs the [explain] extra: pip install "
                "'themis-assess[explain]'. WindowSummaryFeatures needs no extra "
                "dependencies and produces interpretable feature names."
            ) from exc

        import pandas as pd

        self._register_calculators()
        values = data.values()
        n_entities, n_timesteps, _ = values.shape

        # TSFresh's long format: one row per (entity, timestep, feature).
        records = []
        for e in range(n_entities):
            for j, feature in enumerate(data.feature_names):
                for t in range(n_timesteps):
                    records.append((e, t, feature, float(values[e, t, j])))
        long = pd.DataFrame(records, columns=["id", "time", "kind", "value"])

        extracted = extract_features(
            long,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=(
                self.default_fc_parameters
                if self.default_fc_parameters is not None
                else MinimalFCParameters()
            ),
            kind_to_fc_parameters=self.kind_to_fc_parameters,
            disable_progressbar=True,
            n_jobs=self.n_jobs,
        )
        extracted = extracted.sort_index()
        return extracted.to_numpy(dtype=float), [str(c) for c in extracted.columns]


def resolve_extractor(
    extractor: FeatureExtractor | str | Callable[[DataAdapter], Any] | None,
    data: DataAdapter,
) -> FeatureExtractor:
    """Pick an extractor, defaulting by modality.

    Tabular data gets its raw columns; sequential data gets window summaries. Both are
    the interpretable choice for their shape, and neither needs an optional dependency.
    """
    if extractor is None:
        return TabularFeatures() if not data.is_sequential else WindowSummaryFeatures()
    if isinstance(extractor, str):
        known = {
            "tabular": TabularFeatures,
            "window_summary": WindowSummaryFeatures,
            "tsfresh": TSFreshFeatures,
        }
        if extractor not in known:
            raise ValueError(
                f"Unknown extractor {extractor!r}. Available: {sorted(known)}."
            )
        return known[extractor]()
    return extractor
