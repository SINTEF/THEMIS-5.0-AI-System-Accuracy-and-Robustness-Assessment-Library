"""Name -> callable, for every metric the library can compute.

The four repos each carried their own parallel dictionaries — ``default_metric_functions``
as a *list* in repo A's template and repo D, as a *dict* in repo C, alongside
``default_metric_ranges`` and ``default_acceptable_metric_values`` keyed by
``function.__name__``. Keying on ``__name__`` is what made them fragile: wrapping a metric
in a ``partial`` silently broke the lookup, and the three dictionaries could (and did)
drift out of sync, with repo C listing ``f1_score`` in its ranges while repo A's template
did not.

Here there is one mapping. Ranges, orientation and bounds live in
:mod:`themis_assess.metrics.registry`; this module holds only *how to compute*, and the
two are kept in step by a test asserting every catalogued name is a registered name.

Two things this module deliberately does **not** do.

*No re-implementation.* Where scikit-learn has the metric, the entry is scikit-learn's own
function. The only wrappers are for genuine call-convention problems, and each one is
documented at its definition.

*No positional guessing.* Every entry declares which quantities it needs — ``actual``,
``predicted``, ``probabilities``, ``lower``/``upper``, ``samples`` — so the assessment
runner can be handed whatever a model handler produced and ask what is computable from it,
rather than calling everything and catching exceptions. A metric that needs probabilities
from a handler that only returns labels is reported as skipped, up front, with a reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from themis_assess.core.types import AccessLevel, TaskType

__all__ = [
    "MetricEntry",
    "MetricCatalog",
    "default_catalog",
]


# --------------------------------------------------------------------------------------
# call-convention wrappers
# --------------------------------------------------------------------------------------


def _average_for(actual: ArrayLike) -> str:
    """``"binary"`` for two classes, ``"macro"`` beyond.

    All three use-case repos called ``precision_score(y_true, y_pred)`` with no
    ``average``, which raises on more than two classes. Repo D's fake-news task is binary
    so it never surfaced; repo C's is not always. Choosing the average from the data fixes
    it without asking every caller to specify. ``macro`` rather than ``weighted`` because
    a robustness study cares about the rare class — weighting by support hides exactly the
    degradation the assessment is looking for.
    """
    classes = np.unique(np.asarray(actual))
    return "binary" if classes.size <= 2 else "macro"


def _precision(actual: ArrayLike, predicted: ArrayLike) -> float:
    from sklearn.metrics import precision_score

    return float(
        precision_score(actual, predicted, average=_average_for(actual), zero_division=0)
    )


def _recall(actual: ArrayLike, predicted: ArrayLike) -> float:
    from sklearn.metrics import recall_score

    return float(
        recall_score(actual, predicted, average=_average_for(actual), zero_division=0)
    )


def _f1(actual: ArrayLike, predicted: ArrayLike) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(actual, predicted, average=_average_for(actual), zero_division=0))


def _roc_auc(actual: ArrayLike, probabilities: ArrayLike) -> float:
    """ROC AUC from a probability matrix, binary or multiclass.

    ``roc_auc_score`` wants the positive class's column in the binary case and the whole
    matrix plus ``multi_class`` beyond it. Handlers return the full matrix either way, so
    the choice is made here rather than in every caller.
    """
    from sklearn.metrics import roc_auc_score

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        return float(roc_auc_score(actual, probs))
    if probs.shape[1] == 2:
        return float(roc_auc_score(actual, probs[:, 1]))
    return float(roc_auc_score(actual, probs, multi_class="ovr", average="macro"))


def _rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error.

    ``sklearn.metrics.root_mean_squared_error`` exists only from 1.4 and
    ``mean_squared_error(squared=False)`` was removed in 1.6, so neither spelling covers
    the supported range. The square root is written out instead of pinning scikit-learn
    tighter than the science requires.
    """
    from sklearn.metrics import mean_squared_error

    return float(np.sqrt(mean_squared_error(actual, predicted)))


# --------------------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """A computable metric: the callable plus what it needs to be called with.

    Parameters
    ----------
    requires:
        Names of the quantities the callable takes, **in positional order**. The runner
        holds a bag of whatever the handler produced and supplies these by name. Known
        keys: ``actual``, ``predicted``, ``probabilities``, ``logits``, ``lower``,
        ``upper``, ``samples``, ``mean``, ``std``, ``weights``, ``reference``.
    optional:
        Further quantities passed as keywords when available. ``weights`` is the case
        this exists for: the weighted errors work unweighted and are strictly better
        with weights.
    """

    name: str
    fn: Callable[..., Any]
    requires: tuple[str, ...]
    tasks: frozenset[TaskType]
    access: AccessLevel = AccessLevel.BLACK_BOX
    optional: tuple[str, ...] = ()

    def missing(self, available: Mapping[str, Any]) -> tuple[str, ...]:
        """Which required quantities are absent from ``available``."""
        return tuple(k for k in self.requires if available.get(k) is None)

    def compute(self, available: Mapping[str, Any], **kwargs: Any) -> Any:
        absent = self.missing(available)
        if absent:
            raise KeyError(
                f"Cannot compute '{self.name}': missing {', '.join(absent)}. "
                f"It needs {', '.join(self.requires)}; got "
                f"{', '.join(sorted(k for k, v in available.items() if v is not None)) or 'nothing'}."
            )
        args = [available[k] for k in self.requires]
        extra = {k: available[k] for k in self.optional if available.get(k) is not None}
        return self.fn(*args, **extra, **kwargs)


class MetricCatalog:
    """Name -> :class:`MetricEntry`, with lookup by task and by what is available."""

    def __init__(self, *, with_defaults: bool = True) -> None:
        self._entries: dict[str, MetricEntry] = {}
        if with_defaults:
            self._register_defaults()

    # -- registration ----------------------------------------------------------

    def register(self, entry: MetricEntry, *, overwrite: bool = False) -> None:
        if not overwrite and entry.name in self._entries:
            raise ValueError(
                f"Metric '{entry.name}' is already in the catalog. Pass overwrite=True "
                f"to replace it."
            )
        self._entries[entry.name] = entry

    def add(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        requires: tuple[str, ...] = ("actual", "predicted"),
        tasks: frozenset[TaskType] | None = None,
        access: AccessLevel = AccessLevel.BLACK_BOX,
        optional: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> None:
        """Convenience registration for a use case's own metric.

        This is the seam repo B needed: its berthing metrics are not general, but they
        must run through the same machinery as the general ones so the report and the
        normalisation treat them identically.
        """
        self.register(
            MetricEntry(
                name=name,
                fn=fn,
                requires=requires,
                tasks=tasks if tasks is not None else frozenset(TaskType),
                access=access,
                optional=optional,
            ),
            overwrite=overwrite,
        )

    # -- lookup ----------------------------------------------------------------

    def get(self, name: str) -> MetricEntry:
        try:
            return self._entries[name]
        except KeyError:
            near = [k for k in self._entries if name.lower() in k.lower()]
            hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
            raise KeyError(f"No callable registered for metric: '{name}'.{hint}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def for_task(self, task: TaskType) -> list[str]:
        return sorted(n for n, e in self._entries.items() if task in e.tasks)

    def computable(
        self, available: Mapping[str, Any], *, task: TaskType | None = None
    ) -> list[str]:
        """Metrics that can be computed from the quantities in ``available``.

        The runner calls this once and reports the complement as skipped, with the
        missing quantity named, before starting a long evaluation.
        """
        return sorted(
            n
            for n, e in self._entries.items()
            if not e.missing(available) and (task is None or task in e.tasks)
        )

    def compute(self, name: str, available: Mapping[str, Any], **kwargs: Any) -> Any:
        return self.get(name).compute(available, **kwargs)

    def compute_all(
        self,
        names: list[str],
        available: Mapping[str, Any],
        *,
        skip_uncomputable: bool = False,
    ) -> dict[str, Any]:
        """Compute several metrics from one bag of quantities."""
        out: dict[str, Any] = {}
        for name in names:
            entry = self.get(name)
            if skip_uncomputable and entry.missing(available):
                continue
            out[name] = entry.compute(available)
        return out

    # -- defaults --------------------------------------------------------------

    def _register_defaults(self) -> None:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            matthews_corrcoef,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        from themis_assess.metrics.accuracy.weighted import (
            mean_signed_deviation,
            weighted_mae,
            weighted_mse,
        )
        from themis_assess.metrics.calibration import (
            brier_score,
            crps_ensemble,
            crps_gaussian,
            energy_score,
            expected_calibration_error,
            interval_score,
            mean_prediction_interval_width,
            negative_log_likelihood,
            prediction_interval_coverage,
        )

        clf = frozenset({TaskType.CLASSIFICATION})
        fc = frozenset({TaskType.FORECASTING})
        prob = AccessLevel.PROBABILISTIC

        # --- classification, from labels ---------------------------------------
        # scikit-learn's own functions, unwrapped: the signature is already
        # (actual, predicted).
        self.add("accuracy", accuracy_score, tasks=clf)
        self.add("balanced_accuracy", balanced_accuracy_score, tasks=clf)
        self.add("matthews_corrcoef", matthews_corrcoef, tasks=clf)
        self.add("precision", _precision, tasks=clf)
        self.add("recall", _recall, tasks=clf)
        self.add("f1", _f1, tasks=clf)

        # --- classification, from probabilities --------------------------------
        self.add(
            "auc", _roc_auc, requires=("actual", "probabilities"), tasks=clf, access=prob
        )
        self.add(
            "expected_calibration_error",
            expected_calibration_error,
            requires=("probabilities", "actual"),
            tasks=clf,
            access=prob,
        )
        self.add(
            "brier_score",
            brier_score,
            requires=("probabilities", "actual"),
            tasks=clf,
            access=prob,
        )
        self.add(
            "negative_log_likelihood",
            negative_log_likelihood,
            requires=("probabilities", "actual"),
            tasks=clf,
            access=prob,
        )

        # --- forecasting, point ------------------------------------------------
        self.add("mae", mean_absolute_error, tasks=fc)
        self.add("mse", mean_squared_error, tasks=fc)
        self.add("rmse", _rmse, tasks=fc)
        self.add("r2", r2_score, tasks=fc)
        self.add("weighted_mae", weighted_mae, tasks=fc, optional=("weights",))
        self.add("weighted_mse", weighted_mse, tasks=fc, optional=("weights",))
        self.add(
            "mean_signed_deviation", mean_signed_deviation, tasks=fc, optional=("weights",)
        )

        # --- forecasting, distributional ---------------------------------------
        self.add(
            "prediction_interval_coverage",
            prediction_interval_coverage,
            requires=("lower", "upper", "actual"),
            tasks=fc,
            access=prob,
        )
        self.add(
            "mean_prediction_interval_width",
            mean_prediction_interval_width,
            requires=("lower", "upper"),
            tasks=fc,
            access=prob,
        )
        self.add(
            "interval_score",
            interval_score,
            requires=("lower", "upper", "actual"),
            tasks=fc,
            access=prob,
        )
        self.add(
            "crps",
            crps_ensemble,
            requires=("samples", "actual"),
            tasks=fc,
            access=prob,
        )
        self.add(
            "crps_gaussian",
            crps_gaussian,
            requires=("mean", "std", "actual"),
            tasks=fc,
            access=prob,
        )
        self.add(
            "energy_score",
            energy_score,
            requires=("samples", "actual"),
            tasks=fc,
            access=prob,
        )


def default_catalog() -> MetricCatalog:
    """A fresh catalog with the built-in metrics.

    A new instance per call, so a use case adding its own metrics does not mutate anyone
    else's catalog — the same rule as :func:`themis_assess.metrics.registry.default_registry`.
    """
    return MetricCatalog()
