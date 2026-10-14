"""Metric semantics: orientation, bounds, task applicability, and access requirements.

Recovered from ``themis_valencia/algorithms/accuracy/metrics/metric_metadata.py`` and
extended. Two defects in that version are fixed here (see ``docs/DECISIONS.md``, D19
and the architecture document §2.1):

* ``best`` conflated two distinct ideas — the value at which a metric is *ideal*, and
  the *upper bound* used for min-max normalisation. They are now ``target`` and
  ``upper_bound``. ``mean_signed_deviation`` has ``target=0.0`` and no bounds at all;
  ``f1`` has ``upper_bound=1.0`` and no target.
* ``higher_is_better`` was annotated ``bool`` while ``None`` was passed for symmetric
  metrics. It is now ``bool | None``, and ``None`` means "neither direction is better;
  see ``target``".
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from themis_assess.core.types import AccessLevel, TaskType

__all__ = ["MetricSpec", "MetricRegistry", "default_registry"]

_BOTH_TASKS = frozenset({TaskType.CLASSIFICATION, TaskType.FORECASTING})


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Everything the library needs to know about a metric other than how to compute it.

    Parameters
    ----------
    higher_is_better:
        ``True`` for accuracy-type scores, ``False`` for errors, ``None`` for metrics
        that are best at a specific ``target`` and worse in either direction.
    lower_bound, upper_bound:
        Known bounds, used for min-max normalisation. ``None`` where unbounded.
    target:
        The ideal value — where the metric is as good as it gets. Distinct from
        ``upper_bound``, which is only the bound min-max normalisation divides by.
    neutral:
        The value at which the metric says *nothing changed*, for the ratio metrics
        where that point is not the ideal one. A corruption error of 1.0 means this
        model degrades exactly like its reference; an error ratio of 1.0 means the
        perturbation cost nothing. Neither is ideal — the ideal is 0 in both cases — and
        conflating the two is how ``flip_rate`` came to declare that flipping every
        prediction was its target value. The distinction is the same one D19 drew
        between ``target`` and ``upper_bound``, one level down.
    scale:
        Characteristic magnitude, used when normalising an unbounded metric.
    tasks:
        Which task types the metric is defined for. The flip-rate family is
        classification-only; PICP and CRPS are forecasting-only.
    access:
        The least capable model access that can produce this metric.
    requires_baseline:
        Whether the metric is defined relative to a baseline model. True for the
        corruption-error and flip-rate families (survey §5.1.3).
    paper_reference:
        Where the metric is defined, e.g. ``"survey §5.1.3 Eq. 19"``. Machine-checkable
        provenance: since the survey is ours, this is what stops implementation and
        paper drifting apart.
    """

    higher_is_better: bool | None
    lower_bound: float | None = None
    upper_bound: float | None = None
    target: float | None = None
    neutral: float | None = None
    scale: float | None = None
    tasks: frozenset[TaskType] = _BOTH_TASKS
    access: AccessLevel = AccessLevel.BLACK_BOX
    requires_baseline: bool = False
    paper_reference: str | None = None

    def __post_init__(self) -> None:
        if self.higher_is_better is None and self.target is None:
            raise ValueError(
                "A metric with no direction (higher_is_better=None) must declare a "
                "target value, otherwise it cannot be normalised or ranked."
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError(
                f"lower_bound ({self.lower_bound}) must be below "
                f"upper_bound ({self.upper_bound})."
            )
        if self.scale is not None and self.scale <= 0:
            raise ValueError("scale must be positive.")
        # A directed metric's ideal value is, by definition, its good bound: if the
        # metric is monotone in goodness then nothing interior can be better than the
        # end. So a declared target must agree with the bounds. Without this check the
        # registry accepted ``flip_rate`` declaring ``target=1.0`` — that every
        # prediction changing class under perturbation was the ideal outcome — and
        # nothing complained, because until the report layer began putting ``target`` on
        # a metric card, nothing read the field for a directed metric at all.
        if self.higher_is_better is not None and self.target is not None:
            good_end = self.upper_bound if self.higher_is_better else self.lower_bound
            bad_end = self.lower_bound if self.higher_is_better else self.upper_bound
            wrong_side = bad_end is not None and (
                self.target <= bad_end if self.higher_is_better else self.target >= bad_end
            )
            not_at_the_end = good_end is not None and self.target != good_end
            if wrong_side or not_at_the_end:
                direction = "higher" if self.higher_is_better else "lower"
                raise ValueError(
                    f"target={self.target} disagrees with the bounds "
                    f"({self.lower_bound}..{self.upper_bound}) for a metric where "
                    f"{direction} is better: a monotone metric's ideal value is its "
                    f"good bound. If {self.target} is where the metric says nothing "
                    f"changed rather than where it is ideal, it belongs in `neutral`."
                )

    def applies_to(self, task: TaskType) -> bool:
        return task in self.tasks


class MetricRegistry:
    """Name -> :class:`MetricSpec`.

    The registry is what lets the rest of the library stay generic: normalisation,
    report rules and the runner's up-front capability check all read from here rather
    than special-casing metric names.
    """

    def __init__(self, *, with_defaults: bool = True) -> None:
        self._metrics: dict[str, MetricSpec] = {}
        if with_defaults:
            self._register_defaults()

    # -- registration ----------------------------------------------------------

    def register(self, name: str, spec: MetricSpec, *, overwrite: bool = False) -> None:
        if not overwrite and name in self._metrics:
            raise ValueError(
                f"Metric '{name}' is already registered. Pass overwrite=True to replace it."
            )
        self._metrics[name] = spec

    def register_many(
        self, specs: dict[str, MetricSpec], *, overwrite: bool = False
    ) -> None:
        for name, spec in specs.items():
            self.register(name, spec, overwrite=overwrite)

    def amend(self, name: str, **changes: object) -> MetricSpec:
        """Replace selected fields of a registered spec, returning the new spec."""
        spec = replace(self.get(name), **changes)  # type: ignore[arg-type]
        self._metrics[name] = spec
        return spec

    # -- lookup ----------------------------------------------------------------

    def get(self, name: str) -> MetricSpec:
        try:
            return self._metrics[name]
        except KeyError:
            near = [k for k in self._metrics if name.lower() in k.lower()]
            hint = f" Did you mean: {', '.join(sorted(near))}?" if near else ""
            raise KeyError(f"Unknown metric: '{name}'.{hint}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def for_task(self, task: TaskType) -> list[str]:
        return sorted(n for n, s in self._metrics.items() if s.applies_to(task))

    def requiring_baseline(self) -> list[str]:
        return sorted(n for n, s in self._metrics.items() if s.requires_baseline)

    def unavailable_at(self, access: AccessLevel) -> list[str]:
        """Metrics a handler at ``access`` cannot compute.

        The runner calls this once, up front, so the user is told what will be
        skipped before a long evaluation starts rather than after it fails.
        """
        return sorted(n for n, s in self._metrics.items() if not access.satisfies(s.access))

    # -- defaults --------------------------------------------------------------

    def _register_defaults(self) -> None:
        clf = frozenset({TaskType.CLASSIFICATION})
        fc = frozenset({TaskType.FORECASTING})

        # --- accuracy-type scores, bounded ------------------------------------
        for name in ("accuracy", "balanced_accuracy", "f1", "precision", "recall"):
            self.register(
                name,
                MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0, tasks=clf),
            )
        # AUC is not a label metric: it needs scores, so a handler that returns only
        # predicted classes cannot produce it. Registered alongside the others in an
        # earlier revision, which had it claiming BLACK_BOX access.
        self.register(
            "auc",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC),
        )
        self.register(
            "matthews_corrcoef",
            MetricSpec(higher_is_better=True, lower_bound=-1.0, upper_bound=1.0, tasks=clf),
        )
        self.register("r2", MetricSpec(higher_is_better=True, upper_bound=1.0, tasks=fc))

        # --- error-type scores -------------------------------------------------
        self.register("rmse", MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc))
        self.register("mae", MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc))
        self.register("mse", MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc))
        self.register(
            "mean_signed_deviation",
            MetricSpec(
                higher_is_better=None,
                target=0.0,
                tasks=fc,
                paper_reference="repo-A extension; see themis_valencia custom_accuracy_metrics",
            ),
        )
        for name in ("weighted_mae", "weighted_mse"):
            self.register(
                name,
                MetricSpec(
                    higher_is_better=False,
                    lower_bound=0.0,
                    tasks=fc,
                    paper_reference=(
                        "repo-A extension; see themis_valencia custom_accuracy_metrics"
                    ),
                ),
            )
        self.register("log_likelihood", MetricSpec(higher_is_better=True))

        # --- survey §5.1.1, domain-level degradation ---------------------------
        self.register(
            "macro_error",
            MetricSpec(higher_is_better=False, lower_bound=0.0,
                       paper_reference="survey §5.1.1 Eq. 2"),
        )
        self.register(
            "accuracy_drop",
            MetricSpec(higher_is_better=False, paper_reference="survey §5.1.1 Eq. 3"),
        )
        self.register(
            "relative_accuracy_drop",
            MetricSpec(higher_is_better=False, paper_reference="survey §5.1.1 Eq. 4"),
        )
        self.register(
            "error_ratio",
            MetricSpec(higher_is_better=False, lower_bound=0.0, target=0.0, neutral=1.0,
                       paper_reference="survey §5.1.1 Eq. 5"),
        )

        # --- survey §5.1.2, subgroup ------------------------------------------
        self.register(
            "worst_group_accuracy",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       paper_reference="survey §5.1.2 Eq. 6"),
        )
        self.register(
            "accuracy_gap",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       paper_reference="survey §5.1.2 Eq. 7"),
        )
        self.register(
            "accuracy_variance",
            MetricSpec(higher_is_better=False, lower_bound=0.0,
                       paper_reference="survey §5.1.2 Eq. 8"),
        )
        self.register(
            "group_disparity",
            MetricSpec(higher_is_better=False, lower_bound=0.0,
                       paper_reference="survey §5.1.2 Eq. 9"),
        )
        self.register(
            "equal_weight_risk",
            MetricSpec(higher_is_better=False, lower_bound=0.0,
                       paper_reference="survey §5.1.2 Eq. 10"),
        )

        # --- survey §5.1.3, synthetic perturbations ----------------------------
        self.register(
            "relative_robustness",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       target=1.0, neutral=1.0,
                       paper_reference="survey §5.1.3 Eq. 11"),
        )
        self.register(
            "total_relative_robustness",
            MetricSpec(higher_is_better=True, lower_bound=0.0,
                       paper_reference="survey §5.1.3 Eq. 12"),
        )
        self.register(
            "relative_error_ratio",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       target=1.0, neutral=1.0,
                       paper_reference="survey §5.1.3 Eq. 13"),
        )
        self.register(
            "error_based_robustness",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       target=1.0, neutral=1.0,
                       paper_reference="survey §5.1.3 Eq. 14"),
        )
        self.register(
            "total_error_based_robustness",
            MetricSpec(higher_is_better=True, lower_bound=0.0,
                       paper_reference="survey §5.1.3 Eq. 15"),
        )
        self.register(
            "error_inflation",
            MetricSpec(higher_is_better=False, lower_bound=0.0, target=0.0,
                       paper_reference="survey §5.1.3 Eq. 16"),
        )
        self.register(
            "relative_error_inflation",
            MetricSpec(higher_is_better=False, target=0.0, requires_baseline=True,
                       paper_reference="survey §5.1.3 Eq. 17"),
        )
        self.register(
            "mean_relative_error_inflation",
            MetricSpec(higher_is_better=False, target=0.0, requires_baseline=True,
                       paper_reference="survey §5.1.3 Eq. 18"),
        )
        self.register(
            "corruption_error",
            MetricSpec(higher_is_better=False, lower_bound=0.0, target=0.0, neutral=1.0,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 19"),
        )
        self.register(
            "mean_corruption_error",
            MetricSpec(higher_is_better=False, lower_bound=0.0, target=0.0, neutral=1.0,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 20"),
        )
        self.register(
            "relative_corruption_error",
            MetricSpec(higher_is_better=False, target=0.0, neutral=1.0,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 21"),
        )
        self.register(
            "mean_relative_corruption_error",
            MetricSpec(higher_is_better=False, target=0.0, neutral=1.0,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 22"),
        )
        self.register(
            "flip_probability",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0, tasks=clf,
                       paper_reference="survey §5.1.3 Eq. 23"),
        )
        self.register(
            "flip_rate",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       target=0.0, tasks=clf,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 24"),
        )
        self.register(
            "mean_flip_rate",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       target=0.0, tasks=clf,
                       requires_baseline=True, paper_reference="survey §5.1.3 Eq. 25"),
        )

        # --- survey §5.1.4, composite ------------------------------------------
        self.register(
            "unified_robustness_metric",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0, tasks=clf,
                       paper_reference="survey §5.1.4 Eq. 26"),
        )
        self.register(
            "composite_reliability_score",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       paper_reference="survey §5.1.4 Eq. 27"),
        )

        # --- survey §5.5, detector evaluation -----------------------------------
        # Kept as general detector-evaluation metrics rather than OOD-specific ones
        # (docs/DECISIONS.md, D20): they score any binary detector's scores.
        self.register(
            "auroc",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.5 Eq. 46"),
        )
        self.register(
            "auprc",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.5 Eq. 47"),
        )
        self.register(
            "fpr_at_tpr",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.5"),
        )
        self.register(
            "autc",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.5 Eq. 48"),
        )
        self.register(
            "weighted_autc",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.5 Eq. 49"),
        )

        # --- survey §5.4, calibration under shift --------------------------------
        # All of these need more than a predicted label, so they carry
        # access=PROBABILISTIC and the runner reports them as skipped for a handler
        # that returns labels only.
        self.register(
            "expected_calibration_error",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=1.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC,
                       paper_reference="survey §5.4 Eq. 42"),
        )
        self.register(
            "brier_score",
            MetricSpec(higher_is_better=False, lower_bound=0.0, upper_bound=2.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4"),
        )
        self.register(
            "negative_log_likelihood",
            MetricSpec(higher_is_better=False, lower_bound=0.0, scale=1.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4"),
        )
        self.register(
            "predictive_entropy",
            MetricSpec(higher_is_better=None, target=0.0, lower_bound=0.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC,
                       paper_reference="repo-A extension; from themis_media"),
        )
        # PICP is best *at the nominal level*, not high: an interval covering everything
        # is perfectly covering and useless. target is set per run by the report layer
        # from the nominal level; 0.9 is the common default and the one recorded here.
        self.register(
            "prediction_interval_coverage",
            MetricSpec(higher_is_better=None, lower_bound=0.0, upper_bound=1.0,
                       target=0.9, tasks=fc, access=AccessLevel.PROBABILISTIC,
                       paper_reference="survey §5.4 Eq. 43"),
        )
        self.register(
            "mean_prediction_interval_width",
            MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4"),
        )
        self.register(
            "interval_score",
            MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc,
                       access=AccessLevel.PROBABILISTIC,
                       paper_reference="repo-A extension; Gneiting & Raftery interval score"),
        )
        self.register(
            "crps",
            MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4 Eq. 44"),
        )
        self.register(
            "crps_gaussian",
            MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4"),
        )
        self.register(
            "energy_score",
            MetricSpec(higher_is_better=False, lower_bound=0.0, tasks=fc,
                       access=AccessLevel.PROBABILISTIC, paper_reference="survey §5.4 Eq. 45"),
        )

        # --- survey §5.3, distribution distance ----------------------------------
        # Not robustness scores (§5.3 is explicit) but context for one. They carry no
        # direction at all: a large distance is neither good nor bad, it is the
        # denominator a degradation is read against. higher_is_better=None with
        # target=0.0 records "0 means the data did not move", which is the only
        # meaningful anchor; the report layer refuses to rank models by them.
        for name, reference in (
            ("wasserstein_distance", "survey §5.3 Eq. 38"),
            ("kl_divergence", "survey §5.3 Eq. 39"),
            ("maximum_mean_discrepancy", "survey §5.3 Eqs. 40-41"),
        ):
            self.register(
                name,
                MetricSpec(higher_is_better=None, target=0.0, paper_reference=reference),
            )

        # --- repo-A extensions, not in the survey --------------------------------
        self.register(
            "absolute_relative_robustness",
            MetricSpec(higher_is_better=False, lower_bound=0.0, target=0.0,
                       paper_reference="repo-A extension: |1 - Eq. 11|"),
        )
        self.register(
            "confidence_retention",
            MetricSpec(higher_is_better=True, lower_bound=0.0, upper_bound=1.0,
                       target=1.0, neutral=1.0, tasks=clf,
                       access=AccessLevel.PROBABILISTIC,
                       paper_reference="repo-A extension; from themis_media"),
        )


def default_registry() -> MetricRegistry:
    """A fresh registry with the built-in metrics.

    Returns a new instance each call, so a caller registering custom metrics does not
    mutate anyone else's registry.
    """
    return MetricRegistry()
