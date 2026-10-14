"""Survey §8's reporting standards, as ten executable checks.

The survey's reporting section is a list of things a robustness claim must not do, and
most of them are mechanically checkable. That is the point of this module: enforcing
them is what makes this package a reporting instrument rather than a metric collection,
because a metric collection cannot tell you that the number you are about to publish is
meaningless on its own.

The ten, in the survey's order:

1. a relative-drop metric stated without absolute performance beside it — a 2% drop from
   0.91 and a 2% drop from 0.52 are not the same claim;
2. ``mCE`` without ``mRCE`` — the plain corruption error penalises a model twice for
   being weak when clean;
3. an aggregate over a perturbation set without naming the set — sum-based aggregation
   scales with its size, so the number is comparable only within a fixed one;
4. a composite without its components and weights — a composite is a weighting decision
   wearing the clothes of a measurement;
5. one severity in place of a severity curve — a single severity is one point on a curve
   whose shape is the actual finding;
6. an aggregate with no per-group or per-class breakdown — the aggregate is an average
   over groups whose experiences may differ substantially;
7. calibration measured only on clean data — calibration is what degrades first under
   shift, and clean ECE says nothing about shifted ECE;
8. a detector or operating-point metric with no threshold named — the metric is a
   property of the threshold, not of the model;
9. a distribution distance presented as a robustness score — the distance says the data
   moved, not that the model broke;
10. ``PICP`` without ``MPIW`` — coverage can be bought with arbitrarily wide intervals.

Each check reads the *serialised* report — the dictionary from
:meth:`~themis_assess.report.bundle.AssessmentReport.to_dict` — rather than the result
objects. That is deliberate. The dictionary is what gets handed on to a renderer, a
database or a language model, so it is the thing whose completeness matters; and it means
a report assembled by hand from loose numbers is checked on exactly the same terms as one
built by the evaluators.

The checks warn; they do not refuse. A library that raised here would be a library that
stopped an honest exploratory run, and the survey's concern is with what gets *published*,
which is not something this code can observe. So the issues land in the report's findings
with ``area="reporting"``, next to the findings about the model, and a reader who ignores
them is doing so knowingly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from themis_assess.report.findings import Finding, rank_findings

__all__ = [
    "check_report",
    "RELATIVE_FAMILY",
    "COMPOSITE_FAMILY",
    "DISTANCE_FAMILY",
    "SET_AGGREGATED",
]

#: Metrics that are meaningless without the absolute performance they are relative to.
RELATIVE_FAMILY = frozenset(
    {
        "accuracy_drop",
        "relative_accuracy_drop",
        "error_inflation",
        "relative_error_inflation",
        "mean_relative_error_inflation",
        "error_ratio",
        "relative_error_ratio",
        "relative_robustness",
        "absolute_relative_robustness",
        "total_relative_robustness",
        "error_based_robustness",
        "total_error_based_robustness",
        "corruption_error",
        "mean_corruption_error",
        "relative_corruption_error",
        "mean_relative_corruption_error",
    }
)

#: Metrics that combine several others behind one number.
COMPOSITE_FAMILY = frozenset(
    {"composite_reliability_score", "unified_robustness_metric", "equal_weight_risk"}
)

#: Distribution distances. Context for a robustness claim, never a robustness score.
DISTANCE_FAMILY = frozenset(
    {"wasserstein_distance", "kl_divergence", "maximum_mean_discrepancy"}
)

#: Metrics whose magnitude depends on how many perturbations were aggregated over.
SET_AGGREGATED = frozenset(
    {
        "total_relative_robustness",
        "total_error_based_robustness",
        "mean_corruption_error",
        "mean_relative_corruption_error",
        "mean_relative_error_inflation",
        "mean_flip_rate",
        "unified_robustness_metric",
    }
)

_CALIBRATION_METRICS = frozenset(
    {"expected_calibration_error", "brier_score", "negative_log_likelihood"}
)

_THRESHOLD_METRICS = frozenset({"fpr_at_tpr"})


def check_report(payload: Mapping[str, Any]) -> list[Finding]:
    """Every §8 issue in a serialised report, most severe first."""
    out: list[Finding] = []
    for check in (
        _relative_without_absolute,
        _mce_without_mrce,
        _aggregate_without_perturbation_set,
        _composite_without_components,
        _single_severity,
        _aggregate_without_breakdown,
        _calibration_on_clean_only,
        _threshold_metric_without_threshold,
        _distance_as_robustness,
        _picp_without_mpiw,
    ):
        out.extend(check(payload))
    return rank_findings(out)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _robustness(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    section = payload.get("robustness")
    return section if isinstance(section, Mapping) else {}


def _accuracy(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    section = payload.get("accuracy")
    return section if isinstance(section, Mapping) else {}


def _scored(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    scored = _robustness(payload).get("scored")
    return scored if isinstance(scored, Mapping) else {}


def _overall_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    overall = _accuracy(payload).get("overall")
    if isinstance(overall, Mapping):
        metrics = overall.get("metrics")
        if isinstance(metrics, Mapping):
            return metrics
    return {}


def _issue(
    rule: str,
    headline: str,
    detail: str,
    *,
    severity: str = "warning",
    evidence: Mapping[str, Any] | None = None,
    suggestion: str = "",
) -> Finding:
    return Finding(
        severity=severity,  # type: ignore[arg-type]
        rule=f"reporting.{rule}",
        area="reporting",
        headline=headline,
        detail=detail,
        evidence=dict(evidence or {}),
        suggestion=suggestion,
    )


# --------------------------------------------------------------------------
# the ten
# --------------------------------------------------------------------------


def _relative_without_absolute(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.1 — a relative drop with nothing to be relative to."""
    scored = _scored(payload)
    relative = sorted(set(scored) & RELATIVE_FAMILY)
    if not relative:
        return []
    clean = _robustness(payload).get("clean")
    if isinstance(clean, (int, float)) and not isinstance(clean, bool):
        return []
    return [
        _issue(
            "relative_without_absolute",
            "relative degradation is reported with no absolute performance",
            "A relative metric says how much was lost and not how much there was to "
            "lose. Losing 2% of 0.91 accuracy and 2% of 0.52 are the same number and "
            "not remotely the same claim, which is why survey §8 asks for the absolute "
            "clean and perturbed values beside every relative one.",
            severity="serious",
            evidence={"relative_metrics": relative, "clean": clean},
            suggestion=(
                "Report the clean and perturbed values of the underlying accuracy metric "
                "in the same table. A report built from a RobustnessAssessment carries "
                "them automatically."
            ),
        )
    ]


def _mce_without_mrce(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.2 — the corruption error without its relative counterpart."""
    scored = _scored(payload)
    pairs = [
        ("mean_corruption_error", "mean_relative_corruption_error"),
        ("corruption_error", "relative_corruption_error"),
    ]
    out: list[Finding] = []
    for plain, relative in pairs:
        if plain in scored and relative not in scored:
            out.append(
                _issue(
                    "corruption_error_without_relative",
                    f"{plain} is reported without {relative}",
                    f"{plain} divides this model's perturbed error by a baseline's, so a "
                    f"model that was already worse when clean is penalised twice — once "
                    f"for its clean error and again for carrying it into the perturbed "
                    f"error. {relative} takes the ratio of the *added* error instead, and "
                    f"survey §8 asks for the pair.",
                    evidence={"present": plain, "missing": relative},
                    suggestion=(
                        f"Add {relative!r} to the robustness metrics. It needs the same "
                        f"baseline sweep, so it costs no extra model calls."
                    ),
                )
            )
    return out


def _aggregate_without_perturbation_set(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.3 — an aggregate whose magnitude depends on an unnamed set."""
    scored = _scored(payload)
    out: list[Finding] = []
    for name in sorted(set(scored) & SET_AGGREGATED):
        entry = scored[name]
        named = entry.get("perturbation_set") if isinstance(entry, Mapping) else None
        if named:
            continue
        out.append(
            _issue(
                "aggregate_without_perturbation_set",
                f"{name} is reported without naming the perturbation set",
                f"{name} aggregates over every perturbation in the set, so its magnitude "
                f"grows with the size of that set. Two runs of {name} over different sets "
                f"are different quantities under one name, and comparing them is "
                f"meaningless — which is why survey §8 requires the set to travel with "
                f"the number.",
                evidence={"metric": name},
                suggestion=(
                    "Record perturbation_set on the RobustnessResult. The evaluators fill "
                    "it from the sweep; a hand-built result has to state it."
                ),
            )
        )
    return out


def _composite_without_components(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.4 — a composite standing alone."""
    scored = _scored(payload)
    out: list[Finding] = []
    for name in sorted(set(scored) & COMPOSITE_FAMILY):
        entry = scored[name] if isinstance(scored[name], Mapping) else {}
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        has_components = bool(metadata.get("components")) and bool(metadata.get("weights"))
        others = sorted(set(scored) - COMPOSITE_FAMILY)
        if has_components or others:
            continue
        out.append(
            _issue(
                "composite_without_components",
                f"{name} is the only robustness number reported",
                f"{name} folds accuracy, degradation and calibration into one value using "
                f"a weighting nobody in the deployment chose. Reported alone it is a "
                f"judgement presented as a measurement: a model can reach the same "
                f"composite by being robust and miscalibrated or the reverse, and those "
                f"are different models.",
                severity="serious",
                evidence={"composite": name, "other_metrics": others},
                suggestion=(
                    "Report the components beside it and state the weights. The composite "
                    "is a screening device for ranking candidates, not the basis of an "
                    "acceptance decision."
                ),
            )
        )
    return out


def _single_severity(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.5 — one severity instead of a curve."""
    robustness = _robustness(payload)
    if not robustness:
        return []
    severities = robustness.get("severities") or []
    if len(severities) > 1:
        return []
    curves = robustness.get("curves")
    if isinstance(curves, Mapping):
        widest = max((len(c) for c in curves.values() if isinstance(c, Mapping)), default=0)
        if widest > 1:
            return []
    return [
        _issue(
            "single_severity",
            "the sweep covers a single severity, so there is no degradation curve",
            "One severity is one point. Whether the model degrades gracefully or falls "
            "off a cliff between severity 2 and 3 is the finding a deployment needs, and "
            "it is invisible from a single point — the same value at severity 3 is "
            "reassuring on a flat curve and alarming on a steep one.",
            evidence={"severities": list(severities)},
            suggestion=(
                "Sweep the full severity axis (1-5 by default). The cost is linear in the "
                "number of severities and the curve shape is usually the result."
            ),
        )
    ]


def _aggregate_without_breakdown(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.6 — an aggregate with nothing broken out of it."""
    accuracy = _accuracy(payload)
    if not accuracy:
        return []
    if accuracy.get("subgroups") or accuracy.get("per_class"):
        return []
    return [
        _issue(
            "aggregate_without_breakdown",
            "accuracy is reported as an aggregate only",
            "An aggregate is an average over whatever groups are in the data, and it is "
            "high exactly when the majority group is well served. Survey §8 asks for the "
            "breakdown alongside it because the aggregate cannot show a group the model "
            "fails — a group at 0.55 inside an overall 0.90 leaves no trace in the "
            "overall number.",
            evidence={"metrics": sorted(_overall_metrics(payload))},
            suggestion=(
                "Pass subgroups= to the AccuracyEvaluator, or per_class=True for a "
                "classifier. Both are one argument and neither needs another model call."
            ),
        )
    ]


def _calibration_on_clean_only(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.7 — calibration measured where it does not matter."""
    clean_calibration = sorted(set(_overall_metrics(payload)) & _CALIBRATION_METRICS)
    if not clean_calibration:
        return []
    robustness = _robustness(payload)
    if not robustness:
        return []
    swept = robustness.get("accuracy_metric")
    swept_metrics = set(robustness.get("plan", {}).get("metrics", ()) or ())
    if swept in _CALIBRATION_METRICS or (swept_metrics & _CALIBRATION_METRICS):
        return []
    if "confidence_retention" in _scored(payload):
        return []
    return [
        _issue(
            "calibration_on_clean_only",
            "calibration was measured on clean data only",
            "Calibration is usually the first thing to go under shift, and it goes "
            "silently: a model whose accuracy falls while its stated confidence does not "
            "is the dangerous failure, because nothing downstream can tell that anything "
            "changed. Clean ECE does not bound shifted ECE.",
            evidence={
                "clean_calibration_metrics": clean_calibration,
                "swept_metric": swept,
            },
            suggestion=(
                "Sweep a calibration metric as well, or score confidence_retention, which "
                "asks the same question with one number: how much of its confidence did "
                "the model keep when it stopped being right?"
            ),
        )
    ]


def _threshold_metric_without_threshold(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.8 — an operating-point metric with no operating point recorded."""
    overall = _accuracy(payload).get("overall")
    metrics = _overall_metrics(payload)
    present = sorted(set(metrics) & _THRESHOLD_METRICS)
    if not present:
        return []
    metadata = overall.get("metadata") if isinstance(overall, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    recorded = any(
        key in metadata for key in ("threshold", "target_tpr", "operating_point", "tpr")
    )
    if recorded:
        return []
    return [
        _issue(
            "threshold_metric_without_threshold",
            f"{', '.join(present)} is reported with no operating point named",
            "This metric is a property of a threshold, not of a model: the same model "
            "gives a different value at every operating point. Without the threshold the "
            "number cannot be reproduced and cannot be compared with anyone else's.",
            evidence={"metrics": present, "metadata_keys": sorted(metadata)},
            suggestion=(
                "Record the target true-positive rate (or the threshold) in the "
                "evaluation's metadata, and state it wherever the metric appears."
            ),
        )
    ]


def _distance_as_robustness(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.9 — a distance in the robustness column."""
    misplaced = sorted(set(_scored(payload)) & DISTANCE_FAMILY)
    if not misplaced:
        return []
    return [
        _issue(
            "distance_as_robustness",
            f"{', '.join(misplaced)} appears among the robustness scores",
            "A distribution distance measures how far the data moved. It says nothing "
            "about whether the model still works — a large shift in a feature the model "
            "ignores costs nothing, and a small shift in the feature it leans on can be "
            "fatal. Presenting the distance as a robustness score invites the reader to "
            "conclude the opposite of what was measured.",
            severity="serious",
            evidence={"metrics": misplaced},
            suggestion=(
                "Move these to the report's 'distances' section, which exists for them: "
                "AssessmentReport.add_distance() keeps them as context beside the "
                "robustness scores rather than among them."
            ),
        )
    ]


def _picp_without_mpiw(payload: Mapping[str, Any]) -> list[Finding]:
    """§8.10 — coverage without width."""
    metrics = set(_overall_metrics(payload))
    for section in _robustness(payload).get("scored", {}).values() or ():
        if isinstance(section, Mapping):
            metrics.add(section.get("accuracy_metric"))
    if "prediction_interval_coverage" not in metrics:
        return []
    if "mean_prediction_interval_width" in metrics or "interval_score" in metrics:
        return []
    return [
        _issue(
            "coverage_without_width",
            "interval coverage is reported without interval width",
            "Coverage alone is trivially satisfiable: an interval from minus infinity to "
            "infinity covers everything. The width is what makes the coverage a claim, "
            "and survey §8 requires the pair.",
            evidence={"present": "prediction_interval_coverage"},
            suggestion=(
                "Add mean_prediction_interval_width, or interval_score, which combines "
                "the two into one proper scoring rule."
            ),
        )
    ]
