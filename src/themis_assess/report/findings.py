"""Turning results into findings: what is wrong with this model, ranked.

A results dictionary answers "what are the numbers". A *finding* answers "what should
someone do about them", and the gap between the two is where an assessment usually stalls:
the numbers are all there and nobody reads far enough to notice that the worst subgroup is
eleven points behind the aggregate.

Each finding carries a severity, the evidence it rests on, and a suggested direction. They
are deliberately mechanical — threshold comparisons over the result objects, not judgement
— because a mechanical finding is auditable: every one names the numbers it came from, so
a reader can disagree with it by checking them.

Why this exists in a library rather than being left to the reader
-----------------------------------------------------------------

Findings are the structure a summarising language model needs. Handing a model the raw
results dictionary asks it to notice a subgroup gap among several hundred numbers, and
what it notices will vary between runs. Handing it findings — each with a severity, a
metric, the values, and the rule that fired — leaves it the job it is actually good at:
turning "worst-group accuracy 0.68 against aggregate 0.79, chi-square p = 0.004" into a
sentence a clinician can read, and grouping several findings into a theme.

So the contract here is that :meth:`Finding.as_dict` is stable and self-describing. Nothing
downstream needs to know how a finding was derived to render it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "Severity",
    "Finding",
    "findings_from_accuracy",
    "findings_from_robustness",
    "findings_from_sensitivity",
    "findings_from_attribution",
    "findings_from_groups",
    "rank_findings",
]

Severity = Literal["critical", "serious", "warning", "info"]

_ORDER: dict[str, int] = {"critical": 0, "serious": 1, "warning": 2, "info": 3}


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing worth someone's attention, with the evidence attached.

    Parameters
    ----------
    severity:
        ``critical`` — a stated requirement is not met, or the model is unfit as
        assessed. ``serious`` — a substantial weakness with no requirement attached to
        it. ``warning`` — something that needs a human decision, including the absence of
        a requirement where one belongs. ``info`` — worth recording, not worth acting on.
    rule:
        Which check produced this, as a stable identifier. The audit trail: a reader who
        disagrees with a finding can look up the rule and see the threshold.
    evidence:
        The numbers the finding rests on. Every value here appears somewhere in the
        results, so nothing is asserted that cannot be checked.
    suggestion:
        A direction, not an instruction. The library does not know the deployment, so it
        says what kind of thing would address the finding rather than what to do.
    """

    severity: Severity
    rule: str
    headline: str
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    area: str = "accuracy"

    def as_dict(self) -> dict[str, Any]:
        """The serialisable form. Stable — downstream renderers depend on these keys."""
        return {
            "severity": self.severity,
            "rule": self.rule,
            "area": self.area,
            "headline": self.headline,
            "detail": self.detail,
            "evidence": _plain(self.evidence),
            "suggestion": self.suggestion,
        }

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.headline} — {self.detail}"


def _plain(value: Any) -> Any:
    """Make a value JSON-serialisable without losing precision.

    numpy scalars and arrays are the reason this exists: ``json.dumps`` refuses
    ``np.float64``, and a results bundle full of them is unusable as a handover format.
    """
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _gap_fraction(metric: str, gap: float, aggregate: float | None) -> float | None:
    """The subgroup gap as a fraction, so one threshold can serve every metric.

    A raw gap is not comparable between metrics, and thresholding it directly is a real
    mistake rather than an approximation: 0.05 is a substantial gap in accuracy and
    nothing at all in a negative log-likelihood, which is unbounded above. So the gap is
    expressed relative to something metric-specific before it meets a threshold —

    * the width of the metric's range, where the registry knows both bounds (a gap of
      0.15 in Matthews' correlation, which runs from -1 to 1, is 7.5% of the range and
      not 15%);
    * the aggregate value otherwise, giving a proportional gap.

    Returns ``None`` when neither is available — an unbounded metric whose aggregate is
    zero has no scale to compare against, and inventing one would be worse than
    declining to report.
    """
    from themis_assess.metrics.registry import default_registry

    try:
        spec = default_registry().get(metric)
    except KeyError:
        spec = None
    if spec is not None and spec.lower_bound is not None and spec.upper_bound is not None:
        width = spec.upper_bound - spec.lower_bound
        if width > 0:
            return float(gap) / width
    for denominator in (
        None if aggregate is None else abs(float(aggregate)),
        None if spec is None else spec.scale,
    ):
        if denominator and np.isfinite(denominator) and denominator > 0:
            return float(gap) / float(denominator)
    return None


def rank_findings(findings: Sequence[Finding]) -> list[Finding]:
    """Most severe first, stable within a severity.

    Stable rather than sorted-by-magnitude within a band on purpose: two critical
    findings are both critical, and ordering them by how far past the threshold they are
    invites reading the ordering as a priority it does not carry.
    """
    return sorted(findings, key=lambda f: _ORDER.get(f.severity, 9))


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------


def findings_from_accuracy(
    result: Any,
    *,
    subgroup_gap_threshold: float = 0.05,
    class_error_threshold: float = 0.30,
    calibration_threshold: float = 0.10,
) -> list[Finding]:
    """Findings from an :class:`~themis_assess.assessment.accuracy.AccuracyResult`."""
    out: list[Finding] = []

    # -- acceptance criteria ------------------------------------------------
    if result.acceptance is None:
        out.append(
            Finding(
                severity="warning",
                rule="accuracy.no_criteria",
                area="accuracy",
                headline="No acceptance criteria were supplied",
                detail=(
                    "Every metric was computed but none was checked against a "
                    "requirement, so this assessment reports numbers and not a verdict. "
                    "Whether 0.74 accuracy is acceptable is a property of the "
                    "deployment, which the library cannot know."
                ),
                evidence={"metrics": sorted(result.overall.metrics)},
                suggestion=(
                    "Agree thresholds with whoever owns the deployment and pass them as "
                    "AcceptanceCriteria. themis_50_criteria() is an example to edit, not "
                    "a default to adopt."
                ),
            )
        )
    else:
        for verdict in result.acceptance.failures:
            out.append(
                Finding(
                    severity="critical" if verdict.requirement.critical else "serious",
                    rule="accuracy.criterion_failed",
                    area="accuracy",
                    headline=f"{verdict.metric} is below its requirement",
                    detail=(
                        f"{verdict.metric} is {verdict.value:.4f} against a required "
                        f"{verdict.requirement.acceptable:g} — short by "
                        f"{abs(verdict.margin):.4f}."
                    ),
                    evidence={
                        "metric": verdict.metric,
                        "value": verdict.value,
                        "acceptable": verdict.requirement.acceptable,
                        "margin": verdict.margin,
                        "rationale": verdict.requirement.rationale,
                    },
                    suggestion=(
                        "Either the model needs work or the requirement needs revisiting;"
                        " the rationale recorded on the requirement is where to start."
                    ),
                )
            )
        for metric in result.acceptance.provenance.get("not_measured", ()):
            out.append(
                Finding(
                    severity="warning",
                    rule="accuracy.requirement_not_measured",
                    area="accuracy",
                    headline=f"{metric} has a requirement but was not measured",
                    detail=(
                        f"A threshold is set for {metric} and the assessment did not "
                        f"compute it, so the overall verdict does not cover it. This is "
                        f"a gap in the evaluation rather than a fault in the model."
                    ),
                    evidence={"metric": metric},
                    suggestion=(
                        "Check the assessment plan: the metric is probably unavailable "
                        "at the handler's declared access level."
                    ),
                )
            )

    # -- subgroups ----------------------------------------------------------
    # Grouped by which subgroup is worst rather than one finding per metric. Ten metrics
    # showing a gap is usually one fact — a single group the model serves badly — and
    # reporting it ten times buries the rest of the assessment under repetitions of it.
    gaps: dict[str, dict[str, dict[str, float]]] = {}
    orientation = result.metadata.get("registry_orientation", {})
    for metric, disparity in result.disparity.items():
        gap = disparity.get("accuracy_gap")
        if gap is None:
            continue
        fraction = _gap_fraction(metric, gap, result.overall.metrics.get(metric))
        if fraction is None or fraction < subgroup_gap_threshold:
            continue
        worst = result.worst_subgroup(
            metric, higher_is_better=bool(orientation.get(metric, True))
        )
        gaps.setdefault(str(worst), {})[metric] = {
            "gap": float(gap),
            "fraction": float(fraction),
            "aggregate": result.overall.metrics.get(metric),
            "in_subgroup": (
                result.subgroups[worst].metrics.get(metric)
                if worst in result.subgroups
                else None
            ),
        }

    for subgroup, metrics_with_gaps in sorted(gaps.items()):
        worst_fraction = max(entry["fraction"] for entry in metrics_with_gaps.values())
        names = sorted(metrics_with_gaps)
        out.append(
            Finding(
                severity=(
                    "serious"
                    if worst_fraction >= 2 * subgroup_gap_threshold
                    else "warning"
                ),
                rule="accuracy.subgroup_gap",
                area="subgroups",
                headline=(
                    f"subgroup {subgroup!r} is the worst served, on "
                    f"{len(names)} of {len(result.disparity)} metric(s)"
                ),
                detail=(
                    "The widest gap is "
                    + ", ".join(
                        f"{m} {metrics_with_gaps[m]['gap']:.3f} "
                        f"({metrics_with_gaps[m]['fraction']:.0%} of its range)"
                        for m in sorted(
                            names,
                            key=lambda m: -metrics_with_gaps[m]["fraction"],
                        )[:3]
                    )
                    + ". An aggregate hides this by construction, which is why survey "
                    "§5.1.2 asks for the breakdown alongside it."
                ),
                evidence={
                    "subgroup": subgroup,
                    "metrics": {m: metrics_with_gaps[m] for m in names},
                    "n_samples": (
                        result.subgroups[subgroup].n_samples
                        if subgroup in result.subgroups
                        else None
                    ),
                },
                suggestion=(
                    "A gap of this size is worth a significance test before acting on "
                    "it — themis_assess.metrics.accuracy.significance.compare_groups "
                    "says whether it is distinguishable from sampling noise, which "
                    "matters most when the subgroup is small."
                ),
            )
        )

    # -- per class ----------------------------------------------------------
    for label, metrics in result.per_class.items():
        error_rate = metrics.get("error_rate")
        if error_rate is None or error_rate < class_error_threshold:
            continue
        out.append(
            Finding(
                severity="serious" if error_rate >= 0.5 else "warning",
                rule="accuracy.class_error_rate",
                area="classes",
                headline=f"class {label} is missed {error_rate:.1%} of the time",
                detail=(
                    f"Recall on class {label} is {metrics['recall']:.3f} over "
                    f"{int(metrics['support'])} sample(s). A high aggregate accuracy can "
                    f"sit on top of a class this weak when the class is small."
                ),
                evidence={"class": label, **{k: v for k, v in metrics.items()}},
                suggestion=(
                    "Look at the confusion matrix for where these go instead; the "
                    "top_confusions ranking puts the proportionally worst first."
                ),
            )
        )

    # -- calibration --------------------------------------------------------
    ece = result.overall.metrics.get("expected_calibration_error")
    if ece is not None and ece >= calibration_threshold:
        out.append(
            Finding(
                severity="warning",
                rule="accuracy.miscalibrated",
                area="calibration",
                headline=f"the model's confidence is off by {ece:.3f} (ECE)",
                detail=(
                    "The stated confidence does not match the observed accuracy. This "
                    "matters wherever a probability is acted on — a review threshold, a "
                    "triage queue, an expected-cost calculation — and not at all where "
                    "only the argmax is used."
                ),
                evidence={"expected_calibration_error": ece},
                suggestion=(
                    "Temperature scaling on a held-out set costs nothing and cannot "
                    "change any prediction: see calibration.fit_temperature."
                ),
            )
        )

    # -- diagnostics --------------------------------------------------------
    outliers = result.diagnostics.get("outliers")
    if (
        outliers is not None
        and outliers.p_value is not None
        and outliers.p_value < 0.05
        and outliers.difference > 0
    ):
        out.append(
            Finding(
                severity="warning",
                rule="accuracy.errors_on_outliers",
                area="diagnostics",
                headline="errors concentrate on unusual inputs",
                detail=(
                    f"The error rate is {outliers.outlier_error_rate:.1%} on the most "
                    f"unusual {outliers.n_outliers} sample(s) against "
                    f"{outliers.inlier_error_rate:.1%} elsewhere "
                    f"(chi-square p = {outliers.p_value:.4g})."
                ),
                evidence={
                    "outlier_error_rate": outliers.outlier_error_rate,
                    "inlier_error_rate": outliers.inlier_error_rate,
                    "p_value": outliers.p_value,
                    "n_outliers": outliers.n_outliers,
                },
                suggestion=(
                    "The model is already weak on the tails of its own training "
                    "distribution, so degradation under a synthetic shift is expected "
                    "and is not the first thing to fix."
                ),
            )
        )

    ambiguity = result.diagnostics.get("ambiguity")
    if ambiguity is not None:
        flagged = ambiguity.get("error_rate_flagged")
        unflagged = ambiguity.get("error_rate_unflagged")
        if flagged is not None and unflagged is not None and flagged <= unflagged * 1.1:
            out.append(
                Finding(
                    severity="info",
                    rule="accuracy.uncertainty_uninformative",
                    area="calibration",
                    headline="the model's uncertainty does not identify its errors",
                    detail=(
                        f"The {ambiguity['flagged_fraction']:.0%} of predictions the "
                        f"model is least sure about are wrong {flagged:.1%} of the time, "
                        f"against {unflagged:.1%} for the rest. A review queue built on "
                        f"this flag would be sorting at close to random."
                    ),
                    evidence={
                        "error_rate_flagged": flagged,
                        "error_rate_unflagged": unflagged,
                        "flagged_fraction": ambiguity["flagged_fraction"],
                    },
                    suggestion=(
                        "Do not staff a review queue on this signal as it stands. "
                        "Calibration may help; a different uncertainty estimate may be "
                        "needed."
                    ),
                )
            )

    return rank_findings(out)


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def findings_from_robustness(
    assessment: Any,
    *,
    retention_threshold: float = 0.9,
) -> list[Finding]:
    """Findings from a :class:`~themis_assess.assessment.robustness.RobustnessAssessment`."""
    out: list[Finding] = []

    if assessment.tolerance is None:
        out.append(
            Finding(
                severity="warning",
                rule="robustness.no_tolerance",
                area="robustness",
                headline="No degradation tolerance was supplied",
                detail=(
                    "Survey §8 calls the tolerance the one element no evaluation "
                    "instrument can provide, so this is an incomplete robustness claim: "
                    "it says how much the model degrades and not whether that is "
                    "acceptable."
                ),
                evidence={"scored_metrics": assessment.metric_names},
                suggestion=(
                    "Pass RobustnessCriteria with a tolerance and a basis. Note that "
                    "the three bases disagree — the same sweep can pass on the "
                    "aggregate and fail on the worst perturbation."
                ),
            )
        )
    else:
        for verdict in assessment.tolerance.failures:
            out.append(
                Finding(
                    severity="critical" if verdict.requirement.critical else "serious",
                    rule="robustness.tolerance_exceeded",
                    area="robustness",
                    headline=f"{verdict.metric} is outside tolerance",
                    detail=(
                        f"On the {verdict.basis} basis, {verdict.metric} over "
                        f"{verdict.accuracy_metric} is {verdict.value:.4f} against a "
                        f"tolerated {verdict.requirement.acceptable:g}"
                        + (f", worst at {verdict.worst_at}" if verdict.worst_at else "")
                        + "."
                    ),
                    evidence={
                        "metric": verdict.metric,
                        "accuracy_metric": verdict.accuracy_metric,
                        "basis": verdict.basis,
                        "value": verdict.value,
                        "acceptable": verdict.requirement.acceptable,
                        "worst_at": verdict.worst_at,
                    },
                    suggestion=(
                        "The named perturbation is where to look. A sensitivity map over "
                        "it says which feature carries the damage."
                    ),
                )
            )
        for metric in assessment.tolerance.unchecked:
            out.append(
                Finding(
                    severity="warning",
                    rule="robustness.tolerance_not_checked",
                    area="robustness",
                    headline=f"the tolerance on {metric} was not checked",
                    detail=(
                        f"{metric} was not scored over "
                        f"{assessment.accuracy_metric}, so the overall tolerance verdict "
                        f"does not cover it."
                    ),
                    evidence={"metric": metric},
                    suggestion=(
                        "Score it over a metric of the right orientation — the error "
                        "family needs an error metric, the Eq. 11-12 family an accuracy."
                    ),
                )
            )

    # -- the worst cell -----------------------------------------------------
    perturbation, severity, value = assessment.worst_cell()
    clean = assessment.clean
    if clean:
        retained = value / clean
        if retained < retention_threshold:
            out.append(
                Finding(
                    severity="serious" if retained < 0.75 else "warning",
                    rule="robustness.worst_cell",
                    area="robustness",
                    headline=(
                        f"{perturbation} at severity {severity} costs "
                        f"{abs(1 - retained):.0%} of {assessment.accuracy_metric}"
                    ),
                    detail=(
                        f"Clean {assessment.accuracy_metric} is {clean:.4f} and falls to "
                        f"{value:.4f} at the worst cell. Survey §8 asks for the worst "
                        f"case beside the average precisely because an aggregate over "
                        f"severities hides it."
                    ),
                    evidence={
                        "perturbation": perturbation,
                        "severity": severity,
                        "clean": clean,
                        "worst": value,
                        "retained": retained,
                    },
                    suggestion=(
                        "Whether severity 5 is within anything the deployment will see "
                        "is a judgement the tolerance basis encodes; 'worst_cell' is the "
                        "conservative reading."
                    ),
                )
            )

    return rank_findings(out)


# --------------------------------------------------------------------------
# sensitivity and attribution
# --------------------------------------------------------------------------


def findings_from_sensitivity(
    sensitivity_map: Any, *, dominance_ratio: float = 3.0
) -> list[Finding]:
    """Findings from a :class:`~themis_assess.sensitivity.maps.SensitivityMap`."""
    out: list[Finding] = []
    ranked = sensitivity_map.most_sensitive_rows(n=len(sensitivity_map.row_labels))
    if len(ranked) < 2:
        return out

    top, second = ranked[0], ranked[1]
    if second["score"] > 0 and top["score"] / second["score"] >= dominance_ratio:
        out.append(
            Finding(
                severity="warning",
                rule="sensitivity.single_dominant_input",
                area="sensitivity",
                headline=f"the model leans overwhelmingly on {top['label']!r}",
                detail=(
                    f"Perturbing {top['label']!r} moves the model "
                    f"{top['score'] / second['score']:.1f}x more than the next most "
                    f"influential input ({second['label']!r}). A single point of failure "
                    f"in the input pipeline."
                ),
                evidence={
                    "most_sensitive": top["label"],
                    "score": top["score"],
                    "runner_up": second["label"],
                    "runner_up_score": second["score"],
                    "measure": sensitivity_map.measure,
                },
                suggestion=(
                    "Worth knowing what happens to this input in production — a sensor "
                    "outage or a schema change there takes the model with it."
                ),
            )
        )

    hottest = sensitivity_map.hottest_cell()
    out.append(
        Finding(
            severity="info",
            rule="sensitivity.hottest_cell",
            area="sensitivity",
            headline=(
                f"most sensitive cell: {hottest.get(sensitivity_map.row_axis)} at "
                f"{hottest.get(sensitivity_map.column_axis)}"
            ),
            detail=(
                f"Measured as {sensitivity_map.measure.replace('_', ' ')}, value "
                f"{hottest['value']:.4g}."
            ),
            evidence=dict(hottest) | {"measure": sensitivity_map.measure},
        )
    )
    return rank_findings(out)


def findings_from_attribution(model: Any, *, top_n: int = 3) -> list[Finding]:
    """Findings from a fitted :class:`~themis_assess.attribution.error_model.ErrorModel`."""
    fit = model.fit_report
    if fit is None:
        return []
    if not fit.is_trustworthy:
        return [
            Finding(
                severity="info",
                rule="attribution.surrogate_untrustworthy",
                area="attribution",
                headline="the error model learned nothing, so it has nothing to attribute",
                detail=(
                    f"The surrogate's {fit.gate} is {fit.gate_value:.3f}, below "
                    f"{fit.gate_threshold}. Its feature attributions would be "
                    f"indistinguishable from noise — which is a finding about the error "
                    f"being unstructured, not a failure of the method."
                ),
                evidence={
                    "gate": fit.gate,
                    "gate_value": fit.gate_value,
                    "threshold": fit.gate_threshold,
                    "n_train": fit.n_train,
                    "n_features": fit.n_features,
                },
                suggestion=(
                    "More entities, a different feature extractor, or a different error "
                    "kind. There is one row per entity, so a few long series give few "
                    "rows however many timesteps each has."
                ),
            )
        ]

    attribution = model.importance()
    top = attribution.top(top_n)
    out = [
        Finding(
            severity="info",
            rule="attribution.error_drivers",
            area="attribution",
            headline=(
                "the model's errors are predictable from "
                + ", ".join(entry["feature"] for entry in top)
            ),
            detail=(
                f"A surrogate reaches {fit.gate} {fit.gate_value:.3f} predicting this "
                f"model's error from its input, and these features carry that prediction. "
                f"These are characteristics of the *inputs it gets wrong*, not "
                f"necessarily causes."
            ),
            evidence={
                "top_features": [dict(entry) for entry in top],
                "gate": fit.gate,
                "gate_value": fit.gate_value,
            },
            suggestion=(
                "Check the marginal relationship with feature_error_scatter before "
                "reading any of these as a mechanism."
            ),
        )
    ]
    if fit.looks_overfitted:
        out.append(
            Finding(
                severity="warning",
                rule="attribution.surrogate_overfitted",
                area="attribution",
                headline="the error model's attributions may be memorisation",
                detail=(
                    f"Train score is far above test ({fit.train_r2:.3f} against "
                    f"{fit.test_r2:.3f} R²) over {fit.n_train} training row(s) and "
                    f"{fit.n_features} feature(s)."
                ),
                evidence={
                    "train_r2": fit.train_r2,
                    "test_r2": fit.test_r2,
                    "n_train": fit.n_train,
                    "n_features": fit.n_features,
                },
                suggestion="Fewer features, or more entities.",
            )
        )
    return rank_findings(out)


def findings_from_groups(comparison: Any) -> list[Finding]:
    """Findings from a
    :class:`~themis_assess.metrics.accuracy.significance.GroupComparison`."""
    out: list[Finding] = []
    test = comparison.kruskal or comparison.welch
    if test is None:
        return out

    metric = comparison.metric or "the metric"
    if test.rejected:
        pairs = comparison.significant_pairs
        out.append(
            Finding(
                severity="serious" if pairs else "warning",
                rule="groups.significant_difference",
                area="subgroups",
                headline=f"{metric} differs significantly across groups",
                detail=(
                    f"{test.test} gives p = {test.p_value:.4g}"
                    + (
                        ". The pairs that differ are "
                        + "; ".join(f"{p.left} vs {p.right}" for p in pairs)
                        + "."
                        if pairs
                        else ", though no individual pair survives the multiplicity "
                        "correction."
                    )
                ),
                evidence={
                    "test": test.test,
                    "p_value": test.p_value,
                    "effect_size": test.effect_size,
                    "per_group": _plain(comparison.per_group),
                    "significant_pairs": [
                        {"left": p.left, "right": p.right,
                         "difference": p.difference,
                         "p_value_adjusted": p.p_value_adjusted}
                        for p in pairs
                    ],
                },
                suggestion=(
                    "A significant difference is not necessarily a large one — read the "
                    "effect size and the per-group means beside the p-value."
                ),
            )
        )
    else:
        out.append(
            Finding(
                severity="info",
                rule="groups.no_significant_difference",
                area="subgroups",
                headline=f"no clear sign that {metric} differs across groups",
                detail=test.describe(),
                evidence={
                    "test": test.test,
                    "p_value": test.p_value,
                    "per_group": _plain(comparison.per_group),
                },
                suggestion=(
                    "Not evidence of equivalence: with these group sizes the test may "
                    "lack the power to detect a difference that would matter."
                ),
            )
        )

    if comparison.tests_agree is False:
        out.append(
            Finding(
                severity="warning",
                rule="groups.tests_disagree",
                area="subgroups",
                headline="the parametric and non-parametric tests disagree",
                detail=(
                    f"Welch's ANOVA gives p = {comparison.welch.p_value:.4g} and "
                    f"Kruskal-Wallis p = {comparison.kruskal.p_value:.4g}. The group "
                    f"means and the group distributions are telling different stories, "
                    f"which usually means a skewed metric or one group with an outlier."
                ),
                evidence={
                    "welch_p": comparison.welch.p_value,
                    "kruskal_p": comparison.kruskal.p_value,
                    "per_group": _plain(comparison.per_group),
                },
                suggestion="Look at the box plot before quoting either p-value.",
            )
        )
    return rank_findings(out)
