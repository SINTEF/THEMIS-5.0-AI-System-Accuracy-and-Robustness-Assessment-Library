"""One dictionary holding everything the assessment found.

All four repos ended up building a results dictionary, and each built a different one:
repo B's ``get_results_dict`` returned nested metric dictionaries with the plot objects
mixed in; repo C's flattened to one level and lost which subgroup a number belonged to;
repo D's was assembled inline in a notebook cell. None of the three could be read by
anything that had not been written against that repo, and none of them recorded what the
numbers meant.

:class:`AssessmentReport` is the one form. It collects whatever parts of an assessment
were run — accuracy, a robustness sweep, sensitivity maps, error attribution, group
comparisons, distribution distances — and serialises them to a plain dictionary of
JSON-safe values, alongside three things the four repos' dictionaries did not carry:

* **the glossary** — a card per metric, with the plain-language description, the
  orientation and the bounds, so a reader never has to look up whether lower is better;
* **the findings** — the ranked, evidence-carrying observations from
  :mod:`themis_assess.report.findings`;
* **the reporting issues** — where the report itself is incomplete by survey §8's
  standards, from :mod:`themis_assess.report.rules`.

Why a dictionary rather than a class hierarchy
----------------------------------------------

Because the consumer is often not Python. The intended downstream is a summarising
language model that reads the assessment and says what is weak about the model; the
format that reaches it is JSON. A dictionary with a declared ``schema`` key and stable
section names is the thing that survives that trip, and :meth:`AssessmentReport.to_dict`
is the contract. The rich objects remain available on the report for anything in-process
that wants them — ``report.accuracy`` is the :class:`AccuracyResult` itself, not a copy
of its numbers.

The numbers are not summarised or rounded on the way out. A report is the record, and a
rounded record cannot be checked against the run that produced it.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from themis_assess.metrics.glossary import glossary_for
from themis_assess.report.findings import (
    Finding,
    findings_from_accuracy,
    findings_from_attribution,
    findings_from_groups,
    findings_from_robustness,
    findings_from_sensitivity,
    rank_findings,
)

__all__ = ["SCHEMA", "AssessmentReport", "build_report", "to_plain"]

SCHEMA = "themis-assess/report@1"


def to_plain(value: Any) -> Any:
    """Convert a value to something :func:`json.dumps` accepts, losing nothing.

    numpy types are the whole reason this exists: a results dictionary assembled from
    numpy computations is full of ``np.float64`` and ``np.int64``, and ``json.dumps``
    refuses every one of them. Non-finite floats are converted to ``None`` rather than
    to the ``NaN`` literal, which is JSON that most parsers reject.

    Mapping keys become strings, because JSON object keys are strings — so a severity
    curve keyed by ``int`` round-trips as ``{"1": ..., "2": ...}``. That is a real change
    of type and is why every reader of a curve in this package sorts keys numerically
    after loading.
    """
    # Enums first: AccessLevel is an IntEnum, so the int branch below would catch it and
    # a report would carry access="1" — true, and unreadable by anyone.
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return {str(_key(k)): to_plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return to_plain(value.tolist())
    if isinstance(value, np.generic):
        return to_plain(value.item())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_plain(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if hasattr(value, "as_dict"):
        return to_plain(value.as_dict())
    return str(value)


def _key(key: Any) -> Any:
    if isinstance(key, np.generic):
        return key.item()
    return key


@dataclass
class AssessmentReport:
    """Everything one assessment produced, in one object.

    Every section is optional: a report of accuracy alone is a valid report, and so is a
    robustness sweep with no accuracy assessment beside it — though survey §8 has
    something to say about the latter, which :mod:`themis_assess.report.rules` will put
    in ``reporting_issues``.

    Parameters
    ----------
    model_name:
        What is being assessed. Carried through to the comparison layer, which ranks
        several reports against each other by this name.
    accuracy:
        An :class:`~themis_assess.assessment.accuracy.AccuracyResult`.
    robustness:
        A :class:`~themis_assess.assessment.robustness.RobustnessAssessment`.
    sensitivity:
        ``{name: SensitivityMap}``. Named because a run usually has more than one — a
        feature-by-severity map and a feature-by-time map are different questions.
    attribution:
        ``{name: ErrorModel}``, fitted. Only the fit report and the importances are
        serialised; the surrogate itself is not.
    groups:
        ``{name: GroupComparison}`` from
        :mod:`themis_assess.metrics.accuracy.significance`.
    distances:
        ``{name: value}`` distribution distances. Kept in their own section on purpose:
        survey §8 warns against presenting a distance as if it were a robustness score,
        and a separate section is how this package refuses to.
    context:
        Anything about the run a reader would need and the library cannot know — the
        split protocol, the data vintage, who ran it, which deployment the criteria came
        from. Serialised verbatim.
    """

    model_name: str = "model"
    accuracy: Any = None
    robustness: Any = None
    sensitivity: Mapping[str, Any] = field(default_factory=dict)
    attribution: Mapping[str, Any] = field(default_factory=dict)
    groups: Mapping[str, Any] = field(default_factory=dict)
    distances: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    created: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.UTC))

    # -- assembly -----------------------------------------------------------

    def add_sensitivity(self, name: str, sensitivity_map: Any) -> AssessmentReport:
        """Attach a sensitivity map under ``name``. Returns self, so calls chain."""
        self.sensitivity = {**self.sensitivity, name: sensitivity_map}
        return self

    def add_attribution(self, name: str, model: Any) -> AssessmentReport:
        """Attach a fitted error model under ``name``. Returns self."""
        self.attribution = {**self.attribution, name: model}
        return self

    def add_groups(self, name: str, comparison: Any) -> AssessmentReport:
        """Attach a group comparison under ``name``. Returns self."""
        self.groups = {**self.groups, name: comparison}
        return self

    def add_distance(self, name: str, value: float) -> AssessmentReport:
        """Attach a distribution distance under ``name``. Returns self.

        Distances go in their own section rather than among the robustness scores. A
        Wasserstein distance of 0.4 between train and deployment features says the
        distribution moved; it says nothing whatever about whether the model still works,
        and survey §8 lists conflating the two as a reporting failure.
        """
        self.distances = {**self.distances, name: value}
        return self

    # -- derived ------------------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        """Every metric name appearing anywhere in the report, for the glossary."""
        names: set[str] = set()
        if self.accuracy is not None:
            names |= set(self.accuracy.overall.metrics)
            for sub in self.accuracy.subgroups.values():
                names |= set(sub.metrics)
            names |= set(self.accuracy.disparity)
        if self.robustness is not None:
            names |= set(self.robustness.metric_names)
            names.add(self.robustness.accuracy_metric)
        names |= set(self.distances)
        for comparison in self.groups.values():
            if comparison.metric:
                names.add(comparison.metric)
        return sorted(names)

    def findings(self) -> list[Finding]:
        """Every finding from every section, most severe first.

        Includes the §8 reporting issues, because "no tolerance was supplied" is exactly
        as actionable as "the worst subgroup is eleven points behind" — both are things a
        reader has to decide about before the assessment means anything.
        """
        from themis_assess.report.rules import check_report

        out: list[Finding] = []
        if self.accuracy is not None:
            out.extend(findings_from_accuracy(self.accuracy))
        if self.robustness is not None:
            out.extend(findings_from_robustness(self.robustness))
        for smap in self.sensitivity.values():
            out.extend(findings_from_sensitivity(smap))
        for model in self.attribution.values():
            out.extend(findings_from_attribution(model))
        for comparison in self.groups.values():
            out.extend(findings_from_groups(comparison))
        out.extend(check_report(self._sections()))
        return rank_findings(out)

    def worst_findings(self, n: int = 5) -> list[Finding]:
        """The ``n`` most severe findings — what to read first."""
        return self.findings()[:n]

    # -- serialisation ------------------------------------------------------

    def _sections(self) -> dict[str, Any]:
        """The result sections, without findings or glossary.

        Separated from :meth:`to_dict` because the reporting rules read this and the
        findings go into ``to_dict``: building the whole dictionary first would mean the
        rules inspected their own output.
        """
        out: dict[str, Any] = {}
        if self.accuracy is not None:
            out["accuracy"] = _accuracy_section(self.accuracy)
        if self.robustness is not None:
            out["robustness"] = _robustness_section(self.robustness)
        if self.sensitivity:
            out["sensitivity"] = {
                name: _sensitivity_section(smap) for name, smap in self.sensitivity.items()
            }
        if self.attribution:
            out["attribution"] = {
                name: _attribution_section(model)
                for name, model in self.attribution.items()
            }
        if self.groups:
            out["groups"] = {
                name: _groups_section(comparison)
                for name, comparison in self.groups.items()
            }
        if self.distances:
            out["distances"] = to_plain(dict(self.distances))
        return out

    def to_dict(self, *, glossary: bool = True) -> dict[str, Any]:
        """The whole report as a plain, JSON-safe dictionary.

        This is the stable contract. Section names and the keys inside them do not change
        without a bump to ``SCHEMA``, so anything downstream — a renderer, a database
        row, a language model prompt — can be written against it.
        """
        from themis_assess import __version__

        findings = self.findings()
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "library_version": __version__,
            "model_name": self.model_name,
            "created": self.created.isoformat(),
            "context": to_plain(dict(self.context)),
        }
        payload.update(self._sections())
        payload["findings"] = [f.as_dict() for f in findings]
        payload["summary"] = {
            "n_findings": len(findings),
            "by_severity": _counts(f.severity for f in findings),
            "by_area": _counts(f.area for f in findings),
            "accuracy_passed": (
                None if self.accuracy is None else self.accuracy.passed
            ),
            "robustness_passed": (
                None if self.robustness is None else self.robustness.passed
            ),
            "headlines": [f.headline for f in findings[:5]],
        }
        if glossary:
            payload["glossary"] = glossary_for(self.metric_names)
        return payload

    def to_json(self, *, indent: int | None = 2, glossary: bool = True) -> str:
        """The report as a JSON string. ``allow_nan`` is off — see :func:`to_plain`."""
        return json.dumps(
            self.to_dict(glossary=glossary), indent=indent, allow_nan=False, default=str
        )

    def save(self, path: str | Path, *, glossary: bool = True) -> Path:
        """Write the report to ``path`` as JSON, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(glossary=glossary), encoding="utf-8")
        return target

    # -- reading ------------------------------------------------------------

    def summary(self) -> str:
        """A few lines for a terminal or a notebook cell: verdicts, then what is wrong."""
        lines = [f"{self.model_name} — assessment report"]
        verdict = {True: "PASS", False: "FAIL", None: "no criteria"}
        if self.accuracy is not None:
            lines.append(f"  accuracy:   {verdict[self.accuracy.passed]}")
        if self.robustness is not None:
            lines.append(
                f"  robustness: {verdict[self.robustness.passed]} "
                f"(over {self.robustness.accuracy_metric})"
            )
        findings = self.findings()
        counts = _counts(f.severity for f in findings)
        if counts:
            lines.append(
                "  findings:   "
                + ", ".join(f"{n} {sev}" for sev, n in counts.items())
            )
        for finding in findings[:5]:
            lines.append(f"    [{finding.severity}] {finding.headline}")
        if len(findings) > 5:
            lines.append(f"    ... and {len(findings) - 5} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


# --------------------------------------------------------------------------
# section builders
# --------------------------------------------------------------------------


def _evaluation(result: Any) -> dict[str, Any]:
    return {
        "label": result.label,
        "n_samples": int(result.n_samples),
        "task": None if result.task is None else result.task.name,
        "metrics": to_plain(dict(result.metrics)),
        "metadata": to_plain(dict(result.metadata)),
    }


def _verdict(verdict: Any) -> dict[str, Any]:
    requirement = verdict.requirement
    return {
        "metric": verdict.metric,
        "value": to_plain(verdict.value),
        "passed": verdict.passed,
        "margin": to_plain(verdict.margin),
        "higher_is_better": verdict.higher_is_better,
        "requirement": (
            None
            if requirement is None
            else {
                "acceptable": to_plain(requirement.acceptable),
                "tolerance": to_plain(getattr(requirement, "tolerance", None)),
                "critical": requirement.critical,
                "rationale": requirement.rationale,
            }
        ),
    }


def _accuracy_section(result: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "overall": _evaluation(result.overall),
        "passed": result.passed,
        "metadata": to_plain(dict(result.metadata)),
    }
    if result.acceptance is not None:
        section["acceptance"] = {
            "passed": result.acceptance.passed,
            "label": result.acceptance.label,
            "verdicts": [_verdict(v) for v in result.acceptance.verdicts],
            "failures": [v.metric for v in result.acceptance.failures],
            "no_requirement": result.acceptance.unchecked,
            "not_measured": list(result.acceptance.provenance.get("not_measured", ())),
            "provenance": to_plain(dict(result.acceptance.provenance)),
        }
    if result.per_class:
        section["per_class"] = to_plain(dict(result.per_class))
    if result.confusion is not None:
        from themis_assess.metrics.accuracy.classification import top_confusions

        section["confusion"] = {
            "labels": to_plain(list(result.confusion.labels)),
            "matrix": to_plain(result.confusion.matrix),
            "normalize": result.confusion.normalize,
            "top_confusions": to_plain(top_confusions(result.confusion)),
        }
    if result.subgroups:
        section["subgroups"] = {
            name: _evaluation(sub) for name, sub in result.subgroups.items()
        }
    if result.disparity:
        section["disparity"] = to_plain(dict(result.disparity))
    if result.baseline is not None:
        section["baseline"] = _evaluation(result.baseline)
    if result.diagnostics:
        section["diagnostics"] = to_plain(dict(result.diagnostics))
    return section


def _robustness_section(assessment: Any) -> dict[str, Any]:
    sweep = assessment.sweep
    plan = sweep.plan
    section: dict[str, Any] = {
        "accuracy_metric": assessment.accuracy_metric,
        "passed": assessment.passed,
        "clean": to_plain(assessment.clean),
        "clean_evaluation": _evaluation(sweep.clean),
        "perturbations": list(plan.perturbations),
        "severities": [int(s) for s in plan.severities],
        "curves": to_plain(assessment.curves()),
        "seed": sweep.seed,
        "plan": {
            "metrics": list(plan.metrics),
            "skipped": to_plain(dict(plan.skipped)),
            "n_model_calls": int(plan.n_model_calls),
            "task": None if plan.task is None else plan.task.name,
            "access": plan.access.name,
        },
        "metadata": to_plain(dict(assessment.metadata)),
    }
    perturbation, severity, value = assessment.worst_cell()
    section["worst_cell"] = {
        "perturbation": perturbation,
        "severity": int(severity),
        "value": to_plain(value),
    }
    section["scored"] = {
        name: _robustness_result(result) for name, result in assessment.scored.items()
    }
    if sweep.baseline_clean is not None:
        section["baseline_clean"] = _evaluation(sweep.baseline_clean)
    if sweep.subgroups:
        section["subgroups"] = {
            name: _evaluation(sub) for name, sub in sweep.subgroups.items()
        }
    if assessment.tolerance is not None:
        section["tolerance"] = {
            "passed": assessment.tolerance.passed,
            "verdicts": [_tolerance_verdict(v) for v in assessment.tolerance.verdicts],
            "failures": [v.metric for v in assessment.tolerance.failures],
            "unchecked": assessment.tolerance.unchecked,
            "provenance": to_plain(dict(assessment.tolerance.provenance)),
        }
    return section


def _robustness_result(result: Any) -> dict[str, Any]:
    return {
        "metric_name": result.metric_name,
        "accuracy_metric": result.accuracy_metric,
        "clean": to_plain(result.clean),
        "scores": to_plain(dict(result.scores)),
        "aggregate": to_plain(result.aggregate),
        "per_perturbation": to_plain(dict(result.per_perturbation)),
        "perturbation_set": list(result.perturbation_set),
        "severities": to_plain(result.severities),
        "has_baseline": result.has_baseline,
        "baseline_clean": to_plain(result.baseline_clean),
        "access_used": result.access_used.name,
        "tolerance": to_plain(result.tolerance),
        "is_complete_claim": result.is_complete_claim,
        "n_samples": int(result.n_samples),
        "seed": result.seed,
        "metadata": to_plain(dict(result.metadata)),
    }


def _tolerance_verdict(verdict: Any) -> dict[str, Any]:
    return {
        "metric": verdict.metric,
        "accuracy_metric": verdict.accuracy_metric,
        "value": to_plain(verdict.value),
        "passed": verdict.passed,
        "margin": to_plain(verdict.margin),
        "basis": verdict.basis,
        "worst_at": verdict.worst_at,
        "aggregate": to_plain(verdict.aggregate),
        "requirement": {
            "acceptable": to_plain(verdict.requirement.acceptable),
            "basis": verdict.requirement.basis,
            "critical": verdict.requirement.critical,
            "rationale": verdict.requirement.rationale,
        },
    }


def _sensitivity_section(sensitivity_map: Any) -> dict[str, Any]:
    return {
        "measure": sensitivity_map.measure,
        "metric": sensitivity_map.metric,
        "row_axis": sensitivity_map.row_axis,
        "column_axis": sensitivity_map.column_axis,
        "row_labels": list(sensitivity_map.row_labels),
        "column_labels": list(sensitivity_map.column_labels),
        "grid": to_plain(sensitivity_map.grid),
        "most_sensitive": to_plain(sensitivity_map.most_sensitive_rows(n=10)),
        "hottest_cell": to_plain(sensitivity_map.hottest_cell()),
        "metadata": to_plain(dict(sensitivity_map.metadata)),
    }


def _attribution_section(model: Any) -> dict[str, Any]:
    fit = model.fit_report
    if fit is None:
        return {"fitted": False}
    section: dict[str, Any] = {
        "fitted": True,
        "fit": {
            "surrogate": fit.surrogate,
            "extractor": fit.extractor,
            "error_kind": fit.error_kind,
            "is_binary_error": fit.is_binary_error,
            "n_train": int(fit.n_train),
            "n_test": int(fit.n_test),
            "n_features": int(fit.n_features),
            "error_mean": to_plain(fit.error_mean),
            "error_std": to_plain(fit.error_std),
            "train_r2": to_plain(fit.train_r2),
            "test_r2": to_plain(fit.test_r2),
            "train_auroc": to_plain(fit.train_auroc),
            "test_auroc": to_plain(fit.test_auroc),
            "gate": fit.gate,
            "gate_value": to_plain(fit.gate_value),
            "gate_threshold": to_plain(fit.gate_threshold),
            "is_trustworthy": fit.is_trustworthy,
            "looks_overfitted": fit.looks_overfitted,
        },
    }
    if fit.is_trustworthy:
        attribution = model.importance()
        section["importance"] = {
            "method": attribution.method,
            "scope": attribution.scope,
            "features": list(attribution.features),
            "values": to_plain(attribution.importance),
            "top": to_plain(attribution.top(10)),
            "metadata": to_plain(dict(attribution.metadata)),
        }
    return section


def _groups_section(comparison: Any) -> dict[str, Any]:
    return {
        "metric": comparison.metric,
        "per_group": to_plain(dict(comparison.per_group)),
        "group_names": to_plain(comparison.group_names),
        "dropped_groups": to_plain(list(comparison.dropped_groups)),
        "welch": None if comparison.welch is None else _group_test(comparison.welch),
        "kruskal": (
            None if comparison.kruskal is None else _group_test(comparison.kruskal)
        ),
        "tests_agree": comparison.tests_agree,
        "pairwise": [_pairwise(p) for p in comparison.pairwise],
        "significant_pairs": [
            {"left": to_plain(p.left), "right": to_plain(p.right)}
            for p in comparison.significant_pairs
        ],
        "metadata": to_plain(dict(comparison.metadata)),
    }


def _group_test(test: Any) -> dict[str, Any]:
    return {
        "test": test.test,
        "statistic": to_plain(test.statistic),
        "p_value": to_plain(test.p_value),
        "df_between": to_plain(test.df_between),
        "df_within": to_plain(test.df_within),
        "n_groups": int(test.n_groups),
        "n_samples": int(test.n_samples),
        "significance_level": to_plain(test.significance_level),
        "effect_size": to_plain(test.effect_size),
        "effect_size_name": test.effect_size_name,
        "rejected": test.rejected,
    }


def _pairwise(pair: Any) -> dict[str, Any]:
    return {
        "left": to_plain(pair.left),
        "right": to_plain(pair.right),
        "difference": to_plain(pair.difference),
        "statistic": to_plain(pair.statistic),
        "p_value": to_plain(pair.p_value),
        "p_value_adjusted": to_plain(pair.p_value_adjusted),
        "rejected": pair.rejected,
    }


def build_report(
    *,
    model_name: str = "model",
    accuracy: Any = None,
    robustness: Any = None,
    sensitivity: Mapping[str, Any] | Sequence[Any] | None = None,
    attribution: Mapping[str, Any] | None = None,
    groups: Mapping[str, Any] | None = None,
    distances: Mapping[str, float] | None = None,
    context: Mapping[str, Any] | None = None,
) -> AssessmentReport:
    """Assemble a report in one call.

    A convenience over the constructor: ``sensitivity`` may be a sequence, in which case
    the maps are named by their axes (``feature_severity``, ``feature_time``), which is
    what a caller who ran the two standard maps wants and would otherwise have to spell.
    """
    if isinstance(sensitivity, Mapping) or sensitivity is None:
        maps = dict(sensitivity or {})
    else:
        maps = {f"{m.row_axis}_{m.column_axis}": m for m in sensitivity}
    return AssessmentReport(
        model_name=model_name,
        accuracy=accuracy,
        robustness=robustness,
        sensitivity=maps,
        attribution=dict(attribution or {}),
        groups=dict(groups or {}),
        distances=dict(distances or {}),
        context=dict(context or {}),
    )
