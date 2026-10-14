"""The report layer: one results dictionary, ranked findings, and §8's rules.

Three things, in the order a reader meets them.

:mod:`~themis_assess.report.bundle` collects every part of an assessment into one
:class:`~themis_assess.report.bundle.AssessmentReport` and serialises it to a plain
dictionary with a declared schema — the handover format, and the descendant of the four
repos' four incompatible results dictionaries.

:mod:`~themis_assess.report.findings` turns result objects into ranked
:class:`~themis_assess.report.findings.Finding` records: what is wrong, how badly, on
what evidence. This is the structure a summarising language model needs, and it is what
lets a reader start at the top of a report rather than reading all of it.

:mod:`~themis_assess.report.rules` checks the report against survey §8's reporting
standards — ten things a robustness claim must not do — and adds what it finds to the
findings as ``area="reporting"``. Those issues are about the *report*, not the model: a
model can be excellent and its report still incomplete.
"""

from themis_assess.report.bundle import (
    SCHEMA,
    AssessmentReport,
    build_report,
    to_plain,
)
from themis_assess.report.findings import (
    Finding,
    Severity,
    findings_from_accuracy,
    findings_from_attribution,
    findings_from_groups,
    findings_from_robustness,
    findings_from_sensitivity,
    rank_findings,
)
from themis_assess.report.rules import check_report

__all__ = [
    "SCHEMA",
    "AssessmentReport",
    "build_report",
    "to_plain",
    "Finding",
    "Severity",
    "findings_from_accuracy",
    "findings_from_attribution",
    "findings_from_groups",
    "findings_from_robustness",
    "findings_from_sensitivity",
    "rank_findings",
    "check_report",
]
