"""Running an assessment: the runner that calls the model, and the scoring that follows.

Two entry points, for the two halves of the library.

:class:`~themis_assess.assessment.accuracy.AccuracyEvaluator` assesses one model on one
dataset: metrics against the deployment's acceptance criteria, the confusion matrix and
per-class breakdown, subgroups and their disparity, a baseline comparison, and diagnostics
saying where the errors are. One model call. Cheap enough to run on every commit.

:class:`~themis_assess.assessment.robustness.RobustnessEvaluator` adds the perturbation
sweep, which costs thousands of model calls. Its ``assess()`` returns the same shape of
answer as the accuracy side: the sweep, the scored §5.1.3 metrics, and a verdict against
the deployment's tolerances. Its ``run()`` returns the unscored sweep, for the sensitivity
and attribution layers that need the raw cells.

Scoring stays a separate module because it is arithmetic on what the sweep produced, so
six robustness metrics over one sweep cost six pieces of arithmetic rather than six
sweeps.

Both evaluators take a criteria object the library refuses to invent —
:class:`~themis_assess.assessment.criteria.AcceptanceCriteria` for accuracy thresholds,
:class:`~themis_assess.assessment.tolerance.RobustnessCriteria` for degradation
tolerances — and both report ``passed is None`` when none was supplied, because survey §8
is explicit that neither can come from the instrument.
"""

from themis_assess.assessment.accuracy import AccuracyResult, AccuracyEvaluator
from themis_assess.assessment.criteria import (
    AcceptanceCriteria,
    AcceptanceReport,
    MetricVerdict,
    Requirement,
    themis_50_criteria,
)
from themis_assess.assessment.robustness import (
    AssessmentPlan,
    RobustnessAssessment,
    RobustnessEvaluator,
    SweepResult,
)
from themis_assess.assessment.tolerance import (
    RobustnessCriteria,
    RobustnessReport,
    ToleranceRequirement,
    ToleranceVerdict,
)
from themis_assess.assessment.scoring import (
    BASELINE_METRICS,
    SUPPORTED_METRICS,
    score,
    score_all,
)

__all__ = [
    # accuracy
    "AccuracyEvaluator",
    "AccuracyResult",
    "AcceptanceCriteria",
    "AcceptanceReport",
    "Requirement",
    "MetricVerdict",
    "themis_50_criteria",
    # robustness
    "RobustnessEvaluator",
    "RobustnessAssessment",
    "RobustnessCriteria",
    "RobustnessReport",
    "ToleranceRequirement",
    "ToleranceVerdict",
    "AssessmentPlan",
    "SweepResult",
    "score",
    "score_all",
    "SUPPORTED_METRICS",
    "BASELINE_METRICS",
]
