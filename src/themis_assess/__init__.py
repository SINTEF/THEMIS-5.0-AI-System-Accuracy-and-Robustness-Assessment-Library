"""themis_assess — accuracy and robustness assessment for classification and forecasting models.

Developed in the EU Horizon Europe project THEMIS 5.0 (grant No. 101121042).

The library is organised in these layers:

* **metrics** — the survey's robustness metrics as pure functions of numbers, with a
  registry describing each metric's orientation, bounds, task applicability and
  model-access requirement.
* **core** — the contracts everything else is written against: data adapters, the
  perturbation specification, subgroups as masks, and the result containers that keep
  the survey's reporting rules enforceable.
* **perturbations** — nineteen transformations on a shared five-level severity scale,
  spanning feature space, the time-series value and time domains, STL-structural, and
  text. Numeric magnitudes are relative to each feature's own spread, so results
  transfer between datasets.
* **protocols** — the split protocols of survey §3.1, which introduce a *natural* shift
  by partitioning real data rather than by corrupting it.
* **handlers** — the only part of the library that knows what a model is. A handler
  declares what it can produce, and everything downstream reads that declaration.
* **sensitivity** — which input, perturbed where, moves the model most: feature-by-severity
  and feature-by-time grids, plus a free perturbation-by-severity map from any sweep.
* **attribution** — a surrogate model of the assessed model's error, and its feature
  attributions. Answers what the cases the model already gets wrong have in common,
  without perturbing anything.
* **viz** — plotly figures, in two shapes: pure functions from a result to a figure, and
  the *bundled* form the use-case repos used, which computes and draws in one call and
  returns both halves.
* **assessment** — two entry points. ``AccuracyEvaluator`` assesses one model on one dataset
  against the deployment's own acceptance criteria, with the confusion matrix, per-class
  and subgroup breakdowns, and diagnostics saying where the errors are; one model call.
  ``RobustnessEvaluator`` adds the perturbation sweep, and the scoring layer turns that into
  the survey's §5.1.3 robustness metrics.

See ``docs/ARCHITECTURE.md`` for the design and ``docs/DECISIONS.md`` for the choices
behind it.
"""

__version__ = "0.12.0"

from themis_assess.core.adapters import ArrayAdapter, DataAdapter, FrameAdapter, TextAdapter
from themis_assess.core.results import EvaluationResult, RobustnessResult
from themis_assess.core.subgroups import SubgroupSpec
from themis_assess.assessment.accuracy import AccuracyEvaluator, AccuracyResult
from themis_assess.attribution.error_model import (
    ErrorModel,
    ErrorModelFit,
    FeatureAttribution,
)
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
from themis_assess.assessment.scoring import score, score_all
from themis_assess.assessment.tolerance import (
    RobustnessCriteria,
    RobustnessReport,
    ToleranceRequirement,
    ToleranceVerdict,
)
from themis_assess.metrics.accuracy.significance import (
    GroupComparison,
    compare_groups,
    kruskal_wallis,
    welch_anova,
)
from themis_assess.sensitivity.maps import (
    SensitivityMap,
    SensitivityMapper,
    map_from_sweep,
)
from themis_assess.core.types import (
    AccessLevel,
    Alignment,
    Modality,
    PerturbationSpec,
    Scope,
    TaskType,
    TimeScope,
)
from themis_assess.handlers.base import BaseHandler, ModelHandler, Prediction
from themis_assess.handlers.baselines import (
    MajorityClassHandler,
    MovingAverageHandler,
    NaiveLastValueHandler,
    PriorProbabilityHandler,
)
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler
from themis_assess.handlers.sklearn import CallableHandler, SklearnHandler
from themis_assess.metrics.accuracy import MetricCatalog, MetricEntry, default_catalog
from themis_assess.metrics.normalize import normalize_metric, normalize_metrics
from themis_assess.metrics.registry import MetricRegistry, MetricSpec, default_registry
from themis_assess.perturbations.base import Perturbation, PerturbationRegistry
from themis_assess.perturbations.registry import default_registry as default_perturbations
from themis_assess.comparison.compare import ModelComparison, compare_models
from themis_assess.config import AssessmentConfig, load_config
from themis_assess.report.bundle import AssessmentReport, build_report
from themis_assess.summary import Audience, Summary, summarise
from themis_assess.report.findings import Finding, rank_findings
from themis_assess.report.rules import check_report
from themis_assess.metrics.glossary import describe_metric
from themis_assess.protocols.splits import (
    Split,
    compositional_split,
    domain_split,
    near_far_split,
    resample_subsets,
    subpopulation_split,
    temporal_split,
)

__all__ = [
    "__version__",
    # enums and specs
    "AccessLevel",
    "Alignment",
    "Modality",
    "Scope",
    "TaskType",
    "TimeScope",
    "PerturbationSpec",
    # adapters
    "DataAdapter",
    "ArrayAdapter",
    "FrameAdapter",
    "TextAdapter",
    # subgroups and results
    "SubgroupSpec",
    "EvaluationResult",
    "RobustnessResult",
    # metrics
    "MetricRegistry",
    "MetricSpec",
    "default_registry",
    "MetricCatalog",
    "MetricEntry",
    "default_catalog",
    "normalize_metric",
    "normalize_metrics",
    # perturbations
    "Perturbation",
    "PerturbationRegistry",
    "default_perturbations",
    # handlers
    "ModelHandler",
    "Prediction",
    "BaseHandler",
    "SklearnHandler",
    "CallableHandler",
    "ForecastingHandler",
    "ForecastingConfig",
    "MajorityClassHandler",
    "PriorProbabilityHandler",
    "NaiveLastValueHandler",
    "MovingAverageHandler",
    # assessment: accuracy
    "AccuracyEvaluator",
    "AccuracyResult",
    "AcceptanceCriteria",
    "AcceptanceReport",
    "Requirement",
    "MetricVerdict",
    "themis_50_criteria",
    # assessment: robustness
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
    # sensitivity and attribution
    "SensitivityMapper",
    "SensitivityMap",
    "map_from_sweep",
    "ErrorModel",
    "ErrorModelFit",
    "FeatureAttribution",
    # group significance
    "compare_groups",
    "GroupComparison",
    "welch_anova",
    "kruskal_wallis",
    # config
    "AssessmentConfig",
    "load_config",
    # comparison
    "compare_models",
    "ModelComparison",
    # the summary layer
    "summarise",
    "Summary",
    "Audience",
    # the report layer
    "AssessmentReport",
    "build_report",
    "Finding",
    "rank_findings",
    "check_report",
    "describe_metric",
    # split protocols
    "Split",
    "domain_split",
    "temporal_split",
    "subpopulation_split",
    "compositional_split",
    "near_far_split",
    "resample_subsets",
]
