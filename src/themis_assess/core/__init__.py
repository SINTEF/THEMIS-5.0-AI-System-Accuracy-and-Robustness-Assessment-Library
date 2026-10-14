"""Dependency-free contracts shared by the rest of the library."""

from themis_assess.core.adapters import (
    ArrayAdapter,
    BaseAdapter,
    DataAdapter,
    FrameAdapter,
    TextAdapter,
    infer_categorical_features,
)
from themis_assess.core.errors import (
    MissingExtraError,
    ThemisAssessError,
    ZeroDenominatorError,
    require_extra,
    safe_divide,
)
from themis_assess.core.results import EvaluationResult, RobustnessResult, severity_curve
from themis_assess.core.subgroups import SubgroupSpec, from_criteria, resolve_masks
from themis_assess.core.types import (
    DEFAULT_SEVERITY_LEVELS,
    AccessLevel,
    Alignment,
    Modality,
    PerturbationSpec,
    Scope,
    TaskType,
    TimeScope,
)

__all__ = [
    # types
    "AccessLevel",
    "Alignment",
    "Modality",
    "Scope",
    "TaskType",
    "TimeScope",
    "PerturbationSpec",
    "DEFAULT_SEVERITY_LEVELS",
    # adapters
    "DataAdapter",
    "BaseAdapter",
    "ArrayAdapter",
    "FrameAdapter",
    "TextAdapter",
    "infer_categorical_features",
    # subgroups
    "SubgroupSpec",
    "resolve_masks",
    "from_criteria",
    # results
    "EvaluationResult",
    "RobustnessResult",
    "severity_curve",
    # errors
    "ThemisAssessError",
    "ZeroDenominatorError",
    "MissingExtraError",
    "safe_divide",
    "require_extra",
]
