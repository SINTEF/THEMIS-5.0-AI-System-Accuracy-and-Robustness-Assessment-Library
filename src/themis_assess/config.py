"""Assessment settings in a plain YAML file.

No hydra, no framework, no plugin registry (docs/DECISIONS.md, D18). A config here is a
YAML file read by :func:`load_config` into ordinary dataclasses, and that is the whole
mechanism. The four repos each grew their own configuration arrangement — repo B a hydra
tree of thirty-odd files, repo C module-level constants, repo D a dictionary in a
notebook cell — and the common failure was not the format but that the *thresholds* lived
somewhere different from the code that used them, so nobody could tell what a given run
had been checked against.

What a config governs, and what it deliberately does not
--------------------------------------------------------

It governs the settings: which metrics, which perturbations, which severities, how many
variants, the seed, and above all the **criteria** — the accuracy thresholds and the
degradation tolerances. Those belong in a versioned file rather than in a call, because
survey §8 makes the tolerance a property of the deployment: it is the one input no
evaluation instrument can supply, so it should be reviewable, diffable, and attributable
to whoever set it.

It does not load your model or your data. A config that names a Python class to import
turns a settings file into executable code, which is how a configuration system becomes a
framework — and then a model has to be constructible from YAML to be assessable at all,
which is exactly the coupling this library was extracted to remove. So the shape is:

.. code-block:: python

    config = load_config("assessment.yaml")
    handler = SklearnHandler(my_model)              # your code, your objects
    accuracy = config.accuracy_evaluator(handler).run(data, actual)
    robustness = config.robustness_evaluator(handler).assess(data, actual)
    report = config.report(accuracy=accuracy, robustness=robustness)

Every field is optional. An empty file is a valid config and gives the library's defaults,
which is what makes this worth using for a first run rather than only for a governed one.

An example file
---------------

.. code-block:: yaml

    model_name: plovdiv_lightgbm
    context:
      split: temporal, 2019-2022 train / 2023 test
      owner: clinical analytics
      data_vintage: "2024-03"

    accuracy:
      metrics: [accuracy, balanced_accuracy, auc, expected_calibration_error]
      per_class: true
      diagnose_outliers: true
      criteria:
        - metric: balanced_accuracy
          acceptable: 0.85
          rationale: below this the triage queue is longer than the clinic can staff
          critical: true
        - metric: expected_calibration_error
          acceptable: 0.05
          rationale: the score is shown to clinicians as a probability
          critical: false

    robustness:
      perturbations: [gaussian_noise, missing_values, feature_scaling]
      severities: [1, 2, 3, 4, 5]
      n_variants: 3
      seed: 0
      accuracy_metric: balanced_accuracy
      criteria:
        - metric: relative_robustness
          acceptable: 0.9
          basis: worst_cell
          rationale: sensor drift at severity 5 is within a year's normal calibration drift

    report:
      output: reports/plovdiv.json
      glossary: true
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AccuracyConfig",
    "RobustnessConfig",
    "ReportConfig",
    "AssessmentConfig",
    "load_config",
    "dump_config",
]


def _reject_unknown(section: str, data: Mapping[str, Any], known: Sequence[str]) -> None:
    """Refuse keys nothing reads.

    A misspelt key in a settings file is worse than an error, because the run proceeds
    with the default and the file says otherwise — so ``severities: [1,2,3]`` under
    ``severity:`` would silently sweep all five. Hydra's strict mode existed for this
    reason and repo B had it switched off.
    """
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown key(s) in the '{section}' section: {unknown}. "
            f"Known keys: {sorted(known)}. Nothing reads an unrecognised key, so it "
            f"would silently have no effect."
        )


@dataclass(frozen=True, slots=True)
class AccuracyConfig:
    """Settings for the accuracy assessment. Every field optional."""

    metrics: tuple[str, ...] | None = None
    per_class: bool = True
    diagnose_outliers: bool = False
    diagnose_features: bool = False
    diagnose_ambiguity: bool = False
    criteria: tuple[Mapping[str, Any], ...] = ()
    source: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AccuracyConfig:
        _reject_unknown("accuracy", data, [f.name for f in cls.__dataclass_fields__.values()])
        metrics = data.get("metrics")
        return cls(
            metrics=None if metrics is None else tuple(metrics),
            per_class=bool(data.get("per_class", True)),
            diagnose_outliers=bool(data.get("diagnose_outliers", False)),
            diagnose_features=bool(data.get("diagnose_features", False)),
            diagnose_ambiguity=bool(data.get("diagnose_ambiguity", False)),
            criteria=tuple(dict(entry) for entry in data.get("criteria", ())),
            source=str(data.get("source", "")),
        )

    def acceptance_criteria(self) -> Any:
        """The criteria object, or ``None`` when the file sets no thresholds.

        ``None`` rather than an empty criteria set, so that a config with no thresholds
        produces the three-valued "nothing was checked" verdict instead of a vacuous
        pass.
        """
        if not self.criteria:
            return None
        from themis_assess.assessment.criteria import AcceptanceCriteria, Requirement

        return AcceptanceCriteria(
            [Requirement(**entry) for entry in self.criteria],
            source=self.source or "config",
        )


@dataclass(frozen=True, slots=True)
class RobustnessConfig:
    """Settings for the robustness sweep. Every field optional."""

    metrics: tuple[str, ...] | None = None
    perturbations: tuple[str, ...] | None = None
    severities: tuple[int, ...] | None = None
    n_variants: int | None = None
    seed: int | None = None
    accuracy_metric: str | None = None
    robustness_metrics: tuple[str, ...] | None = None
    criteria: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RobustnessConfig:
        _reject_unknown("robustness", data, [f.name for f in cls.__dataclass_fields__.values()])
        def maybe_tuple(key: str) -> tuple[Any, ...] | None:
            value = data.get(key)
            return None if value is None else tuple(value)

        severities = data.get("severities")
        return cls(
            metrics=maybe_tuple("metrics"),
            perturbations=maybe_tuple("perturbations"),
            severities=None if severities is None else tuple(int(s) for s in severities),
            n_variants=None if data.get("n_variants") is None else int(data["n_variants"]),
            seed=None if data.get("seed") is None else int(data["seed"]),
            accuracy_metric=data.get("accuracy_metric"),
            robustness_metrics=maybe_tuple("robustness_metrics"),
            criteria=tuple(dict(entry) for entry in data.get("criteria", ())),
        )

    def tolerance_criteria(self) -> Any:
        """The tolerance object, or ``None`` when the file sets none.

        ``None`` is survey §8's incomplete claim, and the report layer says so rather
        than treating an unstated tolerance as satisfied.
        """
        if not self.criteria:
            return None
        from themis_assess.assessment.tolerance import (
            RobustnessCriteria,
            ToleranceRequirement,
        )

        return RobustnessCriteria(
            [ToleranceRequirement(**entry) for entry in self.criteria]
        )


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Where the results dictionary goes, and what it carries."""

    output: str | None = None
    glossary: bool = True
    indent: int | None = 2

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReportConfig:
        _reject_unknown("report", data, [f.name for f in cls.__dataclass_fields__.values()])
        return cls(
            output=data.get("output"),
            glossary=bool(data.get("glossary", True)),
            indent=data.get("indent", 2),
        )


@dataclass(frozen=True, slots=True)
class AssessmentConfig:
    """One assessment's settings, read from YAML.

    Parameters
    ----------
    model_name:
        Carried into the report and used as the label in a multi-model comparison.
    context:
        Free-form notes about the run — the split protocol, the data vintage, who owns
        the deployment. Copied verbatim into the report's ``context`` section, because
        these are the facts a reader needs and the library cannot discover.
    """

    model_name: str = "model"
    context: Mapping[str, Any] = field(default_factory=dict)
    accuracy: AccuracyConfig = field(default_factory=AccuracyConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    path: str | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str | None = None) -> AssessmentConfig:
        _reject_unknown(
            "top level",
            data,
            ["model_name", "context", "accuracy", "robustness", "report"],
        )
        return cls(
            model_name=str(data.get("model_name", "model")),
            context=dict(data.get("context", {})),
            accuracy=AccuracyConfig.from_mapping(data.get("accuracy") or {}),
            robustness=RobustnessConfig.from_mapping(data.get("robustness") or {}),
            report=ReportConfig.from_mapping(data.get("report") or {}),
            path=path,
        )

    def to_mapping(self) -> dict[str, Any]:
        """The config as a plain dictionary, in the file's own shape.

        Round-trips through :func:`load_config`, so a config assembled in Python can be
        written out and reviewed like a hand-written one.
        """
        out: dict[str, Any] = {"model_name": self.model_name}
        if self.context:
            out["context"] = dict(self.context)
        accuracy: dict[str, Any] = {}
        if self.accuracy.metrics is not None:
            accuracy["metrics"] = list(self.accuracy.metrics)
        if not self.accuracy.per_class:
            accuracy["per_class"] = False
        for flag in ("diagnose_outliers", "diagnose_features", "diagnose_ambiguity"):
            if getattr(self.accuracy, flag):
                accuracy[flag] = True
        if self.accuracy.criteria:
            accuracy["criteria"] = [dict(entry) for entry in self.accuracy.criteria]
        if self.accuracy.source:
            accuracy["source"] = self.accuracy.source
        if accuracy:
            out["accuracy"] = accuracy

        robustness: dict[str, Any] = {}
        for name in (
            "metrics",
            "perturbations",
            "severities",
            "robustness_metrics",
        ):
            value = getattr(self.robustness, name)
            if value is not None:
                robustness[name] = list(value)
        for name in ("n_variants", "seed", "accuracy_metric"):
            value = getattr(self.robustness, name)
            if value is not None:
                robustness[name] = value
        if self.robustness.criteria:
            robustness["criteria"] = [dict(entry) for entry in self.robustness.criteria]
        if robustness:
            out["robustness"] = robustness

        report: dict[str, Any] = {}
        if self.report.output is not None:
            report["output"] = self.report.output
        if not self.report.glossary:
            report["glossary"] = False
        if report:
            out["report"] = report
        return out

    # -- what the config builds ---------------------------------------------

    def accuracy_evaluator(self, handler: Any, **overrides: Any) -> Any:
        """An :class:`~themis_assess.assessment.accuracy.AccuracyEvaluator` from the file.

        Keyword overrides win over the file, which is what makes a config usable in an
        exploratory session: the file holds the agreed settings and the call can vary one
        of them without editing it.
        """
        from themis_assess.assessment.accuracy import AccuracyEvaluator

        kwargs: dict[str, Any] = {
            "metrics": None if self.accuracy.metrics is None else list(self.accuracy.metrics),
            "per_class": self.accuracy.per_class,
            "criteria": self.accuracy.acceptance_criteria(),
        }
        kwargs.update(overrides)
        return AccuracyEvaluator(handler, **kwargs)

    def run_accuracy(self, handler: Any, data: Any, actual: Any, **overrides: Any) -> Any:
        """Run the accuracy assessment, including the file's diagnostic flags.

        The diagnostics are arguments of ``run`` rather than of the evaluator, so they
        cannot be set by :meth:`accuracy_evaluator` alone; this is the method that applies
        the whole accuracy section.
        """
        evaluator = self.accuracy_evaluator(
            handler, **{k: v for k, v in overrides.items() if k in _ASSESSOR_KEYS}
        )
        run_kwargs = {
            "diagnose_outliers": self.accuracy.diagnose_outliers,
            "diagnose_features": self.accuracy.diagnose_features,
            "diagnose_ambiguity": self.accuracy.diagnose_ambiguity,
        }
        run_kwargs.update({k: v for k, v in overrides.items() if k not in _ASSESSOR_KEYS})
        return evaluator.run(data, actual, **run_kwargs)

    def robustness_evaluator(self, handler: Any, **overrides: Any) -> Any:
        """A :class:`~themis_assess.assessment.robustness.RobustnessEvaluator` from the file."""
        from themis_assess.assessment.robustness import RobustnessEvaluator

        kwargs: dict[str, Any] = {}
        if self.robustness.metrics is not None:
            kwargs["metrics"] = list(self.robustness.metrics)
        if self.robustness.perturbations is not None:
            kwargs["perturbations"] = list(self.robustness.perturbations)
        if self.robustness.severities is not None:
            kwargs["severities"] = tuple(self.robustness.severities)
        if self.robustness.n_variants is not None:
            kwargs["n_variants"] = self.robustness.n_variants
        if self.robustness.seed is not None:
            kwargs["seed"] = self.robustness.seed
        kwargs.update(overrides)
        return RobustnessEvaluator(handler, **kwargs)

    def assess_robustness(self, handler: Any, data: Any, actual: Any, **overrides: Any) -> Any:
        """Sweep, score and check the tolerance, using the whole robustness section."""
        evaluator = self.robustness_evaluator(handler)
        kwargs: dict[str, Any] = {"criteria": self.robustness.tolerance_criteria()}
        if self.robustness.accuracy_metric is not None:
            kwargs["accuracy_metric"] = self.robustness.accuracy_metric
        if self.robustness.robustness_metrics is not None:
            kwargs["robustness_metrics"] = list(self.robustness.robustness_metrics)
        kwargs.update(overrides)
        return evaluator.assess(data, actual, **kwargs)

    def build_report(self, **sections: Any) -> Any:
        """A report carrying this config's model name and context.

        The context matters more than it looks: it is where the split protocol and the
        data vintage reach the report, and those are the two things a reader most often
        cannot reconstruct afterwards.
        """
        from themis_assess.report.bundle import build_report as _build

        context = dict(self.context)
        if self.path:
            context.setdefault("config", self.path)
        return _build(model_name=self.model_name, context=context, **sections)

    def save_report(self, report: Any, path: str | Path | None = None) -> Path | None:
        """Write the report to the configured output, if there is one."""
        target = path if path is not None else self.report.output
        if target is None:
            return None
        return report.save(target, glossary=self.report.glossary)


_ASSESSOR_KEYS = frozenset(
    {"metrics", "per_class", "criteria", "subgroups", "baseline", "registry", "catalog"}
)


def load_config(path: str | Path) -> AssessmentConfig:
    """Read a YAML config file.

    An empty file, or one containing only comments, is valid and gives the defaults —
    ``yaml.safe_load`` returns ``None`` for both, and treating that as an error would
    make a config file all-or-nothing.
    """
    import yaml

    target = Path(path)
    text = target.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{target} does not contain a YAML mapping (got {type(data).__name__}). "
            f"A config file's top level is a mapping of sections."
        )
    return AssessmentConfig.from_mapping(data, path=str(target))


def dump_config(config: AssessmentConfig, path: str | Path | None = None) -> str:
    """Serialise a config back to YAML, optionally writing it to ``path``."""
    import yaml

    text = yaml.safe_dump(config.to_mapping(), sort_keys=False, allow_unicode=True)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text
