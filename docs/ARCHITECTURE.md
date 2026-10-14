# Repository A — Architecture and Consolidation Plan

**Status:** draft for review — revision 3
**Date:** 2026-09-01

| | Repo | Branch | Domain | Task | Model | Layout |
|---|---|---|---|---|---|---|
| **A** | `themis-accuracy-robustness` | **`main`** | — (generic) | — | — | flat, submodules, GitLab CI |
| **B** | `themis_valencia` | – | Valencia port, vessel berthing | forecasting | TSMixer (darts) | flat mirror of A |
| **C** | `themis_medical` | – | Plovdiv, clinical | tabular classification | LightGBM | flat mirror of A |
| **D** | `themis_media` | – | fake news, hate speech | text classification | RoBERTa, DistilBERT | `src/` package, `pyproject`, `uv`, pytest |

**Reference:** Michałowska & Johnsen, *Evaluating Machine Learning Robustness Under Natural
Distribution Shifts: A Survey of Benchmarks, Perturbations, Metrics, and Assumptions* (2026) —
hereafter **the survey**.

---

## 0. Constraints

1. **No merging of repositories.** Repo A is not a union of the four.
2. **No edits to B, C or D.** They are read-only reference material, ported *from*, never modified.
3. **Repo A `main` is the only thing that changes.**

Consequence: the acceptance criterion cannot be "re-point each use case at repo A", since that would
require editing them. It must live inside repo A — §9, Step 5.

Second consequence: repo A must become **pip-installable**, so a use-case owner can adopt it later by
adding a dependency at a time of their choosing. Repo D already demonstrates the layout.

---

## 1. Where repo A `main` actually stands

`main` is a bare template, but not an empty one, and — importantly — **not a superset of what the use-case
repos contain**. Two things have to be understood before planning any migration.

### 1.1 What `main` has

```
algorithms/accuracy/assessment/classification.py    484 lines — substantial, generic
algorithms/accuracy/assessment/forecasting.py        50 lines — stub, and broken (§1.3)
algorithms/robustness/assessment/base.py            176 lines — BaseRobustnessAssessment
algorithms/robustness/assessment/classification.py   57 lines — flip_rate only
algorithms/robustness/assessment/forecasting.py     110 lines — a second compute_robustness
data_handler/base_data_handler.py                            — DataHandler (ABC)
model_handler/base_model_handler.py                          — BaseModelHandler (ABC)
model_handler/model_handler_forecasting.py                   — ForecastingModelHandler
Metrics_list.yaml, requirements.txt, .gitlab-ci.yml, tests/ (2 notebooks)
algorithms/{accuracy/submodules/safetycage, robustness/submodules/OODGuard}   ← both to be removed (§3)
```

`ClassificationAccuracyAssessment` on `main` is the strongest asset: a generalised descendant of repo C's
version with the clinical specifics stripped, carrying `default_metric_functions`,
`default_metric_ranges`, `default_acceptable_metric_values`, a `_call_metric` that introspects signatures,
confusion matrix, polar chart, gauges, PDP/ICE. This is the shape the rest of the library should follow.

`.gitlab-ci.yml` exists and runs `pytest --nbmake tests/` on `python:3.10` — so there is already a CI
harness, currently executing two tutorial notebooks that came from repo C.

### 1.2 What `main` is missing that already exists elsewhere — the regression

Generic, use-case-agnostic components have drifted **out** of repo A and now live only in repo B:

| Component | On `main` | In `themis_valencia` | Nature |
|---|---|---|---|
| `MetricRegistry` / `MetricSpec` | **absent** | `algorithms/accuracy/metrics/metric_metadata.py` (2.6 kB, newer) | fully generic |
| `normalize_metric` | **absent** | `algorithms/utils/utils.py` (1.9 kB, newer) | fully generic |
| `ForecastingAccuracyAssessment` | 50-line broken stub | 400-line ABC with polar chart, interactive + animated forecast plots | fully generic |
| epsilon guard in robustness metrics | **absent** | present in B's `base.py` | bug fix |

So consolidation is **bidirectional**. It is not only "extract from B/C/D into A" — several things that
belong in A were last edited in B and have to come home. `algorithms/accuracy/metrics/` and
`algorithms/utils/` on `main` contain nothing but stale `__pycache__` directories, which is the fingerprint
of exactly this.

### 1.3 Defects on `main`

These are in the *template* that every future use case would inherit, so they matter more than defects in
a use case.

**Tuple-assignment in both forecasting classes.** `ForecastingAccuracyAssessment.__init__` and
`ForecastingRobustnessAssessment.__init__` both assign with trailing commas:

```python
self.model_handler = model_handler,      # -> a 1-tuple
self.data_handler  = data_handler,
self.metrics       = metrics,
```

Every attribute becomes a one-element tuple, so `self.model_handler.historical_forecasts(...)` raises
`AttributeError`. Neither class can run as written.

**Missing `self` in `ForecastingRobustnessAssessment`.** `perturbate(data, perturbation_method)`,
`compute_robustness(clean_metrics, all_pertub_metrics, aggregator)` and `pertubation_method_1(data)` are
defined without `self`. Called on an instance, `self` binds to the first argument.

**`ForecastingModelHandler.__init__` calls `super(BaseModelHandler, self).__init__()`** — that resolves to
`object.__init__`, skipping `BaseModelHandler` entirely. The parent's initialisation never runs.

**Module-level import that only works from one directory.** `model_handler_forecasting.py` does
`from base_model_handler import BaseModelHandler` — a bare module name, resolvable only if that directory
is on `sys.path`. Meanwhile `robustness/assessment/classification.py` calls
`pyrootutils.setup_root(search_from=Path(os.getcwd()), ...)`, making the import depend on the *working
directory*. Two incompatible import strategies in one repo.

**`base.py::compute_robustness` — the two defects, plus one `main`-only.**

1. *Double normalisation.* `all_perturb_robustness[pert][metric]` already holds robustness values; the
   `within_aggregator` branch then computes `np.abs(1 - (d[m] / clean_metrics[m]))` over those same values,
   applying the relative-difference transform a second time.
2. *Unbound name.* `metric_names` is assigned only inside the `within_aggregator` branch but read in the
   cross-method branch. More than one perturbation method, with `aggregator` and `within_aggregator=None`,
   raises `NameError`.
3. *No epsilon guard.* `main`'s `rel_robustness_metric` and `abs_rel_robustness_metric` divide by
   `clean_accuracy_metric` unprotected. Repo B's copy has the `epsilon = 10e-6` fix; `main` does not.
   A clean metric of zero — an entirely reachable state for an error metric — is a `ZeroDivisionError`.

**Severity encoded in a string key.** `compute_robustness` keys perturbations as `"normal_0_01"` and
recovers method and intensity with `pert_name.split("_", 1)`. Any name containing an underscore
(`perturb_only_numerical`) is mis-parsed; any name without one raises `IndexError`; and severity cannot be
*ordered*, so the severity curves the survey requires for `CE`, `EI` and `mFR` cannot be produced at all.

**Requirements do not match the code.** `requirements.txt` on `main` lists `pyrootutils`, but the base
handlers no longer import it — while `robustness/assessment/classification.py` still does. There is no
`darts`, yet `ForecastingModelHandler` is built around `historical_forecasts`. The CI pins `python:3.10`
while the README instructs `conda create -n themis python=3.11`.

**The CI artifact glob is wrong.** `paths: - ./**/.ipynb` matches files literally named `.ipynb`;
it should be `./**/*.ipynb`.

### 1.4 Repo B cannot run against `main` today

`ValenciaModelHandler_NaiveMovingAverage` subclasses
`ForecastingModelHandler(model_paths, data_handler, input_chunk_length, forecast_horizon, stride)`.
`main`'s `ForecastingModelHandler` signature is `(model_filename, model)`. The base classes have diverged
to the point of incompatibility. Any consolidation must settle this signature deliberately rather than by
whichever branch was edited last.

---

## 2. The cost of the current arrangement, measured

The same quantities exist repeatedly, under different names, with different defects.

**`A_perturbed / A_clean` — survey Eq. 11 — five implementations:**

| Location | Name | Notes |
|---|---|---|
| A `robustness/assessment/base.py` | `rel_robustness_metric` | no epsilon guard |
| A `robustness/assessment/forecasting.py` | inline `abs(1 - p/c)` | inside a second `compute_robustness` |
| B `robustness/assessment/base.py` | `rel_robustness_metric` | has the epsilon fix |
| C `robustness/assessment/classification.py` | inline in `compute_robustness` | class is non-functional (§2.1) |
| D `utils/robustness_utils.py` | `robustness_ratio` | guards `unperturbed_acc > 0` instead |

**`compute_robustness` — four distinct implementations across five files:** A `base.py`,
A `forecasting.py`, C `forecasting.py` (byte-identical to A's), C `classification.py`.

**Prediction flip proportion — survey Eq. 23 — two implementations:** A
`robustness/assessment/classification.py::flip_rate`, D `utils/robustness_utils.py::flip_rate`. Both name
it `flip_rate`, which is the survey's Eq. 24 (`FR`, baseline-normalised) — a name collision with our own
reference paper.

**Gaussian feature noise — three:** B `perturb_numerical_data`, C `noise_sensitivity` (twice, in two
classes in one file).

**`plot_polar_chart_metric` — four:** A `classification.py`, B `forecasting.py`, B
`accuracy_assessment_valencia.py`, D `accuracy_evaluator.py`.

The survey's own gap statement is the design brief:

> "No single toolkit, however, supports the full range of robustness evaluations discussed in this review.
> Combining natural shifts, controlled perturbations, calibration analysis, and OOD detection still
> requires several libraries with different interfaces and conventions."

### 2.1 Defects in the use-case repos worth knowing before porting

*Repo C, `robustness/assessment/classification.py`:* `__init__` sets `self.model_hander` (typo) and never
`self.model`, but `_evaluate` uses `self.model` — every path raises `AttributeError`. `accuracy_score`,
`f1_score`, `log_loss`, `brier_score_loss` and `resample` are used but never imported.
`calibration_under_shift` simply delegates to `distribution_shift` and measures no calibration.
`ClassificationRobustnessAssessment` and `RobustnessEvaluator` in the same file duplicate
`noise_sensitivity`; only the latter binds `self.model`, so only it works. **Port from `RobustnessEvaluator`
only.**

*Repo B, `utils.py::normalize_metric`:* `best = spec.best if spec.best != None else best` silently
overrides the caller's `best` argument, and `MetricSpec.best` conflates two distinct ideas — "the ideal
value" (`mean_signed_error` has `best=0`) and "the upper bound for min–max normalisation". `MetricSpec`
also annotates `higher_is_better: bool` while `None` is passed for symmetric metrics. Fix during the port
by splitting `best` (target value) from `upper_bound` (normalisation bound) and typing
`higher_is_better: bool | None`.

*Repo D, `MaskingPerturbation`:* draws from global `np.random` state; not reproducible.

---

## 3. Dependencies and packaging

### 3.1 SafetyCage becomes a dependency

SafetyCage is now on PyPI (`safetycage`, 0.0.62, Aug 2026; [docs](https://safetycage.readthedocs.io/),
[source](https://github.com/SINTEF/safetycage)). Actions on `main`:

- **Remove the submodule** `algorithms/accuracy/submodules/safetycage` and its `.gitmodules` entry.
- **Add `safetycage` to dependencies.** The package exposes `safetycage.datamodule.DataModule` and
  `safetycage.modelmodule.ModelModule` as the two adapters a project implements, plus methods
  `safetycage.methods.{msp,doctor,spardacus,mahalanobis,red}`.
- **Integrate through those adapters, not by re-wrapping the methods.** Repo A provides thin bridges —
  `algorithms/integrations/safetycage.py`: `data_handler -> DataModule`, `model_handler -> ModelModule`.
  The old `SafetyCage_themis.py` / `ModelHandler_SafetyCage_themis.py` wrappers (still present in repo C)
  must not be ported: they mutate `sys.path` with a hardcoded backslash path
  (`'algorithms\\accuracy\\submodules\\safetycage'`, broken off Windows), hand-build Hydra-style config
  dataclasses, and maintain a private name→class map. The package makes all of it unnecessary.
- Method requirements: `MSP` and `DOCTOR` need class probabilities and work with any classifier;
  `SPARDACUS`, `Mahalanobis` and `RED` need layer activations and are neural-network-only. `MSP` needs no
  fitting — the right default.

**An unplanned benefit.** `ModelModule` is precisely the activation-extraction plumbing that survey §5.2
needs. Neuron sensitivity (Eq. 37), NCCR, ROBY (Eqs. 32–34) and RDI (Eqs. 35–36) all require white-box
access, which was the reason for deferring them. Building the model-centric metrics on `ModelModule` moves
that tier from "distant" to "reachable" — §9, Step 7.

### 3.2 OODGuard is cancelled — OOD detection becomes native

- **Remove the submodule** `algorithms/robustness/submodules/OODGuard` and its `.gitmodules` entry
  (`main`'s entry also carries a stray `branch= main` line).
- **Do not port** `OODGuard_themis.py` / `ModelHandler_OODGuard_themis.py` from repo C — same `sys.path`
  and config problems as the SafetyCage wrappers.
- **Create `algorithms/robustness/ood/` as first-class repo-A code**, absorbing what the submodule and the
  use cases already contain:

| Module | Content | Source |
|---|---|---|
| `ood/base.py` | `OODScore` ABC: `fit(reference)`, `score(x) -> float[]`, `threshold(alpha)` | new |
| `ood/scores.py` | `KNNDistanceScore` | vendored `OODGuard/src/OODGuard/KNN.py` (read before deleting the submodule) |
| | `MahalanobisScore`, `compute_embedding_stats` | D `utils/callibration_utils.py` |
| | `MSPScore`, `EntropyScore` | D `utils/callibration_utils.py` |
| `metrics/robustness/ood.py` | AUROC, AUPRC, FPR@95TPR, AUTC, wAUTC | survey §5.5 |

The separation to hold: **`ood/` provides score functions; `metrics/robustness/ood.py` provides the metrics
that evaluate any score function** — including `safetycage`'s and a user's own. That is what makes the loss
of OODGuard-as-a-package a net simplification rather than a gap.

> **Do this before Step 1.** The OODGuard source exists today only inside the working tree of repo A's
> submodule checkout. Copy `KNN.py` and `OODGuard.py` out to a scratch location before removing the
> submodule, or the code becomes unreachable.

### 3.3 Python version — decide before Step 3

| | Python | numpy | pandas | other |
|---|---|---|---|---|
| A `main` (requirements) | – | 1.26.4 | 2.2.3 | sklearn 1.6.1, seaborn, `pyrootutils` |
| A `main` (CI) | **3.10** | | | |
| A `main` (README) | **3.11** | | | |
| B | 3.11 | 1.26.4 | 2.3.3 | `darts==0.39.0` |
| C | 3.11 | 1.26.4 | 2.2.3 | `lightgbm` |
| D | **≥3.13** | **≥2.4.3** | **≥3.0.1** | `torch`, `transformers`, `textattack` |
| `safetycage` | **≥3.13** | – | – | `joblib`, `matplotlib` |

Repo A `main` currently disagrees with itself in three places (3.10 in CI, 3.11 in README, unpinned in
requirements). Taking `safetycage` as a hard dependency forces ≥3.13, and with it numpy 2.x / pandas 3.x.
Repo B's `darts==0.39.0` is the risk and must be verified against that stack. Options:

1. **Repo A targets ≥3.13 with `safetycage` required.** Aligns with D and the package. Requires resolving
   darts — or dropping the hard darts dependency from the core, which §5.1 already forbids below
   `core/adapters.py`.
2. **`safetycage` as an optional extra** (`pip install themis-assess[safetycage]`), core at ≥3.11.
   Keeps repo C's stack viable; misclassification detection becomes opt-in.
3. Target ≥3.13 and drop darts entirely, treating forecasting data as `(entity, time, feature)` arrays at
   the adapter boundary.

**Recommendation: option 2 now, option 1 once darts is verified.** Either way, fix the 3.10/3.11
disagreement in the same commit.

### 3.4 Packaging

Adopt repo D's layout, since repo A must be installable without editing its consumers:

```
pyproject.toml            hatchling; name e.g. "themis-assess"
src/themis_assess/…       the package
tests/                    mirrors src/  (keep the notebook tests; add unit tests)
examples/                 one notebook per method family
docs/
.gitlab-ci.yml            extend: pytest on src/, keep --nbmake for examples/
```

This retires `pyrootutils`, `.project-root`, the `os.getcwd()`-dependent import, and the bare
`from base_model_handler import …`. Repo D shows it works: five test modules, `uv.lock`, `.python-version`.

---

## 4. Design principles

1. **Separate perturbation, evaluation, and scoring.** Repo D already does this; generalise its shape.
2. **Robustness metrics are pure functions of numbers.** `f(clean, perturbed, baseline=…) -> float`.
   No model, no data, no I/O. Classification and forecasting then share *identical* metric code — which is
   what would have prevented four divergent `compute_robustness` lineages.
3. **Data types are handled by adapters, never by the algorithm.** No module under `perturbations/` or
   `metrics/` may import `darts`, `torch`, or `transformers`.
4. **Severity is a first-class ordered axis.**
5. **Declare access requirements, and also introspect them.** Declared for up-front reporting;
   introspected (repo D's `_missing_metric_args`, `main`'s `_call_metric`) for per-call dispatch.
6. **The baseline is another model handler, not methods on one.** Repo B's
   `ValenciaModelHandler_NaiveMovingAverage` is the better pattern; the `create_baseline_model()` /
   `get_baseline_model_predictions()` abstract methods that appear on some branches are not needed.
   `CE/mCE`, `RCE/mRCE`, `REI/mREI` and `FR/mFR` then take a second handler, not a mode flag.
7. **Subgroups are masks, never feature indices** (§5.6).
8. **Every plot has a `compute_*_data` counterpart** (repo D's convention).
9. **Use the survey's names, exactly.** Two repos call Eq. 23 `flip_rate` when Eq. 24 owns that name.
10. **Fail loudly on unsupported combinations.** Flip-rate on a forecasting model is a configuration error,
    not a silent NaN.

---

## 5. Module layout and core contracts

```
src/themis_assess/
  core/
    adapters.py       DataAdapter protocol; ArrayAdapter, FrameAdapter, TextAdapter, DartsAdapter
    types.py          PerturbationSpec, Severity, TaskType, AccessLevel, Scope, TimeScope, Modality
    subgroups.py      SubgroupSpec, mask resolution
    results.py        EvaluationResult, RobustnessResult, SeverityCurve
    dispatch.py       signature introspection (from repo D / main's _call_metric)

  perturbations/
    base.py           BasePerturbationMethod (from repo D), extended with severity
    registry.py       PerturbationHandler (from repo D) + YAML instantiation
    tabular.py        gaussian noise, feature scaling, offsets, missing values,
                      category swap, conditional-distribution sampling
    timeseries/
      value_domain.py jitter, scaling, offset/drift, damping, flat-sensor, spikes,
                      magnitude warp, missing segments, alpha-stable noise
      time_axis.py    window slicing, time warping, speed change, window shift
      frequency.py    spectrogram perturbation, VTLP
      structural.py   STL decomposition; perturb residual / seasonal / trend selectively
    text.py           CharNoise, Masking, WordEmbeddingSwap, ContextualWordSwap (from repo D)

  protocols/
    splits.py         domain, temporal, subpopulation, compositional, near/far OOD

  metrics/
    registry.py       MetricRegistry (recovered from repo B, fixed per §2.1)
    normalize.py      normalize_metric (recovered from repo B, fixed per §2.1)
    accuracy/
      classification.py                    (from main — already generic)
      forecasting.py                       (recovered from repo B)
      weighted.py     event_weights, weighted_mae/mse, mean_signed_deviation (from repo B)
    robustness/
      degradation.py  AccDrop, RelAccDrop, ErrRatio, macro error, R_phi, R_Phi,
                      EI, REI, mREI, CE, mCE, RCE, mRCE
      stability.py    FP, FR, mFR, confidence_retention
      subgroup.py     A_g, WGA, Gap, VarAcc, group disparity, equal-weight risk, Welch ANOVA
      margins.py      EBD, EBD2, CLEVER
      embedding.py    ROBY, RDI
      activation.py   neuron sensitivity, NCCR
      distance.py     Wasserstein, MMD, KL
      calibration.py  ECE, PICP, MPIW, CRPS, energy score, Brier, NLL, TemperatureScaler
      ood.py          AUROC, AUPRC, FPR@95TPR, AUTC, wAUTC
      composite.py    URM, s_HR

  robustness/
    ood/              base.py, scores.py

  assessment/
    base.py
    accuracy/{classification,forecasting}.py
    robustness/{classification,forecasting}.py
    sensitivity.py    SensitivityMap — feature x time, feature x sample, token x position
    error_model.py    TSFresh + gradient-boosting error meta-model with SHAP (from repo B)
    error_analysis.py outlier / feature / class error attribution (from repo C)

  integrations/
    safetycage.py     handler -> DataModule / ModelModule bridges

  report/             rules.py, render.py
  viz/                forecast.py, heatmap.py, polar.py, severity.py, flow.py, calibration.py
```

### 5.1 `DataAdapter`

The abstraction that lets one perturbation serve darts `TimeSeries` lists, numpy `(N, T, F)` arrays,
pandas frames and text corpora — and where the `darts` / `torch` dependencies stop.

```python
class DataAdapter(Protocol):
    @property
    def modality(self) -> Modality: ...            # TABULAR | SEQUENCE | TEXT
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def categorical_features(self) -> list[str]: ...
    @property
    def numerical_features(self) -> list[str]: ...
    @property
    def n_entities(self) -> int: ...               # vessels, patients, documents
    @property
    def n_timesteps(self) -> int | None: ...       # None for non-sequential

    def values(self) -> np.ndarray: ...            # (n_entities, n_timesteps, n_features)
    def with_values(self, values: np.ndarray) -> "DataAdapter": ...
    def category_pool(self, feature: str) -> np.ndarray: ...
```

**Where feature types come from.** Repo B has already moved on this: its
`list_categorical_and_numerical_features` no longer infers types from the data (the earlier heuristic was
"constant within every entity"), it parses an external declaration file,
`data/categorical_and_numerical_features.txt`. That is the right direction — a declaration, not an
inference — but the wrong home. Put it on the **data handler** as a declared property, with the
constant-within-entity heuristic as an overridable default. A loose text file beside the data is not
discoverable and cannot be validated.

### 5.2 `BasePerturbationMethod` and `PerturbationSpec`

Take repo D's ABC as the base and add the severity axis it lacks:

```python
@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    severity: int                                  # ordered, 1..S
    params: Mapping[str, Any] = field(default_factory=dict)
    scope: Scope = Scope.ALL_FEATURES              # ALL | NUMERICAL | CATEGORICAL | NAMED
    features: tuple[str, ...] = ()
    time_scope: TimeScope = TimeScope.CONSTANT     # CONSTANT | PER_TIMESTEP | AT_INDICES
    time_indices: tuple[int, ...] = ()
    n_variants: int = 1                            # D's transformations_per_example, B's n_copies

class BasePerturbationMethod(ABC):
    name: str
    task_types: ClassVar[frozenset[TaskType]]
    modalities: ClassVar[frozenset[Modality]]
    label_preserving: ClassVar[bool] = True
    severity_levels: ClassVar[int] = 5

    @abstractmethod
    def perturb(self, data: DataAdapter, spec: PerturbationSpec,
                rng: np.random.Generator) -> list[DataAdapter]:
        """Return `spec.n_variants` perturbed copies."""

    def sequence(self, data: DataAdapter, spec: PerturbationSpec,
                 rng: np.random.Generator) -> list[DataAdapter]:
        """Monotonically increasing severity 1..S — required by FP/FR/mFR (Eqs. 23-25)."""

    def metadata(self) -> dict: ...                # keep repo D's reporting hook
```

`scope` and `time_scope` collapse repo B's sixteen `perturb_*` methods into a handful of implementations
plus a spec: `perturb_only_numerical_over_time`, `perturb_all_data` and `perturb_one_feature_at_k_times`
become configurations of `GaussianNoise` and `CategorySwap`, not separate functions.

`sequence()` is genuinely new. The flip-rate family needs *monotonically increasing severity sequences*,
not independent draws; both existing `flip_rate` implementations compare a single perturbed pair, which is
a different and weaker quantity.

Every perturbation takes an explicit `rng`. Repo D's `MaskingPerturbation` uses global `np.random`;
repo B's methods accept a `seed` but reseed globally. Neither is reproducible under concurrency.

Keep `PerturbationHandler` from repo D essentially as-is, adding YAML instantiation (§11.5) and a
`filter(task, modality)` query so the runner can report which registered methods apply.

### 5.3 Extended `MetricSpec`

Recover repo B's version, fix §2.1, and extend:

```python
@dataclass(frozen=True)
class MetricSpec:
    higher_is_better: bool | None                  # None for symmetric metrics
    lower_bound: float | None = None
    upper_bound: float | None = None
    target: float | None = None                    # was `best`; the ideal value, e.g. 0
    scale: float | None = None
    # new:
    tasks: frozenset[TaskType] = frozenset({TaskType.CLASSIFICATION, TaskType.FORECASTING})
    access: AccessLevel = AccessLevel.BLACK_BOX
    requires_baseline: bool = False
    paper_reference: str | None = None             # e.g. "survey §5.1.3 Eq. 19"
```

Renaming `best` to `target` resolves the conflation flagged in §2.1: `target` is where the metric is
ideal, `upper_bound` is where normalisation ends. `mean_signed_error` has `target=0` and no bounds.

`AccessLevel` ∈ {`BLACK_BOX`, `PROBABILISTIC`, `QUERY`, `EMBEDDING`, `ACTIVATION`, `GRADIENT`}, ordered,
so the runner checks a handler's declared capability once and lists every unavailable assessment before
running anything.

`paper_reference` is not decoration: since the survey is our own, a machine-checkable link between
implementation and equation number is what keeps the two from drifting.

`main`'s `default_metric_ranges` / `default_acceptable_metric_values`, repo C's `default_metric_ranges`
and repo D's `metric_ranges` / `acceptable_metric_values` are four parallel informal versions of this
registry. Fold all four in.

### 5.4 Robustness metric signature

```python
def corruption_error(perturbed_errors: Sequence[float],
                     baseline_perturbed_errors: Sequence[float]) -> float:
    """CE — survey §5.1.3 Eq. 19. Sequences are indexed by severity s = 1..S."""
```

Pure, numeric, no model, no data.

### 5.5 `RobustnessResult`

The clean value, the per-severity values and the baseline values must all survive to the end so the report
layer can enforce its rules — never collapsed early.

```python
@dataclass
class RobustnessResult:
    metric_name: str
    clean: float
    per_perturbation: dict[str, SeverityCurve]     # name -> {severity: value}
    baseline: dict[str, SeverityCurve] | None
    robustness_scores: dict[str, float]
    aggregate: float | None
    access_used: AccessLevel
    perturbation_metadata: dict[str, dict]         # from BasePerturbationMethod.metadata()
    n_samples: int
    seed: int | None
```

### 5.6 Subgroups as masks

Repo C states the problem plainly:

> "I don't think this is so easy to abstract so that it is applicable to other use-cases. The medical use
> case has ohe features in a tabular format — which allow me to select a subset of the population very
> easily. But this is not the case for the media use case for instance."

Correct as stated — *feature indexing* does not abstract. Masks do:

```python
@dataclass(frozen=True)
class SubgroupSpec:
    name: str
    mask_fn: Callable[[Any], np.ndarray]           # dataset -> boolean mask
    description: str = ""
```

| Repo | Mechanism | As a mask |
|---|---|---|
| C | one-hot column selection | `lambda ds: ds[col] == 1` |
| D | embedding-similarity themes | `lambda ds: sim(ds, theme) > τ` |
| B | callable `category(series) -> label`; lead-time bins | `lambda ds: category(ds) == label` |
| A `main` | `{"age": {"min": 30, "max": 40}, "job": ["nurse"]}` dict | compile the dict to a mask |

`metrics/robustness/subgroup.py` then only ever sees `dict[str, np.ndarray]` — no feature semantics, no
modality. `main`'s existing dict form stays as a convenience constructor over the mask.

---

## 6. Metric catalogue

Mapped to survey sections. **Access:** BB = predictions only, P = probabilities/intervals, Q = repeated
queries, E = embeddings, A = activations, G = gradients. **Base** = requires a baseline model.
**Have** = present today, with source (A = repo A `main`).

### 6.1 Data-centric — domain-level shift (§5.1.1)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Macro-average error | 2 | both | BB | – | – |
| AccDrop | 3 | both | BB | – | – |
| RelAccDrop | 4 | both | BB | – | – |
| ErrRatio | 5 | both | BB | – | – |

### 6.2 Data-centric — subgroup shift (§5.1.2)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Per-group accuracy `A_g` | – | both | BB | – | ✓ A, B, C, D¹ |
| Worst-group accuracy | 6 | both | BB | – | – |
| Accuracy gap | 7 | both | BB | – | – |
| Variance of group accuracies | 8 | both | BB | – | – |
| Group disparity `Δ_group` | 9 | both | BB | – | – |
| Equal-weight risk | 10 | both | BB | – | – |
| *Welch ANOVA significance test* | – | both | BB | – | ✓ B² |
| *Accuracy per lead time to event* | – | fc | BB | – | ✓ B³ |

¹ A `compute_accuracy_by_subgroup`; B `compute_aggregated_accuracy_per_category` (+ bar variant);
C `compute_accuracy_by_subgroup` + `print_error_demographic`; D `per_class_assessment` +
`per_theme_assessment`. Four mechanisms, one metric — §5.6.
² Not in the survey. Tests whether an observed subgroup gap is significant rather than leaving it to the eye.
³ `compute_accuracy_per_lead_time` — accuracy binned by the interval between forecast origin and a target
event. Not in the survey; a forecasting-native subgroup axis and a strong repo-A candidate.

### 6.3 Data-centric — synthetic perturbations (§5.1.3)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Accuracy-based robustness `R_φ` | 11 | clf | BB | – | ✓ ×5 (§2) |
| Total accuracy robustness `R_Φ` | 12 | clf | BB | – | ✓ A, B |
| Relative error ratio | 13 | both | BB | – | – |
| Error-based robustness per disturbance | 14 | both | BB | – | – |
| Error-based robustness, multiplicative | 15 | both | BB | – | – |
| Error inflation `EI` | 16 | both | BB | – | – |
| Relative error inflation `REI` | 17 | both | BB | ✓ | – |
| Mean relative error inflation `mREI` | 18 | both | BB | ✓ | – |
| Corruption error `CE` | 19 | clf | BB | ✓ | – |
| Mean corruption error `mCE` | 20 | clf | BB | ✓ | – |
| Relative corruption error `RCE` | 21 | clf | BB | ✓ | – |
| Mean relative corruption error `mRCE` | 22 | clf | BB | ✓ | – |
| Flip probability `FP` | 23 | clf | BB | – | partial A, D⁴ |
| Flip rate `FR` | 24 | clf | BB | ✓ | – |
| Mean flip rate `mFR` | 25 | clf | BB | ✓ | – |
| *Absolute relative robustness* | – | both | BB | – | ✓ A, B⁵ |
| *Confidence retention* | – | clf | P | – | ✓ D⁶ |

⁴ Both compute the flip proportion over a *single* perturbed pair, not a severity sequence. Keep the
quantity, name it `FP`, and add `sequence()` support (§5.2).
⁵ `abs_rel_robustness_metric` — `|1 − R_φ|`, symmetric in the direction of change, zero-is-best. Not in
the survey; keep as a repo-A original and document it as such.
⁶ Mean probability retained on the *originally predicted* class. Preserve repo D's docstring caution
verbatim in the port — using the perturbed sample's own top-class probability would make a confident label
flip read as retained confidence. A soft-margin proxy that, unlike §5.2.1, needs no gradients.

### 6.4 Composite (§5.1.4)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Unified Robustness Metric (URM) | 26 | clf | BB | – | – |
| Composite reliability score `s_HR` | 27 | both | BB+ | – | – |

Both must be reported with their components — §8.

### 6.5 Model-centric (§5.2) — now reachable via `safetycage`'s `ModelModule`

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Empirical boundary distance (EBD) | 28–30 | clf | Q | – | – |
| EBD2 (gradient-guided) | – | clf | G | – | – |
| CLEVER | 31 | clf | G | – | – |
| ROBY | 32–34 | clf | E | – | – |
| Robustness Difference Index (RDI) | 35–36 | clf | E | – | – |
| Neuron sensitivity | 37 | both | A | – | – |
| Neuron cover change rate (NCCR) | – | both | A | – | – |

`EBD` needs only repeated black-box queries and no gradients — the cheapest of the group, and worth doing
in Tier 1 rather than Tier 3.

### 6.6 Distribution distance (§5.3)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Wasserstein `W_p` (+ Sinkhorn) | 38 | both | model-free | – | – |
| KL divergence | 39 | both | model-free | – | – |
| Maximum Mean Discrepancy | 40–41 | both | model-free | – | – |
| *Mahalanobis distance to train distribution* | – | both | E | – | ✓ D |

Context, not robustness scores. The report layer must present them as such — §8.

### 6.7 Calibration under shift (§5.4)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| Expected calibration error (ECE) | 42 | clf | P | – | ✓ D |
| Reliability diagram | – | clf | P | – | ✓ D |
| PICP | 43 | fc | P | – | – |
| MPIW | – | fc | P | – | – |
| Brier score, NLL / log loss | – | clf | P | – | ✓ C |
| CRPS | 44 | fc | P | – | – |
| Energy score | 45 | fc (multi-horizon) | P | – | – |
| *Predictive entropy* | – | clf | P | – | ✓ D |
| *Temperature scaling* | – | clf | P | – | ✓ D |
| *Monte Carlo dropout spread* | – | both | P | – | ✓ D |
| *Ambiguity threshold from entropy* | – | clf | P | – | ✓ D |

PICP and MPIW must always be reported as a pair; either alone is gameable. Repo C's
`calibration_under_shift` is the right *idea* implemented wrongly (§2.1) — a correct version is simply
"compute the §5.4 metrics on perturbed data", which the survey insists on:

> "calibration established on clean data need not survive a shift… a robustness report that cites only
> clean-data calibration can be misleading"

Monte Carlo dropout and ensemble spread are also the survey's recommended substitute in chaotic systems
(§6.3), where invariance to small perturbations is not a valid criterion at all — relevant to forecasting.

### 6.8 OOD detection (§5.5)

| Metric | Eq. | Task | Access | Base | Have |
|---|---|---|---|---|---|
| AUROC | 46 | score-based | BB | – | partial (OODGuard) |
| AUPRC | 47 | score-based | BB | – | partial (OODGuard) |
| FPR@95%TPR | – | score-based | BB | – | – |
| AUTC | 48 | score-based | BB | – | – |
| Weighted AUTC | 49 | score-based | BB | – | – |

### 6.9 Label-free OOD performance estimation (§6.1)

Covered by no existing toolkit and directly useful where OOD labels are exactly what is missing:

- **Agreement-on-the-line** — agreement between two models on *unlabelled* OOD data predicts OOD accuracy.
- **Accuracy-on-the-line regime diagnosis** — probit-transformed OOD vs ID accuracy across models,
  classified into positive-linear / vertical / horizontal / none.

---

## 7. Perturbation and instrument catalogue

### 7.1 Tabular / feature-space (from repo C)

| Method | Source | Survey mapping |
|---|---|---|
| Gaussian feature noise | `noise_sensitivity` ×2 | §4.2 value-domain |
| Feature scaling on a random subset | `distribution_shift` | covariate shift, §2 Def. 2 |
| Large fixed offsets on chosen features | `adversarial_edge_cases` | §4.2 sensor offset (extreme) |
| Stratified subsampling | `class_imbalance_sensitivity` | §3.1 subpopulation split |
| Missing values at given rates | Plovdiv `compute_robustness_against_missing_values` | §4.2 missing-data |
| Resampling (+ ECDF plot) | Plovdiv `compute_robustness_against_resamplings` | §3.1 subpopulation split |
| Conditional-distribution sampling | **missing** | §7 realism caveat — **high priority** |

`class_imbalance_sensitivity` and the resampling methods are really **split protocols** (§3.1
subpopulation) implemented as perturbations. They belong in `protocols/splits.py`, which makes them
reusable for the §5.1.2 subgroup metrics.

The multivariate-realism caveat applies directly to repo B's `perturb_numerical_data` and repo C's
`noise_sensitivity`, both of which add independent per-feature noise. Survey §7:

> "perturbing individual features independently can produce physically implausible combinations that
> violate the statistical structure of real data. Perturbation protocols that respect observed feature
> correlations, for instance, by sampling from estimated conditional distributions, are needed"

### 7.2 Time series (survey §4.2, organised by acting domain)

| Family | Method | Status |
|---|---|---|
| Value-domain | additive Gaussian noise (jitter) | ✓ B `perturb_numerical_data_over_time` |
| Value-domain | constant sensor offset | ✓ B `perturb_numerical_data` |
| Value-domain | amplitude scaling | missing |
| Value-domain | sensor drift | missing |
| Value-domain | exponential damping | missing |
| Value-domain | flat-sensor episode | missing |
| Value-domain | outlier spikes | missing |
| Value-domain | magnitude warping | missing |
| Value-domain | missing-data segments | missing |
| Value-domain | alpha-stable impulsive noise | missing |
| Time-axis | **prediction-window shift** | ✓ B `time_robustness_heatmap`⁷ |
| Time-axis | window slicing | missing |
| Time-axis | time warping | missing |
| Time-axis | speed change | missing |
| Frequency | spectrogram perturbation, VTLP | missing (low priority — no speech use case) |
| Structural | STL component-selective perturbation | missing — **high priority** |
| Categorical | category swap, constant / per-timestep / at-indices | ✓ B, repo-A original |

⁷ New in repo B since the earlier survey of this code: measures how much the prediction changes when the
prediction window shifts one block forward in time. A *temporal* stability instrument distinct from
feature sensitivity, and the closest thing anyone has built to the survey's time-axis family. Generalise
as an `InputWindowShift` perturbation feeding the standard stability metrics.

**STL** is the survey's recommended way to keep time-series perturbations structure-faithful: perturb the
residual while preserving trend and seasonality, or alter seasonal amplitude alone. Cheap relative to its
value. The **categorical** family has no analogue in the survey — a genuine repo-A contribution, and the
one repo-B perturbation that transfers directly to repo C's tabular setting.

### 7.3 Text (survey §4.3 — from repo D, port essentially unchanged)

| Method | Survey mapping |
|---|---|
| `CharNoisePerturbation` (deletion, insertion, neighbouring swap, QWERTY, homoglyph) | §4.3 character-level noise |
| `WordEmbeddingSwapPerturbation` (counter-fitted embeddings) | §4.3 embedding-NN substitution |
| `ContextualWordSwapPerturbation` (BAE / masked LM) | §4.3 paraphrasing |
| `MaskingPerturbation` (mask-token replacement) | not in the survey — repo-A original, a partial-observability probe |
| Round-trip machine translation | missing — a few lines given `transformers` |
| Counterfactual / demographic term substitution | missing — connects §5.1.2 subgroups to §4.3, where robustness meets fairness (§2.1) |

### 7.4 Sensitivity maps

Repo B's heatmap family is the strongest original instrument anywhere in the four repos: for every
(feature, timestep) pair, perturb in isolation `n_copies` times and measure the change in prediction
against the clean prediction. `one_vessel_heatmap`, `all_vessels_heatmap`, four typed variants,
`time_robustness_heatmap`, plus `_prepare_heatmap_color_scaling` for linear/log colour scaling.

Generalise to `assessment/sensitivity.py`:

- **feature × time** for sequence models (repo B as-is);
- **feature × sample** for tabular models (immediately applicable to Plovdiv);
- **token × position** for text — repo D's `compute_masking_data` is already this, restricted to the
  masking perturbation; the two are the same instrument and should share one implementation;
- end-aligned aggregation across entities (implemented for vessels) as one of several alignment strategies.

Vessel-specific parts to strip: the `data`/`series` past-covariate/target split, `prediction_end_offset`,
`cutoff_at_berth`, and the darts calls. Behind a `DataAdapter` and a model handler `predict`, none of it is
maritime.

### 7.5 Learned error attribution — the most novel capability in any repo

Repo B has grown a family that appears in neither the survey nor any inventoried toolkit:

| Method | What it does |
|---|---|
| `TrainErrorModel` | fits an XGBoost meta-model predicting forecast error from **TSFresh** features of the input window |
| `RankFeaturesForSample` | local **SHAP** attribution — which input characteristics drove *this* sample's error |
| `GlobalFeatureImportance` | global SHAP importance, ranked by pressure toward *worse* accuracy |
| `FeatureErrorScatter` | per-feature scatter of error against feature value, with a feature selector |

This is the forecasting analogue of repo C's `compute_error_analysis_across_features` (which compares
feature distributions between correct and misclassified samples), but far stronger: a *model* of the error
rather than a group comparison. Promote to `assessment/error_model.py` with the feature extractor
pluggable (TSFresh for sequences, raw columns for tabular, embeddings for text) and the regressor
pluggable. `shap` and `tsfresh` become optional extras.

Worth stating plainly in any paper follow-up: this answers a question the survey raises but does not
instrument — *why* the model degrades, not just *how much*.

### 7.6 Event-weighted accuracy metrics (from repo B)

`custom_accuracy_metrics.py` mixes genuinely Valencia-specific event definitions with a generic weighting
mechanism:

| Function | Verdict |
|---|---|
| `berth_prediction`, `berth_timing`, `recall_at_berth_time_point` | use-case specific — leave in B |
| `_berthing_onsets` | use-case specific *definition*, generic *role* — becomes an injected `onset_fn` |
| `event_weights` (Gaussian bump per onset + floor, normalised) | **promote** — generic given an onset callable |
| `weighted_mae`, `weighted_mse` | **promote** |
| `mean_signed_deviation` | **promote** — explains the `target=0` MetricSpec field (§5.3) |

Event-weighted error is the right metric family whenever accuracy near a rare transition matters more than
average accuracy — true of most operational forecasting, not just berthing.

### 7.7 Error attribution by group (from repo C)

`compute_error_outlier_analysis` (are errors concentrated on outliers?), `error_analysis_by_true_class`,
`compute_error_analysis_across_features`. Promote to `assessment/error_analysis.py`. The closest thing in
any repo to the survey's §7 component-level fault attribution, applied to features instead of pipeline
stages.

---

## 8. The report layer

The survey's reporting prescriptions are testable rules, and enforcing them is where repo A goes beyond
being a metric collection. `report/rules.py` should refuse — or warn loudly — when a report:

1. states a relative-drop metric (`AccDrop`, `RelAccDrop`, `ErrRatio`, `EI`) without absolute performance
   alongside it;
2. reports `mCE` without `mRCE`;
3. reports `R_Φ` without naming the perturbation set `Φ` (sum-based aggregation scales with `|Φ|`);
4. presents a composite (`URM`, `s_HR`) without its components and weights;
5. reports a single severity instead of a severity curve;
6. reports an aggregate without per-domain or per-group breakdowns;
7. reports calibration measured on clean rather than shifted data;
8. reports OOD detection without naming the operating threshold;
9. presents a distribution distance as if it were a robustness score;
10. reports `PICP` without `MPIW`.

Per survey §8, a complete robustness claim states four things — what is varied, what is monitored, how the
change is measured, and **how much change the application tolerates**. The first three we compute. The
fourth cannot come from the library: "The tolerance is the one element no evaluation instrument provides,
as it must come from the requirements of the deployment itself." Make it a required field of the report
object, supplied by the user, so its absence is visible rather than implicit.

`main`'s `default_acceptable_metric_values` and repo D's `acceptable_metric_values` are informal
precursors of exactly this field. `Metrics_list.yaml` — identical in A and C, describing each metric in
prose — is the precursor of the rendered report's glossary; keep it and grow it.

---

## 9. Migration plan

**Step 0 — Rescue the OODGuard source** before removing the submodule (§3.2). It exists only in the
working tree.

**Step 1 — Extraction, not merging.**
No `git merge-base` diffing, no subtree merges, no history import: repo D is not a mirror of A, and B and C
have diverged structurally (B's handlers target a base-class signature `main` does not have — §1.4). Each
method is ported by hand into the new layout, with the source repo and symbol recorded in a docstring line
(`Ported from themis_valencia ValenciaRobustnessAssessment.time_robustness_heatmap`) so provenance
survives. B, C and D remain frozen reference material.

**Step 2 — Recover what drifted out of A** (§1.2): `MetricRegistry`, `normalize_metric` and the 400-line
`ForecastingAccuracyAssessment` from repo B, fixing the defects in §2.1 during the port. Do this early —
several later steps depend on the registry.

**Step 3 — Write and freeze the contracts** in `core/`, plus the ABCs in `perturbations/` and `metrics/`.
Nothing else moves until these are agreed. Settle the `ForecastingModelHandler` signature here (§1.4).

**Step 4 — Repackage** as `src/`-layout with `pyproject.toml` and a `tests/` tree mirroring `src/`. This
removes `pyrootutils`, the `os.getcwd()`-dependent import and the bare `from base_model_handler import …`.
Resolve the Python-version question (§3.3) and the CI/README disagreement in the same commit. Extend
`.gitlab-ci.yml` to run unit tests as well as notebooks, and fix the artifact glob.

**Step 5 — Promote in dependency order.**

| Order | Layer | Rationale |
|---|---|---|
| 1 | `metrics/robustness/*` | pure functions, no dependencies, immediately unit-testable; canonicalises five `R_φ` copies and four `compute_robustness` lineages |
| 2 | `metrics/accuracy/*`, `metrics/registry.py`, `metrics/normalize.py` | folds in four parallel `metric_ranges` dicts |
| 3 | `core/adapters.py`, `core/subgroups.py` | prerequisite for generic perturbations and subgroup metrics |
| 4 | `perturbations/*` | repo D's registry as the base; repo B's sixteen methods become spec-driven; repo C's tabular set added |
| 5 | `protocols/splits.py` | absorbs repo C's resampling/imbalance methods, which are splits not perturbations |
| 6 | `robustness/ood/*`, `integrations/safetycage.py` | replaces both submodules |
| 7 | `assessment/*` | orchestration only, now thin; repo D's `RobustnessEvaluator` is the closest existing model |
| 8 | `assessment/error_model.py`, `error_analysis.py`, `sensitivity.py` | the three original instrument families (§7.4, §7.5, §7.7) |
| 9 | `viz/*` | de-duplicates four `plot_polar_chart_metric` and two `plot_sankey` copies |
| 10 | `report/*` | new capability |

**Step 6 — Acceptance criteria, inside repo A.**
Since B, C and D cannot be modified, correctness has to be demonstrated in repo A itself:

1. **Golden-value tests.** For each ported method, a pytest case with hand-computed expected values, or
   values captured from the source implementation where it is known-correct. Several survey equations are
   simple enough to verify by hand.
2. **Fixture-based reproduction.** Small committed synthetic fixtures — one tabular, one multivariate
   sequence, one text — exercising every perturbation × metric × task combination. No use-case data.
3. **`examples/` wiring demonstrations.** One notebook per use-case *shape* (forecasting-with-darts,
   tabular-classification, text-classification), showing what a data handler and model handler for that
   shape look like against the new API. These stand in for B, C and D without touching them.
4. **Equivalence checks where feasible.** Where a use-case repo's numbers appear in its notebooks, assert
   the ported implementation reproduces them on the same fixture. Read-only comparison, no edits.

**Step 7 — Fill the survey gaps**, prioritised by access level:

- *Tier 1 (black-box)* — §5.1 degradation, subgroup and perturbation metrics; §5.3 distances; §5.5 OOD
  metrics; plus `EBD` from §5.2.1, which needs only repeated queries. Cheap, covers most of the survey.
- *Tier 2 (probabilistic)* — §5.4 calibration. Extend `BaseModelHandler` with optional `predict_proba` /
  `predict_interval`. Repo D already has the classification half.
- *Tier 3 (white-box)* — §5.2 embeddings and activations, on `safetycage`'s `ModelModule` (§3.1).

**Step 8 — Documentation.** One example per method family. The survey's §6 pitfalls belong in prose beside
the methods they qualify — in particular that synthetic robustness does not imply natural robustness, and
that label invariance breaks down as severity grows.

---

## 10. Cross-repo inventory

### 10.1 Repo A `main` — keep, fix, or delete

| Component | Verdict |
|---|---|
| `ClassificationAccuracyAssessment` (484 lines) | **keep as the model for the rest**; move plots to `viz/` |
| `default_metric_functions` / `_ranges` / `_acceptable_values` | fold into `MetricRegistry` |
| `_call_metric` (signature introspection) | promote → `core/dispatch.py` |
| `BaseRobustnessAssessment.rel_/abs_rel_robustness_metric` | promote; add epsilon from repo B |
| `BaseRobustnessAssessment.compute_robustness` | **rewrite** (§1.3) |
| `ClassificationRobustnessAssessment.flip_rate` | promote as `FP`; add `sequence()` |
| `ForecastingRobustnessAssessment` (110 lines) | **delete** — broken duplicate |
| `ForecastingAccuracyAssessment` (50 lines) | **replace** with repo B's 400-line version |
| `DataHandler`, `BaseModelHandler`, `ForecastingModelHandler` | keep contract; fix `super()`, imports, signature (§1.3, §1.4) |
| `Metrics_list.yaml` | keep; grow into the report glossary |
| `.gitlab-ci.yml` | keep; extend and fix the artifact glob |
| `tests/*.ipynb` | keep as `examples/`; add real unit tests |
| both submodules | **remove** (§3) |

### 10.2 Repo B — `themis_valencia`

| Method | Coupling | Verdict |
|---|---|---|
| 16 × `perturb_*` | darts, vessel semantics | split → `perturbations/timeseries` + spec |
| `list_categorical_and_numerical_features` + `_parse_feature_file` | external txt file | promote → data-handler declared property (§5.1) |
| `one_vessel_heatmap`, `all_vessels_heatmap` (+ 4 typed variants) | darts, vessel/series split | promote → `SensitivityMap` |
| `time_robustness_heatmap` | darts, `cutoff_at_berth` | promote → `InputWindowShift` + `SensitivityMap` |
| `_prepare_heatmap_color_scaling` | none | promote → `viz/heatmap.py` |
| `MetricRegistry`, `normalize_metric` | none | **recover into A** (§1.2), fix §2.1 |
| `ForecastingAccuracyAssessment` (400-line ABC) | darts | **recover into A** |
| `rel_/abs_rel_robustness_metric` with epsilon | none | promote (epsilon version wins) |
| `compute_accuracy_over_time`, `compute_aggregated_accuracy` | darts | promote behind adapter |
| `compute_aggregated_accuracy_per_category` (+ bar) + Welch ANOVA | callable category | promote → `subgroup.py` |
| `compute_accuracy_per_lead_time` | event callable | promote → `subgroup.py` |
| `TrainErrorModel`, `RankFeaturesForSample`, `GlobalFeatureImportance`, `FeatureErrorScatter` | TSFresh, XGBoost, SHAP, darts | promote → `assessment/error_model.py` (§7.5) |
| `event_weights`, `weighted_mae`, `weighted_mse`, `mean_signed_deviation` | onset callable | promote → `metrics/accuracy/weighted.py` |
| `berth_prediction`, `berth_timing`, `recall_at_berth_time_point`, `incident_locator` | berthing semantics | leave in B |
| `plot_accuracy_over_time`, `plot_accuracy_at_forecast_horizon_n` | darts | promote → `viz/forecast.py` |
| `plot_interactive_forecast_plot`, `animate_interactive_forecast_plot` | darts | promote, de-duplicate |
| `plot_polar_chart_metric` | none | promote → `viz/polar.py` (canonical of four) |
| `get_first_paragraph`, `wrap_title` | none | promote → `viz/` helpers |
| `ValenciaModelHandler_NaiveMovingAverage` | darts | pattern: **baseline as a second handler** (§4.6) |

### 10.3 Repo C — `themis_medical`

| Method | Coupling | Verdict |
|---|---|---|
| `RobustnessEvaluator.noise_sensitivity` | none | split → `perturbations/tabular.py` |
| `.distribution_shift` | none | split → `perturbations/tabular.py` |
| `.adversarial_edge_cases` | none | split → `perturbations/tabular.py` |
| `.class_imbalance_sensitivity` | none | promote → `protocols/splits.py` |
| `.calibration_under_shift` | none | rewrite (§6.7) |
| `ClassificationRobustnessAssessment` | none | **discard** — non-functional duplicate (§2.1) |
| `robustness/assessment/forecasting.py` | none | **discard** — identical to A's broken copy |
| Plovdiv missing-values robustness | one-hot categories | split → perturbation + subgroup metric |
| Plovdiv resampling robustness, ECDF plot | none | promote → `protocols/` + `viz/` |
| `compute_error_outlier_analysis` | none | promote → `assessment/error_analysis.py` |
| `error_analysis_by_true_class` | none | promote → `assessment/error_analysis.py` |
| `compute_error_analysis_across_features` | tabular features | promote behind adapter |
| `compute_accuracy_by_subgroup` | one-hot indexing | rewrite → `SubgroupSpec` masks (§5.6) |
| `print_error_demographic` | tabular | rewrite as a report section, not `print` |
| `get_metrics_table` (model vs baseline) | none | promote → `report/` |
| `plot_gauges`, `plot_metrics_by_subgroups` | none | promote → `viz/` |
| SafetyCage / OODGuard `*_themis.py` wrappers | `sys.path`, Hydra configs | **do not port** (§3.1, §3.2) |

### 10.4 Repo D — `themis_media`

| Method | Coupling | Verdict |
|---|---|---|
| `BasePerturbationMethod`, `PerturbationHandler` | none | **promote as the basis** (§5.2) |
| `CharNoise`, `WordEmbeddingSwap`, `ContextualWordSwap` | TextAttack | promote → `perturbations/text.py` |
| `MaskingPerturbation` | none | promote; fix global RNG |
| `robustness_ratio`, `confidence_retention`, `flip_rate` | none | promote; rename to survey names (§6.3) |
| `_compute_metrics`, `_missing_metric_args` | none | **promote → `core/dispatch.py`** |
| `_resolve_label_permutation` | none | promote → `core/` |
| `build_perturbation_confusion_matrix` | none | promote → `assessment/robustness` |
| `compute_masking_data` | tokenizer | merge with `SensitivityMap` (§7.4) |
| `per_class_assessment`, `assess_subset`, `global_assessment` | dataloaders | promote → assessment orchestration |
| `per_theme_assessment` (embedding similarity) | RoBERTa embeddings | rewrite → `SubgroupSpec` mask |
| `compute_semantic_similarity` | model embeddings | promote → label-preservation check⁸ |
| `expected_calibration_error`, `reliability_plot` | none | promote → `calibration.py` |
| `predictive_entropy`, `get_ambiguity_threshold` | none | promote → `calibration.py` |
| `TemperatureScaler` | torch | promote → `calibration.py` (optional torch extra) |
| `mahalanobis_distance`, `compute_embedding_stats` | none | promote → `robustness/ood/scores.py` |
| `compute_monte_carlo_dropout_data` | torch dropout | promote → uncertainty; relevant to §6.3 |
| `plotting/robustness.py` (9 functions) | none | promote → `viz/` wholesale |
| `BaseDataHandler` (465 lines) | HF tokenizer, text | pattern to follow — see §11.6 |
| `tests/` (5 modules), `pyproject`, `uv.lock` | – | **promote the structure** |

⁸ Semantic similarity between original and perturbed text is an *empirical check on the label-preservation
assumption* — survey §6.3's central pitfall, for which the paper offers no concrete instrument.
Generalising it per modality (embedding similarity for text, feature-space distance for tabular,
STL-component preservation for series) would be a repo-A contribution beyond the paper.

---

## 11. Open questions

1. **Python version and the darts pin** (§3.3). Blocking Step 4. Someone must try `darts==0.39.0`
   (or later) on Python 3.13 / numpy 2.x.
2. **The `ForecastingModelHandler` signature** (§1.4). Repo B is built against
   `(model_paths, data_handler, input_chunk_length, forecast_horizon, stride)`; `main` has
   `(model_filename, model)`. Which is canonical? Recommendation: B's, since it is the one with working
   subclasses — but with `model_paths` renamed and the extra arguments moved into `**kwargs` or a config
   object, so the base class does not assume a forecasting-shaped model.
3. **Package name.** `themis-assess`? The repo name need not change.
4. **Do we support image data?** Survey §4.1 is large and none of the three use cases is vision. Design the
   interface so image corruptions *could* be added; do not implement them.
5. **Severity convention.** `S = 5` discrete levels (ImageNet-C's convention) as the default, each
   perturbation mapping severity to its own natural parameter scale. Without a fixed convention,
   cross-perturbation aggregation (`mCE`, `mFR`, `R_Φ`) is meaningless. Agree before Step 5.
6. **Configuration format.** `Metrics_list.yaml` already establishes a human-readable metric-description
   file. Extend to one `assessment.yaml` declaring metrics, perturbations, severities, subgroups and
   protocols for a run — or adopt Hydra, which the retired submodules used. YAML plus the registry is
   probably enough; Hydra is a large dependency for the benefit.
7. **How much of repo D's data handler generalises?** Its 465-line `BaseDataHandler` does splitting,
   caching, dataloaders and serialisation — genuinely more than `main`'s five abstract methods. Proposal:
   keep `main`'s minimal contract, add an optional `BatchedDataHandler` mixin carrying the reusable parts.
8. **Optional-dependency budget.** The full promotion list implies `darts`, `torch`, `transformers`,
   `textattack`, `tsfresh`, `xgboost`, `shap`, `lightgbm`, `safetycage`, `plotly`, `seaborn`. Core must
   stay at numpy/pandas/sklearn/matplotlib, with everything else behind extras
   (`[timeseries]`, `[text]`, `[explain]`, `[safetycage]`) — otherwise the library is unusable for anyone
   who wants one metric.
9. **The notebooks.** Substantial notebooks across all four repos (Plovdiv robustness 172 kB, OOD
   detection tutorial 252 kB, Valencia heatmaps 487 kB, and the two in `main/tests/`) have not been read.
   They may contain assessment logic that never reached a module.

---

## Appendix — reference

Survey sections referenced: §2 definitions and shift taxonomy; §3 split protocols and benchmark suites;
§4.2 time-series perturbations; §4.3 text perturbations; §5.1 data-centric metrics; §5.2 model-centric
metrics; §5.3 distribution distances; §5.4 calibration under shift; §5.5 OOD detection; §6 pitfalls;
§7 composite systems, tooling inventory, and the EU AI Act Article 15 driver; §8 reporting standards.

SafetyCage: <https://pypi.org/project/safetycage/> · <https://safetycage.readthedocs.io/> ·
<https://github.com/SINTEF/safetycage>
