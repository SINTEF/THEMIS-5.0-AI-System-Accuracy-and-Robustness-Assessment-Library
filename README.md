# THEMIS 5.0 AI System Accuracy and Robustness Assessment Library

Accuracy and robustness assessment for classification and forecasting models.

Give it a fitted model and a test set, and it answers three questions: **how good is
this model**, **how much does it degrade when the data shifts**, and **is either answer
good enough for what you are about to do with it**. The third question is the one most
evaluation code leaves out, and it is why this library exists.

It is task-agnostic and framework-agnostic. Anything you can wrap in a five-line handler
can be assessed — scikit-learn, LightGBM, a time series forecaster, a transformer, a REST
endpoint behind a function. Nothing in the core depends on any packages; those are extras, imported lazily by the one module that
needs them.

This repository is created as a result of the project THEMIS 5.0 (Human-centered Trustworthiness Optimisation in Hybrid Decision Support) 
funded by the Horizon Europe program (2023-2026).
The methods included in this repository is a result of separate assessments of three AI systems within different domains; health care, news media, and port logistics.
See https://www.themis-trust.eu/projects for more information. 
The methodology follows [Michałowska & Johnsen, *Evaluating Machine Learning Robustness
Under Natural Distribution Shifts* (SINTEF Digital, 2026)](https://www.researchgate.net/publication/410991476_Evaluating_Machine_Learning_Robustness_Under_Natural_Distribution_Shifts_A_Survey_of_Benchmarks_Perturbations_Metrics_and_Assumptions?channel=doi&linkId=6a6af068f475b23f400f1ffc&showFulltext=true), referred to throughout the
code as "the survey". 

Note: This code base was orchestrated using Claude Opus 5, and was created based on the separate
accuracy and robustness assessments of the three AI system use cases explored during the THEMIS 5.0 project
funded by the Horizon Europe program between 2023 and 2026. The code base is for most parts reviewed, and edited where needed. The code base may - using help of SWE AI agents or not - include errors, or needs improvement or clarifications. We encourage users to create an "Issue" in this case. The code base includes all methods that were applied
for these three use cases, with the aim and purpose that this code base can be valuable no matter what type of
AI system that is investigated. Note that the code base for now only includes analysis for classification and forecasting tasks. Assessments for tasks with continuous outcomes/dependent variables (classical regression) or counting outcomes (Poisson regression) are not yet taken into account, and so errors may occurs for this type of tasks when using the code base.

---

## Contents

- [Install](#install)
- [Running the tutorial notebook](#running-the-tutorial-notebook)
- [The framework described in a few lines of code](#the-framework-described-in-a-few-lines-of-code)
- [The seven things the framework does](#the-seven-things-the-framework-does)
- [Three levels: global, group, individual](#three-levels-global-group-individual)
- [Architecture: the layers and what each is for](#architecture-the-layers-and-what-each-is-for)
- [Where to find what](#where-to-find-what)
- [How to extend each layer](#how-to-extend-each-layer)
- [Concepts you need in order to read the code](#concepts-you-need-in-order-to-read-the-code)
- [The results dictionary](#the-results-dictionary)
- [Comparing several models](#comparing-several-models)
- [Configuration](#configuration)
- [Where each capability came from](#where-each-capability-came-from)
- [Developing](#developing)
- [If something looks wrong](#if-something-looks-wrong)

---

## Install

Python 3.11 or newer. The version floor is deliberate: it is what lets the type
annotations stay readable without a compatibility shim, and it is old enough by now that
nothing is excluded by it.

### Three ways to work with the code

The library is meant to be *read and edited*, not just called, so pick whichever of these
fits how you already work. All three leave `src/themis_assess/` as the code that actually
runs — none of them makes a frozen copy.

**1 — Editable install (recommended).** `-e` puts a link to this source tree on the path
rather than copying it, so an edit to `src/themis_assess/` takes effect on the next
import. This is the normal way to work on a package you are also changing.

```bash
pip install -e ".[dev]"             # + pytest and ruff
python -m pytest                    # 1329 tests
```

**2 — Requirements files, no package install.** For conda, or for a venv you want to
build from a file, or when you would simply rather not install the package at all.
`pyproject.toml` puts `src/` on the test path, so the suite runs with nothing installed:

```bash
# venv
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest                                      # works: src/ is on the test path

# conda
conda env create -f environment.yml                   # includes `-e . --no-deps` + Jupyter
conda activate themis-assess
python -m pytest
```

**Do you also need `pip install -e ".[dev]"` in a conda environment?** No — and it would
be the wrong command there, because it asks pip to resolve dependencies conda has already
placed. What you *do* want is the editable **link** without the dependency resolution:

```bash
pip install -e . --no-deps
```

`environment.yml` already includes that line, so a `conda env create` gives you a working
environment in one step. Add it by hand if you built the environment another way.

Without the link, `pytest` still works — `pyproject.toml` puts `src/` on the test path —
but `import themis_assess` from a notebook or from a script in another directory does not.
For those, either install the link or set the path yourself:

```bash
PYTHONPATH=src python my_assessment.py                 # Windows: set PYTHONPATH=src
```

**3 — Just read it.** Everything is plain Python under `src/themis_assess/`, with no code
generation, no plugin discovery and no import-time magic. `git clone` and open the files.

### The requirements files

| File | What it is |
|---|---|
| `requirements.txt` | the core runtime — numpy, pandas, scikit-learn, scipy, plotly, pyyaml |
| `requirements-dev.txt` | the core plus pytest and ruff. What CI installs. |
| `requirements-tutorial.txt` | the core plus Jupyter, for `docs/tutorial.ipynb` |
| `requirements-optional.txt` | the five extras as one annotated menu, all commented out |
| `environment.yml` | conda: dev + tutorial requirements, plus `-e . --no-deps` |

**These files are generated from `pyproject.toml`.** That is the single source of truth;
after changing a dependency, run `python tools/sync_requirements.py`, and
`tests/test_requirements.py` fails if they are stale. Two hand-maintained lists of
dependencies drift, the drift is silent, and the day one is bumped and the other is not,
the conda user and the pip user are running different libraries with nothing to say so.

The extras stay commented out in `requirements-optional.txt` on purpose: each pulls in a
substantial dependency tree, and [`Darts`](https://unit8co.github.io/darts/) and `safetycage` have transitive pins that may
not co-exist in one environment. Uncomment what you need, or install one directly:

```bash
pip install -e ".[timeseries]"      # darts, statsmodels — the forecasting handler
pip install -e ".[text]"            # transformers, textattack — text perturbations
pip install -e ".[explain]"         # tsfresh, xgboost, shap — error attribution
pip install -e ".[safetycage]"      # safetycage — model-centric robustness metrics (§5.2)
pip install -e ".[tutorial]"        # jupyter, nbformat, nbclient
```

### Running the tutorial notebook

`docs/tutorial.ipynb` is the fastest way in — one model taken through every layer, with
the plots. It needs the core dependencies plus Jupyter, and nothing else: the dataset
ships with scikit-learn, so there is nothing to download.

```bash
# conda — environment.yml already includes Jupyter and the editable link
conda env create -f environment.yml
conda activate themis-assess
jupyter notebook docs/tutorial.ipynb

# or with pip, in a venv
pip install -e ".[tutorial]"
jupyter notebook docs/tutorial.ipynb
```

To regenerate it rather than read it — `docs/build_tutorial.py` executes every cell and
fails loudly if any raises, which is what keeps the prose honest about its own outputs:

```bash
python docs/build_tutorial.py
```

If the plotly figures come out blank, that is a Jupyter renderer issue rather than a
library one: `import plotly.io as pio; pio.renderers.default = "notebook"` in the first
cell usually settles it, and `figure.write_html("out.html")` always works.

## The framework described in a few lines of code

This is the whole library in one screen. It runs as written — the data ships with
scikit-learn, so there is nothing to download.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from themis_assess import (
    AccuracyEvaluator, AcceptanceCriteria, ArrayAdapter, Requirement,
    RobustnessEvaluator, RobustnessCriteria, SklearnHandler, ToleranceRequirement,
    build_report,
)

# --- a fitted model, however you like ------------------------------------
bundle = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    bundle.data, bundle.target, test_size=0.3, random_state=0, stratify=bundle.target
)
model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)

# --- wrap it, and describe the data --------------------------------------
handler = SklearnHandler(model, name="random_forest")
data = ArrayAdapter(X_test, feature_names=list(bundle.feature_names), categorical_features=[])

# --- how accurate is it, and is that enough? -----------------------------
accuracy = AccuracyEvaluator(
    handler,
    criteria=AcceptanceCriteria(
        [Requirement("balanced_accuracy", acceptable=0.95,
                     rationale="a missed malignancy is the expensive error")],
        source="example",
    ),
).run(data, y_test, diagnose_outliers=True)

# --- how much does it degrade, and is that acceptable? -------------------
robustness = RobustnessEvaluator(
    handler, perturbations=["gaussian_noise", "missing_values"], severities=(1, 2, 3, 4, 5)
).assess(
    data, y_test,
    accuracy_metric="balanced_accuracy",
    criteria=RobustnessCriteria(
        [ToleranceRequirement("relative_robustness", acceptable=0.9, basis="worst_cell",
                              rationale="instrument drift over one year")]
    ),
)

# --- one object holding everything, and what is wrong with it ------------
report = build_report(model_name="random_forest", accuracy=accuracy, robustness=robustness)
print(report.summary())
report.save("reports/random_forest.json")
```

```
random_forest — assessment report
  accuracy:   FAIL
  robustness: FAIL (over balanced_accuracy)
  findings:   2 critical, 1 serious
    [critical] balanced_accuracy is below its requirement
    [critical] relative_robustness is outside tolerance
    [serious] missing_values at severity 5 costs 46% of balanced_accuracy
```

A model at 0.947 balanced accuracy that loses **46% of it** when a fifth of the
measurements go missing. Neither number is visible from the other, and the second is the
one that decides whether this model can be deployed against an instrument that sometimes
fails.

`docs/tutorial.ipynb` walks the same dataset through every layer, with the plots.

## The seven things the framework does

| | What you get | Entry point |
|---|---|---|
| **1** | **Accuracy**, with a verdict against thresholds you set — per class, per subgroup, calibration, error diagnostics | `AccuracyEvaluator` |
| **2** | **Robustness**: a severity sweep over 18 perturbations, scored with the survey's §5.1.3 metrics, checked against a degradation tolerance | `RobustnessEvaluator` |
| **3** | **Sensitivity**: which single input the model leans on, and how hard it can be pushed before the model notices | `SensitivityMapper` |
| **4** | **Attribution**: what the inputs it gets wrong have in common, via a surrogate that refuses to speak when it has learned nothing | `ErrorModel` |
| **5** | **Significance**: whether an accuracy gap between groups is distinguishable from sampling noise | `compare_groups` |
| **6** | **Figures**: 13 plotly builders, one shared theme, analysis never hidden inside a plotting function | `themis_assess.viz` |
| **7** | **Reporting**: one results dictionary, ranked findings, and survey §8's ten reporting rules enforced against your own report | `build_report`, `check_report` |

## Three levels: global, group, individual

The same model can be strong on average, weak for one group, and indefensible on the one
prediction in front of you. Those are three different questions, and an answer to one is
not an answer to the others — which is why both halves of the library are reachable at
each level. A *group* here is any named partition of the data: a subgroup mask, a class,
a feature bin, a forecast window.

Below is a table which summarizes the different methods available for each of the two trustworthiness characteristics across all three levels. Where applicable, we refer to sections from the survey where the reader can go more into detail about each method.

| Level | Accuracy | Robustness |
|---|---|---|
| **Global** — the whole test set, one number for the model | point metrics on held-out data ([Rainio et al.](https://www.nature.com/articles/s41598-024-56706-x)); calibration and proper scoring rules (§5.4); Performance comparison of AI models using polar charts with normalized metrics; acceptance thresholds, which the survey makes a property of the deployment and not of the model (§8) | the severity sweep over synthetic perturbations (§5.1.3); domain-level shift between two splits (§5.1.1); composite indices (§5.1.4); distribution distance, as context and never as a score (§5.3); detector evaluation (§5.5); a degradation tolerance (§8) |
| **Group** — a named partition: subgroup, class, bin, window | subgroup breakdown and disparity (§5.1.2); per-class breakdown and confusion matrices; error rate by feature bin; accuracy over time and by forecast horizon; significance testing of a group gap — the inferential counterpart to §5.1.2's descriptive measures; reliability by confidence bin (§5.4) | degradation curves per perturbation and per severity (§5.1.3); sensitivity mapping over feature × severity and feature × time; subgroup degradation (§5.1.2); a tolerance read at the worst perturbation or the worst cell (§8); the split protocols that define the partitions (§3.1) |
| **Individual** — one prediction | outlier and novelty scoring ([Sikder and Batarseh](https://arxiv.org/pdf/2112.00588)), with the error rate of the flagged samples; predictive uncertainty and ambiguity flagging; per-sample error; local attribution from a surrogate error model, misclassification detection ([Johnsen et al.](https://ceur-ws.org/Vol-4132/short41.pdf)) | per-sample perturbation — every row perturbed, at every severity; prediction stability and flip analysis (§5.1.3) |

For the entry point behind each method, see [Where to find what](#where-to-find-what).

## Architecture: the layers and what each is for

Read top to bottom; each layer only knows about the ones above it.

```
                        ┌─────────────────────────────────────────┐
   your code ───────────│  handlers/     model → Prediction       │
   your model           │  core/         data  → DataAdapter      │
                        └──────────────────┬──────────────────────┘
                                           │  canonical contracts
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                 │
┌───────▼────────┐              ┌──────────▼─────────┐            ┌──────────▼─────────┐
│ perturbations/ │              │ metrics/           │            │ protocols/         │
│ 18 shifts, 1-5 │              │ 61 metrics, with   │            │ 6 split protocols  │
│ severities     │              │ registry + catalog │            │ (survey §3.1)      │
└───────┬────────┘              └──────────┬─────────┘            └────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────▼──────────────────────────────────────────┐
        │ assessment/                                             │
        │   AccuracyEvaluator    → AccuracyResult   (+ verdict)    │
        │   RobustnessEvaluator  → RobustnessAssessment (+ verdict)│
        │   criteria.py / tolerance.py  — the thresholds          │
        └──────────────┬──────────────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────┬──────────────────┐
     │                 │                 │                  │
┌────▼──────┐  ┌───────▼──────┐  ┌───────▼───────┐  ┌───────▼────────┐
│sensitivity│  │ attribution/ │  │ comparison/   │  │ viz/           │
│ maps      │  │ error model  │  │ x vs y vs z   │  │ figures + theme│
└────┬──────┘  └───────┬──────┘  └───────┬───────┘  └───────┬────────┘
     └─────────────────┴─────────────────┴──────────────────┘
                       │
        ┌──────────────▼──────────────────────────────────────────┐
        │ report/    one dictionary · ranked findings · §8 rules  │
        └─────────────────────────────────────────────────────────┘
```

### `core/` — the contracts everything else is written against

| File | What it holds |
|---|---|
| `core/types.py` | `TaskType`, `AccessLevel`, `Modality`, `Alignment`, `Scope`, `PerturbationSpec` |
| `core/adapters.py` | `DataAdapter` protocol; `ArrayAdapter`, `FrameAdapter`, `TextAdapter` |
| `core/results.py` | `EvaluationResult` (metrics from one evaluation), `RobustnessResult` (one robustness metric with its provenance) |
| `core/subgroups.py` | `SubgroupSpec` — a named mask over the data |
| `core/seeding.py` | `derive_seed` — stable per-cell seeds, so a recorded seed reproduces one cell of a sweep |
| `core/dispatch.py` | How a metric name reaches the right computation |
| `core/errors.py` | The exception types, with messages that say what to do instead |

Every adapter presents data as `(n_entities, n_timesteps, n_features)`. Tabular data has
a singleton time axis; ragged sequences are padded with NaN and carry a `valid_mask()`.
That one shape is why a perturbation written for time series also applies to a table.

### `handlers/` — the five lines that make a model assessable

A handler turns your model into a `Prediction` and declares its `AccessLevel`. That
declaration is a promise the library checks: `AccuracyEvaluator.plan()` and
`RobustnessEvaluator.plan()` tell you *before* a long run which metrics will be skipped
and why.

| File | What it holds |
|---|---|
| `handlers/base.py` | `ModelHandler` protocol, `BaseHandler`, `Prediction` |
| `handlers/sklearn.py` | `SklearnHandler` (any `predict`/`predict_proba` estimator), `CallableHandler` (a bare function) |
| `handlers/forecasting.py` | `ForecastingHandler`, `ForecastingConfig` — covariates, horizons, forecast origins |
| `handlers/baselines.py` | `MajorityClassHandler`, `PriorProbabilityHandler`, `NaiveLastValueHandler`, `MovingAverageHandler` |

### `perturbations/` — 18 named shifts, on a severity axis

Severity is an ordered axis from 1 to 5, and magnitudes are **relative**: each scales the
feature's own standard deviation, so one severity level means something comparable across
features measured in different units.

| File | Perturbations |
|---|---|
| `perturbations/tabular.py` | `gaussian_noise`, `uniform_bias`, `missing_values`, `category_swap`, `outlier_spikes` |
| `perturbations/correlated.py` | `correlated_noise`, `alpha_stable_noise` |
| `perturbations/timeseries/value_domain.py` | `amplitude_scaling`, `sensor_drift`, `flat_sensor`, `exponential_damping` |
| `perturbations/timeseries/time_axis.py` | `time_warp`, `input_window_shift`, `window_slice` |
| `perturbations/timeseries/structural.py` | `missing_segment` |
| `perturbations/text.py` | `character_noise`, `token_masking`, `word_dropout` |
| `perturbations/base.py` | `Perturbation` ABC, `PerturbationRegistry`, the severity→parameter contract |
| `perturbations/registry.py` | `default_perturbations()` |

### `metrics/` — 61 metrics, and two things that keep them straight

Two separate structures, deliberately:

- **The registry** (`metrics/registry.py`) holds each metric's *semantics* — which
  direction is better, its bounds, which tasks it applies to, which access level
  produces it, and its survey equation. This is what lets normalisation, ranking and the
  reporting rules stay generic instead of special-casing metric names.
- **The catalog** (`metrics/accuracy/catalog.py`) holds the *callables*. A test asserts
  the two agree, so a metric cannot be computable but unregistered, or registered but
  uncomputable.
- **The glossary** (`metrics/glossary.py`) holds the *sentence* — a plain-language
  description of every one of the 61, which travels with the numbers in a report.

| File | Metrics |
|---|---|
| `metrics/accuracy/classification.py` | confusion matrix, per-class metrics, `top_confusions` |
| `metrics/accuracy/diagnostics.py` | outlier-error analysis, error by feature bin, ambiguity flagging |
| `metrics/accuracy/temporal.py` | accuracy over time, by horizon, worst windows |
| `metrics/accuracy/weighted.py` | weighted MAE/MSE, mean signed deviation, event weights |
| `metrics/accuracy/significance.py` | Welch's ANOVA, Kruskal-Wallis, pairwise Welch with Holm correction |
| `metrics/calibration.py` | ECE, reliability bins, Brier, temperature scaling, PICP/MPIW, CRPS |
| `metrics/robustness/degradation.py` | survey §5.1.1 and §5.1.3: drops, ratios, corruption error, AUTC |
| `metrics/robustness/stability.py` | flip rate, flip probability, confidence retention |
| `metrics/robustness/subgroup.py` | accuracy gap, worst-group accuracy, group disparity, macro error |
| `metrics/robustness/distance.py` | Wasserstein, KL, MMD |
| `metrics/robustness/composite.py` | URM, composite reliability score, equal-weight risk |
| `metrics/robustness/detection.py` | detector metrics (survey §5.5) |
| `metrics/normalize.py` | `normalize_metric` — every metric onto [0, 1] with 1 best |

### `assessment/` — the two evaluators and the two sets of thresholds

| File | What it holds |
|---|---|
| `assessment/accuracy.py` | `AccuracyEvaluator`, `AccuracyResult`. One model call (two with a baseline). Nothing perturbed. |
| `assessment/robustness.py` | `RobustnessEvaluator`, `SweepResult`, `RobustnessAssessment`, `AssessmentPlan` |
| `assessment/criteria.py` | `AcceptanceCriteria`, `Requirement`, `AcceptanceReport`, `MetricVerdict`, `themis_50_criteria` |
| `assessment/tolerance.py` | `RobustnessCriteria`, `ToleranceRequirement`, `RobustnessReport`, `ToleranceVerdict` |
| `assessment/scoring.py` | `score`, `score_all` — curves → robustness metrics, when you want that step alone |

### `sensitivity/`, `attribution/`, `comparison/`

| File | What it holds |
|---|---|
| `sensitivity/maps.py` | `SensitivityMapper`, `SensitivityMap` — feature × severity and feature × time grids, `hottest_cell`, `tipping_point` |
| `attribution/error_model.py` | `ErrorModel` — a surrogate predicting *this model's error* from its input, gated so it refuses to attribute when it has learned nothing |
| `attribution/features.py` | `TabularFeatures`, `WindowSummaryFeatures`, `TSFreshFeatures` |
| `comparison/compare.py` | `compare_models`, `ModelComparison` — x vs y vs z on normalised axes, with what could not be compared recorded |

### `viz/` — figures, and never analysis hidden inside them

Every builder in `viz/figures.py` is a pure function of a result object: same result in,
same figure out, no model calls and no statistical tests. Where the four ancestor repos
had functions that computed *and* plotted — a box plot that ran an ANOVA on the way past
and left the p-value reachable only through a dictionary keyed on the plot's title —
`viz/bundles.py` provides the one-call convenience with the analysis returned beside the
figure.

| File | What it holds |
|---|---|
| `viz/theme.py` | `Palette`, `apply_layout`, `status_colour`. The one place colour is decided. |
| `viz/figures.py` | 13 builders: severity curves, sensitivity heatmap, confusion matrix, reliability diagram, accuracy over time, group boxes, subgroup bars, feature bins, attribution, metrics table, tolerance table, **model comparison polar chart**, comparison table |
| `viz/bundles.py` | `assess_groups`, `assess_calibration`, `assess_outliers`, `assess_feature_bins`, `assess_severity`, `assess_accuracy_over_time`, `assess_models` — each returns `Bundle(figure=..., analysis=...)` |

### `report/` — the handover

| File | What it holds |
|---|---|
| `report/bundle.py` | `AssessmentReport`, `build_report`, `to_plain`. One dictionary, `to_dict()` / `to_json()` / `save()`. |
| `report/findings.py` | `Finding`, `rank_findings`, and five generators — accuracy, robustness, sensitivity, attribution, groups |
| `report/rules.py` | `check_report` — survey §8's ten reporting rules, as executable checks |

### `protocols/` and `integrations/`

| File | What it holds |
|---|---|
| `protocols/splits.py` | `domain_split`, `temporal_split`, `subpopulation_split`, `compositional_split`, `near_far_split`, `resample_subsets` (survey §3.1) |
| `integrations/darts_adapter.py` | `TimeSeries` ↔ `DataAdapter`, for the forecasting extra |

## Where to find what

| You want to… | Go to |
|---|---|
| assess a scikit-learn classifier | `SklearnHandler` + `AccuracyEvaluator` |
| assess a [Darts](https://unit8co.github.io/darts/) forecaster | `handlers/forecasting.py`, `integrations/darts_adapter.py` |
| assess something else entirely | subclass `BaseHandler` — see [extending](#a-new-model-type) |
| know what a metric means | `metrics/glossary.py`, or `describe_metric("mae")` |
| know which direction is better | `default_registry().get("mae").higher_is_better` |
| find out what a run will cost before it runs | `RobustnessEvaluator(...).plan(data, actual)` |
| set accuracy thresholds | `assessment/criteria.py`, or the `accuracy.criteria` section of a config |
| set a degradation tolerance | `assessment/tolerance.py`, or `robustness.criteria` in a config |
| see the severity curve | `viz.plot_severity_curves(sweep, "accuracy")` |
| find which feature the model leans on | `SensitivityMapper(...).feature_severity_map(...)` |
| find what the inputs it fails on have in common | `attribution/error_model.py` |
| test whether a subgroup gap is real | `compare_groups(values, groups)` |
| compare model x, y and z | `compare_models([report_x, report_y, report_z])` |
| get everything as one dictionary or JSON | `AssessmentReport.to_dict()` / `.to_json()` |
| find out what is wrong with a model, ranked | `AssessmentReport.findings()` |
| check your own report for completeness | `check_report(report.to_dict())` |
| add a metric / perturbation / handler / figure | [extending](#how-to-extend-each-layer) |
| add or change a dependency | `pyproject.toml`, then `python tools/sync_requirements.py` |
| understand why something is built the way it is | `docs/DECISIONS.md` (D1–D60) |

## How to extend each layer

Each of these is a small, local change. Nothing needs a plugin system, and none of them
requires editing more than two files.

### A new metric

Two steps, because a metric is two things: a computation and a set of semantics.

```python
import numpy as np
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.metrics.accuracy import default_catalog
from themis_assess.metrics.registry import MetricSpec, default_registry
from themis_assess.metrics.glossary import METRIC_GLOSSARY


def cost_weighted_error(actual, predicted):
    """A false negative costs ten times a false positive, in this clinic."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(
        (10.0 * ((actual == 1) & (predicted == 0)).sum()
         + 1.0 * ((actual == 0) & (predicted == 1)).sum()) / actual.size
    )


# 1. the semantics — what makes it normalisable, rankable and reportable
registry = default_registry()
registry.register(
    "cost_weighted_error",
    MetricSpec(
        higher_is_better=False,
        lower_bound=0.0,
        scale=1.0,                     # an unbounded metric you intend to normalise needs one
        tasks=frozenset({TaskType.CLASSIFICATION}),
        access=AccessLevel.BLACK_BOX,
        paper_reference="internal: clinic cost model v2",
    ),
)

# 2. the callable
catalog = default_catalog()
catalog.add(
    "cost_weighted_error",
    cost_weighted_error,
    requires=("actual", "predicted"),
    tasks=frozenset({TaskType.CLASSIFICATION}),
)

# 3. the sentence, so it reaches a reader as more than a name
METRIC_GLOSSARY["cost_weighted_error"] = (
    "The average cost of the model's mistakes under the clinic's cost model, where a "
    "missed positive costs ten times a false alarm. Lower is better; 0 means no mistake "
    "cost anything."
)

AccuracyEvaluator(
    handler, registry=registry, catalog=catalog, metrics=["cost_weighted_error"]
).run(data, actual)
```

`catalog.add` is the seam a use case needs for a metric that is not general — a berthing
metric or a clinic cost model — so it still runs through the same machinery, and the
report and the normalisation treat it identically to a survey metric. Register the
callable without the semantics and you get a clear error rather than a number normalised
by a default nobody chose.

### A new perturbation

Subclass `Perturbation` and implement `perturb`, which returns `spec.n_variants`
independently perturbed copies. Draw every random quantity from the `rng` you are handed
and never from global numpy state — that is what makes a sweep reproducible.

For anything acting on numeric feature columns there is less to write:
`_NumericFeaturePerturbation` already resolves the scope (which features), the time scope
(which timesteps), the per-variant loop and the relative-severity reference, so only the
one-column transformation is left.

```python
import numpy as np
from themis_assess.perturbations.tabular import _NumericFeaturePerturbation


class Quantisation(_NumericFeaturePerturbation):
    """Round each reading to a coarse grid — a cheaper sensor, or a truncated field."""

    name = "quantisation"
    severity_scale = (0.1, 0.25, 0.5, 1.0, 2.0)   # one multiplier per severity level

    def _apply_column(self, column, valid, timesteps, magnitude, spec, rng):
        # `magnitude` already multiplies the feature's own standard deviation, so the
        # same severity means something comparable across features in different units.
        step = max(magnitude, 1e-12)
        column[:, timesteps] = np.round(column[:, timesteps] / step) * step
        return np.where(valid, column, np.nan)


from themis_assess import default_perturbations

perturbations = default_perturbations()
perturbations.register(Quantisation())

# everything in that registry that applies to your data:
RobustnessEvaluator(handler, perturbations=perturbations)

# or a named subset, looked up in your registry:
RobustnessEvaluator(
    handler,
    perturbations=["quantisation", "gaussian_noise"],
    perturbation_registry=perturbations,
)
```

`perturbation_registry` is only needed for perturbations of your own:
`default_perturbations()` builds a fresh registry on each call, so a name list alone
resolves against the built-in 18. Naming a perturbation that does not apply to your data
raises rather than being silently skipped — a silently skipped perturbation overstates
robustness.

### A new model type

Declare what your model can produce and return a `Prediction`.

```python
from themis_assess.handlers.base import BaseHandler, Prediction
from themis_assess.core.types import AccessLevel

class MyServiceHandler(BaseHandler):
    access = AccessLevel.PROBABILISTIC      # a promise the library checks

    def predict(self, data):
        probabilities = call_my_service(data.values())
        return Prediction(predicted=probabilities.argmax(1), probabilities=probabilities)
```

`AccessLevel` is a total order: declaring `QUERY` asserts you can also do everything
`PROBABILISTIC` can. Declare the lowest level that is true, and the planner will tell you
which metrics that rules out rather than failing halfway through a sweep.

### A new data shape

Implement the `DataAdapter` protocol in `core/adapters.py`. The contract is the canonical
`(n_entities, n_timesteps, n_features)` view, `feature_names`, and `with_values()` so a
perturbation can hand back a modified copy. Tabular adapters set `n_timesteps == 1`.

### A new figure

Add a pure builder to `viz/figures.py` taking a result object and returning a plotly
figure, and read your colours from `viz/theme.py` rather than choosing them in the
builder. If your figure needs an analysis run first, put the pair in `viz/bundles.py` and
return both — that separation is the point.

### A new finding or reporting rule

Add a generator to `report/findings.py` (a function from a result object to a list of
`Finding`) or a check to `report/rules.py` (a function from the serialised report
dictionary to a list of `Finding`). Every rule needs a stable `rule` identifier, the
evidence it rests on, and a suggestion — a reader who disagrees with a finding has to be
able to check the numbers behind it.

## Concepts you need in order to read the code

**Access level.** What your model lets you see: `BLACK_BOX` (labels only) →
`PROBABILISTIC` (scores) → `QUERY` (can be called repeatedly) → `EMBEDDING` →
`GRADIENT` → `ARCHITECTURE`. A total order. It decides which metrics are computable, and
the planner reports the consequences before the run.

**Severity.** An ordered axis 1–5. Not a physical unit — the mapping from severity to
parameter lives in each perturbation, and magnitudes are relative to each feature's own
spread so that severity 3 means something comparable across features.

**Relative vs absolute.** A robustness metric says how much was *lost*. It is meaningless
without the absolute clean and perturbed values, which is why `RobustnessResult` carries
them and why `check_report` complains when a report omits them.

**Three-valued verdicts.** `passed` is `True`, `False`, or `None`. `None` means nothing
was checked, and it is not a pass. This distinction is load-bearing: an assessment with no
criteria must not read as approval, and survey §8 calls the unstated tolerance the one
element no evaluation instrument can provide.

**Tolerance basis.** How a tolerance reads the sweep: `aggregate` (averaged over the
grid), `worst_perturbation`, or `worst_cell`. The three disagree — the same sweep can pass
on the aggregate and fail on the worst cell — so the basis is part of the requirement, not
a detail.

**Findings.** A threshold comparison over a result, carrying a severity, the evidence, and
a suggestion. Mechanical on purpose: every one names the numbers it came from, so you can
disagree with it by checking them.

**Composite.** Any metric that folds several others behind one number. Useful for ranking
candidates, never a basis for an acceptance decision, and never reported without its
components — the components are what a decision is actually made on.

## The results dictionary

`AssessmentReport` collects whatever parts of an assessment you ran and serialises them to
one plain dictionary with a declared schema.

```python
report = build_report(
    model_name="random_forest",
    accuracy=accuracy,
    robustness=robustness,
    sensitivity=[feature_map],          # named by their axes
    attribution={"errors": error_model},
    groups={"by_site": comparison},
    context={"split": "temporal 2019-2022 / 2023", "owner": "clinical analytics"},
)
report.add_distance("wasserstein_distance", 0.42)   # context, never a robustness score

payload = report.to_dict()
report.save("reports/random_forest.json")
```

The payload carries the sections you filled plus three things the ancestor repos'
dictionaries did not:

- **`glossary`** — a card per metric: the plain-language description, the direction, the
  bounds, the survey reference. So a reader never has to look up whether lower is better.
- **`findings`** — every observation, ranked most severe first, each with its evidence.
- **reporting issues** — where the report itself is incomplete by survey §8's standards,
  in the findings under `area: "reporting"`.

Numbers are recorded, not rounded. Non-finite values become `null` rather than the `NaN`
literal, so the JSON parses everywhere. Integer mapping keys become strings, as JSON
requires — a severity curve round-trips as `{"1": ..., "2": ...}`, so sort numerically
after loading.

### Findings, and why they exist

```python
for finding in report.worst_findings(5):
    print(f"[{finding.severity}] {finding.headline}")
    print(f"    {finding.detail}")
    print(f"    → {finding.suggestion}")
```

The findings layer exists because a results dictionary answers "what are the numbers" and
nobody reads far enough to notice that the worst subgroup is eleven points behind the
aggregate. It is also the structure a summarising language model needs: handing a model
several hundred raw numbers and asking what is wrong gives a different answer every run,
while handing it findings — each with a severity, a metric, the values, and the rule that
fired — leaves it the job it is good at, turning
`worst-group accuracy 0.68 against aggregate 0.79, chi-square p = 0.004` into a sentence a
clinician can read.

`Finding.as_dict()` is a stable contract. Nothing downstream needs to know how a finding
was derived in order to render it.

### Survey §8's reporting rules

```python
from themis_assess import check_report
for issue in check_report(report.to_dict()):
    print(issue.rule, "—", issue.headline)
```

Ten checks, each a way a robustness claim can mislead: a relative drop with no absolute
value beside it; `mCE` without `mRCE`; an aggregate over an unnamed perturbation set; a
composite without its components; one severity in place of a curve; an aggregate with no
breakdown; calibration measured only on clean data; an operating-point metric with no
threshold; a distribution distance among the robustness scores; coverage without width.

They warn, they never raise — a library that refused here would stop an honest exploratory
run, and §8 is about what gets *published*, which this code cannot observe. The issues
land in the report's findings so that ignoring them is a decision rather than an accident.

## Comparing several models

```python
from themis_assess import compare_models
from themis_assess.viz import assess_models

comparison = compare_models([report_x, report_y, report_z])
print(comparison.summary())

bundle = assess_models([report_x, report_y, report_z],
                       axes=["balanced_accuracy", "auc", "f1",
                             "expected_calibration_error", "relative_robustness"])
bundle.figure    # the polar chart
bundle.analysis  # the ModelComparison behind it
```

Every value is normalised to [0, 1] with 1 always better, so an error metric and an
accuracy can share a chart without the worse model looking better. The raw values stay
beside the normalised ones and appear on the hover, because a normalised score is not a
number anyone can act on.

The comparison **refuses** three things rather than drawing them:

- a metric one model has and another does not — averaging over "whatever each model
  reported" rewards the model that reported fewer of the hard metrics;
- two robustness scores computed over different underlying accuracy metrics — different
  quantities under one name;
- a sweep aggregate over two different perturbation sets — a sum-based aggregate scales
  with the size of the set.

Everything refused appears in `comparison.dropped` with the reason. And the output most
worth reading is `comparison.disagreements()`: the pairs of metrics that prefer different
models. Where every metric agrees you did not need a comparison layer; where they disagree
there is a trade-off, and a trade-off is a decision for the deployment rather than for a
scoring rule.

## Configuration

Plain YAML, read into dataclasses. No hydra, no plugin registry, no framework
(`docs/DECISIONS.md`, D18 and D54). See `examples/assessment.yaml` for a fully commented file.

```python
from themis_assess import load_config

config = load_config("examples/assessment.yaml")
handler = SklearnHandler(my_model)                      # your objects stay yours
accuracy = config.run_accuracy(handler, data, actual)
robustness = config.assess_robustness(handler, data, actual)
report = config.build_report(accuracy=accuracy, robustness=robustness)
config.save_report(report)
```

A config governs the settings and, above all, the **criteria** — the accuracy thresholds
and the degradation tolerances, which belong in a versioned, reviewable file because the
survey makes them properties of the deployment rather than of the model.

It deliberately does **not** load your model or your data. A config that names a class to
import turns a settings file into executable code, and then a model has to be
constructible from YAML to be assessable at all — exactly the coupling this library was
extracted to remove.

Every field is optional; an empty file is valid and gives the defaults. A misspelt key is
an error, because the alternative is a run that proceeds with the default while the file
says otherwise.

## Where each capability came from

This library consolidates three repositories that had drifted apart as each repository assessed a different AI system. Every ported symbol
records its origin in a docstring line, so provenance survives.

| Source | Domain | What came from it |
|---|---|---|
| `themis-accuracy-robustness` (this repo, `main`) | generic | the perturbation and metric skeleton, `Metrics_list.yaml` (now `metrics/glossary.py`) |
| `themis_valencia` | Vessel berthing at Valencia Port, forecasting (TSMixer/[Darts](https://unit8co.github.io/darts/)) | `MetricRegistry`, `normalize_metric`, the forecasting accuracy assessment, accuracy over time, Welch's ANOVA over groups, the time × feature heatmap |
| `themis_medical` | Pancreatic Cancer risk prediction at the Medical University of Plovdiv, tabular (LightGBM) | outlier-error analysis, error by feature bin, error analysis by true class, the metrics table |
| `themis_media` | Fake news and hate speech detection, text (RoBERTa) | text perturbations, the ambiguity threshold, confidence retention, the polar chart, calibration under shift |

Six capabilities were recovered that had been computed and then not returned — per-class
error rates behind a commented-out print, the outlier chi-square test, a formatted metrics
table, a Welch ANOVA whose function returned the figure instead of the results, an
abstract aggregate with no body, and an accuracy-over-time function that could not run.
Each is now returned, and each has a test.

## Developing

```bash
pip install -e ".[dev]"             # or: pip install -r requirements-dev.txt
python -m pytest                    # 1329 tests
python -m pytest tests/report -q    # one layer
ruff check src tests tools
python tools/sync_requirements.py --check    # are the requirements files current?
```

The test suite is also run against **numpy 1.26** as well as the current release, because
the numpy 2 migration changed enough behaviour that a library claiming `numpy>=1.26` has
to prove it. Tests needing `shap` are marked and skipped where it is absent.

It also runs with the package **not installed at all**, which is what makes way 2 above
work. The four tests that spawn a child interpreter — two pinning cross-process seed
stability, two pinning that the core does not import an optional extra — go through
`run_child` in `tests/conftest.py`, which puts `src/` on the child's `PYTHONPATH`. A child
process does not inherit pytest's own path setting, and without that helper those four
failed on the no-install path while everything else passed: a failure in the harness
looking exactly like a broken property.

Tests are named after the property they defend, not the function they call —
`test_the_baseline_improvement_is_sign_corrected` rather than
`test_improvement_over_baseline`. Several of them exist because the behaviour they pin was
once wrong; the docstring says which, so nobody removes the test as redundant.

`docs/DECISIONS.md` records every design decision, D1 to D60, with the alternative that
was rejected and why. It is the authority where it disagrees with anything else.

## If something looks wrong or could be approved

Users are welcome to report in "Issues" at this repository.

## Documents in this repository

| File | What it is |
|---|---|
| `README.md` | this file |
| `docs/DECISIONS.md` | every design decision, D1–D60, each with the alternative rejected and why. The authority where it disagrees with anything else. |
| `docs/ARCHITECTURE.md` | the survey-section-by-survey-section map of what was ported from where, including the metrics not yet implemented |
| `docs/tutorial.ipynb` | the full walkthrough on the breast cancer Wisconsin dataset |
| `examples/assessment.yaml` | a fully commented config file |
| `requirements*.txt`, `environment.yml` | generated from `pyproject.toml` by `tools/sync_requirements.py` |

---

If this assessment supports a published result, cite the survey as well as the code — the
metrics and the equation numbers in the docstrings are how a reader gets from a
number in a report back to its definition.

> Michałowska, K. & Johnsen, P. (2026). *Evaluating Machine Learning Robustness Under
> Natural Distribution Shifts.* SINTEF Digital.
