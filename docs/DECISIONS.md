# Decisions — repo A consolidation

**Date:** 2026-09-01 · Companion to `docs/ARCHITECTURE.md` (revision 3)
**Status:** all 16 questions resolved. This file is now a decision record; it supersedes any conflicting
statement in the architecture document, which will be revised to match.

---

## Scope and workflow

**D1 — Branch and review.** A long-lived `restructure` branch, one commit per tranche, merged to `main`
when the test suite is green. CI is updated in the same branch so `main` never goes red. Review before
each merge.

**D2 — Layout.** Move to `src/` layout with `pyproject.toml`, mirroring repo D. Retires `pyrootutils`,
`.project-root`, the `os.getcwd()` import hack and the bare `from base_model_handler import …`.
Repo A becomes pip-installable.

**D3 — Package name.** `themis_assess`. Repo name stays `themis-accuracy-robustness`.

**D4 — First tranche.** The pure metric layer: all survey §5.1 robustness metrics as pure functions of
numbers, plus `MetricRegistry` and `normalize_metric` recovered from repo B, plus unit tests. Depends on
none of the other decisions, and canonicalises five `R_φ` copies and four `compute_robustness` lineages.

**D5 — Deletions.** Delete all three on the `restructure` branch:

- `algorithms/robustness/assessment/forecasting.py` — missing `self` on every method, tuple-assignment
  `__init__`, duplicate `compute_robustness` byte-identical to repo C's;
- `algorithms/accuracy/assessment/forecasting.py` — 50-line stub, tuple-assignment `__init__`, superseded
  by repo B's 400-line version;
- both submodule directories (`safetycage`, `OODGuard`) and their `.gitmodules` entries.

Git history preserves them. Nothing is rescued from OODGuard.

**D6 — Public release.** Internal SINTEF GitLab for now. `pyproject` metadata stays minimal; no licence
decision and no PyPI naming needed this round.

---

## Dependencies

**D7 — Python floor: `>=3.11`.** The governing principle stated with the answer: *repo A should be as
flexible as possible and must not depend on any specific package such as darts or safetycage.* This rules
out `>=3.13` (which `safetycage` would force) and drives D8–D10.
*Superseded 2026-10-23: floor raised to `>=3.13` so that the `[safetycage]` extra (`safetycage>=0.0.62`
requires 3.13) resolves under the project's declared Python range. safetycage itself stays optional.*

**D8 — Core dependencies.** `numpy`, `pandas`, `scikit-learn`, `plotly`, `pyyaml`. **Note the swap:
plotly is core, matplotlib is not.**

**D9 — Extras.**

| Extra | Contents |
|---|---|
| `[timeseries]` | `darts`, `statsmodels` (STL) |
| `[text]` | `transformers`, `textattack` |
| `[explain]` | `tsfresh`, `xgboost`, `shap` |
| `[safetycage]` | `safetycage` (pulls Python ≥3.13 for those who want it) |
| `[mpl]` | `matplotlib` — only for `sklearn.inspection.PartialDependenceDisplay` |

Every optional package is imported lazily inside the module that needs it, so a missing extra raises a
clear message naming the extra rather than an `ImportError` at package import.

**D10 — Plotting.** Plotly everywhere. The matplotlib and seaborn plots inherited from repos A/B/C
(confusion matrices, error analysis, ECDF) are ported to plotly. `matplotlib` survives only in `[mpl]`
because `plot_pdp` / `plot_ice` go through sklearn's `PartialDependenceDisplay`, which has no plotly
equivalent.

---

## Core contracts

**D11 — Forecasting data crosses at an array boundary.** Core speaks `(entity, time, feature)` numpy
arrays. A `DartsAdapter` converting to and from darts `TimeSeries` ships under `[timeseries]`. No module
under `perturbations/` or `metrics/` may import darts — and now neither may `core/`, except inside the
adapter that `[timeseries]` gates.

**D12 — `ForecastingModelHandler` signature.** Repo B's, with a config dataclass:

```python
@dataclass(frozen=True)
class ForecastingConfig:
    input_chunk_length: int
    forecast_horizon: int
    stride: int = 1

class ForecastingModelHandler(BaseModelHandler):
    def __init__(self, model_paths, data_handler, config: ForecastingConfig, **kwargs): ...
```

Repo B's handlers are the only ones with working subclasses, but the dataclass stops the base class
hardcoding a windowed-forecasting shape. `main`'s `(model_filename, model)` signature is retired.

**D13 — Class naming.** `BaseDataHandler`, `BaseModelHandler`. All modules lowercase (repo B's
`Classification.py` is a case error not to be reproduced).

**D14 — Baseline is a second model handler instance.** Repo B's pattern. `CE/mCE`, `RCE/mRCE`,
`REI/mREI`, `FR/mFR` take two handlers. The `create_baseline_model()` /
`get_baseline_model_predictions()` abstract methods are dropped.

**D15 — Data handler contract.** `main`'s five-method ABC stays the required base. An optional
`BatchedDataHandler` mixin carries the reusable parts of repo D's 465-line handler — stratified splitting,
caching, partition accessors, pickle round-trip — with the HuggingFace tokenizer specifics left to a text
subclass.

**D16 — Severity.** `S = 5` fixed by default (ImageNet-C's convention), each perturbation mapping
`s ∈ {1..5}` to its own natural parameter scale — e.g. Gaussian noise → `{0.25, 0.5, 1, 2, 4} × σ_feature`.
`severity_levels` is an overridable class variable. **The severity→parameter mapping must appear in each
perturbation's `metadata()`**, so a report can state exactly what "severity 3" meant.

**D17 — Modalities.** `Modality` enum includes `IMAGE` and the adapter protocol accommodates it, but no
§4.1 image corruptions are implemented.

**D18 — Configuration.** Plain YAML plus the registry. Extend the existing `Metrics_list.yaml` pattern to
one `assessment.yaml` declaring metrics, perturbations, severities, subgroups and protocols for a run.
No Hydra.

---

## Naming

**D19 — Survey names are canonical, with no aliases.** The library reads the same as the paper:

| Survey | Eq. | Replaces |
|---|---|---|
| `flip_probability` | 23 | `flip_rate` in A `robustness/assessment/classification.py` and D `robustness_utils.py` |
| `flip_rate` | 24 | *reserved for the baseline-normalised metric* |
| `mean_flip_rate` | 25 | — |
| `relative_robustness` | 11 | `rel_robustness_metric` (A, B), `robustness_ratio` (D), inline `abs(1-p/c)` (A, C) |
| `total_relative_robustness` | 12 | `R_Φ` |

No deprecated aliases. Nothing in B, C or D calls repo A's functions — they vendor their own copies, and
we are not editing them — so aliases would be dead weight.

Repo-A originals with no survey equivalent keep their own names and are documented as extensions:
`absolute_relative_robustness` (`|1 − R_φ|`), `confidence_retention`.

---

## Feature scope

**D20 — OOD.** Detectors are descoped: no `robustness/ood/`, nothing rescued from OODGuard. The survey
§5.5 **evaluation metrics are kept** — AUROC, AUPRC, FPR@95%TPR, AUTC, wAUTC — relocated to
`metrics/robustness/detection.py` and documented as detector-evaluation metrics rather than OOD-specific.
They are pure functions of `(scores, labels)` and will score SafetyCage's misclassification flags when
that lands.

**D21 — SafetyCage.** Submodule removed now, integration deferred to a later tranche via
`integrations/safetycage.py` bridging our handlers to its `DataModule` / `ModelModule`, under the
`[safetycage]` extra. Knock-on to accept: the white-box §5.2 metrics (ROBY, RDI, neuron sensitivity, NCCR)
wait with it, since `ModelModule` is their activation plumbing.

**D22 — Notebooks.** Read the four repos' notebooks (Plovdiv robustness 172 kB, Valencia heatmaps 487 kB,
the OOD tutorial 252 kB, `main/tests/` ×2) **before the perturbation tranche**, not before the metric
tranche. They may hold assessment logic that never reached a module.

---

## Decisions taken while implementing (D23-D26)

These were not open questions when the tranches started; each is a choice forced by something the
implementation turned up, recorded here because a reader of the code would otherwise have to
reconstruct the reasoning.

**D23 — Metric callables live in a catalog, separate from the registry.** `metrics/registry.py`
answers *what a metric means* (orientation, bounds, task, access); `metrics/accuracy/catalog.py`
answers *how to compute it*. The four repos instead carried parallel dictionaries keyed by
`function.__name__` — `default_metric_functions` (a list in repos A and D, a dict in repo C),
`default_metric_ranges` and `default_acceptable_metric_values` — which drifted (repo C listed
`f1_score` in its ranges, repo A's template did not) and broke whenever a metric was wrapped in a
`partial`. A test now asserts every catalogued name is a registered name, and that the two agree on
task and access, so the drift cannot recur silently.

**D24 — A catalog entry declares the quantities it needs, by name.** `MetricEntry.requires` names
what the callable takes (`actual`, `predicted`, `probabilities`, `lower`/`upper`, `samples`, …), so
the runner can ask `catalog.computable(bag)` and report up front which metrics will be skipped and
why, rather than calling everything and catching exceptions. This is what makes the access-level
promise in the registry enforceable rather than documentation.

**D25 — Ensemble CRPS and the energy score share one spread estimator, defaulting to the plug-in
form.** They were written with different divisors (`m^2` and `m(m-1)`) and so disagreed on
one-dimensional data where the energy score *is* CRPS; a test comparing them caught it. Both now take
`fair=False` by default, matching the standard ensemble CRPS and the survey's plug-in reading of
Eq. 44, with `fair=True` available for the unbiased spread. The default is not neutral: the plug-in
estimator is minimised by an ensemble narrower than the truth, which a test demonstrates directly, so
`fair=True` is the better choice below roughly fifty members.

**D26 — `mean_signed_error` is renamed `mean_signed_deviation`.** The registry used one name and repo
B's implementation the other. Repo B's wins, under D19's principle that the name in the source of the
metric is canonical.

Two defects fixed in passing, both found by tests written against the survey rather than against the
code:

* `auc` was registered alongside the label metrics and inherited `BLACK_BOX` access, though it needs
  scores. It now carries `PROBABILISTIC`, which is what makes the runner skip it for a label-only
  handler instead of failing mid-run.
* `precision`, `recall` and `f1` are called with an `average` chosen from the data (binary below
  three classes, macro above). All three repos called scikit-learn's versions bare, which raises on
  multiclass input; repo D's task is binary so it never surfaced. Macro rather than weighted, because
  weighting by support hides degradation in the rare class — the thing a robustness study is looking
  for.


## Decisions taken while implementing (D27-D31)

**D27 — A handler is a Protocol, and loading is not part of it.** All four repos made
`load_trained_model` an abstract method, so every handler had to be a loader and a model
already in memory had no way in. `ModelHandler` requires one method, `predict`, plus three
declared attributes (`name`, `task`, `access`); where the model object came from is the use
case's business. Being a `typing.Protocol` also means a use case's existing class satisfies
it without inheriting anything — which is what repo D's `ModelHandlerRoBERTa`, that did not
inherit its own base class, was already doing in practice.

**D28 — Predictions travel as one object.** `Prediction` carries labels, probabilities,
logits, intervals, ensemble samples and embeddings together, with the fields that are
absent left as `None`. Repo C returned argmax labels and stashed the probabilities on
`self.y_pred_raw`, so a calibration metric could only be computed by reaching into the
handler afterwards. `Prediction.available()` feeds the catalog's `computable()` directly,
which is what turns the declared access level into something the runner can act on rather
than merely document. `check_prediction` enforces the declaration: a handler claiming
`PROBABILISTIC` that returns labels only is refused, because the runner repeats that claim
to the user.

**D29 — The runner plans before it runs.** `AssessmentRunner.plan()` reports which metrics
will be computed, which will be skipped and why, and how many model calls the sweep will
make — for one clean prediction, before anything expensive starts. None of the four repos
could tell a user that their handler returns labels only and so nine of their twelve
requested metrics would be missing, until after the run.

**D30 — The ground truth is captured once, from clean data.** In forecasting the target is
a column of the input, so a perturbation touching it would move the truth along with the
prediction and score a broken model as perfect. Survey §6.3's label-preservation assumption
is enforced structurally — `run()` takes `actual` as an argument and never recomputes it —
rather than trusted. Subgroups are scored by `Prediction.take(mask)` for the same reason:
re-predicting a subgroup on perturbed data would draw different noise.

**D31 — Cell-derived seeding.** Each `(perturbation, severity, variant)` cell derives its
own generator from the run seed and the cell's identity, so a cell's draw does not depend
on how many cells ran before it. Repo B seeded numpy globally once per run, so re-running a
single severity gave different noise than that severity had inside the full sweep.

Five defects found and fixed during this tranche, four of them by the first end-to-end run:

* **`_as_list` iterated a mapping's keys.** Every severity curve passed as
  `{severity: value}` silently became `[1, 2, 3, 4, 5]` — identical for model and baseline
  — so `CE` came out at exactly 1.0, which reads as "matches the baseline" rather than as
  an error. Mappings are now read by value, sorted by severity.
* **`relative_error_inflation` was listed as supported with no dispatch branch**, so it
  fell through to the corruption-error branch and returned that metric's number under
  Eq. 17's name. Eq. 17 is per (perturbation, severity) and does not fit a per-perturbation
  result; only its Eq. 18 aggregate is offered, and the dispatch now ends in a raise rather
  than a catch-all.
* **The error-family equations accepted accuracy metrics.** Eqs. 13-22 are written over an
  error E; handing them an accuracy inverts every ratio, so a model that held its accuracy
  scored as *less* robust. The registry knew the orientation and the scoring layer did not
  read it. It does now, and refuses the mismatch in both directions.
* **A NaN metric propagated silently.** scikit-learn warns and returns NaN for AUC on a
  single-class subgroup rather than raising, and that NaN would spread through every mean
  and ratio downstream. Non-finite results are now recorded as undefined, by name, in the
  evaluation's metadata, alongside metrics that raised.
* **A binary `decision_function` was passed off as a probability.** Its output is shaped
  exactly like the `P(class 1)` vector a handler may legitimately return; only the range
  distinguishes them. Squashing margins into probabilities would make every calibration
  metric a measurement of the squashing rather than of the model, so the handler refuses and
  points at `CalibratedClassifierCV`.

`AccessLevel`'s total order is now documented as meaning what a total order means: declaring
`QUERY` asserts `PROBABILISTIC` too. Being callable many times and exposing probabilities
are in principle independent, so a model that can be queried freely but returns bare labels
should declare `BLACK_BOX`. Nothing is lost by that while the query-based §5.2 metrics
remain deferred with SafetyCage (D21).



## Decisions taken while implementing (D32-D36)

Prompted by a question from the project owner: the repo had drifted robustness-heavy, and
the accuracy half — the half the repository is named for — had one and a half tranches
against robustness's four. The four repos hold 3,423 lines of accuracy assessment; audited
against them, seven capabilities were unported. Tranche 7 became a dedicated accuracy
tranche and the roadmap grew to ten.

**D32 — Acceptance thresholds are a requirements object, not a registry field.** All four
repos carried `default_acceptable_metric_values` (accuracy 0.8, MCC 0.6). Tranche 5 folded
in their *ranges* as bounds and dropped the thresholds entirely, which was inconsistent:
the robustness side already had a `tolerance` slot the library refuses to guess. A
threshold is a property of the deployment, not of the metric — 0.8 accuracy is fine for a
triage queue and not for a clinical aid — so `AcceptanceCriteria` is supplied per
assessment and the registry keeps only what is true of a metric everywhere. The repos'
numbers ship as `themis_50_criteria()`, a function rather than a constant so it cannot
apply itself, with `source` recording that they have no derivation on file.

Three consequences worth stating. The direction always comes from the registry, so a
caller never says which way a metric runs and the two cannot disagree. The margin is
signed so positive is good for every metric, which makes a table of margins sortable. And
the overall verdict is three-valued: no criteria yields `None`, never `True`, so an
assessment with nothing checked cannot read as a green light.

**D33 — `AccuracyRunner` is a first-class entry point, not a sweep with zero
perturbations.** One model call, two with a baseline, nothing perturbed — cheap enough for
every commit, where a robustness sweep is a periodic exercise. It delegates metric
selection and the undefined-metric bookkeeping to `AssessmentRunner`, so a metric skipped
in an accuracy check is skipped for the same reason and reported the same way as in a
sweep.

**D34 — Calibration moved to `metrics/calibration.py`.** Whether a model's confidence is
honest is an accuracy property of its uncertainty, measurable on clean data with no shift
in sight; only its *change* under shift is a robustness question, and the accuracy runner
needs ECE and Brier as ordinary metrics. `metrics/robustness/calibration.py` remains as a
two-line re-export, because the survey numbers these equations §5.4 and a reader following
its structure will look there.

**D35 — Descriptive error diagnostics belong to accuracy; learned attribution does not.**
Error rate by true class, by outlier status and by feature bin need only predictions, so
they are here. Repo B's TSFresh/XGBoost/SHAP family trains a second model and stays in the
attribution tranche behind the `[explain]` extra.

**D36 — Both readings of "ambiguous" are offered, explicitly.** Repo D computed
`max_entropy * sensitivity` and documented `sensitivity=0.8` as "flagging the top 20% most
uncertain samples". Those are different things: a fraction of maximum possible entropy is a
fixed absolute bar, and what share of samples it flags depends on how confident the model
happens to be. Measured on a confident model it flags near 0% while claiming 20%, and on a
diffuse one it flags everything. `ambiguity_threshold` gives the absolute bar without
`entropies` and the empirical quantile with them.

Defects found and fixed while porting. Three of the four functions ported here computed
their headline output and discarded it:

* **`error_analysis_by_true_class` (repo C)** computed per-class error rates in a loop
  whose only consumer was a commented-out `print`, built a margin-annotated crosstab and
  never returned it, and returned only a per-sample frame. The numbers its name promises
  never reached its caller. `error_rate_by_class` returns them.
* **`compute_error_outlier_analysis` (repo C)** ran a chi-square test and discarded the
  result, returning two error rates with no test — and the outlier group is small by
  construction, so its rate is precisely the noisy one. The test is now returned.
* **`compute_accuracy_over_time` (repo B)** could not execute in the general repo: it
  called `historical_forecasts(series=..., covariates=..., forecast_horizon=..., stride=...)`
  while the handler it targets takes `past_covariates` and reads horizon and stride from
  itself, so the call raises `TypeError` on the keyword. `compute_aggregated_accuracy` was
  an `@abstractmethod` with no body. The rolling evaluation existed only in notebooks.
* **Argument order.** Repo B called `metric(y_pred, y_true)`, following darts; this library
  is `(actual, predicted)` throughout. Invisible for MAE and RMSE, sign-inverting for
  anything asymmetric. `accuracy_over_time` passes actual first, with a test that pins it.
* **A sign error in repo C's combined outlier score.** Isolation Forest and LOF both run
  low-for-anomalous while mean kNN distance runs high-for-anomalous; all three were
  normalised and combined as `1 - mean(...)`, correct for the first two and inverted for
  the third. Measured over eight seeds the two correctly-oriented detectors outvote it and
  ranking quality is unchanged (AUROC 0.954 either way) — so this did not break repo C's
  results. What it costs is separation, which more than halves, and it makes the third
  detector's contribution actively harmful as soon as it is weighted differently or used
  alone.
* **A quadratic memory footprint.** The same function built the full `(n, n)`
  `pairwise_distances` matrix to read k columns of it. `NearestNeighbors` replaces it; a
  test runs 20,000 samples, which the original would have needed 3.2 GB for.
* **`get_metrics_table` (repos A and D)** built a formatted copy of the table and returned
  the unformatted one. The formatting was dead code, and the verdict was in a dictionary
  kept elsewhere. `AcceptanceReport.to_frame()` puts the verdict in the table.

One thing added that none of the repos had: `top_confusions`, ranking specific mistakes by
the fraction of the true class confused rather than by raw count. In the tranche's own test
fixture a 70%-of-class failure contributes only 14 errors, so ranking by count puts it
last — which is where a reader scanning a confusion matrix by eye also puts it.



## Decisions taken while implementing (D37-D42)

**D37 — `AccuracyAssessor` and `RobustnessAssessor`.** `AccuracyRunner` and
`AssessmentRunner` renamed at the project owner's request, and `assessment/runner.py`
renamed to `assessment/robustness.py` so the module names match. No aliases: the library is
pre-1.0 on a single branch with no external users, and a compatibility shim would be cruft
with nothing to be compatible with.

**D38 — `RobustnessAssessor.assess()` returns a scored, verdict-bearing result.** The
rename alone would have left the two halves asymmetric: `AccuracyAssessor.run()` returned a
rich object carrying a pass/fail verdict, while the robustness side returned raw severity
curves the caller had to feed back into `score()` themselves — so the cheap half of the
library was the more finished one. `assess()` sweeps, scores and checks in one call and
returns `RobustnessAssessment`. `run()` remains and still returns the unscored
`SweepResult`, because the sensitivity and attribution layers need the raw cells; nothing
is duplicated, `assess()` calls `run()`.

**D39 — `RobustnessCriteria`, the §8 counterpart of `AcceptanceCriteria`.** The survey
sentence D32 rests on — "the tolerance is the one element no evaluation instrument
provides" — is about robustness, so the robustness side needs the same object. One thing
differs: an accuracy criterion is checked against a number, a tolerance against a *shape*,
and the three defensible readings of "acceptable degradation" disagree — the aggregate
(which averages away a single catastrophic perturbation), the worst perturbation, and the
worst cell over perturbation and severity. All three are available, the basis is explicit
with no fallback, and the default is `worst_perturbation`: the aggregate is too easy to
pass and the worst cell is dominated by severity 5, often outside anything the deployment
will see. Both assessors report `passed is None` when no criteria were supplied.

An unchecked *critical* tolerance is surfaced in the summary. "overall: PASS" beside a
tolerance whose metric was never scored is the most misleading thing the report could say.

**D40 — Sensitivity maps separate prediction shift from degradation.** Repo B's heatmap
compared each perturbed prediction against the **clean prediction**, not the truth, and
never named the distinction. That is a prediction-shift map, and it is the more broadly
useful of the two because it needs no labels and can therefore be computed on production
data. A degradation map compares against the truth and answers a different question:
sensitivity is not harm, since a model can move a lot and still be right, or barely move
and be confidently wrong. `measure` selects, the result records which, and a degradation
map is oriented so higher always means more harm — without that, a map over accuracy and a
map over an error would read in opposite directions.

Also: the analysis window is a parameter (repo B hardcoded 120 timesteps), the plot is
gone from the computation (repo B's function imported matplotlib and suppressed
PyTorch-Lightning loggers inside it), and `map_from_sweep` builds the coarse
perturbation-by-severity map from a sweep that already ran, for no model calls at all.

**D41 — No optional dependency is load-bearing in the attribution layer.** Repo B's error
model required XGBoost, SHAP and TSFresh together. Here the default surrogate is
scikit-learn's `HistGradientBoostingRegressor` and the default attribution is permutation
importance, both core dependencies, so the whole capability works with no extras. The
default feature extractor is twelve interpretable window summaries computed in numpy
rather than TSFresh — and interpretability is the substantive argument, not just the
dependency: "the covariate's range over the input window was high" is a finding someone can
act on, and `cov__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2` is not.
XGBoost, SHAP and TSFresh remain available under `[explain]` and are sharper.

**D42 — The surrogate's own held-out score gates its attributions, and the gate statistic
depends on the target.** An error model that cannot predict the error has attributions
indistinguishable from noise — the output looks identical either way — so
`is_trustworthy` guards the attribution methods and overriding it takes `force=True`. Repo
B returned `test_r2` and nothing stopped a caller reaching for `GlobalFeatureImportance` at
an R² of 0.02.

The refinement is mine and it matters: **R² is the wrong yardstick for a 0/1
misclassification target.** Whether a particular borderline sample flips is close to
irreducibly random even when the risk of flipping is highly predictable. Measured on
synthetic data whose error rate doubles across terciles of a hidden noise feature, the
surrogate scored R² = −0.04 and AUROC = 0.64 — the R² gate would have discarded a real
finding. A binary error target is gated on AUROC (0.55), a continuous one on R² (0.1).

Local attribution for a sample the surrogate trained on now raises rather than printing a
warning: repo B's `print` reaches nobody, and the optimistic output is indistinguishable
from an honest one.

### A defect in my own tranche-6 code, found here

**Cell-derived seeding was not reproducible across processes.** `_cell_rng` derived each
cell's generator with `hash((seed, perturbation, severity, variant))`, and Python
randomises string hashing per process unless `PYTHONHASHSEED` is set. So a sweep's recorded
seed did not reproduce it on the next run — the entire point of recording one. Tranche 6's
reproducibility test passed because both of its runs were in the same process.

The symptom was an intermittently failing assertion in an unrelated test. The derivation
now lives in `core/seeding.py` and uses BLAKE2b over the cell's string form, with parts
separated by a byte that cannot occur in them so `("ab", "c")` and `("a", "bc")` cannot
collide. Two subprocess tests pin it, because a same-process test cannot see this class of
bug at all.

Two smaller things fixed while porting: repo B's `TrainErrorModel` contained three lines of
copy-pasted XGBoost hyperparameter setup that ran twice, and computed R² by hand rather
than with `r2_score`.



## Decisions taken while implementing (D43-D46)

Prompted by a second observation from the project owner: many of repo B's functions bundle
analysis *with* the plot — a box plot of accuracy by group also runs a statistical test for
whether the groups differ — and the separation this library has imposed risks losing both
the bundled convenience and, worse, the analysis itself.

An audit of every ``plot_*`` function across the four repos found one piece of analysis
that existed **only** inside plotting code and had not been ported: **Welch's ANOVA across
groups**, at two call sites in repo B. Everything else buried in a plot was already in the
library — repo C recomputes per-class error rates inside
``plot_error_analysis_by_true_class`` (D35's ``error_rate_by_class``), the chi-square
outlier test is in ``error_rate_by_outlier_status``, reliability bins in
``reliability_bins``.

The repo B call sites are worth stating exactly, because the pattern is worse than
bundling. ``compute_aggregated_accuracy_per_category`` assembled

    results = {"figure": fig, "summary": df.describe(),
               "hypothesis_test_result": res, "conclusion": conclusion}
    self.results[title] = results
    return fig

— the right bundle, stashed on a side-effect dictionary keyed by the plot's title string,
with only the figure returned. The caller got the picture and never the statistics.

**D43 — Group significance testing is its own module.** ``metrics/accuracy/significance.py``
is the inferential counterpart to §5.1.2's descriptive disparity metrics: ``accuracy_gap``
of 0.085 across three sites is a number, and whether 0.085 is distinguishable from
sampling variation is a different question that the gap alone invites a reader to get
wrong.

Welch's ANOVA is implemented rather than imported. Repo B used ``pingouin``; SciPy has only
``f_oneway``, which assumes equal variances — wrong here by construction, since a site
contributing 40 samples and one contributing 400 have accuracy estimates of different
precision. On the tranche's own fixture the standard ANOVA reports F = 56.7, p = 3e-21
where Welch reports F = 9.6, p = 3e-4: both reject, but only one is honest about how much
of the evidence rests on a small noisy group. The implementation is validated against an
exact identity rather than a captured value — at two groups, Welch's ANOVA F must equal
Welch's t statistic squared, and the denominator degrees of freedom must equal the
Welch-Satterthwaite value; both hold to machine precision.

Three additions beyond repo B's version. Kruskal-Wallis runs alongside, because both ANOVA
variants compare means under approximate normality and a bounded accuracy near its ceiling
is skewed — ``tests_agree`` surfaces a disagreement rather than hiding it behind one
p-value. ``pairwise_welch`` says *which* groups differ, with Holm correction, since an
omnibus rejection says only that the groups are not all alike. And ``describe()`` keeps
repo B's careful wording that a failure to reject is not evidence of equivalence, and adds
the sample count that explains why.

**D44 — Both shapes of figure exist, and the bundled one returns everything.**
``viz/figures.py`` holds pure functions from a result object to a plotly figure — no model
calls, no fitting, no tests. ``viz/bundles.py`` holds the bundled form the repos used, and
every bundle returns a frozen object carrying the figure **and** the analysis. That is the
fix rather than a restatement of the problem: the separation is preserved for anything that
needs a number without a rendering step, and the notebook workflow is preserved without
the analysis becoming unreachable. Bundles also implement ``_ipython_display_``, so
``assess_groups(...)`` as a cell's last expression shows the plot exactly as repo B's
function did while ``.analysis`` still holds the statistics.

The group box plot annotates the test verdict in its subtitle, so the gap and whether the
gap is distinguishable from noise cannot be read apart from each other. Repo B's
sample-count-under-each-tick trick is kept: a box drawn from six series and one drawn from
six hundred look identical otherwise.

**D45 — Colour is decided in one place, by the job the quantity does.** ``viz/theme.py``
holds the palette and layout; the four repos made four independent colour decisions, none
comparable — repo A's confusion matrix took ``colormap="viridis"`` per call, repo C
hard-coded hex lists, repo D's plotting module carried its own, repo B's heatmap took a
``scale="log"`` switch.

Four rules, each of which one repo broke. Sequential encoding is **one hue, light to dark**
(``viridis`` sweeps purple through green to yellow, which makes the hue look meaningful
when only the lightness is, and readers see boundaries in it that are not in the data).
Diverging encoding has a **neutral grey midpoint**, so the midpoint reads as "nothing".
Categorical hues are assigned in a **fixed order and never cycled** — a ninth series
raises rather than repeating a colour, because a repeat makes two series indistinguishable
with no way to tell which two. Status colours are **reserved and never carry meaning
alone**: every verdict gets a symbol and a word, since three categorical slots and two
status colours sit below 3:1 contrast on a light surface. The eight-slot palette was run
through a colour-vision-deficiency validator rather than eyeballed.

No figure has two y-axes, pinned by a test. It is the single most common chart mistake and
the temptation is real here — a forecast-error curve invites overlaying the series being
forecast, which makes two unrelated magnitudes look comparable and their crossings look
meaningful.

**D46 — Figures are rendered and looked at, not just built.** The figures were exported to
PNG and inspected, which caught two things no assertion had: every metric with *no*
requirement was painted with the warning colour, so half the acceptance table glowed orange
and buried the rows that had actually failed; and ``expected_calibration_error`` was
truncated in the table's first column. Both fixed, the first pinned by a test.


**D47 — One results dictionary with a declared schema, not four.** Each of the four repos
grew its own results dictionary and none could be read by anything not written against it:
repo B's mixed plot objects in with the numbers, repo C's flattened to one level and lost
which subgroup a value belonged to, repo D's was assembled in a notebook cell.
`report/bundle.py` has one shape, a `schema` key (`themis-assess/report@1`), and stable
section names. The rich objects stay reachable on the report for in-process use;
`to_dict()` is the contract for everything else. Numbers are recorded, not rounded — a
rounded record cannot be checked against the run that produced it. Non-finite floats
serialise as `null` rather than the `NaN` literal, which most JSON parsers reject.

**D48 — Findings are a first-class layer, because the summarising consumer is a language
model.** A results dictionary answers "what are the numbers"; nobody reads far enough to
notice that the worst subgroup is eleven points behind the aggregate. `report/findings.py`
turns result objects into ranked `Finding` records — severity, evidence, rule identifier,
suggested direction — and `Finding.as_dict()` is declared stable so no renderer needs to
know how a finding was derived. The alternative considered was leaving this to the reader
(or to the future LM's prompt), and rejected because an LM handed several hundred raw
numbers notices something different on every run, while an LM handed findings does the job
it is actually good at: turning "worst-group accuracy 0.68 against aggregate 0.79,
chi-square p = 0.004" into a sentence a clinician can read.

Findings are mechanical threshold comparisons on purpose. A mechanical finding is
auditable: every one names the numbers it came from, so a reader can disagree with it by
checking them.

**D49 — The subgroup-gap threshold is scale-free, and findings are grouped by subgroup.**
Two corrections found by looking at real output. A single absolute threshold (0.05) on the
gap is meaningless across metrics: it fires on any gap at all in an unbounded
`negative_log_likelihood` and needs twice the gap in Matthews' correlation, whose range is
two wide. `_gap_fraction` expresses the gap as a fraction of the metric's declared range,
or of the aggregate value where the metric is unbounded, and declines to report where
neither exists. Separately, ten metrics showing a gap is usually *one* fact — a single
badly served group — so the findings are grouped by which subgroup is worst rather than
emitted once per metric, which had buried the rest of the assessment under repetitions.

**D50 — Survey §8's reporting rules read the serialised report, not the objects.** The ten
checks in `report/rules.py` operate on the dictionary from `to_dict()`, because that is
what gets handed to a renderer, a database or a language model, and so it is the thing
whose completeness matters. It also means a report assembled by hand from loose numbers is
checked on identical terms. The rules **warn and never raise**: a library that refused here
would stop an honest exploratory run, and §8 is about what gets *published*, which this
code cannot observe. The issues join the model findings with `area="reporting"`, so
ignoring one is a decision rather than an accident.

**D51 — `target` and `neutral` are different fields.** The same class of defect as D19, one
level down. `MetricSpec.target` was documented as the ideal value and was being used for
the *no-change reference point* of the ratio metrics — so `flip_rate` declared
`target=1.0`, which says that every prediction changing class under perturbation is the
ideal outcome. Nothing read the field for a directed metric, so nothing complained, until
the report layer began putting `target` on a metric card handed to a reader. Fixed three
ways: `neutral` now holds the reference point (a corruption error of 1.0 means "degrades
like the baseline"), `target` holds the ideal, and `MetricSpec.__post_init__` refuses a
directed metric whose target disagrees with its bounds — a monotone metric's ideal *is* its
good bound. The retention ratios (`relative_robustness`, `error_based_robustness`,
`confidence_retention`, `relative_error_ratio`) also gained `upper_bound=1.0`, without
which they normalised through an unbounded sigmoid and a perfect retention of 1.0 mapped to
0.73 — which made every polar-chart axis carrying one look mediocre.

**D52 — A comparison refuses rather than drawing something misleading.** `comparison/`
normalises every metric to [0, 1] with 1 best before comparing, and then declines three
things: a metric one model has and another does not (averaging over "whatever each model
reported" rewards the model that reported fewer hard metrics); two robustness scores over
different underlying accuracy metrics (different quantities under one name, and the
underlying metric is picked by a default nobody looks at); and a sweep aggregate over
different perturbation sets (§8: sum-based aggregation scales with the size of the set).
Everything refused appears in `dropped` with the reason. The output held to be most useful
is `disagreements()` — the pairs of metrics that prefer different models — since where they
agree no comparison layer was needed, and where they disagree the trade-off is a deployment
decision rather than something a scoring rule settles.

The mean normalised score is a method rather than a field, and `ranking()` returns the
per-metric components beside it. That is §8 rule 4 applied to this package's own ranking
device.

**D53 — The polar chart caps its axes at eight and drops its fill past two models.** Both
found by rendering the figure and looking at it (D46's habit). Thirteen axes is a circle
with colliding labels; the builder raises above eight and names the table form and
`disagreements()` as the alternatives. Three translucent fills overlap into one muddy
region, so the fill is kept for one or two models and dropped beyond that, where the
outlines carry it. The radial scale sits at 0° with the metrics rotated half a sector, so
the tick numbers fall in the gap between two spokes instead of on a model's vertex — plotly
rotates radial labels to follow the axis angle and `tickangle` cannot override it, so the
chart has to move rather than the text.

Repo D's ancestor plotted **raw** values on a radial axis fixed to [0, 1], which is correct
only while every metric is a bounded accuracy: an error metric drawn there puts the better
model closer to the centre, and an unbounded one leaves the chart. The raw value is on the
hover instead, because a normalised score is not a number anyone can act on.

**D54 — Config is plain YAML into dataclasses, and it does not load your model.** D18 ruled
out hydra; `config.py` is what replaces it. A config governs the settings and, above all,
the criteria — the thresholds and tolerances, which belong in a versioned, reviewable,
attributable file because the survey makes them properties of the deployment. It
deliberately cannot name a class to import: that turns a settings file into executable
code, and then a model has to be constructible from YAML to be assessable at all, which is
exactly the coupling this library was extracted to remove. Every field is optional and an
empty file is valid, so a config is worth using for a first run and not only for a governed
one. A misspelt key is an error, because the alternative is a run that proceeds with the
default while the file says otherwise — hydra's strict mode existed for this and repo B had
it switched off.

**D55 — `Metrics_list.yaml` becomes `metrics/glossary.py`, and every metric gets a
sentence.** The original held plain-language descriptions for five metrics; all 61 now have
one, cross-checked against the registry in both directions by a test, so a metric cannot be
registered without a description or described without being registered. `describe_metric`
joins the sentence to the registry facts, producing the metric card that travels with the
numbers in a report — which is what stops a reader, or a language model, having to look up
whether lower is better. Two of the original five entries are deliberately *not* registered
as metrics: specificity is the recall of the negative class and negative predictive value is
its precision, both available from `per_class`, and registering them would mean two names
for one computation.

**D56 — A custom perturbation is selectable by name.** `default_perturbations()` builds a
fresh registry on every call, so registering into one instance was invisible to a later
name lookup and the only way to reach a custom perturbation was to sweep the whole
registry. `RobustnessAssessor` gained `perturbation_registry=`, which is where names are
resolved when given. Passing two registries raises rather than one silently winning.

**D57 — The tutorial is generated and executed, not hand-written.** `docs/tutorial.ipynb`
is built by `docs/build_tutorial.py`, which executes every cell and fails loudly on any
exception, so the notebook cannot drift from the library. Writing it also caught four places
where the narrative asserted something the output contradicted — the surrogate refusing
rather than attributing, temperature scaling's direction, a tipping point never reached, and
a subgroup tercile confounded with the label. All four are now the notebook's most
instructive passages, which is an argument for executing documentation rather than
describing what you expect it to say.

**D58 — Requirements files are generated from `pyproject.toml`, and the suite runs with
the package uninstalled.** `pyproject.toml` stays the single source of truth for
dependencies, and `requirements.txt`, `requirements-dev.txt`,
`requirements-tutorial.txt`, `requirements-optional.txt` and `environment.yml` are
generated from it by `tools/sync_requirements.py`. `tests/test_requirements.py` fails when
they are stale.

Two hand-maintained dependency lists are the same defect this whole consolidation was
about, one scale down: the day someone bumps a version in one and not the other, the conda
user and the pip user are running different libraries and nothing says so. The alternative
considered was hand-maintained files with a note asking people to keep them in step, and
rejected because a note is not a check.

The extras stay commented out in `requirements-optional.txt`. Uncommented it would install
all five at once, and darts and safetycage have transitive pins that may not co-exist in
one environment (the open question at the foot of this file). It is a menu, not an install.

Separately, `[tool.pytest.ini_options] pythonpath = ["src"]` makes `pytest` run against
`src/` with nothing installed, so `pip install -r requirements-dev.txt` followed by
`pytest` is a complete workflow — the "explore it in my own conda environment" path. Note
that an editable `pip install -e .` was *already* editable, so this is not a fix for a
frozen install; it is for people who would rather not install the package at all.

That change exposed a defect in the test harness. Four tests spawn a child interpreter —
two pinning cross-process seed stability (the `hash()` fix, D-numbered nowhere because it
was a bug rather than a decision), two pinning that importing the core does not pull in an
optional extra. A child does not inherit pytest's `pythonpath`, so on the no-install path
those four failed with `No module named 'themis_assess'` while every other test passed:
a harness failure that looks exactly like a broken property, which is the worst kind.
`run_child` in `tests/conftest.py` now puts `src/` on the child's `PYTHONPATH`, and
`test_a_child_interpreter_imports_the_source_tree_not_a_copy` pins that the child imports
this source tree rather than an installed copy.

**D59 — `dev` is an extra, not a PEP 735 dependency-group.** A group is the tidier home
for test tooling, and it was the original choice. It was also silently broken: a group is
reachable only as `pip install --group dev` (pip 25.1 or newer), so
`pip install -e ".[dev]"` — the command in this repo's README *and* in two CI jobs —
resolved `[dev]` as an extra, found none, printed `WARNING: themis-assess does not provide
the extra 'dev'`, and installed **neither pytest nor ruff**. Verified by running it: the
warning is one line in pip's output and nothing else went wrong, so both CI jobs had been
testing against whatever the base image happened to carry.

`dev` moves to `[project.optional-dependencies]`. Correctness of the documented command
beats packaging taste, and `tests/test_requirements.py` now cross-checks every `.[extra]`
appearing in an install command in the README or the CI config against the extras
pyproject actually defines — so the next such mismatch is a red test rather than a warning
nobody reads.

**D60 — `environment.yml` uses conda-forge for everything and has no `pip:` section;
the editable install is a separate, offline command.** Three revisions, two of them broken
in the hands of the first person to run them, so the reasoning is worth recording in full.

The first version put `- -e . --no-deps` in a `pip:` list. Conda does not execute each
`pip:` entry as its own command: it flattens the list into one temporary requirements file
and runs `pip install -U -r` on it. A requirements file accepts a narrower grammar than
the pip command line, and `--no-deps` is command-line-only, so pip failed with
`no such option: --no-deps` — reported by conda as nothing more than
`CondaEnvException: Pip failed`, which never names the line at fault.

The second version dropped the flag. It then failed on the network: pip's build isolation
fetches the build backend from PyPI before it can build an editable install, and the
`hatchling` wheel arrived truncated — `IncompleteRead(14094 bytes read, 815 more
expected)`. That is an institutional-network fault rather than a repository one, but the
file made the repository maximally exposed to it, because a `pip:` section routes *every*
wheel through PyPI and one partial read fails the whole `conda env create`. numpy and
scipy are tens of megabytes each; hatchling was simply the unlucky one.

So the third version inverts the arrangement. conda-forge supplies everything — the
scientific stack, pytest and ruff, Jupyter, and `hatchling` plus `editables` — through
conda's own downloader and a different index. There is no `pip:` section at all, and the
editable link becomes a documented third command:

    pip install -e . --no-deps --no-build-isolation

which, with the build tools already in place, touches the network **zero** times. Verified
with `--no-index`. `--no-build-isolation` cannot be folded back into the conda file for the
same grammar reason as `--no-deps`, and `PIP_NO_BUILD_ISOLATION` is not honoured by pip 24
— tested, not assumed.

The cost is one extra command instead of one; the gain is that the fragile step is the
*only* pip step, it is offline, and a failure costs a retry of a fast local command rather
than a rebuild of the environment. For a research library whose readers sit behind
corporate TLS inspection, that is the right trade.

The conda specs are generated from the same `pyproject.toml` lists as the requirements
files, with the constraints unchanged, so the two cannot disagree — the single-source-of-
truth discipline of D58 extended to a second output format. `CONDA_NAMES` exists for the
case where a PyPI name and a conda-forge name differ; it is empty for this project's
dependencies, and it is there so that a future dependency which does differ has an obvious
home rather than being silently wrong.

Four tests pin the result: the specs match pyproject, the build tools are present, there is
no `pip:` section, and no requirements file carries a command-line-only flag. Each was
checked to fail on the bug it describes before being trusted.


**D61 — the conda path is a CI job, because it was the one install path no check ever
ran.** `environment.yml` went through three revisions in two days, two of them broken, and
every breakage was found by the first person to run the file rather than by a test. The
common cause was mine: I reasoned about pip's and conda's behaviour instead of executing
it. A check that runs the file is the structural fix.

`conda_environment` uses the `mambaorg/micromamba` image — a single small binary reading
the same `environment.yml` from the same conda-forge, so the job costs seconds rather than
the minutes a full Miniconda install would. It creates the environment, runs the
documented `pip install -e . --no-deps --no-build-isolation --no-index`, imports the
package from outside the repository root, runs the suite, and checks the notebook tooling
is present. The `--no-index` is doing real work there: it is what turns "this step should
not need the network" into an assertion.

Two honest limits. micromamba is a different implementation from conda, sharing the
libmamba solver conda now uses, so the job catches structural mistakes — illegal syntax, a
package absent from conda-forge, a broken install flow — but is not proof about conda
itself. And it cannot reproduce the failure that prompted the current design: CI has a
clean connection, so a proxy that truncates downloads will never appear there.

The job could not be tested before committing, because this development sandbox has no
route to conda channels — verified, not assumed: `conda.anaconda.org` is unreachable and
micromamba is not on PyPI. What *was* checked locally is everything short of running it:
the CI YAML parses, `environment.yml` parses and contains only plain specs (so no `pip:`
section has crept back), every spec is well formed, and the image tag
`2.9-ubuntu24.04` exists — which caught an error, since `mambaorg/micromamba` publishes
**no** `latest` tag and the obvious guess would have failed. Two tests pin the job itself:
that some job runs `micromamba create -f environment.yml` with `--no-build-isolation`, and
that the environment name in the job matches the one the file declares. Both were confirmed
to fail on the mistake they describe before being trusted.

**D62 — the summary is three parts, and only one uses a language model.** (a) State what
was measured: templates in `summary/templates.py`, numbers substituted from each
finding's evidence, nothing generated. (b) Write it as a narrative for a named reader:
the one job a small local model does well, and the only part a model touches. (c)
Recommend actions: retrieved from `summary/actions.yaml`, which people write — 43 entries
across all 30 rules the library emits, cross-checked in both directions by tests. A model
may select and order catalogue entries; it may never author one, because what to do about
a weak model is knowledge about a deployment and a model asked to invent it produces
fluent unfounded advice, which is the hallucination that matters most.

**D63 — the audience shapes the templates, not only the prompt.** Requested by the person
who will use this: a clinician reading about the breast-cancer model needs different prose
from an engineer. Four readers ship. The reason it reaches the *deterministic* layer is
that the fallback has to be acceptable: a fallback producing engineer prose for a
clinician is one nobody will accept, and the moment the fallback is unacceptable the model
stops being optional.

Two details that took work. A percentage is offered only where the registry bounds the
metric on [0, 1] — rendering a negative log-likelihood of 0.2916 as "29%" is confident
nonsense, and formatting by magnitude cannot tell the cases apart, since plenty of
unbounded metrics land between 0 and 1. And a finding spanning several metrics needs one
chosen for the sentence: the widest gap on this model is in the log-likelihood, which is
the right answer to "which metric moved most" and useless to tell a clinician, so
`Audience.prefers_metric` picks the widest gap *among metrics this reader can interpret*.

**D64 — a generated narrative is untrusted input, and the fallback is the deterministic
one.** Four checks in `summary/verify.py`: numeric grounding, vocabulary, citations, and
banned causal language. On failure the model's output is discarded and the templates
stand. So the worst case is the no-model case — adding a model can improve the summary or
leave it unchanged, and cannot make it wrong. `provenance["narrative_source"]` records
which path wrote the prose, and the rendered document says so too.

Three defects were found while building it, each by running something rather than
reasoning about it:

*The allowed-number set is built as rendered strings, not compared with a tolerance.* A
tolerance accepts 93% as "close enough" to 0.9266 *and* accepts an invented 0.93.

*Citation markers had to be stripped before the content checks.* `[F1]` lowercased
contains `f1`, so the vocabulary check rejected every correctly-cited sentence that
happened to cite finding one.

*Grounding is scoped per sentence to the findings it cites.* This is the important one,
and the tutorial found it rather than a test. A model wrote "its correctness is 98%
[F3]" and the check **passed** it, because 0.98 was in the report — as a recall threshold
belonging to a different finding. The number existed; the attribution was invented, which
is the more insidious error. Scoping each sentence to its citations makes the citation
requirement load-bearing rather than decorative.

A residual limit, stated because it is real: even scoped, a model could attach one of a
finding's own numbers to the wrong quantity within that finding. What mitigates it is
that the model is shown the *already-correct sentence* and asked to arrange sentences
rather than compose them from raw numbers — which is also why a 3-to-9-billion-parameter
model suffices.

**D65 — templates declare the evidence keys they require.** Made tolerant of missing keys,
the acceptance template produced "The model's the metric is not measured, short of the not
measured that was agreed as the minimum" — grammatical, meaningless, and worse than the
plain fallback it displaced. `_require` raises so `render_finding` falls back instead. A
template either has what it needs or it stands aside. The catch around it is deliberately
broad `Exception`: this is the fallback path, and narrowing it to
`(KeyError, TypeError, ValueError)` let an `AttributeError` through and crashed the
summary.

**D66 — the model's raw output is kept in the provenance whether or not it was
accepted.** `provenance["model_output"]`, verbatim, alongside the verdict. The first
version recorded only `rejected_because`, which answers "was it accepted?" and not "what
did it say?" — and the second is the question actually asked when a summary looks thin.
Without the text there is no way to tell a model that needs a better prompt from one that
needs replacing, and no way for a reviewer to check that the verifier rejected the right
thing. The text is held as data and never as an instruction; the prompt itself is kept
only under `keep_prompt=True`, because it is several kilobytes in every serialised report
and is reproducible from `build_prompt` at any time.

**D67 — the headline is verified without the citation rule.** A headline is a claim about
the assessment as a whole ("this model does not meet 3 of its requirements") and has no
single finding to cite; the template fallback's own headline carries none either. Checked
together with the body, *every* narrative — accepted ones included — came back carrying a
spurious `uncited` problem against its first sentence, which is how a reader learns to
stop reading the problem list. So the two are verified separately and the verdicts merged
(`Verification.merge`). The headline's numbers are still checked, against the whole
evidence rather than one finding's: that is the weaker form of the grounding check and it
is the most that is available where there is nothing to scope to. Found by executing the
tutorial.

**D68 — `find_local_model` prefers, rather than requires, a named model.** Documentation
and example scripts have to run identically on a laptop with a model pulled and in CI
with none, and `OllamaModel("qwen3.5:9b")` on a machine that has `llama3.2` fails at the
worst moment — while demonstrating that the model is optional. `find_local_model(prefer)`
matches a tag or a family, falls back to whatever is installed, and returns `None` when
nothing is; `list_ollama_models` returns an empty list for all three indistinguishable
reasons (no package, no daemon, nothing pulled) rather than raising. Falling back to an
unexpected model is safe precisely because its output is verified before it is trusted:
the cost of guessing wrong is a rejected narrative, not a wrong one.

---

## Consequences for the architecture document

Revision 4 of `docs/ARCHITECTURE.md` will fold these in. The substantive changes:

1. `robustness/ood/` is removed from the module tree; §5.5 metrics move to
   `metrics/robustness/detection.py` (D20).
2. The module tree is rooted at `src/themis_assess/` (D2, D3).
3. `core/adapters.py` gains the array-boundary rule; `DartsAdapter` moves behind `[timeseries]` (D11).
4. §3.3's Python-version options collapse to `>=3.11`; option 1 (`>=3.13` with required safetycage) is
   struck (D7).
5. Metric names throughout the catalogue switch to survey terminology (D19).
6. The baseline-model discussion drops the abstract-method alternative (D14).
7. All plotting targets plotly; `viz/` gains no matplotlib dependency except the PDP/ICE path (D10).
8. The migration plan's Step 0 (rescue OODGuard source) is deleted (D5).

---

## Still to be tested, not decided

**Does `darts` (0.39.0 or newer) install and run on Python 3.13 with numpy 2.x / pandas 3.x?**
Under D7 this no longer blocks anything — darts is behind `[timeseries]` and core stays at `>=3.11` — but
the answer determines whether a `[timeseries]` install and a `[safetycage]` install can ever coexist in
one environment. Worth knowing before repo B is asked to adopt repo A.
