"""Build (and execute) docs/tutorial.ipynb.

The notebook is generated from this file rather than hand-edited, so that its code is
executed on every rebuild and cannot drift from the library. Run:

    python docs/build_tutorial.py

It writes ``docs/tutorial.ipynb`` with outputs, and fails loudly if any cell raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

CELLS: list = []


def m(text: str) -> None:
    CELLS.append(md(text.strip("\n")))


def c(text: str) -> None:
    CELLS.append(code(text.strip("\n")))


# ==========================================================================
m(
    """
# Assessing a model end to end

### Breast cancer Wisconsin (diagnostic) · random forest · `themis-assess`

This notebook takes one model through every layer of the library and, at each step, says
what the number means and what it does *not* mean.

The data is the [Breast Cancer Wisconsin (Diagnostic)
dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): 569
fine-needle aspirate images of breast masses, 30 features computed from cell nuclei
(radius, texture, perimeter, area, smoothness, and so on — each as a mean, a standard
error and a worst-case), and a diagnosis of **malignant** or **benign**. It ships with
scikit-learn, so nothing is downloaded.

A random forest reaches around 96% on it without tuning, which is exactly why it is a
good teaching case. **A model that looks finished is where an assessment earns its
keep**, because the interesting questions are the ones accuracy cannot answer:

- Is 96% enough? Enough *for what*?
- Which 4% does it get wrong, and do they have anything in common?
- What happens when the imaging pipeline changes slightly, or a measurement is missing?
- Which single input is the model leaning on hardest?
- If its stated confidence is 0.9, is it right 90% of the time?

### Running it

The library's core dependencies are all this notebook needs, plus Jupyter:

```bash
pip install -e ".[dev]" jupyter
```

If that fights your existing environment, a notebook-specific one is quick:

```bash
python -m venv .venv-tutorial
.venv-tutorial/bin/pip install -e . jupyter
.venv-tutorial/bin/jupyter notebook docs/tutorial.ipynb
```
"""
)

# ==========================================================================
m(
    """
## 1. The model, and the two objects the library needs

Two wrappers, and neither one asks you to change your model or your data.

**A handler** turns your model into predictions and — this is the part that matters —
*declares what it can produce*. `PROBABILISTIC` says "I can give you class
probabilities". That declaration is a promise the library checks and plans against: it is
what lets an evaluator tell you which metrics are unavailable before spending anything.

**An adapter** presents your data in the library's canonical
`(entities, timesteps, features)` shape. Tabular data has one timestep, so the adapter is
mostly there to carry the feature names — which is what makes a finding say
`worst mean texture` rather than `column 21`.
"""
)

c(
    """
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import themis_assess as ta

print("themis-assess", ta.__version__)

cancer = load_breast_cancer()

# Three splits, not two. The third is a calibration set, held out from fitting and from
# testing alike: §4 fits a temperature on it, and fitting that on the test set would be
# tuning against the thing being measured. It is a small cost to pay up front and it is
# the difference between demonstrating calibration and faking it.
X_fit, X_test, y_fit, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=0, stratify=cancer.target
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_fit, y_fit, test_size=0.25, random_state=0, stratify=y_fit
)

# In this dataset, target 0 is malignant and 1 is benign. Worth stating out loud: the
# expensive error is calling a malignant mass benign, and that is the *negative* class,
# which is the opposite of the usual convention and the sort of thing that silently
# inverts an interpretation.
print("classes:", dict(zip(cancer.target_names, [0, 1])))
print("train:", X_train.shape, " calibration:", X_cal.shape, " test:", X_test.shape)
print("malignant in test:", int((y_test == 0).sum()), "of", y_test.size)

model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
"""
)

c(
    """
handler = ta.SklearnHandler(model, name="random_forest")
data = ta.ArrayAdapter(
    X_test,
    feature_names=list(cancer.feature_names),
    categorical_features=[],       # all 30 are continuous
)

print("access level:", handler.access.name)
print("task:", handler.task.name)
print(data)
"""
)

# ==========================================================================
m(
    """
## 2. Accuracy — and the thresholds that turn numbers into a verdict

Running the metrics is the easy half. The half that decides anything is the
**acceptance criteria**: what "good enough" means here.

The library will not invent them. Survey §8 puts it plainly — the acceptance bar is a
property of the deployment, not of the model, so it has to come from whoever owns the
deployment. What the library does is make its absence *visible*: with no criteria the
verdict is `None`, which means **nothing was checked**, and that is not a pass.

Each `Requirement` carries a `rationale`. It is not decoration. It is what a reader in a
year's time needs in order to disagree with the number.
"""
)

c(
    """
criteria = ta.AcceptanceCriteria(
    [
        ta.Requirement(
            "recall",
            acceptable=0.98,
            rationale="recall here is on the benign class; see the note below",
            critical=False,
        ),
        ta.Requirement(
            "balanced_accuracy",
            acceptable=0.95,
            rationale="both classes matter and the split is 63/37, so the aggregate flatters",
            critical=True,
        ),
        ta.Requirement(
            "expected_calibration_error",
            acceptable=0.05,
            rationale="the score is shown to a clinician as a probability",
            critical=False,
        ),
    ],
    source="tutorial — illustrative, not a clinical sign-off",
)

evaluator = ta.AccuracyEvaluator(handler, criteria=criteria)
evaluator.plan(data)
"""
)

m(
    """
`plan()` before `run()` is a habit worth keeping. It says what will be computed, what
will be skipped and *why* — a black-box handler has no calibration error, and being told
that up front is better than finding a gap in the results afterwards.

Now the run. One model call.
"""
)

c(
    """
accuracy = evaluator.run(
    data, y_test,
    diagnose_outliers=True,     # are the errors concentrated on unusual inputs?
    diagnose_ambiguity=True,    # does the model's own uncertainty find its errors?
)
print(accuracy.summary())
"""
)

m(
    """
Read the verdict, not the metrics.

`balanced_accuracy` is the critical requirement and it is the one to watch: with a 63/37
split, plain accuracy is flattered by the majority class, and balanced accuracy is the
average of the two per-class recalls, so it cannot be.

The `recall` requirement is the trap in this dataset, deliberately left in. scikit-learn
computes recall for the *positive* label, which here is **benign** — so that number says
how many healthy patients were correctly cleared, and says nothing at all about missed
malignancies. The clinically expensive error lives in the per-class breakdown for class 0,
two cells down. This is the kind of inversion that a metric name alone will not protect
you from.
"""
)

c(
    """
from themis_assess.viz import plot_metrics_table

plot_metrics_table(accuracy.acceptance)
"""
)

m(
    """
The verdict is *in* the table — value, requirement, pass or fail, and by how much — with
a symbol as well as a colour, so it survives a black-and-white print and a colour-blind
reader. Rows with no requirement are left neutral rather than coloured as warnings: they
were simply not part of the criteria, and painting them orange buries the rows that
actually failed.
"""
)

# ==========================================================================
m(
    """
## 3. Which class does it miss, and what does it confuse?

The aggregate is an average over the classes. Here is the breakdown it averages.
"""
)

c(
    """
import pandas as pd

pd.DataFrame(accuracy.per_class).T.rename(index=dict(enumerate(cancer.target_names)))
"""
)

m(
    """
**This is the number the clinical question was actually about**: the `error_rate` row for
`malignant`. It is the share of malignant masses the model called benign, and it is
several times the aggregate error rate — which the aggregate, by construction, cannot
show you.
"""
)

c(
    """
from themis_assess.viz import plot_confusion_matrix

plot_confusion_matrix(accuracy.confusion)
"""
)

c(
    """
# The most actionable output of a confusion matrix, and the one none of this library's
# four ancestor repositories returned: the specific mistakes, worst first, ranked by
# share of the true class rather than by raw count.
accuracy.diagnostics.get("top_confusions")
"""
)

# ==========================================================================
m(
    """
## 4. Is the confidence trustworthy?

A model can rank cases perfectly and still state its confidence wrongly. That matters the
moment a probability is *acted on* — a review threshold, a triage queue, an
expected-cost calculation — and not at all if only the top-scoring class is used. So it is
worth knowing which situation you are in before deciding whether to care.

Expected calibration error is the average gap between stated confidence and observed
accuracy. The reliability diagram is the same information, unaggregated: it shows *where*
along the confidence range the gap is.
"""
)

c(
    """
from themis_assess.viz import assess_calibration

prediction = handler.predict(data)
calibration = assess_calibration(prediction.probabilities, y_test)
print(calibration.summary)
calibration.figure
"""
)

m(
    """
Note that `assess_calibration` returned a `Bundle`: the figure **and** the analysis behind
it. That is a deliberate rule throughout `themis_assess.viz` — a function that computes
something and plots it must return both. In the repositories this library was extracted
from, a box-plot function ran an ANOVA on the way past and left the p-value reachable only
through a dictionary keyed on the plot's title.

The bin count is reported beside the ECE, and it should be: too few bins hide
miscalibration by averaging over-confident and under-confident regions together, too many
leave each bin with a handful of samples and a noisy accuracy estimate. An ECE quoted
without its bin count is not reproducible.

If calibration needs fixing, **temperature scaling** is the standard first move: divide
the logits by a single fitted scalar. Its appeal is that it **cannot change a prediction**
— it is monotone, so every argmax stays where it was — and its cost is one number fitted
on held-out data. Below is the honest version of that demonstration, fitted on the
calibration split, and it does not go the way a tutorial would like.
"""
)

c(
    """
from scipy.special import softmax

from themis_assess.metrics.calibration import expected_calibration_error, fit_temperature

# fit_temperature works on logits. A random forest only gives probabilities, so we use
# their logs: softmax(log p) == p, so a temperature applied to log p is the standard way
# to scale a probability-only model. The clip keeps a zero probability finite — a forest
# of 200 trees produces plenty of exact zeros.
def as_logits(probabilities):
    return np.log(np.clip(probabilities, 1e-12, None))


# Fitted on the calibration split, which the model has never seen and which is not the
# test set.
cal_data = ta.ArrayAdapter(
    X_cal, feature_names=list(cancer.feature_names), categorical_features=[]
)
cal_probabilities = handler.predict(cal_data).probabilities
temperature = fit_temperature(as_logits(cal_probabilities), y_cal)

before = prediction.probabilities
after = softmax(as_logits(before) / temperature, axis=1)

print(f"fitted temperature: {temperature:.4f}  (>1 = the model was over-confident)")
print(f"test ECE before: {expected_calibration_error(before, y_test):.4f}")
print(f"test ECE after:  {expected_calibration_error(after, y_test):.4f}")
print(
    "predictions changed:",
    int((after.argmax(1) != before.argmax(1)).sum()),
    "- monotone, so necessarily zero",
)
"""
)

m(
    """
It helped, and the direction is the interesting part. The fitted temperature came out
**below 1**, which means this forest was *under*-confident: averaging 200 trees pulls
probabilities toward the middle, so a case the ensemble is really sure about still comes
out at 0.9 rather than 0.99. Dividing the logits by 0.75 sharpens them, and the test ECE
falls by about a quarter — with, as promised, not one prediction changed.

Three caveats before reaching for it as a habit:

- It optimises **negative log-likelihood**, not ECE. Those are different objectives — NLL
  is dominated by the few samples the model was most confidently wrong about, ECE averages
  a gap over bins — so a temperature that improves one can worsen the other. It improved
  both here; that is not guaranteed.
- It is **one scalar for the whole range**. A model over-confident at the top of its range
  and about right in the middle cannot be fixed by any single temperature. The reliability
  diagram, not the single ECE number, tells you which situation you are in.
- It was fitted on 100 held-out cases. That is a small calibration set, and the
  temperature carries corresponding uncertainty.

The library gives you the diagnostic rather than applying a fix automatically, because
whether calibration matters at all depends on something it cannot see: whether anything
downstream reads the probability, or only the argmax.
"""
)

# ==========================================================================
m(
    """
## 5. Are the errors concentrated anywhere?

Two diagnostics that ask the same question in different ways: *are the mistakes random,
or do they have a shape?*

**Outliers.** If the errors cluster on the most unusual inputs, the model is already weak
on the tails of its own training distribution — and degradation under a synthetic shift
later on is then expected rather than surprising. The library scores outlyingness by
combining three detectors (isolation forest, local outlier factor, k-nearest-neighbour
distance) and runs a chi-square test on the difference in error rate, so the finding is
either significant or it is not stated.

**The model's own uncertainty.** If the predictions it is least sure about are no more
often wrong than the rest, its uncertainty carries no information — and a human review
queue built on that flag would be sorting at close to random. That is worth knowing before
anyone is hired to staff it.
"""
)

c(
    """
outliers = accuracy.diagnostics["outliers"]
print(outliers.describe())
"""
)

m(
    """
A non-finding, and reported as one. The error rate on the most unusual inputs is
indistinguishable from the rest — the difference is under a percentage point, in either
direction depending on the run, and the chi-square test does not come close to
significance. The honest statement is that **the errors are not concentrated on the
tails**.

That is worth having explicitly, because it rules out an explanation. The degradation seen
under perturbation in §6 is *not* simply the model being weak on unusual inputs and the
perturbation pushing more cases into that region: the two are separate effects here, so the
robustness result needs a different account.

Notice what the library did not do. There *is* a numeric difference, and it was not
reported as a finding, because the test says it is indistinguishable from noise. A
difference that fails its own test is not evidence — and 17 outliers is far too few to
conclude anything from either way.
"""
)

c(
    """
ambiguity = accuracy.diagnostics["ambiguity"]
flagged, unflagged = ambiguity["error_rate_flagged"], ambiguity["error_rate_unflagged"]
print(f"flagged as ambiguous:    {ambiguity['flagged_fraction']:.1%} of predictions")
print(f"error rate when flagged: {flagged:.1%}")
print(f"error rate when not:     {unflagged:.1%}")
print(f"ratio:                   {flagged / unflagged:.1f}x")
"""
)

m(
    """
**This one found something, and it is directly actionable.** The predictions the model is
least sure about are wrong more than ten times as often as the rest. So its uncertainty
*does* carry information, which means a review queue built on that flag would work: sending
the flagged fifth of cases to a human catches most of the errors while leaving four fifths
of the volume untouched.

That is the operational conclusion, and note where it came from — not from the accuracy
number, which cannot express it, but from a diagnostic that asks whether the model knows
when it does not know. Had the two error rates matched, the same three lines would have
told you not to staff that queue.
"""
)

c(
    """
from themis_assess.viz import assess_feature_bins

# Does the error rate vary along one feature? "worst area" is the strongest single
# predictor in this dataset, so it is the natural place to look.
column = X_test[:, list(cancer.feature_names).index("worst area")]
bins = assess_feature_bins(column, y_test, prediction.predicted, feature="worst area")
print(bins.summary)
bins.figure
"""
)

# ==========================================================================
m(
    """
## 6. Robustness — what happens when the data shifts

Everything so far describes the model on *this* test set. Robustness asks what happens
when the input changes in ways it will actually change: an instrument recalibrated, a
measurement missing, a systematic offset in one lab's readings.

Three things to understand before reading the result.

**Severity is an ordered axis, 1 to 5.** Magnitudes are *relative*: each severity scales
the feature's own standard deviation, so severity 3 means something comparable across
features measured in wildly different units — which these are, from `mean smoothness`
around 0.1 to `worst area` in the thousands.

**Cost is knowable in advance.** `plan()` reports the exact number of model calls before
anything runs.

**The tolerance has to come from you.** Same principle as the acceptance criteria, and
survey §8 is explicit that this is the one element no evaluation instrument can supply:
"the tolerance ... must come from the requirements of the deployment itself."
"""
)

c(
    """
sweeper = ta.RobustnessEvaluator(
    handler,
    perturbations=["gaussian_noise", "uniform_bias", "missing_values"],
    severities=(1, 2, 3, 4, 5),
    n_variants=3,          # three independent draws per cell, averaged
    seed=0,                # recorded on the result, so one cell can be re-run alone
)
plan = sweeper.plan(data, y_test)
print(plan.summary())
"""
)

c(
    """
robustness = sweeper.assess(
    data, y_test,
    accuracy_metric="balanced_accuracy",   # name it: two robustness scores over
                                           # different underlying metrics are not
                                           # comparable, and the default is a
                                           # convenience rather than a choice
    criteria=ta.RobustnessCriteria(
        [
            ta.ToleranceRequirement(
                "relative_robustness",
                acceptable=0.90,
                basis="worst_cell",
                rationale="the worst case is the one a clinic has to survive",
                critical=True,
            )
        ]
    ),
)
print(robustness.summary())
"""
)

m(
    """
First, the number that will confuse a reader: `relative_robustness` shows about **2.5**
in the summary while the tolerance verdict reads **0.53**. Those are two different
quantities. The verdict is the *worst cell*, and the 2.5 is the sum-based aggregate over
three perturbations — survey §8 flags exactly this, that a sum-based aggregate scales with
the size of the perturbation set and so is comparable only within a fixed one. It is why
`RobustnessResult` records its `perturbation_set`, and why the comparison layer in §10
refuses to put two such aggregates on one axis unless the sets match.

Second, the `basis` is not a detail. The same sweep can pass on the `aggregate` basis —
averaged over the whole grid — and fail on `worst_cell`. `worst_cell` is the conservative reading
and the right one whenever the deployment has to survive its bad days rather than its
average day.

Now the curve, which is what survey §8 asks for in place of a single severity: whether the
model degrades gracefully or falls off a cliff is the finding, and it is invisible from one
point.
"""
)

c(
    """
from themis_assess.viz import assess_severity

severity = assess_severity(robustness.sweep, "balanced_accuracy")
severity.figure
"""
)

c(
    """
from themis_assess.viz import plot_tolerance_table

plot_tolerance_table(robustness.tolerance)
"""
)

m(
    """
Read the shapes, not just the endpoints. Noise the model shrugs off and missing values it
does not are two completely different engineering conclusions: the first says the
instrument's precision does not matter much, the second says a dropped measurement has to
be *detected* rather than imputed and passed through.

Notice too that the curve tells you where the cliff is. A tolerance that passes at
severity 3 and fails at 4 is a specification, not a failure — provided somebody can say
which severity the deployment will actually see.
"""
)

# ==========================================================================
m(
    """
## 7. Which input is it leaning on?

The sweep perturbs everything at once, so it says *how much* damage a shift does and not
*where* the damage comes from. A sensitivity map perturbs one feature at a time, so its
answer is attributable.

The useful output is not the ranking on its own — it is the **tipping point**: how far one
input can be pushed before the model notices. A feature with a flat row is one you do not
need to instrument carefully. A feature whose row falls off a cliff between severity 2 and
3 is a single point of failure in the data pipeline.
"""
)

c(
    """
mapper = ta.SensitivityMapper(
    handler,
    perturbation="gaussian_noise",
    severities=(1, 2, 3, 4, 5),
    n_variants=3,
    measure="degradation",
    metric="balanced_accuracy",
    seed=0,
)
sensitivity = mapper.feature_severity_map(data, y_test)
for row in sensitivity.most_sensitive_rows(n=6):
    print(f"{row['label']:<26} {row['score']:.4f}")
"""
)

c(
    """
from themis_assess.viz import plot_sensitivity_map

plot_sensitivity_map(sensitivity, title="Balanced-accuracy loss per feature, per severity")
"""
)

c(
    """
hottest = sensitivity.hottest_cell()
print("worst single cell:", hottest)
print()

worst_feature = sensitivity.most_sensitive_rows(n=1)[0]["label"]
for threshold in (0.01, 0.02, 0.05):
    tipping = sensitivity.tipping_point(worst_feature, threshold=threshold)
    reached = f"severity {tipping}" if tipping is not None else "never, within severities 1-5"
    print(f"{worst_feature!r} first costs {threshold:.0%} of balanced accuracy at: {reached}")
"""
)

m(
    """
`None` from `tipping_point` is an answer, not a missing value: it means the threshold was
**never crossed anywhere in the swept range**. Here the worst single feature costs barely a
point of balanced accuracy even at severity 5, so the 1% threshold is only reached at the
very end of the axis and 2% is never reached at all.

That is a genuinely reassuring shape. No single input, pushed to four standard deviations
of noise on its own, takes this model down — the forest has 30 correlated features to fall
back on and no individual point of failure among them. Compare it with the whole-dataset
sweep in §6, which perturbed **all thirty at once** and cost far more: the damage there is
cumulative rather than attributable to one input, and those are different engineering
problems. A single dominant feature would mean instrumenting one sensor carefully;
cumulative degradation means the whole input pipeline's quality matters.
"""
)

# ==========================================================================
m(
    """
## 8. What do the cases it gets wrong have in common?

A different question again. Not "which feature breaks it under perturbation" but "is there
a *kind* of case it fails on".

The method: fit a small surrogate that tries to predict **this model's error** from the
same inputs, then read the surrogate's feature importances. If the surrogate succeeds, the
errors are structured and its importances say along which features. If it fails, that is
itself a result — the errors are unstructured, which is what you want and is the more
common outcome for a good model.

The gate matters. An error model that has learned nothing will still produce feature
importances, and they will be noise wearing the clothes of an explanation. This one
**refuses to attribute** below its threshold, and says so.
"""
)

c(
    """
error_model = ta.ErrorModel(error_kind="auto", test_fraction=0.25, seed=0)
fit = error_model.fit(data, y_test, prediction.predicted)
print(fit.describe())
print()
print("trustworthy:", fit.is_trustworthy)
"""
)

c(
    """
from themis_assess.viz import plot_attribution

if fit.is_trustworthy:
    attribution = error_model.importance()
    for entry in attribution.top(5):
        print(f"{entry['feature']:<26} {entry['importance']:.4f}")
else:
    attribution = None
    print(
        "The surrogate could not predict this model's errors from the input, so it has "
        "nothing to attribute — a finding rather than a failure. Attributions produced "
        "anyway would be noise wearing the clothes of an explanation."
    )
"""
)

c(
    """
plot_attribution(attribution) if attribution is not None else None
"""
)

m(
    """
**The surrogate refused, and the refusal is the most instructive output in this notebook.**

Look at what the fit report says. Train R² is 0.96 and test R² is *negative* — the
surrogate memorised its 128 training rows perfectly and generalised worse than predicting
the mean. And the gate it is actually judged on, test AUROC, is 0.508: for a 0/1 error
target that is a coin flip.

So there is nothing here to attribute, and the library says so instead of producing a
ranking. That matters because **it would have produced one**. Feature importances are
always computable; an importance from a surrogate that cannot predict its target is noise
in the shape of an explanation, and it is the kind of output that gets screenshotted into a
slide and believed.

Note also which number the gate uses. An earlier version of this library gated on R², and
on this exact kind of data that is the wrong yardstick — a 0/1 target with a 6% positive
rate gives a poor R² even to a surrogate that ranks errors usefully, and a good R² to one
that predicts "no error" for everything. Measuring both on synthetic data whose error rate
genuinely doubled across terciles gave R² of −0.04 and AUROC of 0.64: the R² gate would
have rejected a working surrogate. The fit report prints the reason in as many words.

**When would this have worked?** With more data (there is one row per *case*, so 171 test
cases is thin for a 30-feature surrogate), or with errors that actually cluster in feature
space. The honest conclusion here is that this model's mistakes do not share a describable
region — which, for a model going into use, is better news than the alternative.
"""
)

c(
    """
# The check worth doing when a surrogate *does* pass: the raw marginal relationship, with
# no model in between. A feature the surrogate ranks highly should show something here.
if attribution is not None:
    top_feature = attribution.top(1)[0]["feature"]
    scatter = error_model.feature_error_scatter(top_feature)
    values, wrong = scatter["feature_value"], scatter["error"].astype(bool)
    print(f"{top_feature}:")
    print(f"  mean where the model was wrong: {values[wrong].mean():.2f}  (n={wrong.sum()})")
    print(f"  mean where it was right:        {values[~wrong].mean():.2f}  (n={(~wrong).sum()})")
else:
    print(
        "Skipped: there is no trustworthy attribution to check. importance() raises "
        "here rather than returning the noise; force=True overrides that, and should "
        "only be used by someone who intends to look at exactly that."
    )
"""
)

# ==========================================================================
m(
    """
## 9. Does it work equally well for everyone?

An aggregate is an average over whatever groups are in the data, and it is high exactly
when the majority group is well served. So the breakdown belongs beside it — survey §5.1.2
and §8 both ask for this.

There is no demographic column in this dataset, so we use a clinically meaningful one:
tumour size, in terciles of `mean area`.

Two steps, and both are needed. First the per-subgroup metrics — *how large* is the gap.
Then a significance test — *is the gap distinguishable from sampling noise*. A gap of eight
points across 57 cases per group may be nothing at all, and reporting it without the test
is how an evaluation ends up chasing noise.
"""
)

c(
    """
area = X_test[:, list(cancer.feature_names).index("mean area")]
edges = np.quantile(area, [1 / 3, 2 / 3])
size_group = np.where(area <= edges[0], "small", np.where(area <= edges[1], "medium", "large"))

subgroups = [
    ta.SubgroupSpec(name, (lambda name: lambda d: size_group == name)(name))
    for name in ("small", "medium", "large")
]

by_size = ta.AccuracyEvaluator(handler, subgroups=subgroups, criteria=criteria).run(data, y_test)
pd.DataFrame({name: dict(sub.metrics) for name, sub in by_size.subgroups.items()}).T[
    ["accuracy", "balanced_accuracy", "expected_calibration_error"]
]
"""
)

c(
    """
from themis_assess.viz import plot_subgroup_metrics

plot_subgroup_metrics(
    by_size.subgroups,
    ["accuracy", "balanced_accuracy"],
    overall=by_size.overall,
    title="Performance by tumour size tercile",
)
"""
)

c(
    """
from themis_assess.viz import assess_groups

# The per-case correctness, grouped: is the difference real, or is it sampling noise?
correct = (prediction.predicted == y_test).astype(float)
groups = assess_groups(correct, size_group, metric="correct")
print(groups.summary)
groups.figure
"""
)

m(
    """
### The result is not the one a tutorial would choose, and it is the more instructive one

The small tercile is at 1.000 — perfect. That is not the model being brilliant on small
masses; it is the **subgroup being confounded with the label**. Tumour area is one of the
strongest single predictors of malignancy in this dataset, so slicing by area very nearly
slices by diagnosis: the small tercile is almost all benign and therefore almost trivially
separable, and the middle tercile is where the two classes actually overlap.

That makes it a bad fairness comparison and a good lesson. **A subgroup defined by a
feature correlated with the target does not compare like with like** — the groups differ in
base rate, so their accuracies were never going to be comparable, and the model doing
"worse" on the middle group mostly reflects that the middle group is the hard one. What
would make this analysis meaningful is a subgroup axis independent of the label:
which scanner, which site, which technician, which year.

The library cannot detect this for you — nothing in the data says which columns are
confounded with the target. Choosing the subgroup axis is a modelling judgement, and it is
the judgement that decides whether the resulting numbers mean anything.

### The test says the difference is real. That does not make it a fairness finding.

Kruskal-Wallis rejects at p ≈ 0.03, and the pairwise comparison — Holm-corrected, so the
three tests do not inflate each other — names `medium vs small` as the pair that differs.
So the gap is not sampling noise.

It is still not evidence that the model treats a group unfairly, for the reason above: the
groups have different base rates, so this is largely a measurement of which tercile
contains the ambiguous cases. **A significance test tells you a difference is real; it
cannot tell you what the difference is about.** That part is on whoever chose the subgroup
axis.

Two omnibus tests are run rather than one: **Welch's ANOVA** (which compares group means
without assuming equal variances) and **Kruskal-Wallis** (which compares distributions and
assumes nothing about their shape). Here Welch's is skipped — the small group is constant
at 1.0, so its within-group variance is zero and the F statistic is undefined. That is
reported rather than worked around, and it is why the summary quotes Kruskal-Wallis. When
both run and disagree, `tests_agree` surfaces it rather than letting the choice of test
decide the conclusion.

And the caveat to carry away for the cases that come out the other way: a non-significant
result is **not** evidence of equivalence. With 57 cases per group the test can easily lack
the power to detect a difference that would matter clinically.
"""
)

# ==========================================================================
m(
    """
## 10. Model x, model y or model z?

The comparison every project actually needs, and the one that is easiest to get wrong.

Three problems the layer solves, each of which produces a confident and meaningless
answer if ignored:

1. **Metrics point in different directions and live on different scales.** An error and an
   accuracy on one axis makes the worse model look better. Every value is normalised to
   [0, 1] with 1 always better, using the registry's declared orientation and bounds.
2. **A metric one model has and another does not is not a comparison.** Averaging over
   "whatever each model reported" rewards the model that reported fewer of the hard
   metrics. Only shared metrics become axes; everything else is recorded as *dropped*,
   with the reason.
3. **A composite hides the trade-off.** So the most useful output here is not the ranking
   — it is `disagreements()`: the pairs of metrics that prefer different models.
"""
)

m(
    """
### First, a practical obstacle worth understanding

Every model in a comparison has to survive the **same** perturbation set, or the scores
are not comparable. And `missing_values` produces NaN, which scikit-learn's
`LogisticRegression` and `DecisionTreeClassifier` refuse outright — only the forest
accepts it natively.

The library will not quietly drop the perturbation for the models that cannot take it.
That would mean scoring each model over the perturbations it happens to tolerate, which
**overstates robustness** for exactly the models least able to handle real data. So it
raises, and names the fix: put an imputer in the pipeline.

That is not a workaround, it is the honest comparison. A deployed logistic regression
would need an imputation step anyway, so including one is comparing the models as they
would actually be shipped — and the imputer's own contribution to the degradation is part
of what is being measured.
"""
)

c(
    """
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# The forest takes NaN natively; the other two get a median imputer so that all three
# face an identical perturbation set.
candidates = {
    "random forest": model,
    "logistic": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)
    ).fit(X_train, y_train),
    "shallow tree": make_pipeline(
        SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=3, random_state=0)
    ).fit(X_train, y_train),
}

reports = []
for name, candidate in candidates.items():
    h = ta.SklearnHandler(candidate, name=name)
    reports.append(
        ta.build_report(
            model_name=name,
            accuracy=ta.AccuracyEvaluator(h, criteria=criteria).run(data, y_test),
            robustness=ta.RobustnessEvaluator(
                h,
                perturbations=["gaussian_noise", "uniform_bias", "missing_values"],
                severities=(1, 2, 3, 4, 5),
                n_variants=3,
                seed=0,
            ).assess(data, y_test, accuracy_metric="balanced_accuracy"),
        )
    )

comparison = ta.compare_models(reports)
print(comparison.summary())
"""
)

c(
    """
comparison.to_frame().round(4)
"""
)

c(
    """
from themis_assess.viz import assess_models

# A polar chart of a dozen axes is a circle, so choose the handful the decision turns on.
# Ask for more than eight and the builder refuses and says to use the table instead.
compared = assess_models(
    reports,
    axes=[
        "balanced_accuracy",
        "auc",
        "expected_calibration_error",
        "brier_score",
        "relative_robustness",
    ],
)
compared.figure
"""
)

m(
    """
Every axis runs 0 (centre) to 1 (edge) and further out is always better, whatever the
metric's natural direction. The measured value is on the hover, because a normalised score
is not a number anyone can act on.

If one loop encloses another the choice is easy and you did not need the chart. The
interesting case is the one where the loops cross — and that is what `disagreements()`
enumerates, so the trade-off is stated rather than left for the eye to find.
"""
)

c(
    """
for disagreement in comparison.disagreements()[:8]:
    print(disagreement.describe())
print()
print("could not be compared:")
for metric, reason in sorted(comparison.dropped.items()):
    print(f"  {metric}: {reason}")
"""
)

# ==========================================================================
m(
    """
## 11. The report — everything in one place, and what is wrong with it

The last layer collects whatever you ran into one object, serialises it to a plain
dictionary with a declared schema, and — the part worth the most — ranks what is wrong.

The **findings** layer exists because a results dictionary answers "what are the numbers"
and nobody reads far enough to notice that the worst subgroup is eleven points behind the
aggregate. Each finding carries a severity, the evidence it rests on, and a suggested
direction. They are deliberately mechanical: every one names the numbers it came from, so
you can disagree with it by checking them.

They are also the structure a summarising language model needs. Handing a model several
hundred raw numbers and asking what is weak gives a different answer every run; handing it
findings leaves it the job it is good at — turning
`worst-group accuracy 0.68 against aggregate 0.79, chi-square p = 0.004` into a sentence a
clinician can read.
"""
)

c(
    """
report = ta.build_report(
    model_name="random_forest",
    accuracy=by_size,                     # the assessment with the subgroup breakdown
    robustness=robustness,
    sensitivity=[sensitivity],
    groups={"by_tumour_size": groups.analysis},
    context={
        "dataset": "breast cancer Wisconsin (diagnostic), UCI id 17",
        "split": "stratified random holdout, 30% test, seed 0",
        "subgroups": "terciles of mean area",
        "note": "tutorial run — the criteria are illustrative, not a clinical sign-off",
    },
)
print(report.summary())
"""
)

c(
    """
for finding in report.findings():
    print(f"[{finding.severity:<8}] {finding.area:<12} {finding.headline}")
"""
)

c(
    """
worst = report.findings()[0]
print(worst.headline)
print()
print(worst.detail)
print()
print("evidence:", worst.evidence)
print()
print("suggestion:", worst.suggestion)
"""
)

m(
    """
### The report checks itself

Survey §8 lists ten ways a robustness claim can mislead, and most of them are mechanically
checkable. `check_report` runs them against your own report: a relative drop with no
absolute value beside it, `mCE` without `mRCE`, an aggregate over an unnamed perturbation
set, a composite without its components, one severity in place of a curve, an aggregate
with no breakdown, calibration measured only on clean data, an operating-point metric with
no threshold named, a distribution distance among the robustness scores, coverage without
width.

They warn rather than raise — a library that refused here would stop an honest exploratory
run, and §8 is about what gets *published*, which this code cannot observe. The issues land
in the findings under `area: "reporting"`, so ignoring one is a decision rather than an
accident.
"""
)

c(
    """
issues = ta.check_report(report.to_dict())
if issues:
    for issue in issues:
        print(f"[{issue.severity}] {issue.rule}")
        print(f"    {issue.detail}")
        print(f"    → {issue.suggestion}")
        print()
else:
    print("No §8 reporting issues: this report states everything a robustness claim needs.")
"""
)

c(
    """
payload = report.to_dict()
print("sections:", sorted(payload))
print()
print("findings by severity:", payload["summary"]["by_severity"])
print("findings by area:    ", payload["summary"]["by_area"])
print()
# The glossary travels with the numbers, so a reader never has to look up a direction.
card = payload["glossary"]["relative_robustness"]
for key, value in card.items():
    print(f"  {key}: {value}")
"""
)

c(
    """
path = report.save("reports/tutorial_random_forest.json")
print("written:", path, f"({path.stat().st_size / 1024:.0f} KB)")
"""
)

# ==========================================================================
m(
    """
## 12. A readable summary, for a named reader

Everything so far is a number or a picture. This is the layer that turns them into prose
someone will actually read — and it is split into three parts, only one of which involves
a language model at all.

**(a) State what was measured.** Templates, with the numbers substituted from each
finding's evidence. Nothing is generated, so nothing can be invented. This is what runs
below: no model is configured, and the output is complete.

**(b) Write it as a narrative for a particular reader.** The one job a small local model
does well. It is optional, verified, and discarded on failure — see below.

**(c) Recommend what to do.** From a YAML file that people write and review. A model may
select and order entries; it may never author one. What to do about a weak model is
knowledge about *your* deployment, and a model asked to invent it produces fluent,
unfounded advice — the hallucination that matters most, because recommendations are what
people act on.

The reader is a parameter, and it shapes the templates too, not only the model's prompt.
That is deliberate: if the audience only reached the prompt, the fallback would produce
engineer prose for a clinician, and a fallback nobody accepts is a fallback that makes
the model mandatory.
"""
)

c(
    """
from themis_assess.summary import summarise

clinical = summarise(report, audience="clinician")
print(clinical.to_markdown(include_evidence=False))
"""
)

m(
    """
Now the same assessment, same findings, same numbers — written for the person who would
change the model.
"""
)

c(
    """
technical = summarise(report, audience="ml_engineer")
print(technical.to_markdown(include_actions=False, include_evidence=False))
"""
)

m(
    """
Note what differs and what does not. The **findings are identical** — the same five
observations, in the same severity bands. What changes is the order they arrive in
(a clinician meets the subgroup findings first, an engineer the robustness ones), the
number format, the metric names, and which metric each sentence leads with.

That last one is subtler than it looks. The widest subgroup gap by proportion on this
model is in the *negative log-likelihood* — correct as an answer to "which metric moved
most", and useless to tell a clinician. So a reader who works in percentages leads with a
metric that can be one. And a percentage is offered only where the metric registry bounds
the metric on [0, 1]: rendering a log-likelihood of 0.29 as "29%" would be confident
nonsense, and formatting by magnitude alone cannot tell the two cases apart.
"""
)

c(
    """
from themis_assess.summary import CLINICIAN, ML_ENGINEER

for reader in (CLINICIAN, ML_ENGINEER):
    print(f"{reader.name}:")
    print(f"  bounded   balanced_accuracy = 0.9266 -> "
          f"{reader.format_value('balanced_accuracy', 0.9266)}")
    print(f"  unbounded neg. log-likelihood = 0.2916 -> "
          f"{reader.format_value('negative_log_likelihood', 0.2916)}")
    print(f"  leads with: {sorted(['negative_log_likelihood', 'balanced_accuracy'], key=reader.prefers_metric)[0]}")
"""
)

m(
    """
### The actions come from a file, not from a model

`summary/actions.yaml` — 43 entries across all 30 finding rules the library can emit,
each with the condition under which it applies, who would do it, what it costs and what
it trades away. A test asserts every rule has at least one entry, and that no entry is
keyed to a rule nothing emits.

The caveats are the point. An action list without them reads as a recipe and gets
followed rather than considered.
"""
)

c(
    """
from themis_assess.summary import default_catalogue

catalogue = default_catalogue()
print(f"{len(catalogue)} actions across {len(catalogue.rules)} rules")
print()

for action in catalogue.for_rule("accuracy.subgroup_gap"):
    print(f"- {action.action}")
    print(f"    when:   {action.precondition}")
    print(f"    who:    {action.owner}")
    print(f"    caveat: {action.caveat[:150]}")
    print()
"""
)

m(
    """
The order is the file's order, and it is meaningful: *check whether the subgroup axis is
confounded with the label* comes before *reweight the training data*, because it is
cheaper, more diagnostic, and — as §9 showed on this very dataset — frequently the actual
answer.

### What happens when a model is added

The model's output is treated as untrusted input. It goes through four checks, and on any
failure it is discarded in favour of the templates above:

1. **Numeric grounding** — every number must be traceable to the evidence of the finding
   *that sentence cites*, in a form derived from it. The scoping matters more than it
   sounds: writing this notebook, a model produced "its correctness is 98% [F3]" and an
   earlier version of the check passed it, because 0.98 **was** in the report — as a
   recall threshold belonging to a different finding. The number existed and the
   attribution was invented, which is the more insidious error. So citations are not only
   traceability; they are the boundary of what a sentence may claim.
2. **Vocabulary** — no metric name the assessment did not compute.
3. **Citations** — every sentence carries `[F3]`, so a claim can be traced. Uncited
   sentences are dropped rather than the whole narrative being rejected.
4. **No causal language** — nothing here establishes *why* the model fails, so
   "because", "due to", "the reason" and their kin are forbidden outright.

Below, `EchoModel` stands in for a local model so the behaviour is reproducible in a
notebook. In use you would pass `OllamaModel("qwen3.5:9b")` instead. Five responses: one
well-behaved, one that invents a figure outright, one that takes a **real** figure from
the wrong finding, one that asserts a cause, and one that ignores the format entirely.

The well-behaved response is *built from the real statements* rather than written out, so
it stays well-behaved as the assessment changes. Hard-coding it was the first version of
this cell, and the finding numbers shifted underneath it on the next rebuild — which is
its own small argument for executing documentation.
"""
)

c(
    """
import json
import re

from themis_assess.summary import EchoModel

statements = clinical.statements
print(f"the model is shown {len(statements)} numbered statements. The first two:")
for fid in list(statements)[:2]:
    print(f"  {fid}: {statements[fid][:150]} ...")
print()


def figures_in(text):
    \"\"\"The numbers a statement already contains — the only ones a sentence citing it
    may use.\"\"\"
    return re.findall(r"-?\\d[\\d,]*(?:\\.\\d+)?%?", text)


# The well-behaved response is built from the real statements rather than written by
# hand, so it stays well-behaved as the assessment changes. It restates one figure from
# each finding and cites that finding — which is exactly what the prompt asks for.
first, second = list(statements)[:2]
good = {
    "headline": "Two things stand out about this model.",
    "themes": [
        {
            "title": "The first thing to know",
            "findings": [int(first[1:])],
            "prose": f"One measure sits at {figures_in(statements[first])[-1]} [{first}].",
        },
        {
            "title": "The second",
            "findings": [int(second[1:])],
            "prose": f"Another sits at {figures_in(statements[second])[-1]} [{second}].",
        },
    ],
}

responses = {
    "well-behaved": good,
    "invents a figure": {
        "headline": "The model performs strongly.",
        "themes": [{"title": "Accuracy", "findings": [int(first[1:])],
                    "prose": f"Its correctness is 42.4242% [{first}]."}],
    },
    "borrows another finding's number": {
        "headline": "Accuracy falls short.",
        "themes": [{"title": "Accuracy", "findings": [int(first[1:])],
                    "prose": f"Its correctness is "
                             f"{figures_in(statements[second])[-1]} [{first}]."}],
    },
    "asserts a cause": {
        "headline": "Accuracy falls short.",
        "themes": [{"title": "Accuracy", "findings": [int(first[1:])],
                    "prose": f"It sits at {figures_in(statements[first])[-1]} because "
                             f"the training set was small [{first}]."}],
    },
    "ignores the format": "I'd be happy to help summarise that for you!",
}

for label, response in responses.items():
    payload = response if isinstance(response, str) else json.dumps(response)
    result = summarise(report, audience="clinician",
                       model=EchoModel(payload, name="demo-model"))
    source = result.provenance["narrative_source"]
    print(f"{label:34} -> {source}")
    if source == "model":
        print(f"{'':36} accepted. headline: {result.headline}")
    else:
        print(f"{'':36} {result.provenance['rejected_because'][:110]}")
"""
)

m(
    """
Four of the five were rejected, and the summary in each case is the template version —
identical to what you read at the top of this section. **That is the property the whole
design rests on: adding a model can make the summary better or leave it unchanged, and
cannot make it wrong.** The worst case is the no-model case.

The rejection is also visible in the document rather than hidden. A reader can always
tell which path wrote what they are reading.
"""
)

c(
    """
bad = summarise(
    report,
    audience="clinician",
    model=EchoModel(json.dumps(responses["invents a figure"]), name="demo-model"),
)
print(bad.to_markdown(include_actions=False, include_evidence=False)[-700:])
"""
)

m(
    """
### The prompt

One function builds it, so it can be printed, diffed and reviewed. It is small — the model
sees the **findings**, never the results dictionary.
"""
)

c(
    """
from themis_assess.summary import CLINICIAN, build_prompt

prompt = build_prompt(CLINICIAN.order(report.findings()), CLINICIAN)
full_report = json.dumps(report.to_dict())
print(f"prompt:            {len(prompt):>7,} characters  (~{len(prompt)//4:,} tokens)")
print(f"full report JSON:  {len(full_report):>7,} characters  (~{len(full_report)//4:,} tokens)")
print()
print(prompt[:1500])
print("   ... [the findings follow] ...")
"""
)

m(
    """
Feeding the whole dictionary would cost roughly twelve times the context **and triple the
number of figures a model could misquote** — 142 against 40 — for no information the task
needs. The findings are the right input, and building them was most of the work.

### Watching the model work

Everything above shows the *outcome* — accepted or rejected. This shows the working: the
model that answered, the JSON it returned verbatim, what the verifier made of it, and the
prose that survived.

The model's raw output is kept in `summary.provenance["model_output"]` **whether or not it
was accepted**. That matters for a rejection: "what did it actually say?" is the first
question asked when a summary looks thin, and it is the only way to tell a model that
needs a better prompt from one that needs replacing.

The next cell looks for a model this machine already has:

```bash
ollama pull qwen3.5:9b        # or llama3.2, gemma3, phi4 — any instruct model
pip install ollama
```

If it finds one, **everything below is that model's real output**. If not, the same code
runs against a recorded response so the notebook still demonstrates the path, and says so.
Nothing here needs a model in order to execute — which is the same property the library
has.
"""
)

c(
    """
from themis_assess.summary import find_local_model, list_ollama_models

local = find_local_model(["qwen3.5:9b", "qwen3", "llama3.2", "gemma3", "phi4"])
print("models installed here:", list_ollama_models() or "none")
print("using:", local.name if local else "no local model — a recorded response stands in")
"""
)

c(
    """
def show_working(summary, *, note=""):
    prov = summary.provenance
    print(f"model:            {prov.get('model', 'none')}   {note}")
    print(f"prompt:           {prov.get('prompt_characters', 0):,} characters")
    print(f"narrative source: {prov['narrative_source']}")
    print()

    raw = prov.get("model_output")
    if raw is None:
        print("no model was called.")
        return

    print("--- what the model returned, verbatim " + "-" * 40)
    try:
        print(json.dumps(json.loads(raw), indent=2)[:1800])
    except json.JSONDecodeError:
        print(repr(raw[:800]))
    print()

    check = prov.get("verification", {})
    print("--- what the verifier made of it " + "-" * 45)
    print(f"accepted: {check.get('ok')}")
    for problem in check.get("problems", []):
        print(f"  [{problem['kind']}] {problem['detail'][:160]}")
    if not check.get("problems"):
        print("  no objections: every number traced to the finding its sentence cites,")
        print("  every sentence cited, no invented metric name, no asserted cause.")
    print()

    print("--- the prose the reader gets " + "-" * 48)
    print(summary.headline)
    for theme in summary.themes:
        print(f"\\n## {theme.title}\\n{theme.prose}")


recorded = EchoModel(json.dumps(good), name="recorded-response")
summary = summarise(
    report,
    audience="clinician",
    model=local or recorded,
    keep_prompt=True,
)
show_working(
    summary,
    note="" if local else "(recorded — no local model on this machine)",
)
"""
)

m(
    """
And the same view of a rejection, which is where the provenance earns its keep. This is
the subtle failure: the figure is **real** and appears in the report, but it belongs to a
different finding than the one the sentence cites.
"""
)

c(
    """
show_working(
    summarise(
        report,
        audience="clinician",
        model=EchoModel(json.dumps(responses["borrows another finding's number"]),
                        name="demo-model"),
    )
)
"""
)

m(
    """
The narrative was discarded, the templates stand, and the model's words are still on the
record — so the rejection is diagnosable rather than a shrug.

Model size matters less here than you would expect, because the task is *constrained
rewriting* rather than reasoning: the model is arranging sentences that are already
correct. A 3-to-9-billion-parameter model is adequate; asking a larger one to diagnose
causes would not help, because the information is not in the input.

One caution on reading a real model's output above: `narrative_source: model` means the
four checks found nothing to object to, not that the paragraphs are good. Grounded,
cited, non-causal prose can still be badly organised or emphasise the wrong finding, and
no mechanical check will tell you so. That judgement stays with the reader — which is
why the deterministic statements are printed beneath every summary.
"""
)

# ==========================================================================
m(
    """
## 13. Reproducing one cell

Worth demonstrating, because it is the property most sweeps quietly lack. The seed on the
result is enough to reproduce **one cell** of the grid in isolation — not just the whole
sweep from the start.

This works because each cell's random state is derived from the recorded seed plus the
perturbation name, the severity and the variant index, hashed stably. An earlier version
of this library used Python's `hash()` for that, which is randomised per process: the same
recorded seed gave three different numbers in three runs. There are two subprocess tests
pinning it now.
"""
)

c(
    """
same = ta.RobustnessEvaluator(
    handler,
    perturbations=["missing_values"],
    severities=(4,),
    n_variants=3,
    seed=0,                    # the seed the sweep above recorded
).run(data, y_test)

original = robustness.sweep.curve("missing_values", "balanced_accuracy")[4]
repeated = same.curve("missing_values", "balanced_accuracy")[4]
print(f"in the full sweep:  {original:.10f}")
print(f"re-run alone:       {repeated:.10f}")
print("identical:", original == repeated)
"""
)

# ==========================================================================
m(
    """
## What this told us, and what it did not

Working through it in order:

- The aggregate accuracy is high, and the **per-class breakdown** is where the clinically
  expensive error lives. The `recall` requirement measured the benign class — a reminder
  that a metric name is not a specification.
- The **calibration** number decides whether the probability can be shown to a clinician
  as a probability. Temperature scaling improved it here without changing a single
  prediction — and the fitted temperature below 1 said the forest was *under*-confident,
  which is the direction an ensemble average usually errs in.
- The **severity curves** separate shifts the model shrugs off from shifts that break it,
  and those are different engineering conclusions: precision that does not need to be
  bought versus a missing measurement that has to be detected rather than imputed.
- The **sensitivity map** turned that into a specific input and a specific severity — a
  tipping point, which is something a data pipeline can be built against.
- The **error model** refused: its surrogate could not predict this model's mistakes from
  the input at better than chance, so the mistakes do not share a describable region of
  feature space. That is a real answer, and the better of the two.
- The **subgroup gap** was real by the significance test — and still not a fairness
  finding, because the subgroup axis was confounded with the label. The test answers
  "is this difference noise"; nothing answers "is this difference the one you meant to
  measure" except the person who chose the grouping.
- The **comparison** ranked three models, and more usefully named the trade-off: the
  logistic regression won on essentially every accuracy and calibration metric, while the
  forest was the least *disturbed* by perturbation. Where the metrics disagree no scoring
  rule settles it — the deployment does.

**What no part of this could tell us**: whether any of it is good enough. Every threshold
in this notebook was invented for the illustration. The verdicts are real arithmetic
against made-up requirements, and the requirements are the input the library cannot
supply — that is the whole point of survey §8's insistence on the tolerance, and the
reason `passed is None` is a distinct outcome from `passed is False`.

### Where to go next

- `README.md` — the architecture map, where to find what, and how to extend each layer
- `docs/DECISIONS.md` — every design decision with the alternative that was rejected
- `examples/assessment.yaml` — the same settings as a versioned config file, which is
  where criteria belong once anyone other than you depends on them
- `describe_metric("any_metric_name")` — the direction, bounds, ideal value and survey
  equation for any of the 61 metrics
"""
)


def build() -> int:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    target = HERE / "tutorial.ipynb"

    if "--no-execute" not in sys.argv:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(HERE.parent)}},
        )
        client.execute()

    nbf.write(notebook, target)
    print(f"wrote {target} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
