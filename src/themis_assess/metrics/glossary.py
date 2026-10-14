"""Plain-language descriptions of every registered metric.

The registry next door says what a metric *is* formally — its orientation, its bounds,
which tasks it applies to, which access level produces it. That is what the library
needs. It is not what a reader needs, and it is not what a summarising language model
needs either: neither can do much with ``higher_is_better=False, lower_bound=0.0``.

So this module carries the sentence. It is the descendant of repo A's
``Metrics_list.yaml``, which held exactly this and held it for five metrics; the
descriptions there are preserved almost verbatim below for accuracy, precision, recall
and their kin, and the remaining fifty-odd are written in the same register.

Two of that file's five entries are deliberately absent as metrics, because they are the
same two numbers under different names:

* **Specificity** is the recall of the negative class.
* **Negative predictive value** is the precision of the negative class.

Both come out of :attr:`~themis_assess.assessment.accuracy.AccuracyResult.per_class`
without a separate metric, and registering them would mean two names for one computation
— which is how the four repos ended up unable to compare their own numbers.

:func:`describe_metric` joins a description to the registry facts, producing the metric
card that travels with the numbers in a results bundle. That is the point of keeping the
two halves separate but adjacent: the formal properties stay machine-readable and
testable, the prose stays editable by whoever owns the wording, and a cross-check test
asserts that every glossary key names a registered metric and every registered metric has
a description.
"""

from __future__ import annotations

from typing import Any

from themis_assess.metrics.registry import MetricRegistry, default_registry

__all__ = ["METRIC_GLOSSARY", "describe_metric", "glossary_for", "missing_descriptions"]


METRIC_GLOSSARY: dict[str, str] = {
    # -- classification, aggregate ------------------------------------------
    "accuracy": (
        "The proportion of all class predictions that are correct. Ranges from 0 to 1. "
        "On imbalanced data a high value can be reached by always predicting the "
        "majority class, so read it beside the per-class breakdown."
    ),
    "balanced_accuracy": (
        "The mean of the recalls of each class, so every class counts equally however "
        "rare it is. Ranges from 0 to 1, and 0.5 is what random guessing achieves on a "
        "two-class problem regardless of how imbalanced it is."
    ),
    "precision": (
        "Of the samples predicted to be in the positive class, the proportion that "
        "really are. Ranges from 0 to 1. This is the metric that matters when acting on "
        "a positive prediction is expensive."
    ),
    "recall": (
        "Of the samples that really are in the positive class, the proportion the model "
        "found. Ranges from 0 to 1. This is the metric that matters when missing a "
        "positive case is expensive. The recall of the negative class is what is "
        "elsewhere called specificity."
    ),
    "f1": (
        "The harmonic mean of precision and recall, ranging from 0 to 1. A single number "
        "for the precision/recall trade-off, which means it also hides which of the two "
        "is weak — so it is a summary, not a substitute for both."
    ),
    "matthews_corrcoef": (
        "A correlation between the predictions and the truth, from -1 (perfectly wrong) "
        "through 0 (no better than chance) to +1 (perfect). Unlike accuracy and F1 it "
        "takes all four cells of the confusion matrix into account, so it cannot be "
        "inflated by a majority-class-only predictor."
    ),
    "auc": (
        "The area under the receiver-operating-characteristic curve: the probability "
        "that the model scores a randomly chosen positive sample above a randomly chosen "
        "negative one. 0.5 is chance, 1.0 is perfect. It measures the ranking only, so a "
        "model can score well here and still be badly calibrated."
    ),
    "auroc": (
        "The area under the receiver-operating-characteristic curve — the same quantity "
        "as ``auc``, under the name used where a second area-under-curve metric is also "
        "in play. 0.5 is chance, 1.0 is perfect."
    ),
    "auprc": (
        "The area under the precision-recall curve. Preferred over AUROC when the "
        "positive class is rare, because AUROC's false-positive rate has a large "
        "denominator there and moves very little. Its chance level is the positive-class "
        "prevalence, not 0.5."
    ),
    "fpr_at_tpr": (
        "The false-positive rate the model incurs at a chosen true-positive rate — how "
        "many false alarms are the price of catching the required share of positives. "
        "Lower is better. This is the operating-point form of the ROC curve, and the one "
        "an operator can act on."
    ),
    # -- calibration and uncertainty ----------------------------------------
    "expected_calibration_error": (
        "The average gap between the confidence the model states and the accuracy it "
        "actually achieves at that confidence, over bins of confidence. 0 means stated "
        "confidence can be taken at face value. It matters wherever a probability is "
        "acted on — a review threshold, a triage queue — and not at all where only the "
        "top-scoring class is used."
    ),
    "brier_score": (
        "The mean squared difference between the predicted probability and the outcome "
        "(0 or 1). Ranges from 0 (perfect) to 1. It rewards being both right and "
        "appropriately confident, so it penalises a confident mistake far more heavily "
        "than a hedged one."
    ),
    "log_likelihood": (
        "The log-probability the model assigned to what actually happened. Higher is "
        "better, and it is unbounded below: a single outcome the model called impossible "
        "sends it to minus infinity, which is the intended behaviour."
    ),
    "negative_log_likelihood": (
        "The negated log-probability the model assigned to what actually happened, so "
        "lower is better. Unbounded above, and dominated by the samples the model was "
        "most confidently wrong about."
    ),
    "predictive_entropy": (
        "How spread out the model's predicted probabilities are, in nats — 0 when it is "
        "certain of one class, at its maximum when every class is equally likely. It "
        "measures the model's stated uncertainty, and says nothing about whether that "
        "uncertainty is warranted."
    ),
    "prediction_interval_coverage": (
        "The proportion of true values that fell inside the model's stated prediction "
        "interval. A nominal 90% interval should cover about 0.90; well below that means "
        "the intervals are too narrow to be believed. Read it with the interval width, "
        "since an arbitrarily wide interval covers everything."
    ),
    "mean_prediction_interval_width": (
        "The average width of the model's prediction intervals, in the units of the "
        "target. Narrower is more useful, but only at adequate coverage — this metric "
        "exists to stop coverage being bought with uninformative intervals."
    ),
    "interval_score": (
        "A single score combining interval coverage and interval width, penalising both "
        "misses and needless width. Lower is better. It is the proper scoring rule for "
        "interval forecasts, which is why it cannot be gamed in either direction alone."
    ),
    # -- regression and forecasting -----------------------------------------
    "mae": (
        "Mean absolute error: the average size of the errors, in the units of the target. "
        "Lower is better. Every error counts in proportion to its size, so it is the less "
        "outlier-sensitive of the two standard error measures."
    ),
    "mse": (
        "Mean squared error: the average of the squared errors. Lower is better. Squaring "
        "means one large error can dominate the score, which is desirable when large "
        "errors are disproportionately costly and misleading when they are not."
    ),
    "rmse": (
        "The square root of the mean squared error, so it is back in the units of the "
        "target and comparable with the mean absolute error. Lower is better. It is never "
        "below the MAE, and the gap between them measures how uneven the errors are."
    ),
    "r2": (
        "The proportion of the variance in the target that the model explains, where 1 is "
        "perfect and 0 is no better than always predicting the mean. It can go negative, "
        "which means worse than that constant baseline."
    ),
    "mean_signed_deviation": (
        "The average error keeping its sign, so it measures systematic bias rather than "
        "size: positive means the model over-predicts on average, negative that it "
        "under-predicts. The ideal value is 0, and it is the one metric here where large "
        "errors in both directions can cancel to look perfect."
    ),
    "weighted_mae": (
        "Mean absolute error with a weight per sample, for when some samples matter more "
        "— recent timesteps, high-value entities, an under-represented group. Lower is "
        "better, and the weights are part of the metric's definition rather than a "
        "detail: two weightings give two different numbers."
    ),
    "weighted_mse": (
        "Mean squared error with a weight per sample. Lower is better. Same intent as the "
        "weighted absolute error, with the same sensitivity to large errors as the "
        "unweighted squared form."
    ),
    "crps": (
        "The continuous ranked probability score: how far the model's whole predictive "
        "distribution sat from the value that occurred, in the units of the target. Lower "
        "is better, and it reduces to the absolute error when the forecast is a single "
        "number — so probabilistic and point forecasts are comparable on it."
    ),
    "crps_gaussian": (
        "The continuous ranked probability score computed in closed form for a Gaussian "
        "forecast given its mean and standard deviation. Lower is better. Cheaper and "
        "exact where the ensemble form would only approximate."
    ),
    "energy_score": (
        "The multivariate generalisation of the CRPS: it scores a joint forecast over "
        "several targets at once, so a model that gets each margin right while getting "
        "their dependence wrong is penalised. Lower is better."
    ),
    # -- subgroups ----------------------------------------------------------
    "accuracy_gap": (
        "The difference in accuracy between the best-served and worst-served subgroup. 0 "
        "means the model works equally well for all of them; a large value means the "
        "aggregate is an average over groups with materially different experiences."
    ),
    "worst_group_accuracy": (
        "The accuracy on the subgroup the model serves worst; higher is better. This is "
        "the number a deployment guarantee has to be written against, since the aggregate "
        "can only ever be better than it."
    ),
    "accuracy_variance": (
        "The variance of accuracy across subgroups — how unevenly performance is spread "
        "over all of them rather than just between the two extremes. Lower is better."
    ),
    "group_disparity": (
        "The ratio of the worst subgroup's performance to the best subgroup's. 1 means no "
        "disparity. The ratio form rather than the difference, for metrics where a "
        "proportional gap is the meaningful one."
    ),
    "macro_error": (
        "The mean of the per-group error rates, giving every group equal weight whatever "
        "its size. Lower is better. It is what an aggregate error rate would be if the "
        "groups were equally represented, which is usually the fairer question."
    ),
    # -- distribution distance ----------------------------------------------
    "wasserstein_distance": (
        "How far one distribution has to be moved to become another, in the units of the "
        "variable — the 'earth mover's distance'. 0 means identical. Unlike the "
        "divergences it stays finite and meaningful when the two distributions barely "
        "overlap, which is the usual situation under a distribution shift."
    ),
    "kl_divergence": (
        "How much information is lost by using one distribution in place of another. 0 "
        "means identical, and it is asymmetric: the divergence from A to B is not the "
        "divergence from B to A. It goes infinite wherever the reference gives zero "
        "probability to something that occurs."
    ),
    "maximum_mean_discrepancy": (
        "A kernel-based distance between two samples: 0 when they could have come from "
        "the same distribution, larger the more they differ. It works in many dimensions "
        "at once without binning, which is what makes it usable on features rather than "
        "on one variable at a time."
    ),
    # -- robustness, absolute -----------------------------------------------
    "accuracy_drop": (
        "How much accuracy the model loses under perturbation: clean accuracy minus "
        "perturbed accuracy. 0 means the perturbation cost nothing. It is in the units of "
        "the accuracy metric, so a drop of 0.05 is five percentage points and not five "
        "percent."
    ),
    "relative_accuracy_drop": (
        "The accuracy lost under perturbation as a fraction of the clean accuracy, so it "
        "is comparable between models that started at different levels. 0 is no loss, 1 is "
        "total collapse."
    ),
    "error_inflation": (
        "How much larger the model's error becomes under perturbation: perturbed error "
        "minus clean error, in the units of the error metric. 0 means the perturbation "
        "cost nothing."
    ),
    "relative_error_inflation": (
        "The extra error under perturbation as a fraction of the clean error. 0 is no "
        "inflation; 1 means the error doubled. The relative form is the comparable one "
        "across models with different clean errors."
    ),
    "error_ratio": (
        "Perturbed error divided by clean error. 1 means unaffected, 2 means the error "
        "doubled. The ratio is undefined for a model with no clean error at all, which is "
        "the case to watch for on easy test sets."
    ),
    "relative_error_ratio": (
        "The error ratio expressed against a baseline model's error ratio, so it says "
        "whether this model degrades more or less than the reference does under the same "
        "perturbation. 1 means they degrade alike."
    ),
    "relative_robustness": (
        "Perturbed performance as a fraction of clean performance, signed so that "
        "positive is retention. 1 means the perturbation changed nothing; 0 means "
        "everything the model had was lost."
    ),
    "absolute_relative_robustness": (
        "The magnitude of the relative change under perturbation, ignoring its direction; "
        "0 means nothing moved and lower is better. Used where an improvement under "
        "perturbation is as suspicious as a degradation — that usually means the "
        "perturbation removed a feature the model was misusing."
    ),
    "error_based_robustness": (
        "Robustness expressed in the error family rather than the accuracy family: how "
        "little the error grows under perturbation. Higher is better. The error family "
        "exists because a metric with no upper bound cannot be normalised the way an "
        "accuracy can."
    ),
    "total_error_based_robustness": (
        "The error-based robustness aggregated over every perturbation and severity in "
        "the sweep — one number for the whole grid. Higher is better."
    ),
    "total_relative_robustness": (
        "The relative robustness aggregated over the whole sweep. Higher is better. Read "
        "it beside the worst cell, since an aggregate over severities is exactly what "
        "hides the severity that breaks the model."
    ),
    # -- robustness, baseline-relative --------------------------------------
    "corruption_error": (
        "The model's error under perturbation, divided by a reference model's error under "
        "the same perturbation. 1 means it degrades exactly like the reference, below 1 "
        "means it holds up better. Requires a baseline model, which is what makes it "
        "comparable across papers and meaningless without one."
    ),
    "mean_corruption_error": (
        "The corruption error averaged over every perturbation type. 1 means degradation "
        "matching the reference model's overall."
    ),
    "relative_corruption_error": (
        "The corruption error computed on the *extra* error the perturbation caused "
        "rather than on the total, so a model that was already worse when clean is not "
        "penalised twice. 1 means the added error matches the reference's added error."
    ),
    "mean_relative_corruption_error": (
        "The relative corruption error averaged over perturbation types. 1 means the "
        "added error matches the reference model's; lower is better."
    ),
    "mean_relative_error_inflation": (
        "The relative error inflation averaged over every perturbation and severity in "
        "the sweep. 0 means no inflation anywhere."
    ),
    # -- robustness, prediction stability -----------------------------------
    "flip_rate": (
        "The proportion of predictions that changed class under perturbation, whether or "
        "not they were right. 0 means the model's decisions were unmoved. It catches the "
        "case accuracy hides: as many predictions turning right as wrong leaves accuracy "
        "unchanged while every decision has moved."
    ),
    "mean_flip_rate": (
        "The flip rate averaged over every perturbation and severity in the sweep. 0 "
        "means no decision moved anywhere; lower is better."
    ),
    "flip_probability": (
        "The estimated probability that an individual prediction changes class under "
        "perturbation, over repeated random draws of the perturbation. Lower is better. "
        "The per-sample counterpart of the flip rate."
    ),
    "confidence_retention": (
        "How much of its stated confidence the model keeps under perturbation. 1 means "
        "unchanged. A model whose accuracy falls while its confidence does not is the "
        "dangerous case, because nothing downstream can tell that anything went wrong."
    ),
    # -- robustness, curve summaries ----------------------------------------
    "autc": (
        "The area under the degradation curve: performance summarised across the whole "
        "severity axis rather than at one severity. Higher is better. One number for how "
        "the model behaves as a perturbation intensifies."
    ),
    "weighted_autc": (
        "The area under the degradation curve with the severities weighted, for when the "
        "deployment will see mild perturbation often and extreme perturbation rarely. "
        "Higher is better, and the weights are part of the claim: state them."
    ),
    # -- composite ----------------------------------------------------------
    "composite_reliability_score": (
        "A single score combining accuracy, robustness and calibration into one number, "
        "for ranking models. Higher is better. It necessarily embeds a judgement about "
        "how those three trade off against one another, so it is a screening device and "
        "never the basis of an acceptance decision — the components are what a decision "
        "is made on."
    ),
    "unified_robustness_metric": (
        "A composite of the robustness family, aggregating degradation, stability and "
        "calibration retention into one number. Higher is better. Same caveat as any "
        "composite: it hides which component is weak."
    ),
    "equal_weight_risk": (
        "A composite that weights each component equally, stated explicitly so the "
        "weighting is visible rather than tuned. Lower is better. It exists to be the "
        "honest default when nobody has supplied a weighting."
    ),
}


def describe_metric(name: str, *, registry: MetricRegistry | None = None) -> dict[str, Any]:
    """The metric card: the sentence, plus every formal fact the registry holds.

    This is what travels with the numbers in a results bundle, so a reader — or a
    summarising model — never has to look a metric up elsewhere to know which direction
    is good.

    Raises
    ------
    KeyError
        If the metric is not registered. An unregistered metric has no orientation and
        no bounds, so there is nothing truthful to say about its value.
    """
    reg = registry or default_registry()
    spec = reg.get(name)
    return {
        "name": name,
        "description": METRIC_GLOSSARY.get(name, ""),
        "higher_is_better": spec.higher_is_better,
        "lower_bound": spec.lower_bound,
        "upper_bound": spec.upper_bound,
        "target": spec.target,
        "neutral": spec.neutral,
        "requires_baseline": spec.requires_baseline,
        "access": spec.access.name,
        "tasks": sorted(task.name for task in spec.tasks),
        "reference": spec.paper_reference,
    }


def glossary_for(
    names: Any, *, registry: MetricRegistry | None = None, strict: bool = False
) -> dict[str, dict[str, Any]]:
    """Metric cards for ``names``, skipping unregistered names unless ``strict``.

    Lenient by default because a results bundle should not fail to serialise over a
    custom metric someone registered locally and did not describe. ``strict=True`` is for
    a caller who wants to know.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        try:
            out[name] = describe_metric(name, registry=registry)
        except KeyError:
            if strict:
                raise
    return out


def missing_descriptions(*, registry: MetricRegistry | None = None) -> list[str]:
    """Registered metrics with no plain-language description.

    Pinned at empty by a test, so a metric added without a sentence fails the suite
    rather than reaching a reader as a bare name.
    """
    reg = registry or default_registry()
    return sorted(name for name in reg.names() if not METRIC_GLOSSARY.get(name))
