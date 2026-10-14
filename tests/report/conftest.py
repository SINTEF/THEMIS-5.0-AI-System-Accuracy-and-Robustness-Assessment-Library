"""One assessed model, shared by the report tests.

Module-scoped because fitting the forest and running a two-perturbation sweep is the
expensive part and none of these tests mutate it. The problem is deliberately one where
things are *wrong*: a subgroup the model serves worse, a requirement it misses, a
tolerance it exceeds. A report layer tested only on a clean assessment would pass while
being unable to say anything at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from themis_assess.assessment.accuracy import AccuracyEvaluator
from themis_assess.assessment.criteria import AcceptanceCriteria, Requirement
from themis_assess.assessment.robustness import RobustnessEvaluator
from themis_assess.assessment.tolerance import RobustnessCriteria, ToleranceRequirement
from themis_assess.core.adapters import ArrayAdapter
from themis_assess.core.subgroups import SubgroupSpec
from themis_assess.handlers.sklearn import SklearnHandler


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    n = 900
    age = rng.uniform(20, 85, n)
    severity = rng.normal(size=n)
    site = rng.choice(["oslo", "plovdiv"], n, p=[0.7, 0.3])
    logit = 0.06 * (age - 50) + 1.4 * severity + rng.normal(scale=0.9, size=n)
    y = (logit > 0).astype(int)
    X = np.column_stack([age, severity, rng.normal(size=n)])
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X[:600], y[:600])
    holdout = site[600:]
    return {
        "data": ArrayAdapter(
            X[600:], feature_names=["age", "severity", "noise"], categorical_features=[]
        ),
        "actual": y[600:],
        "handler": SklearnHandler(model, name="random_forest"),
        "subgroups": [
            SubgroupSpec(name, (lambda name: lambda d: holdout == name)(name))
            for name in ("oslo", "plovdiv")
        ],
    }


@pytest.fixture(scope="module")
def accuracy(problem):
    return AccuracyEvaluator(
        problem["handler"],
        subgroups=problem["subgroups"],
        criteria=AcceptanceCriteria(
            [
                Requirement(
                    "accuracy", acceptable=0.95, rationale="deliberately out of reach"
                ),
                Requirement(
                    "auprc",
                    acceptable=0.8,
                    rationale="a requirement on a metric this run does not compute",
                ),
            ],
            source="test",
        ),
    ).run(
        problem["data"],
        problem["actual"],
        diagnose_outliers=True,
        diagnose_ambiguity=True,
    )


@pytest.fixture(scope="module")
def accuracy_bare(problem):
    """The same assessment with no criteria and no breakdown — an incomplete report."""
    return AccuracyEvaluator(
        problem["handler"], metrics=["accuracy", "auc"], per_class=False
    ).run(problem["data"], problem["actual"])


@pytest.fixture(scope="module")
def robustness(problem):
    return RobustnessEvaluator(
        problem["handler"],
        perturbations=["gaussian_noise", "uniform_bias"],
        severities=(1, 2, 3),
    ).assess(
        problem["data"],
        problem["actual"],
        accuracy_metric="accuracy",
        criteria=RobustnessCriteria(
            [
                ToleranceRequirement(
                    "relative_robustness",
                    acceptable=0.95,
                    basis="worst_cell",
                    rationale="the conservative reading",
                )
            ]
        ),
    )


@pytest.fixture(scope="module")
def robustness_bare(problem):
    """A sweep with no tolerance supplied — survey §8's incomplete claim."""
    return RobustnessEvaluator(
        problem["handler"], perturbations=["gaussian_noise"], severities=(1, 2, 3)
    ).assess(problem["data"], problem["actual"], accuracy_metric="accuracy")
