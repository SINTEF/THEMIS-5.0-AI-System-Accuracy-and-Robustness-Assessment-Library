"""Two fitted models on one small problem, for the comparison tests that need real
assessment objects rather than metric dictionaries."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from themis_assess.core.adapters import ArrayAdapter
from themis_assess.handlers.sklearn import SklearnHandler


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return {
        "data": ArrayAdapter(
            X, feature_names=["a", "b", "c", "d"], categorical_features=[]
        ),
        "actual": y,
        "handler": SklearnHandler(model, name="logistic"),
    }


@pytest.fixture(scope="module")
def other_model(problem):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] * 1.5 + X[:, 1] > 0).astype(int)
    return SklearnHandler(
        DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y), name="tree"
    )
