"""Learning what makes the model wrong: a surrogate model of its error.

The most novel capability in any of the four repositories, and the one that appears in
neither the survey nor any inventoried toolkit. The idea is to fit a second, small model
that predicts the *first* model's error from a description of its input, and then read that
surrogate's feature attributions. Where a sensitivity map says "perturbing this feature
moves the model", this says "the model is already wrong, and here is what the wrong cases
have in common" — without perturbing anything.

Ported from repo B's ``TrainErrorModel`` / ``RankFeaturesForSample`` /
``GlobalFeatureImportance``, with the Valencia specifics lifted out (the berth incident
locator, the sigmoid over a BCEWithLogitsLoss head, the vessel-series layout) and four
changes of substance.

*No optional dependency is load-bearing.* Repo B required XGBoost, SHAP and TSFresh
together. Here the default surrogate is scikit-learn's
:class:`~sklearn.ensemble.HistGradientBoostingRegressor` and the default attribution is
permutation importance — both core dependencies — so the whole capability works out of the
box. XGBoost as the surrogate and SHAP for local attribution are available and sharper;
they are not required.

*The surrogate's own accuracy gates its attributions.* This is the important one. An error
model that cannot predict the error has attributions that are noise, and reading them
anyway is worse than not computing them: the output looks identical either way. Repo B
returned ``test_r2`` alongside the fitted model and nothing stopped a caller reaching
straight for ``GlobalFeatureImportance`` at an R² of 0.02. Here
:attr:`ErrorModel.is_trustworthy` gates the attribution methods, and overriding the gate
takes an explicit argument.

*Held-out attribution by default.* Repo B warned by ``print`` when a local explanation was
requested for a sample the surrogate had trained on. A print in a library is a warning
nobody reads; :meth:`ErrorModel.explain` reports the sample's role in a returned field and
refuses in-sample explanation unless asked.

*One split, honestly.* Repo B split train/validation/test three ways to support XGBoost
early stopping. The default surrogate needs no validation set, so there are two splits and
the test set is genuinely untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from themis_assess.attribution.features import FeatureExtractor, resolve_extractor
from themis_assess.core.adapters import DataAdapter

__all__ = [
    "ErrorModel",
    "ErrorModelFit",
    "FeatureAttribution",
    "per_sample_error",
]

Role = Literal["train", "test"]

#: Below this test R², a surrogate learning a *continuous* error is not predicting it and
#: its attributions are noise. Deliberately conservative; overridable.
DEFAULT_TRUST_THRESHOLD = 0.1

#: The gate for a *binary* error target — a misclassification indicator — where R² is the
#: wrong yardstick entirely. Whether a particular borderline sample flips is close to
#: irreducibly random, so R² can sit near zero while the surrogate ranks error risk
#: perfectly usefully. Measured on synthetic data whose error rate doubles across terciles
#: of a noise feature, the surrogate scored R² = -0.04 and AUROC = 0.64: the R² gate would
#: have thrown away a real finding. 0.55 is a modest but non-accidental signal above the
#: 0.5 a coin flip gives.
DEFAULT_AUROC_THRESHOLD = 0.55


def per_sample_error(
    actual: ArrayLike,
    predicted: ArrayLike,
    *,
    kind: str = "auto",
) -> NDArray[np.float64]:
    """The quantity the surrogate learns to predict: one error per sample.

    Distinct from a metric, which aggregates. An error model needs the per-sample values,
    and the aggregation is exactly what destroys the signal it is looking for.

    Parameters
    ----------
    kind:
        ``"absolute"`` — ``|actual - predicted|``, for a regression or forecast.
        ``"squared"`` — squares it, which weights the large errors the surrogate is most
        likely to find structure in.
        ``"misclassified"`` — a 0/1 indicator, for classification. The surrogate then
        predicts the probability of being wrong, which is the natural question there.
        ``"auto"`` picks ``misclassified`` for integer labels with few distinct values and
        ``absolute`` otherwise.
    """
    y = np.asarray(actual)
    p = np.asarray(predicted)
    if y.shape[0] != p.shape[0]:
        raise ValueError(
            f"actual and predicted must describe the same samples; got {y.shape[0]} and "
            f"{p.shape[0]}."
        )

    if kind == "auto":
        looks_categorical = (
            y.dtype.kind in "iub"
            or y.dtype.kind in "OUS"
            or (y.dtype.kind == "f" and np.unique(y).size <= 20)
        )
        kind = "misclassified" if looks_categorical else "absolute"

    if kind == "misclassified":
        if y.ndim > 1 or p.ndim > 1:
            raise ValueError(
                "kind='misclassified' needs one label per sample; got shapes "
                f"{y.shape} and {p.shape}."
            )
        return (y != p).astype(float)

    difference = np.abs(y.astype(float) - p.astype(float))
    if difference.ndim > 1:
        # A multi-horizon forecast: one error per sample, averaged over the horizon.
        difference = difference.mean(axis=tuple(range(1, difference.ndim)))
    if kind == "absolute":
        return difference
    if kind == "squared":
        return difference**2
    raise ValueError(
        f"kind must be auto, absolute, squared or misclassified; got {kind!r}."
    )


@dataclass(frozen=True, slots=True)
class ErrorModelFit:
    """How well the surrogate learned the error — the gate on everything downstream."""

    test_r2: float
    train_r2: float
    n_train: int
    n_test: int
    n_features: int
    error_kind: str
    error_mean: float
    error_std: float
    surrogate: str
    extractor: str
    test_auroc: float | None = None
    train_auroc: float | None = None

    @property
    def is_binary_error(self) -> bool:
        """Whether the target is a misclassification indicator rather than a magnitude."""
        return self.test_auroc is not None

    @property
    def gate(self) -> str:
        """Which statistic decides whether the attributions are worth reading."""
        return "test_auroc" if self.is_binary_error else "test_r2"

    @property
    def gate_value(self) -> float:
        return self.test_auroc if self.is_binary_error else self.test_r2

    @property
    def gate_threshold(self) -> float:
        return (
            DEFAULT_AUROC_THRESHOLD if self.is_binary_error else DEFAULT_TRUST_THRESHOLD
        )

    @property
    def is_trustworthy(self) -> bool:
        """Whether the surrogate learned enough for its attributions to mean anything.

        Gated on AUROC for a binary error target and on R² for a continuous one, because
        the two questions are different: a magnitude can be predicted, whereas whether a
        particular borderline classification flips largely cannot, even when the *risk* of
        flipping is highly predictable.
        """
        return self.gate_value >= self.gate_threshold

    @property
    def looks_overfitted(self) -> bool:
        """Train score far above test. Common here: few entities, many features."""
        if self.is_binary_error:
            return (self.train_auroc - self.test_auroc) > 0.25
        return self.train_r2 - self.test_r2 > 0.5

    def describe(self) -> str:
        lines = [
            f"surrogate {self.surrogate} on {self.extractor} features "
            f"({self.n_features} columns, {self.n_train} train / {self.n_test} test)",
            f"error: {self.error_kind}, mean {self.error_mean:.4f} "
            f"(sd {self.error_std:.4f})",
            f"R^2: {self.train_r2:.3f} train, {self.test_r2:.3f} test",
        ]
        if self.is_binary_error:
            lines.append(
                f"AUROC: {self.train_auroc:.3f} train, {self.test_auroc:.3f} test "
                f"(the gate — R^2 is the wrong yardstick for a 0/1 error target)"
            )
        if not self.is_trustworthy:
            lines.append(
                f"  the surrogate does not predict the error ({self.gate} "
                f"{self.gate_value:.3f} below {self.gate_threshold}): its attributions "
                f"would be noise"
            )
        elif self.looks_overfitted:
            lines.append(
                "  train score far above test: attributions may reflect the surrogate's "
                "memorisation rather than structure in the error"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class FeatureAttribution:
    """Which input characteristics are associated with the model being wrong.

    ``signed`` is present only for attribution methods that have a direction (SHAP does,
    permutation importance does not). Where it is present, positive means the feature
    pushes the predicted error *up* — toward worse accuracy.
    """

    features: tuple[str, ...]
    importance: NDArray
    signed: NDArray | None = None
    method: str = "permutation"
    scope: Literal["global", "local"] = "global"
    sample_role: Role | None = None
    baseline: float | None = None
    predicted_error: float | None = None
    actual_error: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def top(self, n: int = 10) -> list[dict[str, Any]]:
        """The ``n`` most important features, most important first."""
        order = np.argsort(np.abs(self.importance))[::-1][:n]
        out = []
        for i in order:
            entry: dict[str, Any] = {
                "feature": self.features[i],
                "importance": float(self.importance[i]),
            }
            if self.signed is not None:
                entry["signed"] = float(self.signed[i])
                entry["direction"] = "worse" if self.signed[i] > 0 else "better"
            out.append(entry)
        return out

    def to_frame(self) -> Any:
        import pandas as pd

        data: dict[str, Any] = {"importance": self.importance}
        if self.signed is not None:
            data["signed"] = self.signed
        return (
            pd.DataFrame(data, index=pd.Index(list(self.features), name="feature"))
            .sort_values("importance", key=np.abs, ascending=False)
        )


class ErrorModel:
    """A surrogate that predicts the assessed model's error from its input.

    Parameters
    ----------
    extractor:
        How to describe the input. ``None`` picks by modality — raw columns for tabular,
        window summaries for sequential. Pass ``"tsfresh"`` for repo B's extractor.
    surrogate:
        Any scikit-learn-compatible regressor. ``None`` uses
        ``HistGradientBoostingRegressor``, which is a core dependency, handles NaN
        natively, and needs no validation set. ``"xgboost"`` uses ``XGBRegressor`` if
        installed.
    error_kind:
        Which per-sample error to learn. See :func:`per_sample_error`.
    test_fraction:
        Held out and never trained on, so the reported R² means something.
    seed:
        The split and the surrogate are both seeded from this.

    Notes
    -----
    The sample size is the binding constraint and it is easy to miss. There is one row per
    *entity*, so on repo B's data — one series per vessel — the surrogate had dozens of
    rows against hundreds of TSFresh columns. That is a regime where a gradient-boosted
    model will fit the training set perfectly and generalise not at all, which is what
    :attr:`ErrorModelFit.looks_overfitted` is for.
    """

    def __init__(
        self,
        *,
        extractor: FeatureExtractor | str | None = None,
        surrogate: Any = None,
        error_kind: str = "auto",
        test_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}.")
        self._extractor_spec = extractor
        self._surrogate_spec = surrogate
        self.error_kind = error_kind
        self.test_fraction = test_fraction
        self.seed = seed

        self._fitted: Any = None
        self._features: list[str] = []
        self._matrix: NDArray | None = None
        self._errors: NDArray | None = None
        self._train_index: NDArray | None = None
        self._test_index: NDArray | None = None
        self.fit_report: ErrorModelFit | None = None

    # -- fitting -----------------------------------------------------------------

    def _build_surrogate(self) -> tuple[Any, str]:
        if self._surrogate_spec is None:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return (
                HistGradientBoostingRegressor(
                    max_iter=200, max_depth=3, random_state=self.seed
                ),
                "HistGradientBoostingRegressor",
            )
        if self._surrogate_spec == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                    "surrogate='xgboost' needs the [explain] extra: pip install "
                    "'themis-assess[explain]'. The default surrogate is a core "
                    "dependency and needs no extra."
                ) from exc
            return (
                XGBRegressor(
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=self.seed,
                ),
                "XGBRegressor",
            )
        return self._surrogate_spec, type(self._surrogate_spec).__name__

    def fit(
        self,
        data: DataAdapter,
        actual: ArrayLike,
        predicted: ArrayLike,
    ) -> ErrorModelFit:
        """Fit the surrogate, and report whether it learned anything.

        ``predicted`` comes from the assessed model — usually
        ``handler.predict(data).predicted``. The surrogate never calls the model, which is
        why this whole capability costs one prediction pass rather than thousands.
        """
        extractor = resolve_extractor(self._extractor_spec, data)
        matrix, names = extractor.extract(data)
        errors = per_sample_error(actual, predicted, kind=self.error_kind)

        if matrix.shape[0] != errors.shape[0]:
            raise ValueError(
                f"The extractor produced {matrix.shape[0]} row(s) for {errors.shape[0]} "
                f"error value(s). Features and errors must describe the same entities."
            )
        if matrix.shape[0] < 8:
            raise ValueError(
                f"An error model needs more than {matrix.shape[0]} sample(s) to say "
                f"anything. There is one row per entity, so a dataset of a few long "
                f"series gives a few rows however many timesteps each has."
            )

        # Drop columns that cannot inform anything: all-NaN, or constant.
        finite_columns = ~np.all(np.isnan(matrix), axis=0)
        with np.errstate(invalid="ignore"):
            varying = np.array(
                [
                    np.unique(matrix[np.isfinite(matrix[:, j]), j]).size > 1
                    for j in range(matrix.shape[1])
                ]
            )
        keep = finite_columns & varying
        if not keep.any():
            raise ValueError(
                "Every feature column is constant or entirely missing, so nothing can be "
                "learned. Check the extractor against the data's shape."
            )
        matrix = matrix[:, keep]
        names = [name for name, k in zip(names, keep, strict=True) if k]

        if np.unique(errors).size == 1:
            raise ValueError(
                f"Every sample has the same error ({errors[0]}), so there is no variation "
                f"to explain. A perfect model, or one that is wrong everywhere, has no "
                f"error structure to attribute."
            )

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(matrix.shape[0])
        n_test = max(2, int(round(self.test_fraction * matrix.shape[0])))
        test_index = np.sort(order[:n_test])
        train_index = np.sort(order[n_test:])
        if train_index.size < 4:
            raise ValueError(
                f"Only {train_index.size} training row(s) after holding out {n_test} for "
                f"testing. Lower test_fraction, or accept that this dataset is too small "
                f"for an error model."
            )

        surrogate, surrogate_name = self._build_surrogate()
        surrogate.fit(matrix[train_index], errors[train_index])

        from sklearn.metrics import r2_score, roc_auc_score

        train_prediction = surrogate.predict(matrix[train_index])
        test_prediction = surrogate.predict(matrix[test_index])
        train_r2 = float(r2_score(errors[train_index], train_prediction))
        test_r2 = float(r2_score(errors[test_index], test_prediction))

        # A misclassification indicator is a ranking problem, not a regression one.
        binary_error = set(np.unique(errors).tolist()) <= {0.0, 1.0}
        train_auroc = test_auroc = None
        if binary_error:
            def safe_auroc(truth: NDArray, scores: NDArray) -> float | None:
                if np.unique(truth).size < 2:
                    return None
                return float(roc_auc_score(truth, scores))

            train_auroc = safe_auroc(errors[train_index], train_prediction)
            test_auroc = safe_auroc(errors[test_index], test_prediction)
            if test_auroc is None:
                raise ValueError(
                    "The held-out set contains only one outcome — every sample right, or "
                    "every one wrong — so the surrogate cannot be scored on it. Use a "
                    "larger test_fraction, or a dataset where the model is sometimes "
                    "right and sometimes wrong."
                )

        self._fitted = surrogate
        self._features = names
        self._matrix = matrix
        self._errors = errors
        self._train_index = train_index
        self._test_index = test_index
        self.fit_report = ErrorModelFit(
            test_r2=test_r2,
            train_r2=train_r2,
            n_train=int(train_index.size),
            n_test=int(test_index.size),
            n_features=len(names),
            error_kind=self.error_kind if self.error_kind != "auto" else (
                "misclassified" if set(np.unique(errors)) <= {0.0, 1.0} else "absolute"
            ),
            error_mean=float(errors.mean()),
            error_std=float(errors.std()),
            surrogate=surrogate_name,
            extractor=getattr(extractor, "name", type(extractor).__name__),
            test_auroc=test_auroc,
            train_auroc=train_auroc,
        )
        return self.fit_report

    # -- guards ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    @property
    def is_trustworthy(self) -> bool:
        return self.fit_report is not None and self.fit_report.is_trustworthy

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before asking for attributions.")

    def _require_trust(self, force: bool) -> None:
        self._require_fit()
        if force or self.is_trustworthy:
            return
        report = self.fit_report
        raise ValueError(
            f"The surrogate's {report.gate} is {report.gate_value:.3f}, below "
            f"{report.gate_threshold}: it is not predicting the error, so its feature "
            f"attributions are noise and would look exactly like a real finding. Either "
            f"improve it — more entities, a different extractor, a different error kind — "
            f"or pass force=True having understood that the output means nothing."
        )

    def role_of(self, index: int) -> Role:
        """Whether a sample was in the surrogate's training or held-out set."""
        self._require_fit()
        return "train" if index in set(self._train_index.tolist()) else "test"

    # -- attribution -------------------------------------------------------------

    def importance(
        self,
        *,
        method: str = "permutation",
        n_repeats: int = 10,
        on: Literal["test", "train", "all"] = "test",
        force: bool = False,
    ) -> FeatureAttribution:
        """Global attribution: which input characteristics go with a larger error.

        Parameters
        ----------
        method:
            ``"permutation"`` (the default, core dependency) measures how much the
            surrogate's R² falls when a column is shuffled. ``"shap"`` uses SHAP's tree
            explainer, which additionally gives a *direction* — but requires the
            ``[explain]`` extra and a tree surrogate.
        on:
            Which rows to compute over. ``"test"`` by default: permutation importance on
            the training set measures what the surrogate *used*, which on a small dataset
            is largely what it memorised.
        """
        self._require_trust(force)
        rows = {
            "test": self._test_index,
            "train": self._train_index,
            "all": np.arange(self._matrix.shape[0]),
        }[on]

        if method == "permutation":
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                self._fitted,
                self._matrix[rows],
                self._errors[rows],
                n_repeats=n_repeats,
                random_state=self.seed,
            )
            return FeatureAttribution(
                features=tuple(self._features),
                importance=result.importances_mean,
                signed=None,
                method="permutation",
                scope="global",
                metadata={
                    "n_repeats": n_repeats,
                    "on": on,
                    "std": result.importances_std,
                    "test_r2": self.fit_report.test_r2,
                },
            )

        if method == "shap":
            values = self._shap_values(self._matrix[rows])
            return FeatureAttribution(
                features=tuple(self._features),
                importance=np.abs(values).mean(axis=0),
                signed=values.mean(axis=0),
                method="shap",
                scope="global",
                metadata={"on": on, "test_r2": self.fit_report.test_r2},
            )

        raise ValueError(f"method must be 'permutation' or 'shap'; got {method!r}.")

    def _shap_values(self, matrix: NDArray) -> NDArray:
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "method='shap' needs the [explain] extra: pip install "
                "'themis-assess[explain]'. Permutation importance is a core dependency "
                "and answers the global question; only the per-sample direction needs "
                "SHAP."
            ) from exc

        explainer = shap.TreeExplainer(self._fitted)
        values = explainer.shap_values(matrix, check_additivity=False)
        return np.asarray(values, dtype=float)

    def explain(
        self,
        index: int,
        *,
        allow_in_sample: bool = False,
        force: bool = False,
    ) -> FeatureAttribution:
        """Local attribution: what drove *this* sample's error. Needs SHAP.

        Parameters
        ----------
        allow_in_sample:
            Explaining a sample the surrogate trained on is optimistic — it has already
            seen this row's features and its error. Repo B warned by ``print``; here it
            raises unless asked, because a warning in a library reaches nobody and the
            output is indistinguishable either way.
        """
        self._require_trust(force)
        if not 0 <= index < self._matrix.shape[0]:
            raise IndexError(
                f"Sample {index} is out of range for {self._matrix.shape[0]} sample(s)."
            )
        role = self.role_of(index)
        if role == "train" and not allow_in_sample:
            raise ValueError(
                f"Sample {index} is in the surrogate's training set, so its local "
                f"attribution is optimistic: the surrogate has already seen this row's "
                f"features and its error. Explain a held-out sample instead — "
                f"{sorted(self._test_index.tolist())[:5]}... are held out — or pass "
                f"allow_in_sample=True."
            )

        row = self._matrix[index : index + 1]
        values = self._shap_values(row)[0]
        explainer_baseline = None
        try:  # pragma: no cover - shap's attribute name varies by version
            import shap

            explainer_baseline = float(
                np.ravel(shap.TreeExplainer(self._fitted).expected_value)[0]
            )
        except Exception:  # noqa: BLE001 - the baseline is a nicety, not the result
            explainer_baseline = None

        return FeatureAttribution(
            features=tuple(self._features),
            importance=np.abs(values),
            signed=values,
            method="shap",
            scope="local",
            sample_role=role,
            baseline=explainer_baseline,
            predicted_error=float(self._fitted.predict(row)[0]),
            actual_error=float(self._errors[index]),
            metadata={"index": index, "test_r2": self.fit_report.test_r2},
        )

    def feature_error_scatter(self, feature: str) -> dict[str, NDArray]:
        """One feature's values against the observed error. Repo B's ``FeatureErrorScatter``.

        The plainest possible check on an attribution, and worth doing: a feature the
        surrogate ranks highly should show *something* here. Where it does not, the
        surrogate is using the feature in an interaction, and the attribution needs
        reading with that in mind rather than as a marginal effect.
        """
        self._require_fit()
        try:
            column = self._features.index(feature)
        except ValueError:
            raise KeyError(
                f"No feature {feature!r}. Available: {self._features[:10]}"
                f"{'...' if len(self._features) > 10 else ''}."
            ) from None
        return {
            "feature_value": self._matrix[:, column],
            "error": self._errors,
            "is_test": np.isin(np.arange(self._matrix.shape[0]), self._test_index),
        }
