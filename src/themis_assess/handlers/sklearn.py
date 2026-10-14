"""Handlers for models with a scikit-learn-style interface.

Covers repo C's LightGBM classifier and anything else exposing ``predict`` and optionally
``predict_proba`` — which is most tabular models, including the sklearn estimators, the
gradient-boosting libraries, and any pipeline wrapping them.

Repo C's ``PlovdivModelHandler`` is the model this replaces. Its ``get_predictions``
returned ``argmax`` labels and left the probabilities on ``self.y_pred_raw``, so a
calibration metric could only be computed by reaching into the handler afterwards and
hoping the attribute belonged to the call you thought it did. Here both come back in one
:class:`~themis_assess.handlers.base.Prediction`, and the probabilities are returned
because the handler *declared* it could produce them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import AccessLevel, TaskType
from themis_assess.handlers.base import BaseHandler, Prediction

__all__ = ["SklearnHandler", "CallableHandler"]


def _flatten(data: DataAdapter) -> np.ndarray:
    """The adapter's canonical ``(n_entities, n_timesteps, n_features)`` as a 2-D matrix.

    Tabular data has a singleton time axis, so this is a squeeze. Sequential data is
    flattened to ``(n_entities, n_timesteps * n_features)``, which is what a tabular
    model expects of a windowed input and what repo C's notebooks did by hand.
    """
    values = data.values()
    if values.ndim != 3:  # pragma: no cover - the adapters guarantee this
        raise ValueError(f"Expected a 3-D adapter array; got shape {values.shape}.")
    n_entities = values.shape[0]
    return values.reshape(n_entities, -1)


def _is_probability(values: np.ndarray) -> bool:
    """Whether an array of scores is actually a probability distribution.

    ``decision_function`` returns signed margins — one column for a binary problem, one
    per class beyond it — and they are not probabilities. Passing them off as such would
    make every calibration metric a measurement of whatever squashing function did the
    conversion rather than of the model, so a non-probability is dropped and the
    calibration metrics are reported as skipped instead.

    The 1-D case is the one that matters: a binary ``decision_function`` looks exactly
    like the ``P(class 1)`` vector handlers legitimately return, and only the range
    distinguishes them.
    """
    if not np.isfinite(values).all():
        return False
    if (values < -1e-9).any() or (values > 1 + 1e-9).any():
        return False
    if values.ndim == 2:
        return bool(np.allclose(values.sum(axis=1), 1.0, atol=1e-4))
    return values.ndim == 1


class SklearnHandler(BaseHandler):
    """Wrap an estimator exposing ``predict`` and optionally ``predict_proba``.

    Parameters
    ----------
    model:
        The fitted estimator.
    task:
        Classification by default. A regressor used for forecasting should say so, since
        it decides which metrics the runner offers.
    access:
        Left to be inferred: ``PROBABILISTIC`` when the model has ``predict_proba``,
        ``BLACK_BOX`` otherwise. Pass it explicitly to *lower* the declaration — for
        instance to evaluate a probabilistic model under black-box conditions, which is
        a legitimate thing to want and something none of the four repos could express.
    probability_source:
        Name of the method producing probabilities. ``predict_proba`` by default;
        ``decision_function`` and custom names are accepted for models that spell it
        differently.

    Notes
    -----
    ``predict`` is called once per perturbed variant, so an estimator whose ``predict``
    is slow makes a severity sweep slow in proportion. The runner reports the number of
    model calls it is about to make for exactly this reason.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str | None = None,
        task: TaskType = TaskType.CLASSIFICATION,
        access: AccessLevel | None = None,
        probability_source: str = "predict_proba",
        **metadata: Any,
    ) -> None:
        if not hasattr(model, "predict"):
            raise TypeError(
                f"{type(model).__name__} has no predict method. Use CallableHandler for "
                f"a plain function, or write a handler implementing the ModelHandler "
                f"protocol."
            )
        self.probability_source = probability_source
        can_do_probabilities = hasattr(model, probability_source)
        inferred = (
            AccessLevel.PROBABILISTIC if can_do_probabilities else AccessLevel.BLACK_BOX
        )
        if access is not None and access > inferred:
            raise ValueError(
                f"{type(model).__name__} has no {probability_source!r}, so it cannot "
                f"serve access={access.name}. Its outputs support {inferred.name}."
            )
        super().__init__(
            model,
            name=name,
            task=task,
            access=inferred if access is None else access,
            **metadata,
        )

    def predict(self, data: DataAdapter) -> Prediction:
        matrix = _flatten(data)
        predicted = np.asarray(self.model.predict(matrix))

        probabilities = None
        if self.access.satisfies(AccessLevel.PROBABILISTIC):
            raw = np.asarray(getattr(self.model, self.probability_source)(matrix))
            if not _is_probability(raw):
                raise ValueError(
                    f"{self.name}.{self.probability_source}() returned values that are "
                    f"not a probability distribution (range "
                    f"[{raw.min():.3g}, {raw.max():.3g}]"
                    + (
                        f", row sums [{raw.sum(axis=1).min():.3g}, "
                        f"{raw.sum(axis=1).max():.3g}]"
                        if raw.ndim == 2
                        else ""
                    )
                    + f"). {self.name} declares access={self.access.name}, which "
                    f"promises probabilities. A decision_function returns signed "
                    f"margins, and squashing them into probabilities here would make "
                    f"every calibration metric a measurement of that squashing rather "
                    f"than of the model. Calibrate the model — sklearn's "
                    f"CalibratedClassifierCV does it properly — or declare "
                    f"access=BLACK_BOX."
                )
            probabilities = raw

        return self.check_prediction(
            Prediction(
                predicted=predicted,
                probabilities=probabilities,
                metadata={"model": self.name, "n_features_in": matrix.shape[1]},
            ),
            data,
        )


class CallableHandler(BaseHandler):
    """Wrap a plain function ``predict(X) -> array``, with optional extras.

    The escape hatch, and the handler most tests use. It is also what a use case reaches
    for when the model is behind an API, in another process, or otherwise not an object
    with the expected methods — the situation none of the four repos' handlers could
    accommodate, since all of them assumed a loadable file on disk.

    Parameters
    ----------
    predict_fn:
        ``(n_entities, n_timesteps * n_features) -> (n_entities, ...)``.
    probability_fn, interval_fn, sample_fn:
        Optional. ``interval_fn`` returns ``(lower, upper)``; ``sample_fn`` returns
        ``(n_entities, n_members)``. Supplying any of them raises the declared access to
        ``PROBABILISTIC``.
    """

    def __init__(
        self,
        predict_fn: Any,
        *,
        probability_fn: Any = None,
        interval_fn: Any = None,
        sample_fn: Any = None,
        name: str = "callable",
        task: TaskType = TaskType.CLASSIFICATION,
        access: AccessLevel | None = None,
        **metadata: Any,
    ) -> None:
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable.")
        self.predict_fn = predict_fn
        self.probability_fn = probability_fn
        self.interval_fn = interval_fn
        self.sample_fn = sample_fn
        richer = any(f is not None for f in (probability_fn, interval_fn, sample_fn))
        inferred = AccessLevel.PROBABILISTIC if richer else AccessLevel.BLACK_BOX
        super().__init__(
            predict_fn,
            name=name,
            task=task,
            access=inferred if access is None else access,
            **metadata,
        )

    def predict(self, data: DataAdapter) -> Prediction:
        matrix = _flatten(data)
        lower = upper = None
        if self.interval_fn is not None:
            lower, upper = self.interval_fn(matrix)
        return self.check_prediction(
            Prediction(
                predicted=np.asarray(self.predict_fn(matrix)),
                probabilities=(
                    None if self.probability_fn is None
                    else np.asarray(self.probability_fn(matrix))
                ),
                lower=None if lower is None else np.asarray(lower),
                upper=None if upper is None else np.asarray(upper),
                samples=(
                    None if self.sample_fn is None else np.asarray(self.sample_fn(matrix))
                ),
                metadata={"model": self.name},
            ),
            data,
        )
