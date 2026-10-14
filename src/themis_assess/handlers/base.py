"""The model handler contract: what the library needs from a model, and nothing more.

A handler is the only thing in the library that knows what kind of object the model is.
Everything else — perturbations, metrics, the runner, the report — sees a
:class:`Prediction` and a declared :class:`~themis_assess.core.types.AccessLevel`.

What the four repos had, and why this differs
---------------------------------------------

All four carried a ``BaseModelHandler`` with ``load_trained_model`` and
``get_predictions``. Four things went wrong with it, and each is fixed here rather than
carried forward.

*The abstract base did not run.* Repo A's ``ForecastingModelHandler`` and repo C's
``PlovdivModelHandler`` both call ``super(BaseModelHandler, self).__init__()``, which
resolves past the base to ``object`` and skips the base's own ``__init__`` entirely. The
attributes survive only because both then reassign them by hand. Nothing here uses
``super()`` with an explicit class, and the base does no attribute assignment a subclass
must remember to repeat.

*Loading was conflated with predicting.* ``load_trained_model`` is an abstract method, so
every handler had to be a loader too, and a model already in memory — the normal case in a
notebook, and the only case in a test — had no way in. Loading is not part of this
contract. A handler is constructed around a model object; where that object came from is
the use case's business.

*What the model can produce was undiscoverable.* Repo C's ``get_predictions`` returns
argmax labels and stashes the probabilities on ``self.y_pred_raw`` as a side effect. Repo
D's ``ModelHandlerRoBERTa`` has ``predict``, ``predict_probability``, ``get_embeddings``
and ``predict_mc_dropout``, and does not inherit the base at all. A caller could not ask
either one what it was capable of. Here a handler declares an
:class:`~themis_assess.core.types.AccessLevel` and returns a :class:`Prediction` carrying
whatever it has, so the runner can say up front which metrics it will be able to compute.

*Predictions were returned bare.* A metric needing probabilities and a metric needing
labels had to be called differently, and calibration metrics could not be computed at all
without reaching into handler internals. :class:`Prediction` bundles them.

The contract is a :class:`typing.Protocol`, so a use case's existing class satisfies it by
having the right methods — it does not have to inherit anything. :class:`BaseHandler` is
offered for convenience, not as a requirement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import AccessLevel, TaskType

__all__ = [
    "Prediction",
    "ModelHandler",
    "BaseHandler",
]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Everything one model produced for one dataset.

    The fields are optional because handlers differ in what they can produce, and the
    difference is exactly what :meth:`available` reports to the runner. A field left
    ``None`` is not an error; it is a metric that will be skipped, with a reason.

    Parameters
    ----------
    predicted:
        Point predictions — class labels for classification, values for forecasting.
        The one field every handler must produce.
    probabilities:
        ``(n_samples, n_classes)`` class probabilities, or ``(n_samples,)`` for the
        positive class of a binary problem. Classification only.
    logits:
        Pre-softmax scores, where the model exposes them. Needed by
        :func:`~themis_assess.metrics.robustness.calibration.fit_temperature`, which
        cannot recover them from probabilities.
    lower, upper:
        Prediction interval bounds. Forecasting only.
    samples:
        ``(n_samples, n_members)`` draws from the predictive distribution, for CRPS —
        or ``(n_samples, n_members, n_dimensions)`` for the multivariate energy score.
    mean, std:
        A Gaussian predictive distribution, where the model reports one directly.
    embeddings:
        ``(n_samples, n_dimensions)`` representations, for distances computed in feature
        space rather than input space. Repo D's ``get_embeddings``.
    metadata:
        Anything the report should be able to cite about how these were produced —
        the number of Monte-Carlo dropout passes, the nominal interval level, the device.
    """

    predicted: NDArray
    probabilities: NDArray | None = None
    logits: NDArray | None = None
    lower: NDArray | None = None
    upper: NDArray | None = None
    samples: NDArray | None = None
    mean: NDArray | None = None
    std: NDArray | None = None
    embeddings: NDArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted)
        if predicted.size == 0:
            raise ValueError("A Prediction must carry at least one predicted value.")
        object.__setattr__(self, "predicted", predicted)

        n = predicted.shape[0]
        for name in ("probabilities", "logits", "lower", "upper", "samples", "mean", "std"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape[0] != n:
                raise ValueError(
                    f"Prediction.{name} has {array.shape[0]} row(s) but there are "
                    f"{n} prediction(s). Every field must describe the same samples, "
                    f"in the same order."
                )
            object.__setattr__(self, name, array)

        if (self.lower is None) != (self.upper is None):
            raise ValueError(
                "Prediction intervals need both bounds; got "
                f"lower={'set' if self.lower is not None else 'None'} and "
                f"upper={'set' if self.upper is not None else 'None'}."
            )
        if self.lower is not None and np.any(self.upper < self.lower):
            raise ValueError("Every upper bound must be at or above its lower bound.")

    def __len__(self) -> int:
        return int(self.predicted.shape[0])

    @property
    def n_samples(self) -> int:
        return len(self)

    def available(self) -> dict[str, NDArray]:
        """The quantities this prediction can supply, as the catalog's bag of inputs.

        Feeds straight into
        :meth:`~themis_assess.metrics.accuracy.catalog.MetricCatalog.computable`, which
        is what lets the runner report skipped metrics before it starts rather than
        after it fails.
        """
        bag = {
            "predicted": self.predicted,
            "probabilities": self.probabilities,
            "logits": self.logits,
            "lower": self.lower,
            "upper": self.upper,
            "samples": self.samples,
            "mean": self.mean,
            "std": self.std,
            "embeddings": self.embeddings,
        }
        return {k: v for k, v in bag.items() if v is not None}

    def implied_access(self) -> AccessLevel:
        """The access level these outputs actually demonstrate.

        Checked against the handler's declaration: a handler that claims
        ``PROBABILISTIC`` and returns no probabilities has a bug, and the runner is
        where it should surface rather than three thousand model calls later.
        """
        if self.embeddings is not None:
            return AccessLevel.EMBEDDING
        distributional = (
            self.probabilities,
            self.logits,
            self.lower,
            self.samples,
            self.std,
        )
        if any(x is not None for x in distributional):
            return AccessLevel.PROBABILISTIC
        return AccessLevel.BLACK_BOX

    def take(self, indices: ArrayLike) -> Prediction:
        """The same prediction restricted to a subset of samples.

        Used for subgroup evaluation and for split protocols, where a metric is computed
        over a mask rather than the whole set. Doing it here rather than re-predicting
        matters: re-running the model on a subgroup would be correct but wasteful, and on
        a *perturbed* subgroup it would draw different noise.
        """
        index = np.asarray(indices)
        if index.dtype == bool:
            if index.shape[0] != len(self):
                raise ValueError(
                    f"A boolean mask must cover all {len(self)} prediction(s); got "
                    f"{index.shape[0]}."
                )
            index = np.flatnonzero(index)
        if index.size == 0:
            raise ValueError("Cannot take an empty subset of predictions.")

        def sub(value: NDArray | None) -> NDArray | None:
            return None if value is None else value[index]

        return Prediction(
            predicted=self.predicted[index],
            probabilities=sub(self.probabilities),
            logits=sub(self.logits),
            lower=sub(self.lower),
            upper=sub(self.upper),
            samples=sub(self.samples),
            mean=sub(self.mean),
            std=sub(self.std),
            embeddings=sub(self.embeddings),
            metadata=dict(self.metadata),
        )


@runtime_checkable
class ModelHandler(Protocol):
    """What the library requires of a model.

    Three members. ``name`` and ``task`` for the report, ``access`` for the capability
    check, and :meth:`predict` to do the work.
    """

    name: str
    task: TaskType
    access: AccessLevel

    def predict(self, data: DataAdapter) -> Prediction:
        """Predict for every entity in ``data``.

        The one method a handler must implement. It is called once per clean evaluation
        and once per perturbed variant, so it should be as cheap as the model allows —
        batching, not a Python loop over samples, wherever the model supports it.
        """
        ...


class BaseHandler:
    """A convenience base for handlers. Implementing :class:`ModelHandler` is enough.

    Provides the three declared attributes, a readable ``repr``, and
    :meth:`check_prediction`, which catches the mismatch between what a handler claims
    and what it returns.
    """

    def __init__(
        self,
        model: Any,
        *,
        name: str | None = None,
        task: TaskType = TaskType.CLASSIFICATION,
        access: AccessLevel = AccessLevel.BLACK_BOX,
        **metadata: Any,
    ) -> None:
        self.model = model
        self.name = name or type(model).__name__
        self.task = task
        self.access = access
        self.metadata = metadata

    def predict(self, data: DataAdapter) -> Prediction:  # pragma: no cover - contract
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict(data) -> Prediction."
        )

    def check_prediction(self, prediction: Prediction, data: DataAdapter) -> Prediction:
        """Verify a prediction against the handler's declaration and the input size.

        Call this at the end of :meth:`predict`. Two failures it catches:

        * a prediction for a different number of entities than were passed in, which
          otherwise surfaces as a shape error inside a metric, far from its cause;
        * a handler declaring more access than it delivers — claiming
          ``PROBABILISTIC`` and returning labels only. The runner promises the user a
          list of computable metrics based on the declaration, so an over-claim makes
          that promise a lie.

        Under-delivering on one call is not always a bug — a forecasting handler may
        return intervals for some horizons and not others — so this raises only when the
        declaration exceeds what any field could support.
        """
        expected = data.n_entities
        if len(prediction) != expected:
            raise ValueError(
                f"{self.name} returned {len(prediction)} prediction(s) for "
                f"{expected} entit(y/ies). Predictions must align with the input, in "
                f"order, so that subgroup masks and perturbation comparisons line up."
            )
        implied = prediction.implied_access()
        if self.access > implied:
            raise ValueError(
                f"{self.name} declares access={self.access.name} but returned only "
                f"{implied.name} outputs. Either return the richer outputs or declare "
                f"access={implied.name} — the runner uses the declaration to tell the "
                f"user which metrics it can compute."
            )
        return prediction

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"{type(self).__name__}(name={self.name!r}, task={self.task.value}, "
            f"access={self.access.name})"
        )
