"""Model handlers: the only part of the library that knows what a model is.

Everything downstream sees a :class:`~themis_assess.handlers.base.Prediction` and a
declared :class:`~themis_assess.core.types.AccessLevel`.
"""

from themis_assess.handlers.base import BaseHandler, ModelHandler, Prediction
from themis_assess.handlers.baselines import (
    MajorityClassHandler,
    MovingAverageHandler,
    NaiveLastValueHandler,
    PriorProbabilityHandler,
)
from themis_assess.handlers.forecasting import ForecastingConfig, ForecastingHandler
from themis_assess.handlers.sklearn import CallableHandler, SklearnHandler

__all__ = [
    # contract
    "ModelHandler",
    "Prediction",
    "BaseHandler",
    # concrete
    "SklearnHandler",
    "CallableHandler",
    "ForecastingHandler",
    "ForecastingConfig",
    # baselines
    "MajorityClassHandler",
    "PriorProbabilityHandler",
    "NaiveLastValueHandler",
    "MovingAverageHandler",
]
