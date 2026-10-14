"""Structure-aware time-series perturbation via STL decomposition. Survey §4.2.

The survey's recommended way to keep a time-series perturbation faithful to the signal's
own structure, and the last of its four time-series families:

    decomposition-based: STL (seasonal-trend decomposition based on locally weighted
    regression) into trend/seasonal/residual, then perturb selected components only
    (e.g., noise into residual preserving trend and seasonality; alter seasonal
    amplitude only).

Why it matters. Adding broadband noise to a seasonal signal tests two things at once:
whether the model tolerates noise, *and* whether it tolerates having its seasonal
structure blurred. Perturbing one component at a time separates them, and the answers
usually differ — a forecaster that shrugs off residual noise can be wrecked by a shift
in seasonal amplitude, because the seasonal term is most of what it is extrapolating.

Needs ``statsmodels``, so it ships under the ``[timeseries]`` extra and imports lazily.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

import numpy as np

from themis_assess.core.errors import require_extra
from themis_assess.core.types import Modality, PerturbationSpec
from themis_assess.perturbations.base import Perturbation
from themis_assess.perturbations.tabular import feature_reference_scale

__all__ = ["Component", "STLComponentPerturbation", "decompose"]

_SEQUENCE_ONLY = frozenset({Modality.SEQUENCE})

#: Shortest series STL can meaningfully decompose, given it needs at least two periods.
MIN_LENGTH_FACTOR = 2


class Component(str, Enum):
    """Which part of the decomposition to disturb.

    ``RESIDUAL``  add noise to the residual, preserving trend and seasonality exactly
    ``SEASONAL``  scale the seasonal amplitude, preserving trend and residual
    ``TREND``     add a proportional offset to the trend, preserving the rest
    """

    RESIDUAL = "residual"
    SEASONAL = "seasonal"
    TREND = "trend"


def decompose(
    series: np.ndarray, period: int, *, robust: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 1-D series into ``(trend, seasonal, residual)`` with STL.

    Parameters
    ----------
    period:
        Number of timesteps in one seasonal cycle. There is no way to infer this from
        an unlabelled array — 24 for hourly data with a daily cycle, 7 for daily data
        with a weekly one — so it must be supplied.
    robust:
        Use STL's robust weighting, which down-weights outliers when fitting. On by
        default: real series have spikes, and a trend estimate dragged around by them
        would make the "structure-preserving" claim false.

    Returns
    -------
    tuple
        Three arrays of the same length as ``series``, summing back to it.
    """
    seasonal_module = require_extra("statsmodels.tsa.seasonal", "timeseries")
    if period < 2:
        raise ValueError(f"period must be at least 2; got {period}.")
    if series.shape[0] < MIN_LENGTH_FACTOR * period:
        raise ValueError(
            f"STL needs at least {MIN_LENGTH_FACTOR} full periods: period={period} "
            f"requires {MIN_LENGTH_FACTOR * period} timesteps, got {series.shape[0]}."
        )
    result = seasonal_module.STL(series, period=period, robust=robust).fit()
    return (
        np.asarray(result.trend, dtype=float),
        np.asarray(result.seasonal, dtype=float),
        np.asarray(result.resid, dtype=float),
    )


class STLComponentPerturbation(Perturbation):
    """Disturb one component of an STL decomposition, leaving the others exact.

    Parameters
    ----------
    period:
        Seasonal period, in timesteps. Required — see :func:`decompose`.
    component:
        Which component to disturb. What severity *means* depends on it, and each is
        reported in :meth:`metadata`:

        * ``RESIDUAL`` — Gaussian noise of ``severity x std(residual)`` added to the
          residual. Trend and seasonality survive bit-for-bit. This is the survey's
          first example, and the mildest structure-preserving perturbation available.
        * ``SEASONAL`` — the seasonal component is scaled by ``1 +- severity``. The
          cycle keeps its shape and phase and changes its depth. The survey's second
          example.
        * ``TREND`` — a constant proportional to ``severity x std(trend)`` is added to
          the trend. Level shift without touching the cycle.
    robust:
        Passed to STL.

    A series too short for its period is left unchanged rather than raising, and the
    count is reported in :meth:`metadata` — a dataset of ragged entities will usually
    have a few that are too short, and failing the whole run for them would be worse
    than reporting them.
    """

    name = "stl_component"
    modalities: ClassVar[frozenset[Modality]] = _SEQUENCE_ONLY
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = False
    severity_scale = (0.10, 0.25, 0.50, 1.00, 2.00)

    #: Severity ladders per component. They differ because severity means a different
    #: quantity in each, and only one of them has a hard ceiling.
    #:
    #: ``SEASONAL`` is capped below 1.0 for a substantive reason: the seasonal component
    #: is scaled by ``1 +- severity``, so a severity of exactly 1.0 can *annihilate* the
    #: cycle and anything above it *inverts* its sign. Neither is an amplitude change —
    #: an inverted cycle is a different signal, not a louder or quieter one, and the
    #: perturbation would stop being label-preserving in any meaningful sense. 0.75 is
    #: the strongest change that keeps the cycle recognisably itself.
    _SEVERITY_SCALES: ClassVar[dict[Component, tuple[float, ...]]] = {
        Component.RESIDUAL: (0.10, 0.25, 0.50, 1.00, 2.00),
        Component.SEASONAL: (0.10, 0.20, 0.35, 0.50, 0.75),
        Component.TREND: (0.10, 0.25, 0.50, 1.00, 2.00),
    }

    def __init__(
        self,
        *,
        period: int,
        component: Component | str = Component.RESIDUAL,
        robust: bool = True,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be at least 2; got {period}.")
        self.period = int(period)
        self.component = Component(component)
        self.robust = bool(robust)
        # Instance attribute shadows the class default, so severity_factor picks up the
        # ladder that matches this component.
        self.severity_scale = self._SEVERITY_SCALES[self.component]
        self._skipped_too_short = 0

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        factor = self.severity_factor(spec)
        reference = feature_reference_scale(data, features)
        mask = data.valid_mask()
        minimum = MIN_LENGTH_FACTOR * self.period
        skipped = 0

        out = []
        for _ in range(spec.n_variants):
            values = data.values().astype(float)
            for feature in features:
                j = data.feature_index(feature)
                for e in range(data.n_entities):
                    positions = np.flatnonzero(mask[e])
                    stretch = values[e, positions, j]
                    if stretch.shape[0] < minimum or not np.isfinite(stretch).all():
                        skipped += 1
                        continue
                    values[e, positions, j] = self._disturb(
                        stretch, factor, reference[feature], rng
                    )
            out.append(data.with_values(values))
        self._skipped_too_short = skipped
        return out

    def _disturb(
        self,
        series: np.ndarray,
        factor: float,
        reference: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        trend, seasonal, residual = decompose(series, self.period, robust=self.robust)

        if self.component is Component.RESIDUAL:
            scale = float(np.std(residual))
            if scale <= 0.0:
                scale = reference
            residual = residual + rng.normal(0.0, factor * scale, residual.shape)
        elif self.component is Component.SEASONAL:
            direction = 1.0 if rng.random() < 0.5 else -1.0
            seasonal = seasonal * (1.0 + direction * factor)
        else:  # TREND
            scale = float(np.std(trend))
            if scale <= 0.0:
                scale = reference
            direction = 1.0 if rng.random() < 0.5 else -1.0
            trend = trend + direction * factor * scale

        return trend + seasonal + residual

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["component"] = self.component.value
        meta["period"] = self.period
        meta["robust_stl"] = self.robust
        meta["preserves"] = sorted(c.value for c in Component if c is not self.component)
        meta["entities_skipped_too_short"] = self._skipped_too_short
        if spec is not None:
            factor = self.severity_factor(spec)
            if self.component is Component.RESIDUAL:
                meta["noise_in_residual_std"] = factor
            elif self.component is Component.SEASONAL:
                meta["seasonal_amplitude_change"] = factor
            else:
                meta["trend_offset_in_trend_std"] = factor
        return meta

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"STLComponentPerturbation(component={self.component.value!r}, "
            f"period={self.period})"
        )
