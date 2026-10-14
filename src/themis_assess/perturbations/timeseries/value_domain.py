"""Value-domain time-series perturbations. Survey §4.2.

The survey organises time-series perturbations by the domain in which they act, and
Figure 3 names six canonical sensor-level distortions, taken from a deep-learning
forecasting benchmark for cyber-physical systems: additive noise, amplitude scaling,
sensor offset or drift, exponential damping, a flat-sensor episode, and outlier spikes.

Additive noise, offset and scaling are in :mod:`themis_assess.perturbations.tabular`,
since they apply equally to tabular data. The four that only make sense along a time
axis are here, plus two the survey lists separately: contiguous missing segments and
alpha-stable impulsive noise.

Every perturbation here is sequence-only and declares so, so the registry will not offer
it for a tabular dataset — a drift over a single timestep is not a milder drift, it is a
meaningless one.

On noise distribution, the survey is specific:

    Most protocols use Gaussian additive noise, although this may poorly represent some
    real sensor disturbances... real measurements may also contain impulsive noise. Such
    noise is often modeled using alpha-stable distributions.

which is why :class:`AlphaStableNoise` exists alongside the Gaussian one.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import Modality, PerturbationSpec
from themis_assess.perturbations.base import Perturbation
from themis_assess.perturbations.tabular import feature_reference_scale

__all__ = [
    "SensorDrift",
    "ExponentialDamping",
    "FlatSensor",
    "OutlierSpikes",
    "MissingSegment",
    "AlphaStableNoise",
    "sample_alpha_stable",
]

_SEQUENCE_ONLY = frozenset({Modality.SEQUENCE})


class _SequencePerturbation(Perturbation):
    """Shared machinery for perturbations that act along the time axis.

    Unlike the tabular base, these need each entity's *own* valid stretch: a drift or a
    damping envelope has to start where the entity's data starts, not where the padded
    array does.
    """

    modalities: ClassVar[frozenset[Modality]] = _SEQUENCE_ONLY
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = False

    def __init__(self, *, relative: bool = True) -> None:
        self.relative = bool(relative)

    def _reference(self, data: DataAdapter, features: list[str]) -> dict[str, float]:
        if self.relative:
            return feature_reference_scale(data, features)
        return dict.fromkeys(features, 1.0)

    def _apply_entity(
        self,
        series: np.ndarray,
        factor: float,
        reference: float,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturb one entity's contiguous 1-D stretch of one feature.

        ``factor`` is the severity multiplier and ``reference`` the feature's own scale
        (1.0 for dimensionless perturbations). They are passed separately rather than
        pre-multiplied because a perturbation with two severity dimensions — spike rate
        and spike height, say — needs the unscaled factor to index its second grid.
        """
        raise NotImplementedError

    def perturb(
        self,
        data: DataAdapter,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> list[DataAdapter]:
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        reference = self._reference(data, features)
        factor = self.severity_factor(spec)
        mask = data.valid_mask()

        out: list[DataAdapter] = []
        for _ in range(spec.n_variants):
            values = data.values().astype(float)
            for feature in features:
                j = data.feature_index(feature)
                for e in range(data.n_entities):
                    positions = np.flatnonzero(mask[e])
                    if positions.size == 0:
                        continue
                    stretch = values[e, positions, j]
                    values[e, positions, j] = self._apply_entity(
                        stretch, factor, reference[feature], spec, rng
                    )
            out.append(data.with_values(values))
        return out

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["relative_to_feature_std"] = self.relative
        return meta


class SensorDrift(_SequencePerturbation):
    """A steadily growing offset. Survey §4.2, Figure 3(c).

    The classic slow failure: a sensor that reads correctly at first and wanders further
    from the truth as time passes. Modelled as a linear ramp from zero to
    ``severity x reference`` across each entity's own valid stretch.

    Distinct from :class:`~themis_assess.perturbations.tabular.GaussianNoise` with
    ``CONSTANT`` time scope, which is a *fixed* offset. A constant offset is a
    calibration error; a drift is a degrading instrument, and a forecasting model that
    survives one may well fail the other — the drift keeps changing the relationship
    between recent and older context, which is exactly what a windowed model relies on.

    The ramp direction is drawn per entity, so a population does not all drift the same
    way.
    """

    name = "sensor_drift"
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)

    def _apply_entity(self, series, factor, reference, spec, rng):
        magnitude = factor * reference
        n = series.shape[0]
        direction = 1.0 if rng.random() < 0.5 else -1.0
        ramp = np.linspace(0.0, magnitude, n) if n > 1 else np.array([magnitude])
        return series + direction * ramp


class ExponentialDamping(_SequencePerturbation):
    """The signal decays toward its own mean. Survey §4.2, Figure 3(d).

    A "dying signal": amplitude fading away while the time axis keeps running. Modelled
    as ``mean + (x - mean) * exp(-rate * t / n)``, so variation collapses toward the
    entity's own level rather than toward zero — decaying toward zero would also inject
    a large spurious offset for any feature that is not zero-centred, which would test
    two things at once.

    Severity is the total decay exponent: at severity 3 the final timestep retains
    ``exp(-1) ~ 37%`` of its deviation from the mean. Dimensionless, so no reference
    scale is needed.
    """

    name = "exponential_damping"
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)

    def __init__(self) -> None:
        super().__init__(relative=False)

    def _apply_entity(self, series, factor, reference, spec, rng):
        n = series.shape[0]
        if n == 0:
            return series
        level = float(np.nanmean(series))
        t = np.arange(n, dtype=float) / max(n - 1, 1)
        return level + (series - level) * np.exp(-factor * t)

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        meta.pop("relative_to_feature_std", None)
        if spec is not None:
            factor = self.severity_factor(spec)
            meta["decay_exponent"] = factor
            meta["retained_deviation_at_end"] = float(np.exp(-factor))
        return meta


class FlatSensor(_SequencePerturbation):
    """A contiguous window frozen at one value. Survey §4.2, Figure 3(e).

    A stuck sensor: the reading holds constant while the world moves on. Modelled by
    picking a window and filling it with the value immediately before it — a "last known
    good" hold, which is what a frozen instrument or a stale cache actually produces. A
    window starting at position zero holds the entity's first value instead.

    Severity is the fraction of the entity's length that freezes, with the window
    position drawn uniformly. This is arguably the most realistic of the family and one
    of the least often tested: the data stays perfectly plausible in range and
    distribution, and only its *dynamics* are wrong, so distribution-distance checks
    (survey §5.3) will not flag it at all.

    Dimensionless — no reference scale needed.
    """

    name = "flat_sensor"
    severity_scale = (0.05, 0.10, 0.25, 0.50, 0.80)

    def __init__(self) -> None:
        super().__init__(relative=False)

    def _apply_entity(self, series, factor, reference, spec, rng):
        n = series.shape[0]
        window = max(1, round(factor * n))
        if window >= n:
            return np.full_like(series, series[0])
        start = int(rng.integers(0, n - window + 1))
        held = series[start - 1] if start > 0 else series[0]
        out = series.copy()
        out[start : start + window] = held
        return out

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        meta.pop("relative_to_feature_std", None)
        if spec is not None:
            meta["frozen_fraction"] = self.severity_factor(spec)
        return meta


class OutlierSpikes(_SequencePerturbation):
    """Sparse, large, isolated deviations. Survey §4.2, Figure 3(f).

    Transmission glitches, quantisation faults, a loose connector. Two things are varied
    together as severity rises, because a spike perturbation has two dimensions and
    treating them separately would double the grid: the **rate** of affected timesteps
    and the **height** of each spike.

    Heights are drawn with a random sign so spikes do not bias the mean, and each spike
    affects a single timestep — that isolation is the point, and it is what distinguishes
    a spike from the drift and damping families.
    """

    name = "outlier_spikes"
    severity_scale = (0.01, 0.02, 0.05, 0.10, 0.20)
    #: Spike height, in reference units, at each severity level.
    spike_height: ClassVar[tuple[float, ...]] = (3.0, 4.0, 5.0, 6.0, 8.0)

    def _apply_entity(self, series, factor, reference, spec, rng):
        rate = factor
        height = self.spike_height[spec.severity - 1] * reference
        n = series.shape[0]
        hit = rng.random(n) < rate
        if not hit.any():  # guarantee at least one spike, else severity 1 is often a no-op
            hit[int(rng.integers(0, n))] = True
        signs = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        out = series.copy()
        out[hit] += signs[hit] * height
        return out

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["spike_rate"] = self.severity_scale[spec.severity - 1]
            meta["spike_height_in_std"] = self.spike_height[spec.severity - 1]
        return meta


class MissingSegment(Perturbation):
    """A contiguous block of missing values. Survey §4.2, missing data.

    Different from :class:`~themis_assess.perturbations.tabular.MissingValues`, which
    drops entries independently. A dropout of the same total size is far harder for a
    sequence model when it is contiguous — an outage, a disconnected logger, a gap
    between deployments — because interpolation across it has no nearby anchor.

    Applies to categorical features as well: a whole record can go missing, not only a
    measurement.

    Severity is the fraction of each entity's length that goes missing, with the gap
    position drawn uniformly.
    """

    name = "missing_segment"
    modalities = _SEQUENCE_ONLY
    severity_scale = (0.05, 0.10, 0.25, 0.50, 0.80)

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        fraction = self.severity_factor(spec)
        mask = data.valid_mask()

        out = []
        for _ in range(spec.n_variants):
            values = data.values().astype(float)
            for feature in features:
                j = data.feature_index(feature)
                for e in range(data.n_entities):
                    positions = np.flatnonzero(mask[e])
                    n = positions.size
                    if n == 0:
                        continue
                    gap = max(1, round(fraction * n))
                    if gap >= n:
                        values[e, positions, j] = np.nan
                        continue
                    start = int(rng.integers(0, n - gap + 1))
                    values[e, positions[start : start + gap], j] = np.nan
            out.append(data.with_values(values))
        return out

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["missing_fraction"] = self.severity_factor(spec)
            meta["contiguous"] = True
        return meta


def sample_alpha_stable(
    alpha: float,
    beta: float,
    scale: float,
    size: tuple[int, ...] | int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from a symmetric-or-skewed alpha-stable distribution.

    Uses the Chambers-Mallows-Stuck method, the standard exact sampler for this family.
    Implemented here rather than via ``scipy.stats.levy_stable`` because CMS is a dozen
    lines, exact, and fast, whereas the scipy path is markedly slower for large draws.

    Parameters
    ----------
    alpha:
        Tail index in ``(0, 2]``. ``alpha = 2`` is Gaussian; smaller values give heavier
        tails and more extreme excursions. The survey names tail heaviness, skewness,
        scale and location as the four parameters of the family.
    beta:
        Skewness in ``[-1, 1]``. ``0`` is symmetric.
    scale:
        Scale parameter. Note this is *not* a standard deviation: for ``alpha < 2`` the
        variance is infinite, which is the whole point of using this family.
    """
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must lie in (0, 2]; got {alpha}.")
    if not -1.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [-1, 1]; got {beta}.")

    u = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    w = rng.exponential(1.0, size=size)

    if abs(alpha - 1.0) < 1e-9:
        # alpha == 1 is a separate limit of the CMS formula, not a substitution into it.
        term = (np.pi / 2.0) + beta * u
        x = (2.0 / np.pi) * (
            term * np.tan(u)
            - beta * np.log((np.pi / 2.0) * w * np.cos(u) / term)
        )
    else:
        zeta = -beta * np.tan(np.pi * alpha / 2.0)
        xi = np.arctan(-zeta) / alpha
        num = np.sin(alpha * (u + xi))
        den = np.cos(u) ** (1.0 / alpha)
        tail = (np.cos(u - alpha * (u + xi)) / w) ** ((1.0 - alpha) / alpha)
        x = (num / den) * tail + zeta

    return scale * np.asarray(x, dtype=float)


class AlphaStableNoise(_SequencePerturbation):
    """Additive impulsive noise from an alpha-stable distribution. Survey §4.2.

    The survey's recommended alternative to Gaussian noise for real sensor disturbance:
    heavy-tailed, so most timesteps are barely touched while a few take large
    excursions. Gaussian noise is the ``alpha = 2`` special case, which makes this a
    strict generalisation of
    :class:`~themis_assess.perturbations.tabular.GaussianNoise`.

    Severity moves **both** the scale and the tail index — heavier tails at higher
    severity — because tail index is what makes this qualitatively different from
    Gaussian noise, and holding it fixed would reduce the perturbation to a rescaling.
    Severity 1 starts near-Gaussian (``alpha = 1.9``) and severity 5 is strongly
    impulsive (``alpha = 1.1``).

    Symmetric by default (``beta = 0``); override via ``params={"beta": ...}`` for a
    one-sided fault mode.
    """

    name = "alpha_stable_noise"
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)
    #: Tail index at each severity level. Lower is heavier-tailed; 2.0 is Gaussian.
    tail_index: ClassVar[tuple[float, ...]] = (1.9, 1.7, 1.5, 1.3, 1.1)

    def _apply_entity(self, series, factor, reference, spec, rng):
        alpha = float(spec.params.get("alpha", self.tail_index[spec.severity - 1]))
        beta = float(spec.params.get("beta", 0.0))
        noise = sample_alpha_stable(alpha, beta, factor * reference, series.shape, rng)
        return series + noise

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["alpha"] = float(
                spec.params.get("alpha", self.tail_index[spec.severity - 1])
            )
            meta["beta"] = float(spec.params.get("beta", 0.0))
            meta["scale_in_std"] = self.severity_factor(spec)
        return meta
