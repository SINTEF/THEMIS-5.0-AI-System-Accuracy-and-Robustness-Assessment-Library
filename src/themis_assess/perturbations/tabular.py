"""Feature-space perturbations, for tabular and sequence data alike.

These are the first *module* implementations of perturbations in this project. In repos
A, B and C the equivalent logic lived inline in the tutorial notebooks:

.. code-block:: python

    # from tests/robustness_multiclass-classif_tutorial.ipynb
    for perturb_name, pertub_args in considered_perturbations.items():
        X_test_pertubed = X_test + pertub_args['func'](**pertub_args['args'])

``ClassificationRobustnessAssessment(considered_perturbations)`` was constructed with the
perturbation dictionary in those notebooks and never used it — the notebook perturbed by
hand and passed precomputed metric dictionaries to ``compute_robustness``.

**Severity is relative, not absolute.** The single most important lesson from those
notebooks. The classification tutorial used ``sigma in {10, 1, 0.1, 0.01}`` and the
regression tutorial ``{0.1, 0.01, 1e-4, 1e-5}`` for the same nominal severity names — a
factor of 100 apart, purely because the iris features are raw centimetres and the
diabetes features are pre-standardised. An absolute severity scale is therefore not
transferable between datasets, and a robustness score built on one is not comparable.
Every numeric perturbation here scales severity by a **data-derived reference**, by
default the per-feature standard deviation. Repo B reached the same conclusion
independently: its sensitivity heatmaps pass ``perturbation_method="std"``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import Modality, PerturbationSpec, TimeScope
from themis_assess.perturbations.base import Perturbation

__all__ = [
    "GaussianNoise",
    "UniformBias",
    "AmplitudeScaling",
    "MissingValues",
    "CategorySwap",
    "feature_reference_scale",
]

_TABULAR_AND_SEQUENCE = frozenset({Modality.TABULAR, Modality.SEQUENCE})


def feature_reference_scale(
    data: DataAdapter,
    features: list[str],
    *,
    minimum: float = 1e-12,
) -> dict[str, float]:
    """Per-feature standard deviation over the real (unpadded) values.

    The reference a relative severity is multiplied by. Computed across all entities and
    timesteps at once, matching what repo B's ``perturb_numerical_data`` did: "std is the
    per-feature standard deviation computed over all vessels and timesteps".

    A constant feature has zero standard deviation and would be immune to a relative
    perturbation, which would silently misreport it as perfectly robust. Two fallbacks,
    in order: a feature constant at some non-zero level uses that level as its scale (a
    reading always at 7.0 is plausibly wrong by a few units, not by 1e-12); a feature
    constant at zero carries no scale information at all and falls back to unit scale.
    """
    values = data.values()
    mask = data.valid_mask()
    out: dict[str, float] = {}
    for name in features:
        j = data.feature_index(name)
        column = values[:, :, j][mask]
        column = column[~pd.isna(column)]
        if column.size == 0:
            out[name] = minimum
            continue
        numeric = np.asarray(column, dtype=float)
        std = float(numeric.std())
        if std <= minimum:
            level = abs(float(numeric.mean()))
            std = level if level > minimum else 1.0
        out[name] = std
    return out


class _NumericFeaturePerturbation(Perturbation):
    """Shared machinery for perturbations acting on numeric feature columns.

    Handles scope and time-scope resolution, the per-variant loop, and the relative
    severity reference, so a concrete perturbation only has to say what it does to one
    column at one set of timesteps.
    """

    modalities: ClassVar[frozenset[Modality]] = _TABULAR_AND_SEQUENCE
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = False

    def __init__(self, *, relative: bool = True) -> None:
        """
        Parameters
        ----------
        relative:
            When ``True`` (the default), severity multiplies each feature's own standard
            deviation. When ``False``, severity is used as an absolute magnitude — which
            reproduces the notebooks' behaviour but is only meaningful if every feature
            is already on a comparable scale.
        """
        self.relative = bool(relative)

    def _reference(self, data: DataAdapter, features: list[str]) -> dict[str, float]:
        if self.relative:
            return feature_reference_scale(data, features)
        return dict.fromkeys(features, 1.0)

    def _column_slots(
        self, data: DataAdapter, spec: PerturbationSpec, feature: str
    ) -> tuple[int, list[int]]:
        return data.feature_index(feature), self.resolve_timesteps(data, spec)

    def _apply_column(
        self,
        column: np.ndarray,
        valid: np.ndarray,
        timesteps: list[int],
        magnitude: float,
        spec: PerturbationSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturb one ``(n_entities, n_timesteps)`` column in place and return it."""
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
                j, timesteps = self._column_slots(data, spec, feature)
                values[:, :, j] = self._apply_column(
                    values[:, :, j],
                    mask,
                    timesteps,
                    factor * reference[feature],
                    spec,
                    rng,
                )
            out.append(data.with_values(values))
        return out

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["relative_to_feature_std"] = self.relative
        return meta


class GaussianNoise(_NumericFeaturePerturbation):
    """Additive Gaussian noise. Survey §4.2, value-domain.

    The workhorse, and the one perturbation every repo used. ``time_scope`` decides
    which of the survey's two value-domain families this becomes:

    * ``CONSTANT`` — one draw per entity, held across the whole series: a **sensor
      offset**. Repo B's ``perturb_numerical_data``.
    * ``PER_TIMESTEP`` — an independent draw at each timestep: **jitter**. Repo B's
      ``perturb_numerical_data_over_time``.

    For tabular data there is one timestep and the two coincide.

    Severity multiplies the feature's own standard deviation, so severity 3 is "one
    standard deviation" by default — the same convention as repo B's
    ``perturbation_method="std"``.

    Independent per-feature draws are *not* realistic for correlated inputs; survey §7
    warns that they "can produce physically implausible combinations". Use this for
    controlled sensitivity analysis, and a correlation-respecting perturbation when the
    question is deployment realism.
    """

    name = "gaussian_noise"
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)

    def _apply_column(self, column, valid, timesteps, magnitude, spec, rng):
        n_entities = column.shape[0]
        if spec.time_scope is TimeScope.PER_TIMESTEP:
            noise = rng.normal(0.0, magnitude, size=(n_entities, len(timesteps)))
            column[:, timesteps] += noise
        else:
            draw = rng.normal(0.0, magnitude, size=(n_entities, 1))
            column[:, timesteps] += draw
        return np.where(valid, column, np.nan)


class UniformBias(_NumericFeaturePerturbation):
    """A single positive offset applied to every value of a feature.

    This is what the tutorial notebooks' ``uniform_*`` entries actually did. Because
    ``norm_args`` (carrying ``size``) was not spread into them:

    .. code-block:: python

        'uniform_0_5': {'func': uniform, 'args': {'high': 0.5}},

    ``uniform(high=0.5)`` returns a *scalar*, so ``X_test + scalar`` shifted every cell
    by the same positive amount. That is not elementwise uniform noise — it is a
    **uniform-random constant bias**, non-zero-mean and one-directional. Whether it was
    intended, it is a genuinely distinct and useful perturbation: a miscalibrated
    sensor, a unit error, a systematically shifted measurement.

    Modelled here as one draw from ``U(0, severity x reference)`` per feature, applied
    to every entity and timestep — the notebook's semantics, made explicit rather than
    accidental.
    """

    name = "uniform_bias"
    severity_scale = (0.25, 0.5, 1.0, 2.0, 4.0)

    def _apply_column(self, column, valid, timesteps, magnitude, spec, rng):
        offset = float(rng.uniform(0.0, magnitude))
        column[:, timesteps] += offset
        return np.where(valid, column, np.nan)


class AmplitudeScaling(_NumericFeaturePerturbation):
    """Multiplicative scaling of a feature. Survey §4.2, value-domain.

    A gain error rather than an offset: the signal keeps its shape and changes its size.
    Repo C's ``distribution_shift`` did this with a fixed ``*= 1.5`` on a random subset
    of columns; here the factor is severity-driven and the columns come from the scope,
    so "scale 30% of the features by 1.5" is a specification rather than a method.

    Severity is a *fractional* change, and the direction is drawn per feature, so
    severity 3 means "scale by 1 +- 0.3". Unlike the additive perturbations this needs
    no reference scale — a ratio is already dimensionless — so ``relative`` has no
    effect here.
    """

    name = "amplitude_scaling"
    severity_scale = (0.05, 0.15, 0.30, 0.50, 0.80)

    def __init__(self) -> None:
        super().__init__(relative=False)

    def _apply_column(self, column, valid, timesteps, magnitude, spec, rng):
        direction = 1.0 if rng.random() < 0.5 else -1.0
        column[:, timesteps] *= 1.0 + direction * magnitude
        return np.where(valid, column, np.nan)

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta.pop("relative_to_feature_std", None)
        if spec is not None:
            meta["fractional_change"] = self.severity_factor(spec)
        return meta


class MissingValues(Perturbation):
    """Replace a fraction of entries with ``NaN``. Survey §4.2, missing data.

    Repo C's Plovdiv notebook is the source, and its markdown states the strategy:

        For each feature category in [category_columns, morbidity_history, symptoms,
        negative_habits, physical_findings] — uniformly NaN xx% of the entries —
        Evaluate model performance

    Two things carry over. First, the rate is per-*entry*, drawn independently, not a
    fixed count. Second, and the reason ``Scope.NAMED`` matters here, the useful question
    was "what happens when *this group of features* goes missing" — a domain grouping,
    expressed here as a named feature set rather than a hardcoded list.

    Repo C swept 100 rates from 0.00 to 0.99 with 100 repeats each, and left a note that
    this "will take about 15mins on a 12 thread laptop". The five-level scale condenses
    that sweep; :meth:`n_model_calls` makes the cost visible before it is paid.

    Applies to categorical features too — a missing category is as real as a missing
    measurement — which is why this does not inherit the numeric base.

    Note the survey's caution: the notebook itself observed that NaN injection is
    "somehow an extreme version" of perturbation. Whether a model's own missing-value
    handling makes the label still recoverable is a question about the pipeline, not
    only the data.
    """

    name = "missing_values"
    modalities = _TABULAR_AND_SEQUENCE
    severity_scale = (0.05, 0.10, 0.25, 0.50, 0.90)

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        rate = self.severity_factor(spec)
        timesteps = self.resolve_timesteps(data, spec)
        mask = data.valid_mask()

        out = []
        for _ in range(spec.n_variants):
            values = data.values().astype(object) if _is_object(data) else data.values().astype(float)
            for feature in features:
                j = data.feature_index(feature)
                column = values[:, :, j]
                if spec.time_scope is TimeScope.CONSTANT and data.is_sequential:
                    # An entity's sensor drops out entirely rather than intermittently.
                    drop = rng.random(column.shape[0]) < rate
                    for t in timesteps:
                        column[drop, t] = np.nan
                else:
                    drop = rng.random((column.shape[0], len(timesteps))) < rate
                    for k, t in enumerate(timesteps):
                        column[drop[:, k], t] = np.nan
                values[:, :, j] = np.where(mask, column, np.nan)
            out.append(data.with_values(values))
        return out

    def n_model_calls(self, data, spec):
        return int(spec.n_variants)

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["missing_rate"] = self.severity_factor(spec)
            meta["dropout_granularity"] = (
                "whole entity" if spec.time_scope is TimeScope.CONSTANT else "per entry"
            )
        return meta


class CategorySwap(Perturbation):
    """Replace a categorical value with a different category observed in the data.

    Repo B's contribution, and it has no analogue in the survey — §4 covers image, time
    series and text, and leaves categorical features alone. Worth keeping as a repo-A
    extension, and it is the one repo-B perturbation that transfers directly to repo C's
    tabular clinical data.

    The replacement always comes from the dataset-wide pool of observed categories, so
    the perturbed input stays a value the model could plausibly have seen. Severity sets
    the *probability* that a given value is swapped rather than the size of the change,
    since categories have no magnitude.

    ``time_scope`` matters, and repo B's notebook records why:

        Since categorical features are constant per vessel, perturbations are applied
        uniformly across the time axis.

    So ``CONSTANT`` draws one replacement per entity and holds it — the semantically
    right behaviour for a per-entity attribute like vessel type. ``PER_TIMESTEP`` draws
    independently at each step, which is a much harsher and usually less realistic test.
    """

    name = "category_swap"
    modalities = _TABULAR_AND_SEQUENCE
    accepts_numerical = False
    accepts_categorical = True
    severity_scale = (0.10, 0.25, 0.50, 0.75, 1.00)

    def applies_to(self, data, task=None):
        """Also requires a categorical feature with at least two observed categories.

        A single-category column has nothing to swap to, and reporting it as applicable
        would hand the caller a perturbation that raises the moment it runs.
        """
        if not super().applies_to(data, task):
            return False
        return any(
            data.category_pool(name).size >= 2 for name in data.categorical_features
        )

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        features = self.eligible_features(data, spec)
        probability = self.severity_factor(spec)
        timesteps = self.resolve_timesteps(data, spec)
        mask = data.valid_mask()

        out = []
        for _ in range(spec.n_variants):
            values = data.values().copy()
            for feature in features:
                j = data.feature_index(feature)
                pool = data.category_pool(feature)
                if pool.size < 2:
                    raise ValueError(
                        f"Feature {feature!r} has {pool.size} observed category; "
                        f"swapping needs at least 2. Check that it really is "
                        f"categorical."
                    )
                column = values[:, :, j]
                if spec.time_scope is TimeScope.CONSTANT:
                    for e in range(column.shape[0]):
                        if rng.random() >= probability:
                            continue
                        current = _first_valid(column[e], mask[e])
                        column[e, timesteps] = _draw_other(pool, current, rng)
                else:
                    for e in range(column.shape[0]):
                        for t in timesteps:
                            if not mask[e, t] or rng.random() >= probability:
                                continue
                            column[e, t] = _draw_other(pool, column[e, t], rng)
                values[:, :, j] = column
            out.append(data.with_values(values))
        return out

    def metadata(self, spec=None):
        meta = super().metadata(spec)
        if spec is not None:
            meta["swap_probability"] = self.severity_factor(spec)
            meta["draw"] = (
                "one replacement per entity"
                if spec.time_scope is TimeScope.CONSTANT
                else "independent per timestep"
            )
        return meta


def _is_object(data: DataAdapter) -> bool:
    return data.values().dtype.kind in "OUS"


def _first_valid(row: np.ndarray, valid: np.ndarray) -> Any:
    positions = np.flatnonzero(valid)
    return row[positions[0]] if positions.size else row[0]


def _draw_other(pool: np.ndarray, current: Any, rng: np.random.Generator) -> Any:
    """Draw a category from ``pool`` that differs from ``current``."""
    alternatives = pool[pool != current] if not pd.isna(current) else pool
    if alternatives.size == 0:
        alternatives = pool
    return alternatives[rng.integers(alternatives.size)]
