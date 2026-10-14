"""Time-axis perturbations. Survey §4.2.

The second of the survey's four time-series families: perturbations that leave the
*values* alone and change *when* they happened, or how much of the record the model gets
to see. The survey names window slicing, time warping, and speed changes.

Two of these shorten a series, which is why
:meth:`~themis_assess.core.adapters.DataAdapter.with_values` accepts new entity lengths:
what a shortened entity loses becomes padding, the array shape is unchanged, and only
the validity mask moves.

:class:`InputWindowShift` is not from the survey. It generalises repo B's
``time_robustness_heatmap``, which measures how much a forecast changes when the
prediction window slides forward — a stability instrument no other repo has, and the
closest anything in this project comes to the survey's time-axis family.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from themis_assess.core.adapters import DataAdapter
from themis_assess.core.types import Alignment, Modality, PerturbationSpec
from themis_assess.perturbations.base import Perturbation

__all__ = ["WindowSlice", "TimeWarp", "InputWindowShift"]

_SEQUENCE_ONLY = frozenset({Modality.SEQUENCE})


class _TimeAxisPerturbation(Perturbation):
    """Shared machinery for perturbations acting on the time axis.

    Unlike the value-domain family these move *all* features together — shifting one
    feature's time axis and not another's would produce a record that never existed,
    rather than a plausible one recorded differently. So these ignore ``Scope`` for the
    numeric/categorical split and act on every feature, categorical included.
    """

    modalities: ClassVar[frozenset[Modality]] = _SEQUENCE_ONLY
    accepts_numerical: ClassVar[bool] = True
    accepts_categorical: ClassVar[bool] = True

    def _entity_slice(self, data: DataAdapter, entity: int) -> np.ndarray:
        """Positions of the entity's real timesteps, in order."""
        return np.flatnonzero(data.valid_mask()[entity])

    def _write_back(
        self,
        values: np.ndarray,
        entity: int,
        block: np.ndarray,
        alignment: Alignment,
        n_timesteps: int,
    ) -> None:
        """Place a (possibly shorter) block into an entity's row, padding the rest."""
        length = block.shape[0]
        values[entity, :, :] = np.nan
        if length == 0:
            return
        if alignment is Alignment.END:
            values[entity, n_timesteps - length :, :] = block
        else:
            values[entity, :length, :] = block


class WindowSlice(_TimeAxisPerturbation):
    """Keep only a contiguous window of each series. Survey §4.2, time axis.

    The survey's description is precise about what this tests: extracting contiguous
    subsequences probes "robustness to incomplete temporal context". A model that only
    works when it can see the full history will fail here even though every value it
    does see is exactly right.

    Severity is the fraction of the series *removed*, so severity 5 leaves a fifth of it.
    The window position is drawn uniformly, and the kept block is re-aligned according to
    the adapter's alignment — under the default ``END`` alignment the kept window sits at
    the right-hand edge, which is what a forecasting model reading the most recent
    context expects.

    Distinct from :class:`~themis_assess.perturbations.timeseries.value_domain.MissingSegment`:
    that leaves a hole *inside* the series with data either side, this hands the model a
    genuinely shorter record. A model with a fixed input window may handle one and not
    the other.
    """

    name = "window_slice"
    severity_scale = (0.10, 0.25, 0.40, 0.60, 0.80)

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        self.eligible_features(data, spec)  # raises if nothing is eligible
        removed = self.severity_factor(spec)
        source = data.values()
        n_timesteps = data.n_timesteps

        out = []
        for _ in range(spec.n_variants):
            values = np.full_like(source, np.nan, dtype=float)
            lengths = []
            for e in range(data.n_entities):
                positions = self._entity_slice(data, e)
                n = positions.size
                keep = max(1, round((1.0 - removed) * n)) if n else 0
                if keep >= n:
                    block = source[e, positions, :]
                else:
                    start = int(rng.integers(0, n - keep + 1))
                    block = source[e, positions[start : start + keep], :]
                lengths.append(block.shape[0])
                self._write_back(values, e, block, data.alignment, n_timesteps)
            out.append(data.with_values(values, entity_lengths=lengths))
        return out

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        if spec is not None:
            removed = self.severity_factor(spec)
            meta["fraction_removed"] = removed
            meta["fraction_kept"] = 1.0 - removed
            meta["changes_series_length"] = True
        return meta


class TimeWarp(_TimeAxisPerturbation):
    """Stretch and compress the time axis. Survey §4.2, time axis.

    The same events, recorded at a varying rate: a process that ran fast then slow, a
    logger whose sampling drifted, a patient monitored at irregular intervals but
    reported on a regular grid. Implemented as a smooth monotone remapping of the time
    axis followed by resampling onto the original grid, so the series keeps its length
    and its values stay in range — only their spacing changes.

    The warp is built from a random cumulative walk, normalised to run from the first
    timestep to the last, which guarantees monotonicity: time never runs backwards, and
    no event is reordered. Severity sets how far the warp may depart from the identity.

    Categorical features are resampled by nearest neighbour rather than interpolated —
    the midpoint between category 1 and category 3 is not category 2.

    A model that is genuinely reading *shape* should be largely unmoved by this; one
    that has memorised absolute positions in its input window will not be.
    """

    name = "time_warp"
    severity_scale = (0.05, 0.10, 0.20, 0.35, 0.50)

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        self.eligible_features(data, spec)
        strength = self.severity_factor(spec)
        source = data.values()
        categorical = {data.feature_index(f) for f in data.categorical_features}
        n_timesteps = data.n_timesteps

        out = []
        for _ in range(spec.n_variants):
            values = np.full_like(source, np.nan, dtype=float)
            for e in range(data.n_entities):
                positions = self._entity_slice(data, e)
                n = positions.size
                if n < 3:  # nothing meaningful to warp
                    self._write_back(
                        values, e, source[e, positions, :], data.alignment, n_timesteps
                    )
                    continue
                warped = _monotone_warp(n, strength, rng)
                grid = np.arange(n, dtype=float)
                block = np.empty((n, source.shape[2]), dtype=float)
                for j in range(source.shape[2]):
                    column = source[e, positions, j]
                    if j in categorical:
                        nearest = np.clip(np.rint(warped).astype(int), 0, n - 1)
                        block[:, j] = column[nearest]
                    else:
                        block[:, j] = np.interp(warped, grid, column)
                self._write_back(values, e, block, data.alignment, n_timesteps)
            out.append(data.with_values(values))
        return out

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        if spec is not None:
            meta["warp_strength"] = self.severity_factor(spec)
            meta["monotone"] = True
            meta["categorical_resampling"] = "nearest neighbour"
        return meta


def _monotone_warp(n: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """A monotone remapping of ``0..n-1`` onto itself, departing from the identity.

    Built from a positive random walk of step sizes, so the cumulative sum is strictly
    increasing by construction. ``strength`` is the coefficient of variation of the step
    sizes: 0 gives the identity, larger values give a more uneven pace.
    """
    steps = 1.0 + strength * rng.normal(0.0, 1.0, n - 1)
    steps = np.clip(steps, 0.05, None)  # keep every step positive
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    return cumulative / cumulative[-1] * (n - 1)


class InputWindowShift(_TimeAxisPerturbation):
    """Move the forecast origin earlier by dropping timesteps from the end.

    Generalises repo B's ``time_robustness_heatmap``, whose notebook implemented the
    same idea inline:

    .. code-block:: python

        # Truncate series & covariates so they end just before desired_pred_start
        if prediction_end_offset > 0:
            trunc_series = vessel_series[:-prediction_end_offset]

    The question it asks is specific to forecasting and asked by nothing else here: how
    much does the prediction change when the model is asked the same question a little
    earlier? A model whose forecast swings sharply as the origin slides by one step is
    unstable in a way no value-domain perturbation reveals — every input value it saw was
    real and correct.

    Severity is the fraction of the series dropped from the end. Deterministic: there is
    no random position to draw, which makes it the one perturbation here whose result
    does not depend on the generator.

    Under ``END`` alignment the remaining window stays right-aligned, so a model reading
    the most recent ``input_chunk_length`` steps sees a genuinely earlier window rather
    than a shifted array.
    """

    name = "input_window_shift"
    severity_scale = (0.02, 0.05, 0.10, 0.20, 0.35)

    def perturb(self, data, spec, rng):
        self.validate(data, spec)
        self.eligible_features(data, spec)
        dropped = self.severity_factor(spec)
        source = data.values()
        n_timesteps = data.n_timesteps

        values = np.full_like(source, np.nan, dtype=float)
        lengths = []
        for e in range(data.n_entities):
            positions = self._entity_slice(data, e)
            n = positions.size
            drop = min(max(1, round(dropped * n)), max(n - 1, 0)) if n else 0
            keep = n - drop
            block = source[e, positions[:keep], :] if keep > 0 else source[e, :0, :]
            lengths.append(block.shape[0])
            self._write_back(values, e, block, data.alignment, n_timesteps)

        shifted = data.with_values(values, entity_lengths=lengths)
        # Deterministic, so every requested variant is the same object's worth of data.
        return [shifted for _ in range(spec.n_variants)]

    def n_model_calls(self, data: DataAdapter, spec: PerturbationSpec) -> int:
        """One call regardless of ``n_variants`` — the shift is deterministic."""
        return 1

    def metadata(self, spec: PerturbationSpec | None = None) -> dict[str, Any]:
        meta = super().metadata(spec)
        meta["deterministic"] = True
        if spec is not None:
            meta["fraction_dropped_from_end"] = self.severity_factor(spec)
            meta["changes_series_length"] = True
        return meta
