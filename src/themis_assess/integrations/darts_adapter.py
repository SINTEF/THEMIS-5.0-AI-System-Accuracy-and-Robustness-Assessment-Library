"""Adapter between darts ``TimeSeries`` and the library's array boundary.

Lives outside ``core/`` and ships under the ``[timeseries]`` extra: repo A's core must
not depend on darts (docs/DECISIONS.md, D7 and D11). ``darts`` is imported inside the
functions that need it, so importing this module without darts installed raises a
message naming the extra rather than an ``ImportError`` from deep in a traceback.

What this buys the Valencia use case: its sixteen perturbation methods and its
sensitivity heatmaps were written against ``List[TimeSeries]`` with vessel semantics.
Through this adapter the same data reaches perturbations that know nothing about darts,
vessels, or forecasting — and a perturbation written for the clinical tabular use case
works on it unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from themis_assess.core.adapters import BaseAdapter
from themis_assess.core.errors import require_extra
from themis_assess.core.types import Alignment, Modality

if TYPE_CHECKING:  # pragma: no cover - typing only
    from darts import TimeSeries

__all__ = ["DartsAdapter"]


def _timeseries_class() -> Any:
    return require_extra("darts", "timeseries").TimeSeries


class DartsAdapter(BaseAdapter):
    """Adapter over a list of darts ``TimeSeries``, one per entity.

    Entities may have different lengths; the value array is padded with ``NaN`` and
    :meth:`valid_mask` marks the real positions. ``alignment`` decides which end the
    padding goes on, and defaults to :attr:`Alignment.END` — in event forecasting the
    meaningful shared axis is time remaining until the event, which is why repo B's
    ``all_vessels_heatmap`` end-aligns its vessels.

    Time indices and static covariates are kept, so :meth:`to_series` returns a list
    that a darts model can consume directly.

    Parameters
    ----------
    series:
        One ``TimeSeries`` per entity. All must share the same components, in the same
        order.
    categorical_features:
        Which components are categorical. Declare these on the data handler and pass
        them through; when omitted, the constant-within-entity heuristic in
        :func:`~themis_assess.core.adapters.infer_categorical_features` is used, which
        is what repo B originally did.
    alignment:
        How unequal lengths line up.
    entity_ids:
        Labels for reporting. Positional indices by default.
    """

    _modality = Modality.SEQUENCE
    _is_sequential = True

    def __init__(
        self,
        series: Sequence[TimeSeries],
        *,
        categorical_features: Sequence[str] | None = None,
        alignment: Alignment = Alignment.END,
        entity_ids: Sequence[object] | None = None,
    ) -> None:
        ts_class = _timeseries_class()
        items = list(series)
        if not items:
            raise ValueError("DartsAdapter needs at least one TimeSeries.")
        for i, s in enumerate(items):
            if not isinstance(s, ts_class):
                raise TypeError(
                    f"Element {i} is a {type(s).__name__}, not a darts TimeSeries."
                )

        components = [str(c) for c in items[0].components]
        for i, s in enumerate(items[1:], start=1):
            other = [str(c) for c in s.components]
            if other != components:
                raise ValueError(
                    f"All series must share the same components in the same order. "
                    f"Series 0 has {components}; series {i} has {other}."
                )

        lengths = np.array([len(s) for s in items], dtype=int)
        n_timesteps = int(lengths.max())
        n_features = len(components)
        alignment = Alignment(alignment)

        array = np.full((len(items), n_timesteps, n_features), np.nan, dtype=float)
        for e, s in enumerate(items):
            block = np.asarray(s.values(copy=False), dtype=float).reshape(len(s), n_features)
            if alignment is Alignment.END:
                array[e, n_timesteps - len(s) :, :] = block
            else:
                array[e, : len(s), :] = block

        self._time_indices = [s.time_index for s in items]
        self._static_covariates = [
            None if s.static_covariates is None else s.static_covariates.copy()
            for s in items
        ]

        super().__init__(
            array,
            components,
            categorical_features=categorical_features,
            entity_lengths=lengths,
            alignment=alignment,
            entity_ids=entity_ids,
        )

    # -- derivation preserving darts metadata ----------------------------------

    def _clone_with(
        self, values: np.ndarray, *, entity_lengths: ArrayLike | None = None
    ) -> DartsAdapter:
        lengths = self._lengths if entity_lengths is None else np.asarray(
            entity_lengths, dtype=int
        ).ravel()
        clone = object.__new__(DartsAdapter)
        self._copy_identity_onto(clone)
        # A shortened series keeps the tail of its own time index, matching END
        # alignment: dropping leading steps drops leading timestamps.
        clone._time_indices = [
            index if new == old else (index[-new:] if new > 0 else index[:0])
            for index, old, new in zip(
                self._time_indices, self._lengths, lengths, strict=True
            )
        ]
        clone._static_covariates = list(self._static_covariates)
        BaseAdapter.__init__(
            clone,
            values,
            self._feature_names,
            categorical_features=self._categorical,
            entity_lengths=lengths,
            alignment=self._alignment,
            entity_ids=self._entity_ids,
        )
        return clone

    def take(self, entity_indices: ArrayLike) -> DartsAdapter:
        idx = np.asarray(entity_indices)
        if idx.dtype == bool:
            if idx.shape[0] != self.n_entities:
                raise ValueError(
                    f"Boolean selector has length {idx.shape[0]} for "
                    f"{self.n_entities} entities."
                )
            idx = np.flatnonzero(idx)
        idx = idx.astype(int).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_entities):
            raise ValueError(
                f"Entity indices must lie in [0, {self.n_entities - 1}]; got "
                f"min={idx.min()}, max={idx.max()}."
            )
        clone = object.__new__(DartsAdapter)
        self._copy_identity_onto(clone)
        clone._time_indices = [self._time_indices[i] for i in idx]
        clone._static_covariates = [self._static_covariates[i] for i in idx]
        BaseAdapter.__init__(
            clone,
            self._values[idx],
            self._feature_names,
            categorical_features=self._categorical,
            entity_lengths=self._lengths[idx],
            alignment=self._alignment,
            entity_ids=[self._entity_ids[i] for i in idx],
        )
        return clone

    def to_series(self) -> list[TimeSeries]:
        """Back to darts ``TimeSeries``, un-padded, with time index and static covariates.

        Round-trips: ``DartsAdapter(s).to_series()`` reproduces ``s`` up to float dtype.
        """
        ts_class = _timeseries_class()
        out: list[TimeSeries] = []
        for e in range(self.n_entities):
            length = int(self._lengths[e])
            if self._alignment is Alignment.END:
                block = self._values[e, self.n_timesteps - length :, :]
            else:
                block = self._values[e, :length, :]
            frame = pd.DataFrame(
                block, columns=self._feature_names, index=self._time_indices[e]
            )
            series = ts_class.from_dataframe(
                frame,
                static_covariates=self._static_covariates[e],
            )
            out.append(series)
        return out
