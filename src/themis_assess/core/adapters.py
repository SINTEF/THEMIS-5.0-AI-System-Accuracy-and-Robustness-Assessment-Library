"""Data adapters — the boundary where data-format concerns stop.

This module is the answer to the single largest obstacle to generalising the use-case
code: repo B's perturbations take ``List[darts.TimeSeries]`` with vessel semantics baked
in, so none of them can serve a classification use case, and none can be tested without
a model. Behind an adapter, a perturbation sees only a numeric array, feature names and
which features are categorical — and one implementation then serves tabular, sequence
and text data.

The canonical value shape is ``(n_entities, n_timesteps, n_features)``:

* **entities** are whatever the dataset counts — vessels, patients, documents;
* **timesteps** are the sequence axis, with length 1 for tabular and text data;
* **features** are the columns.

Tabular data therefore has a singleton time axis rather than a missing one. That is a
deliberate deviation from the architecture sketch, which had ``n_timesteps: int | None``:
a ``None`` that every caller must then treat as 1 is a bug waiting to happen, so
:attr:`DataAdapter.is_sequential` carries the distinction instead and the shape is always
three-dimensional.

**Ragged entities.** Sequence datasets routinely hold entities of different lengths —
repo B's vessels do. :meth:`DataAdapter.values` pads to the longest with ``NaN``,
:meth:`DataAdapter.valid_mask` says which positions are real, and
:attr:`DataAdapter.alignment` says which end the padding went on. End alignment is the
default for sequences because in event forecasting what matters is time remaining until
the event, not time since the record started.

No adapter here imports ``darts``, ``torch`` or ``transformers``. The darts adapter lives
in :mod:`themis_assess.integrations.darts_adapter` and ships under the ``[timeseries]``
extra (docs/DECISIONS.md, D11).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from themis_assess.core.types import Alignment, Modality

__all__ = [
    "DataAdapter",
    "BaseAdapter",
    "ArrayAdapter",
    "FrameAdapter",
    "TextAdapter",
    "infer_categorical_features",
]


@runtime_checkable
class DataAdapter(Protocol):
    """What every perturbation and sensitivity map may assume about its input."""

    @property
    def modality(self) -> Modality: ...

    @property
    def feature_names(self) -> list[str]: ...

    @property
    def categorical_features(self) -> list[str]: ...

    @property
    def numerical_features(self) -> list[str]: ...

    @property
    def n_entities(self) -> int: ...

    @property
    def n_timesteps(self) -> int: ...

    @property
    def is_sequential(self) -> bool: ...

    @property
    def alignment(self) -> Alignment: ...

    @property
    def entity_lengths(self) -> NDArray[np.int_]: ...

    def values(self) -> NDArray: ...

    def valid_mask(self) -> NDArray[np.bool_]: ...

    def with_values(self, values: ArrayLike) -> Self: ...

    def feature_index(self, feature: str) -> int: ...

    def category_pool(self, feature: str) -> NDArray: ...

    def take(self, entity_indices: ArrayLike) -> Self: ...


def infer_categorical_features(
    values: NDArray,
    feature_names: Sequence[str],
    *,
    is_sequential: bool,
    valid_mask: NDArray[np.bool_] | None = None,
) -> list[str]:
    """Guess which features are categorical.

    Repo B originally used the heuristic "constant within every entity", then replaced
    it with a declaration read from ``data/categorical_and_numerical_features.txt``.
    Moving from inference to declaration was the right direction; the right *home* for
    the declaration is the data handler, which is why every adapter here takes an
    explicit ``categorical_features`` argument. This function is only the fallback.

    The heuristic differs by modality, because "constant within entity" is trivially
    true of every column when there is a single timestep:

    * **sequential** — a feature that never varies over time within any entity is
      treated as categorical. This is repo B's original rule, and it correctly picks up
      per-entity attributes (vessel type, flag) as against measurements.
    * **tabular** — only non-numeric dtypes are treated as categorical. Low-cardinality
      integer columns are *not* guessed at: an encoded category and a small count look
      identical, and guessing wrong silently changes what a perturbation does.

    Declare the types on the data handler rather than relying on this.
    """
    names = list(feature_names)
    if not is_sequential:
        if values.dtype.kind in "OUS":  # object / unicode / bytes
            return names
        return []

    mask = np.ones(values.shape[:2], dtype=bool) if valid_mask is None else valid_mask
    categorical: list[str] = []
    for j, name in enumerate(names):
        column = values[:, :, j]
        constant_within_each = True
        for e in range(column.shape[0]):
            observed = column[e][mask[e]]
            if observed.size <= 1:
                continue
            first = observed[0]
            if values.dtype.kind in "OUS":
                varies = bool(np.any(observed != first))
            else:
                finite = observed[~pd.isna(observed)]
                varies = finite.size > 1 and bool(np.any(finite != finite[0]))
            if varies:
                constant_within_each = False
                break
        if constant_within_each:
            categorical.append(name)
    return categorical


class BaseAdapter:
    """Shared implementation behind the concrete adapters.

    Holds the padded value array, the validity mask, the feature names and the
    categorical declaration, and implements everything that does not depend on the
    original data format.
    """

    _modality: Modality = Modality.TABULAR
    _is_sequential: bool = False

    def __init__(
        self,
        values: ArrayLike,
        feature_names: Sequence[str],
        *,
        categorical_features: Sequence[str] | None = None,
        entity_lengths: ArrayLike | None = None,
        alignment: Alignment = Alignment.END,
        entity_ids: Sequence[object] | None = None,
    ) -> None:
        array = np.asarray(values)
        if array.ndim == 2:
            # (entities, features) -> singleton time axis
            array = array[:, None, :]
        if array.ndim != 3:
            raise ValueError(
                f"Expected values with shape (n_entities, n_timesteps, n_features) or "
                f"(n_entities, n_features); got shape {array.shape}."
            )
        self._values = array

        names = [str(n) for n in feature_names]
        if len(names) != array.shape[2]:
            raise ValueError(
                f"Got {len(names)} feature name(s) for {array.shape[2]} feature column(s)."
            )
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Feature names must be unique; duplicated: {duplicates}.")
        self._feature_names = names
        self._index = {name: j for j, name in enumerate(names)}

        if entity_lengths is None:
            self._lengths = np.full(array.shape[0], array.shape[1], dtype=int)
        else:
            lengths = np.asarray(entity_lengths, dtype=int).ravel()
            if lengths.shape[0] != array.shape[0]:
                raise ValueError(
                    f"Got {lengths.shape[0]} entity length(s) for {array.shape[0]} entities."
                )
            if np.any(lengths < 0) or np.any(lengths > array.shape[1]):
                raise ValueError(
                    f"entity_lengths must lie in [0, {array.shape[1]}]; got "
                    f"min={lengths.min()}, max={lengths.max()}."
                )
            self._lengths = lengths

        self._alignment = Alignment(alignment)
        self._entity_ids = list(entity_ids) if entity_ids is not None else list(
            range(array.shape[0])
        )
        if len(self._entity_ids) != array.shape[0]:
            raise ValueError(
                f"Got {len(self._entity_ids)} entity id(s) for {array.shape[0]} entities."
            )

        if categorical_features is None:
            self._categorical = infer_categorical_features(
                self._values,
                self._feature_names,
                is_sequential=self._is_sequential,
                valid_mask=self.valid_mask(),
            )
        else:
            declared = [str(n) for n in categorical_features]
            unknown = [n for n in declared if n not in self._index]
            if unknown:
                raise ValueError(
                    f"categorical_features names unknown feature(s): {unknown}. "
                    f"Known features: {self._feature_names}."
                )
            self._categorical = declared

    # -- identity --------------------------------------------------------------

    @property
    def modality(self) -> Modality:
        return self._modality

    @property
    def is_sequential(self) -> bool:
        return self._is_sequential

    @property
    def alignment(self) -> Alignment:
        return self._alignment

    @property
    def entity_ids(self) -> list[object]:
        """Labels for the entities, for reporting. Positional indices by default."""
        return list(self._entity_ids)

    # -- shape -----------------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return int(self._values.shape[0])

    @property
    def n_timesteps(self) -> int:
        return int(self._values.shape[1])

    @property
    def n_features(self) -> int:
        return int(self._values.shape[2])

    @property
    def entity_lengths(self) -> NDArray[np.int_]:
        return self._lengths.copy()

    # -- features --------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def categorical_features(self) -> list[str]:
        return list(self._categorical)

    @property
    def numerical_features(self) -> list[str]:
        return [n for n in self._feature_names if n not in set(self._categorical)]

    def feature_index(self, feature: str) -> int:
        try:
            return self._index[feature]
        except KeyError:
            raise KeyError(
                f"Unknown feature {feature!r}. Known features: {self._feature_names}."
            ) from None

    def feature_indices(self, features: Sequence[str]) -> list[int]:
        """Positional indices of the named features, in the order given."""
        return [self.feature_index(f) for f in features]

    # -- values ----------------------------------------------------------------

    def values(self) -> NDArray:
        """The padded value array, shape ``(n_entities, n_timesteps, n_features)``.

        A copy: callers perturb freely without touching the adapter's state.
        """
        return self._values.copy()

    def valid_mask(self) -> NDArray[np.bool_]:
        """``(n_entities, n_timesteps)`` boolean array — ``True`` where data is real.

        All ``True`` unless entities have unequal lengths.
        """
        n_entities, n_timesteps = self._values.shape[0], self._values.shape[1]
        mask = np.zeros((n_entities, n_timesteps), dtype=bool)
        for e, length in enumerate(self._lengths):
            if length == 0:
                continue
            if self._alignment is Alignment.END:
                mask[e, n_timesteps - length :] = True
            else:
                mask[e, :length] = True
        return mask

    def category_pool(self, feature: str) -> NDArray:
        """Every category observed for ``feature`` across the whole dataset.

        Dataset-wide rather than per-entity on purpose: a categorical perturbation
        replaces a value with a different category that actually occurs somewhere, so
        the perturbed input stays a plausible one.
        """
        j = self.feature_index(feature)
        column = self._values[:, :, j][self.valid_mask()]
        if column.dtype.kind in "OUS":
            observed = column[~pd.isna(column)]
        else:
            observed = column[~pd.isna(column)]
        return np.unique(observed)

    # -- derivation ------------------------------------------------------------

    def _copy_identity_onto(self, clone: object) -> None:
        """Carry the instance-level modality flags onto a clone.

        ``ArrayAdapter`` decides ``_is_sequential`` and ``_modality`` per instance (from
        the array's shape, or an explicit argument), so a clone built with
        ``object.__new__`` would silently fall back to the class defaults and a sequence
        adapter would come back tabular after ``with_values``.
        """
        clone._is_sequential = self._is_sequential  # type: ignore[attr-defined]
        clone._modality = self._modality  # type: ignore[attr-defined]

    def _clone_with(
        self, values: NDArray, *, entity_lengths: ArrayLike | None = None
    ) -> Self:
        clone = object.__new__(type(self))
        self._copy_identity_onto(clone)
        BaseAdapter.__init__(
            clone,
            values,
            self._feature_names,
            categorical_features=self._categorical,
            entity_lengths=self._lengths if entity_lengths is None else entity_lengths,
            alignment=self._alignment,
            entity_ids=self._entity_ids,
        )
        return clone

    def with_values(
        self, values: ArrayLike, *, entity_lengths: ArrayLike | None = None
    ) -> Self:
        """A new adapter carrying ``values``, keeping names, types, lengths and ids.

        Parameters
        ----------
        entity_lengths:
            New per-entity lengths, for the perturbations that shorten a series —
            window slicing, a shifted forecast origin. Everything a shortened entity
            loses becomes padding, so the shape is unchanged and only the validity mask
            moves. Omit to keep the current lengths.
        """
        array = np.asarray(values)
        if array.ndim == 2:
            array = array[:, None, :]
        if array.shape != self._values.shape:
            raise ValueError(
                f"with_values expects shape {self._values.shape}; got {array.shape}. "
                f"Use take() to change the number of entities."
            )
        return self._clone_with(array, entity_lengths=entity_lengths)

    def take(self, entity_indices: ArrayLike) -> Self:
        """A new adapter over a subset of entities, in the order given."""
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
        clone = object.__new__(type(self))
        self._copy_identity_onto(clone)
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

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"{type(self).__name__}(modality={self._modality.value}, "
            f"entities={self.n_entities}, timesteps={self.n_timesteps}, "
            f"features={self.n_features}, categorical={len(self._categorical)})"
        )


class ArrayAdapter(BaseAdapter):
    """Adapter over a numpy array.

    Accepts ``(n_entities, n_timesteps, n_features)`` for sequences or
    ``(n_entities, n_features)`` for tabular data. The most direct way to get any data
    into the library, and the shape every other adapter converts to.
    """

    def __init__(
        self,
        values: ArrayLike,
        feature_names: Sequence[str] | None = None,
        *,
        categorical_features: Sequence[str] | None = None,
        entity_lengths: ArrayLike | None = None,
        alignment: Alignment = Alignment.END,
        entity_ids: Sequence[object] | None = None,
        sequential: bool | None = None,
    ) -> None:
        array = np.asarray(values)
        inferred_sequential = array.ndim == 3 and array.shape[1] > 1
        self._is_sequential = inferred_sequential if sequential is None else bool(sequential)
        self._modality = Modality.SEQUENCE if self._is_sequential else Modality.TABULAR
        n_features = array.shape[2] if array.ndim == 3 else array.shape[-1]
        names = (
            [f"x{j}" for j in range(n_features)] if feature_names is None else feature_names
        )
        super().__init__(
            array,
            names,
            categorical_features=categorical_features,
            entity_lengths=entity_lengths,
            alignment=alignment,
            entity_ids=entity_ids,
        )


class FrameAdapter(BaseAdapter):
    """Adapter over a pandas DataFrame: one row per entity, one column per feature.

    Tabular only. Object, string and categorical dtypes are treated as categorical
    unless the caller declares otherwise, which is a safer default than for a bare
    array because the dtypes carry real information.
    """

    _modality = Modality.TABULAR
    _is_sequential = False

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        categorical_features: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"FrameAdapter expects a pandas DataFrame; got {type(frame)}.")
        columns = list(frame.columns) if feature_names is None else list(feature_names)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(f"DataFrame is missing column(s): {missing}.")
        subset = frame[columns]

        if categorical_features is None:
            categorical_features = [
                str(c)
                for c in columns
                if not pd.api.types.is_numeric_dtype(subset[c])
                or isinstance(subset[c].dtype, pd.CategoricalDtype)
            ]

        # Mixed dtypes force an object array; homogeneous numerics stay numeric.
        values = subset.to_numpy()
        super().__init__(
            values,
            [str(c) for c in columns],
            categorical_features=categorical_features,
            entity_ids=list(frame.index),
        )
        self._frame_columns = columns

    def to_frame(self) -> pd.DataFrame:
        """Back to a DataFrame, with the original index."""
        flat = self._values[:, 0, :]
        return pd.DataFrame(flat, columns=self._feature_names, index=self._entity_ids)


class TextAdapter(BaseAdapter):
    """Adapter over a collection of text documents.

    Text has no feature columns, so the canonical array is ``(n_documents, 1, 1)`` of
    Python strings and the single feature is named ``text`` by default. That keeps the
    shape uniform — a sensitivity map over token position can still address it — while
    :meth:`texts` gives text perturbations the accessor they actually want.
    """

    _modality = Modality.TEXT
    _is_sequential = False

    def __init__(
        self,
        texts: Sequence[str],
        *,
        feature_name: str = "text",
        entity_ids: Sequence[object] | None = None,
    ) -> None:
        items = list(texts)
        if any(not isinstance(t, str) for t in items):
            offenders = [type(t).__name__ for t in items if not isinstance(t, str)][:3]
            raise TypeError(f"TextAdapter expects strings; found {offenders}.")
        array = np.empty((len(items), 1, 1), dtype=object)
        for i, text in enumerate(items):
            array[i, 0, 0] = text
        super().__init__(
            array,
            [feature_name],
            categorical_features=[],
            entity_ids=entity_ids,
        )

    def texts(self) -> list[str]:
        """The documents, in order."""
        return [str(self._values[i, 0, 0]) for i in range(self.n_entities)]

    def with_texts(self, texts: Sequence[str]) -> TextAdapter:
        """A new adapter over replacement documents of the same count."""
        items = list(texts)
        if len(items) != self.n_entities:
            raise ValueError(
                f"with_texts expects {self.n_entities} document(s); got {len(items)}. "
                f"Use TextAdapter(...) directly to change the count."
            )
        return TextAdapter(
            items,
            feature_name=self._feature_names[0],
            entity_ids=self._entity_ids,
        )
