"""Subgroups as masks.

Repo C, on why per-group accuracy resisted generalisation:

    "I don't think this is so easy to abstract so that it is applicable to other
    use-cases. The medical use case has ohe features in a tabular format - which allow
    me to select a subset of the population very easily. But this is not the case for
    the media use case for instance."

Right about *feature indexing*, and that is the whole point of this module. A subgroup is
a function from the dataset to a boolean mask, and every mechanism across the four repos
is one:

===============================  ================================================
repo C, one-hot column           ``lambda ds: ds["insurance_medicare"] == 1``
repo D, embedding-similar theme  ``lambda ds: similarity(ds, theme) > 0.5``
repo B, callable category        ``lambda ds: category(ds) == "long_haul"``
repo B, lead-time bins           ``lambda ds: (lead_time(ds) >= 6) & (< 12)``
repo A, dict of ranges           :func:`from_criteria`
===============================  ================================================

Downstream, :mod:`themis_assess.metrics.robustness.subgroup` sees only
``{name: boolean array}`` — no feature semantics, no modality, nothing to special-case.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "SubgroupSpec",
    "resolve_masks",
    "from_criteria",
    "from_column_values",
    "complement",
]


@dataclass(frozen=True, slots=True)
class SubgroupSpec:
    """A named subgroup, defined by the mask it selects.

    Parameters
    ----------
    name:
        Label used in reports and metric keys.
    mask_fn:
        ``dataset -> boolean array`` of length ``n_samples``. What "dataset" means is
        the caller's business — a DataFrame, an adapter, a list of documents — because
        the mask function is supplied by whoever knows the data.
    description:
        Human-readable definition, for the report. Worth filling in: "age 30-40" is
        recoverable from a lambda only by reading code.
    """

    name: str
    mask_fn: Callable[[Any], NDArray[np.bool_]]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SubgroupSpec needs a non-empty name.")
        if not callable(self.mask_fn):
            raise TypeError(f"mask_fn must be callable; got {type(self.mask_fn).__name__}.")

    def mask(self, dataset: Any, *, n_samples: int | None = None) -> NDArray[np.bool_]:
        """Evaluate the mask, validating shape and dtype."""
        raw = self.mask_fn(dataset)
        mask = np.asarray(raw)
        if mask.dtype != bool:
            if mask.dtype.kind in "iuf" and set(np.unique(mask)) <= {0, 1}:
                mask = mask.astype(bool)
            else:
                raise TypeError(
                    f"Subgroup {self.name!r} produced dtype {mask.dtype}, which is "
                    f"neither boolean nor 0/1. A mask_fn must return a boolean array."
                )
        mask = mask.ravel()
        if n_samples is not None and mask.shape[0] != n_samples:
            raise ValueError(
                f"Subgroup {self.name!r} produced a mask of length {mask.shape[0]} for "
                f"{n_samples} samples."
            )
        return mask


def resolve_masks(
    specs: Iterable[SubgroupSpec],
    dataset: Any,
    *,
    n_samples: int | None = None,
    require_non_empty: bool = True,
) -> dict[str, NDArray[np.bool_]]:
    """Evaluate several specs against one dataset.

    Parameters
    ----------
    require_non_empty:
        Raise if a subgroup selects nothing. On by default: an empty group is almost
        always a mis-specified criterion rather than a finding, and silently dropping it
        makes an aggregate over "all groups" quietly wrong.

    Returns
    -------
    dict
        ``{name: mask}``, in spec order — ready for
        :func:`themis_assess.metrics.robustness.subgroup.group_scores`.
    """
    specs = list(specs)
    if not specs:
        raise ValueError("resolve_masks needs at least one SubgroupSpec.")

    seen: set[str] = set()
    out: dict[str, NDArray[np.bool_]] = {}
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate subgroup name: {spec.name!r}.")
        seen.add(spec.name)
        mask = spec.mask(dataset, n_samples=n_samples)
        if require_non_empty and not mask.any():
            raise ValueError(
                f"Subgroup {spec.name!r} selects no samples. Check the criterion, or "
                f"pass require_non_empty=False if an empty group is expected here."
            )
        out[spec.name] = mask
    return out


def from_criteria(
    name: str,
    criteria: Mapping[str, Any],
    *,
    description: str = "",
) -> SubgroupSpec:
    """Build a spec from repo A's dictionary form, over a DataFrame.

    Compiles the criteria form documented on ``ClassificationAccuracyAssessment``:

    >>> spec = from_criteria("nurses_30s", {"age": {"min": 30, "max": 40}, "job": ["nurse"]})

    Supported value forms, combined with logical AND:

    * ``{"min": a, "max": b}`` — inclusive range; either bound may be omitted
    * ``[v1, v2, ...]`` — membership
    * a scalar — equality

    Kept because the form is already documented in repo A and is genuinely convenient
    for tabular data. It is a *convenience over* the mask, not an alternative to it —
    anything it cannot express, write as a ``mask_fn``.
    """
    if not criteria:
        raise ValueError(f"Subgroup {name!r} has no criteria.")

    def mask_fn(dataset: Any) -> NDArray[np.bool_]:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"from_criteria builds a mask over a pandas DataFrame; got "
                f"{type(dataset).__name__}. For other data, supply a mask_fn directly."
            )
        missing = [c for c in criteria if c not in dataset.columns]
        if missing:
            raise ValueError(
                f"Subgroup {name!r} references column(s) not in the DataFrame: {missing}."
            )
        mask = np.ones(len(dataset), dtype=bool)
        for column, condition in criteria.items():
            values = dataset[column]
            if isinstance(condition, Mapping):
                unknown = set(condition) - {"min", "max"}
                if unknown:
                    raise ValueError(
                        f"Subgroup {name!r}, column {column!r}: unknown range key(s) "
                        f"{sorted(unknown)}. Use 'min' and/or 'max'."
                    )
                if "min" in condition:
                    mask &= (values >= condition["min"]).to_numpy()
                if "max" in condition:
                    mask &= (values <= condition["max"]).to_numpy()
            elif isinstance(condition, (list, tuple, set, frozenset)):
                mask &= values.isin(list(condition)).to_numpy()
            else:
                mask &= (values == condition).to_numpy()
        return mask

    return SubgroupSpec(
        name=name,
        mask_fn=mask_fn,
        description=description or _describe_criteria(criteria),
    )


def _describe_criteria(criteria: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for column, condition in criteria.items():
        if isinstance(condition, Mapping):
            lo = condition.get("min")
            hi = condition.get("max")
            if lo is not None and hi is not None:
                parts.append(f"{column} in [{lo}, {hi}]")
            elif lo is not None:
                parts.append(f"{column} >= {lo}")
            else:
                parts.append(f"{column} <= {hi}")
        elif isinstance(condition, (list, tuple, set, frozenset)):
            parts.append(f"{column} in {sorted(map(str, condition))}")
        else:
            parts.append(f"{column} == {condition}")
    return "; ".join(parts)


def from_column_values(
    column: str,
    *,
    values: Sequence[Any] | None = None,
    prefix: str = "",
) -> list[SubgroupSpec]:
    """One spec per distinct value of a DataFrame column.

    The common case for a categorical grouping — a class label, a site, a device.

    Parameters
    ----------
    values:
        Which values to build groups for. All observed values if omitted, which is
        resolved lazily so the specs can be built before the data is in hand.
    prefix:
        Prepended to each group name, e.g. ``prefix="site"`` gives ``"site=oslo"``.
    """
    if values is None:
        raise ValueError(
            "from_column_values needs an explicit `values` list. Enumerating the "
            "observed values requires the data, and building specs lazily would hide "
            "which groups a report actually covers — pass the values you intend to "
            "compare."
        )
    label = prefix or column
    return [
        SubgroupSpec(
            name=f"{label}={value}",
            mask_fn=_equality_mask(column, value),
            description=f"{column} == {value}",
        )
        for value in values
    ]


def _equality_mask(column: str, value: Any) -> Callable[[Any], NDArray[np.bool_]]:
    def mask_fn(dataset: Any) -> NDArray[np.bool_]:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                f"from_column_values builds masks over a pandas DataFrame; got "
                f"{type(dataset).__name__}."
            )
        if column not in dataset.columns:
            raise ValueError(f"Column {column!r} is not in the DataFrame.")
        return (dataset[column] == value).to_numpy()

    return mask_fn


def complement(spec: SubgroupSpec, *, name: str | None = None) -> SubgroupSpec:
    """The complement of a subgroup — everything it does not select.

    Useful for the two-group comparison that most disparity questions reduce to: this
    group against everyone else.
    """
    return SubgroupSpec(
        name=name or f"not({spec.name})",
        mask_fn=lambda dataset: ~spec.mask(dataset),
        description=f"complement of: {spec.description or spec.name}",
    )
