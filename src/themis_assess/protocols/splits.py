"""Split protocols — how a shift is introduced. Survey §3.1.

The survey organises robustness evaluation along two dimensions: *how shifts are
introduced* and *how robustness is measured*. The perturbation layer covers the
synthetic half of the first; this module covers the other half — shifts that already
exist in the data, exposed by choosing which samples go where.

Four protocols, as Figure 1 of the survey names them:

``domain_split``         disjoint domains — institution, device, location, protocol
``temporal_split``       a cutoff in time; train before, test after
``subpopulation_split``  the same groups on both sides, in different proportions
``compositional_split``  disjoint *subclasses* of the same superclasses

Plus the near/far grading the survey recommends for reporting, and the resampling that
repo C implemented as a perturbation but which is really a subpopulation protocol.

Every function returns index arrays rather than data, so a protocol never has to know
what the data looks like and the same protocol serves darts series, DataFrames and text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Split",
    "domain_split",
    "temporal_split",
    "subpopulation_split",
    "compositional_split",
    "near_far_split",
    "resample_subsets",
]


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition, by positional index.

    Parameters
    ----------
    name:
        What distinguishes this split — the held-out domain, the cutoff, the mixture.
    train, test:
        Positional indices. Disjoint by construction; they need not cover every sample,
        since a compositional split can legitimately discard some.
    protocol:
        Which protocol produced it, for the report.
    metadata:
        Protocol-specific detail: the cutoff timestamp, the group weights, which
        subclasses were held out. The survey requires a robustness claim to state what
        was varied, and this is where that lives.
    """

    name: str
    train: NDArray[np.int_]
    test: NDArray[np.int_]
    protocol: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train, self.test)
        if overlap.size:
            raise ValueError(
                f"Split {self.name!r} has {overlap.size} sample(s) in both train and "
                f"test. A leaked sample makes the shifted evaluation meaningless."
            )

    @property
    def n_train(self) -> int:
        return int(self.train.size)

    @property
    def n_test(self) -> int:
        return int(self.test.size)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Split({self.name!r}, protocol={self.protocol!r}, "
            f"train={self.n_train}, test={self.n_test})"
        )


def _as_labels(values: ArrayLike, what: str) -> NDArray:
    array = np.asarray(values).ravel()
    if array.size == 0:
        raise ValueError(f"{what} is empty.")
    return array


def _check_non_empty(split: Split, *, minimum: int = 1) -> Split:
    if split.n_train < minimum or split.n_test < minimum:
        raise ValueError(
            f"Split {split.name!r} has {split.n_train} training and {split.n_test} test "
            f"sample(s); at least {minimum} of each is required. A split with an empty "
            f"side produces a metric computed on nothing."
        )
    return split


# --------------------------------------------------------------------------
# domain
# --------------------------------------------------------------------------


def domain_split(
    domains: ArrayLike,
    *,
    held_out: Sequence[Any] | None = None,
    leave_one_out: bool = False,
) -> list[Split]:
    """Train on some domains, test on disjoint ones. Survey §3.1, protocol 1.

    The protocol for "will this work at another hospital, on another device, in another
    city". ``D_train`` and ``D_test`` share no domain, so nothing the model saw in
    training can be memorised about the test environment.

    Parameters
    ----------
    domains:
        Per-sample domain label.
    held_out:
        Which domains form the test side. One split is returned.
    leave_one_out:
        Instead, return one split per domain, each holding that domain out. This is
        what makes a per-domain breakdown possible, and the survey asks for aggregate
        scores to be accompanied by exactly that.
    """
    labels = _as_labels(domains, "domains")
    unique = list(dict.fromkeys(labels.tolist()))  # preserve first-seen order
    if len(unique) < 2:
        raise ValueError(
            f"A domain split needs at least 2 domains; found {unique}. With one domain "
            f"there is no shift to measure."
        )

    if leave_one_out:
        if held_out is not None:
            raise ValueError("Pass either held_out or leave_one_out, not both.")
        targets: list[list[Any]] = [[d] for d in unique]
    else:
        if held_out is None:
            raise ValueError(
                "domain_split needs either held_out (which domains to test on) or "
                "leave_one_out=True."
            )
        unknown = [d for d in held_out if d not in unique]
        if unknown:
            raise ValueError(f"held_out names domain(s) not present: {unknown}.")
        targets = [list(held_out)]

    splits = []
    for target in targets:
        mask = np.isin(labels, target)
        split = Split(
            name=f"domain={'+'.join(map(str, target))}",
            train=np.flatnonzero(~mask),
            test=np.flatnonzero(mask),
            protocol="domain",
            metadata={
                "held_out_domains": [str(d) for d in target],
                "train_domains": [str(d) for d in unique if d not in target],
            },
        )
        splits.append(_check_non_empty(split))
    return splits


# --------------------------------------------------------------------------
# temporal
# --------------------------------------------------------------------------


def temporal_split(
    timestamps: ArrayLike,
    *,
    cutoff: Any = None,
    test_fraction: float | None = None,
    gap: int = 0,
) -> list[Split]:
    """Train before a cutoff, test after. Survey §3.1, protocol 2.

    The protocol for forecasting, and the one that most often exposes a number inflated
    by a random split. The survey is specific that nothing may straddle the boundary —
    "with no sample or time window crossing the boundary" — which is what ``gap``
    enforces for windowed models.

    Parameters
    ----------
    timestamps:
        Per-sample time. Anything comparable and sortable: datetimes, integers, floats.
    cutoff:
        Samples strictly before it train, at-or-after test. Mutually exclusive with
        ``test_fraction``.
    test_fraction:
        Alternatively, take the latest fraction as the test side.
    gap:
        Samples to discard immediately after the cutoff. For a model reading an
        ``input_chunk_length``-step window, set this to that length: otherwise the first
        test windows reach back across the boundary into training data, and the split
        leaks.
    """
    times = _as_labels(timestamps, "timestamps")
    if (cutoff is None) == (test_fraction is None):
        raise ValueError("Pass exactly one of cutoff or test_fraction.")
    if gap < 0:
        raise ValueError(f"gap must be non-negative; got {gap}.")

    order = np.argsort(times, kind="stable")
    if test_fraction is not None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}.")
        boundary = int(round((1.0 - test_fraction) * times.size))
        cutoff_value = times[order[min(boundary, times.size - 1)]]
    else:
        cutoff_value = cutoff

    before = times < cutoff_value
    train = np.flatnonzero(before)
    after = np.flatnonzero(~before)

    dropped: NDArray[np.int_] = np.array([], dtype=int)
    if gap:
        after_sorted = after[np.argsort(times[after], kind="stable")]
        dropped = after_sorted[:gap]
        after = np.setdiff1d(after, dropped)

    split = Split(
        name=f"after={cutoff_value}",
        train=train,
        test=after,
        protocol="temporal",
        metadata={
            "cutoff": str(cutoff_value),
            "gap": gap,
            "n_dropped_to_gap": int(dropped.size),
        },
    )
    return [_check_non_empty(split)]


# --------------------------------------------------------------------------
# subpopulation
# --------------------------------------------------------------------------


def subpopulation_split(
    groups: ArrayLike,
    *,
    test_weights: Mapping[Any, float],
    train_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[Split]:
    """Same groups on both sides, different proportions. Survey §3.1, protocol 3.

    The survey's phrasing: this "changes the *mixture* over groups"; every group appears
    on both sides but in different proportions, so ``P(X|Y)`` shifts through the group
    weights ``pi(g)``.

    Distinct from a domain split, and the distinction matters: nothing here is unseen.
    The model has training examples of every group, so a failure is not about
    unfamiliarity but about a decision rule tuned to the training *mixture*.

    Parameters
    ----------
    groups:
        Per-sample group label.
    test_weights:
        Target proportions for the test side, ``{group: weight}``. Normalised to sum
        to 1. The test set is the largest sample matching them.
    train_fraction:
        Fraction of each group reserved for training before the test mixture is drawn
        from the remainder.
    rng:
        Generator, for reproducibility.
    """
    labels = _as_labels(groups, "groups")
    gen = np.random.default_rng() if rng is None else rng
    unique = list(dict.fromkeys(labels.tolist()))
    if len(unique) < 2:
        raise ValueError(f"A subpopulation split needs at least 2 groups; found {unique}.")
    unknown = [g for g in test_weights if g not in unique]
    if unknown:
        raise ValueError(f"test_weights names group(s) not present: {unknown}.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must lie in (0, 1); got {train_fraction}.")
    total = sum(test_weights.values())
    if total <= 0:
        raise ValueError("test_weights must sum to a positive number.")
    weights = {g: w / total for g, w in test_weights.items()}

    train_parts, pool = [], {}
    for group in unique:
        members = np.flatnonzero(labels == group)
        gen.shuffle(members)
        cut = int(round(train_fraction * members.size))
        train_parts.append(members[:cut])
        pool[group] = members[cut:]

    # Largest test set matching the requested mixture exactly.
    capacity = min(
        (pool[g].size / w for g, w in weights.items() if w > 0), default=0.0
    )
    test_parts = [
        pool[g][: int(np.floor(w * capacity))] for g, w in weights.items() if w > 0
    ]

    split = Split(
        name="mixture=" + ",".join(f"{g}:{w:.2f}" for g, w in weights.items()),
        train=np.sort(np.concatenate(train_parts)),
        test=np.sort(np.concatenate(test_parts)) if test_parts else np.array([], int),
        protocol="subpopulation",
        metadata={
            "test_weights": {str(g): float(w) for g, w in weights.items()},
            "train_fraction": train_fraction,
            "groups": [str(g) for g in unique],
        },
    )
    return [_check_non_empty(split)]


# --------------------------------------------------------------------------
# compositional
# --------------------------------------------------------------------------


def compositional_split(
    subclasses: ArrayLike,
    superclasses: ArrayLike,
    *,
    test_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[Split]:
    """Disjoint subclasses of the same superclasses. Survey §3.1, protocol 4.

    Train on some subclasses of each superclass and test on others, so ``P(X|Y)``
    changes while the labels stay the same — the BREEDS construction. Train on terriers
    and spaniels, test on retrievers, and the label is still "dog".

    The hardest of the four to satisfy accidentally, and the closest to what
    generalisation is supposed to mean: the test examples are of a kind never seen, yet
    the correct answer is one the model has been trained to give.

    Parameters
    ----------
    subclasses, superclasses:
        Per-sample, aligned. Each subclass must belong to exactly one superclass.
    test_fraction:
        Fraction of each superclass's subclasses held out.
    """
    sub = _as_labels(subclasses, "subclasses")
    sup = _as_labels(superclasses, "superclasses")
    if sub.size != sup.size:
        raise ValueError(
            f"subclasses and superclasses must align; got {sub.size} and {sup.size}."
        )
    gen = np.random.default_rng() if rng is None else rng
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must lie in (0, 1); got {test_fraction}.")

    mapping: dict[Any, set[Any]] = {}
    for s, p in zip(sub.tolist(), sup.tolist(), strict=True):
        mapping.setdefault(s, set()).add(p)
    ambiguous = {s: sorted(map(str, p)) for s, p in mapping.items() if len(p) > 1}
    if ambiguous:
        raise ValueError(
            f"Each subclass must belong to exactly one superclass; these do not: "
            f"{ambiguous}."
        )

    held_out: list[Any] = []
    for superclass in dict.fromkeys(sup.tolist()):
        members = [s for s, p in mapping.items() if next(iter(p)) == superclass]
        if len(members) < 2:
            raise ValueError(
                f"Superclass {superclass!r} has {len(members)} subclass(es). A "
                f"compositional split needs at least 2, so that some can be held out "
                f"while the superclass is still represented in training."
            )
        order = gen.permutation(len(members))
        # Floor, not round: "hold out this fraction" should never hold out *more* than
        # the fraction. With 3 subclasses at 0.5, round() would hold out 2 and leave a
        # single subclass to train on.
        count = max(1, min(int(np.floor(test_fraction * len(members))), len(members) - 1))
        held_out.extend(members[i] for i in order[:count])

    mask = np.isin(sub, held_out)
    split = Split(
        name=f"held_out_subclasses={len(held_out)}",
        train=np.flatnonzero(~mask),
        test=np.flatnonzero(mask),
        protocol="compositional",
        metadata={
            "held_out_subclasses": sorted(map(str, held_out)),
            "n_superclasses": len(set(sup.tolist())),
        },
    )
    return [_check_non_empty(split)]


# --------------------------------------------------------------------------
# near / far grading
# --------------------------------------------------------------------------


def near_far_split(
    split: Split,
    shift: ArrayLike,
    *,
    quantile: float = 0.5,
) -> tuple[Split, Split]:
    """Grade an existing split's test side into near-OOD and far-OOD. Survey §3.

    The survey's recommendation for reporting discipline: validate on the near split and
    reserve the far one for final reporting, which gives "graded difficulty and mitigates
    overfitting to the test set".

    Parameters
    ----------
    split:
        Any split from the protocols above. Its training side is shared by both halves.
    shift:
        Per-sample shift magnitude, over *all* samples. Anything ordered will do; the
        §5.3 distances are the principled source, computed per domain or per time bin.
    quantile:
        Where to cut. 0.5 gives equal halves of the test side.

    Returns
    -------
    tuple
        ``(near, far)``.
    """
    magnitudes = np.asarray(shift, dtype=float).ravel()
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must lie in (0, 1); got {quantile}.")
    if split.n_test < 2:
        raise ValueError(
            f"Cannot grade a test side of {split.n_test} sample(s) into near and far."
        )
    test_shift = magnitudes[split.test]
    threshold = float(np.quantile(test_shift, quantile))
    near_mask = test_shift <= threshold

    common = {
        "train": split.train,
        "protocol": f"{split.protocol}+near_far",
    }
    near = Split(
        name=f"{split.name}|near",
        test=split.test[near_mask],
        metadata={**split.metadata, "grading": "near", "shift_threshold": threshold},
        **common,
    )
    far = Split(
        name=f"{split.name}|far",
        test=split.test[~near_mask],
        metadata={**split.metadata, "grading": "far", "shift_threshold": threshold},
        **common,
    )
    return _check_non_empty(near), _check_non_empty(far)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------


def resample_subsets(
    n_samples: int,
    *,
    n_resamples: int = 100,
    fraction: float = 0.8,
    replace: bool = True,
    stratify_by: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> list[NDArray[np.int_]]:
    """Repeated evaluation subsets, for a spread rather than a point estimate.

    Repo C implemented this twice — ``class_imbalance_sensitivity`` in
    ``RobustnessEvaluator`` and ``compute_robustness_against_resamplings`` for Plovdiv —
    both filed under robustness perturbations. It is not a perturbation: no input value
    changes. It is a subpopulation protocol, which is why it lives here.

    What it measures, in the Plovdiv notebook's own words: "The less variation (spread),
    the more robust the model is to resampling of test data." A single test-set metric is
    one draw from a distribution, and a model whose accuracy ranges from 0.6 to 0.95
    across resamples is not the same as one that sits at 0.79 every time, even when both
    average to 0.79.

    Parameters
    ----------
    n_resamples, fraction, replace:
        Repo C used 1000 resamples at 0.8 with replacement.
    stratify_by:
        Per-sample group label. When given, each subset preserves the group proportions,
        so the spread measures sampling noise rather than mixture drift. Leave it out —
        as repo C did — to let the mixture vary too, which is the class-imbalance
        sensitivity question.

    Returns
    -------
    list
        ``n_resamples`` index arrays.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive; got {n_samples}.")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1; got {n_resamples}.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must lie in (0, 1]; got {fraction}.")
    gen = np.random.default_rng() if rng is None else rng

    if stratify_by is None:
        size = max(1, int(round(fraction * n_samples)))
        return [
            np.sort(gen.choice(n_samples, size=size, replace=replace))
            for _ in range(n_resamples)
        ]

    labels = _as_labels(stratify_by, "stratify_by")
    if labels.size != n_samples:
        raise ValueError(
            f"stratify_by has {labels.size} label(s) for {n_samples} samples."
        )
    by_group = {g: np.flatnonzero(labels == g) for g in dict.fromkeys(labels.tolist())}
    out = []
    for _ in range(n_resamples):
        parts = []
        for members in by_group.values():
            size = max(1, int(round(fraction * members.size)))
            parts.append(gen.choice(members, size=size, replace=replace))
        out.append(np.sort(np.concatenate(parts)))
    return out
