"""Map a metric value onto [0, 1] where 1 is always best.

Needed wherever metrics of different kinds are shown together — the polar chart, the
composite reliability score (survey Eq. 27, which requires each component "normalized
to [0,1]"), and any ranking across metrics.

Recovered from ``themis_valencia/algorithms/utils/utils.py``. Two changes:

* That version did ``best = spec.best if spec.best != None else best``, silently
  discarding a ``best`` the caller passed whenever the spec declared one, and then
  used the same value as the min-max upper bound. Here ``target`` (the ideal value)
  and ``upper_bound`` (the normalisation bound) are separate fields and separate
  arguments, so neither can shadow the other.
* Symmetric metrics with ``lower_bound`` set no longer fall into the min-max branch
  and produce a meaningless number; the branch order tests ``target`` first.
"""

from __future__ import annotations

import math

from themis_assess.metrics.registry import MetricRegistry, MetricSpec

__all__ = ["normalize_metric", "normalize_metrics"]

_DEFAULT_SCALE = 1.0


def normalize_metric(
    value: float,
    metric_name: str,
    *,
    registry: MetricRegistry,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    scale: float | None = None,
) -> float:
    """Normalise ``value`` to ``[0, 1]``, 1 = best, using the registry's semantics.

    Explicit ``lower_bound`` / ``upper_bound`` / ``scale`` arguments override the
    registered spec; they do not merge with it.

    The branch taken depends on what the spec declares, in this order:

    1. **Symmetric** (``target`` set, no direction) — exponential decay in the
       absolute distance from the target: ``exp(-|value - target| / scale)``.
       This is the ``mean_signed_error`` case, best at 0 and worse either side.
    2. **Bounded** (both bounds known) — min-max, flipped if lower is better.
    3. **Lower-is-better, unbounded** — ``exp(-value / scale)``.
    4. **Higher-is-better, unbounded** — ``sigmoid(value / scale)``.

    Cases 3 and 4 are scale-dependent by construction: an unbounded metric has no
    intrinsic notion of "how good is good", so ``scale`` sets it. Register a ``scale``
    on the spec for any unbounded metric you intend to normalise, or the default of
    1.0 will silently decide it for you.
    """
    spec: MetricSpec = registry.get(metric_name)

    lb = spec.lower_bound if lower_bound is None else lower_bound
    ub = spec.upper_bound if upper_bound is None else upper_bound
    sc = scale if scale is not None else (spec.scale if spec.scale is not None else _DEFAULT_SCALE)
    if sc <= 0:
        raise ValueError(f"scale must be positive, got {sc}.")

    # 1. Symmetric around a target value.
    if spec.higher_is_better is None:
        assert spec.target is not None  # guaranteed by MetricSpec.__post_init__
        score = math.exp(-abs(value - spec.target) / sc)

    # 2. Both bounds known -> min-max, oriented so 1 is best.
    elif lb is not None and ub is not None:
        if ub == lb:
            raise ValueError(
                f"Cannot normalise '{metric_name}': lower and upper bound are both {lb}."
            )
        score = (value - lb) / (ub - lb)
        if not spec.higher_is_better:
            score = 1.0 - score

    # 3. Lower is better, unbounded above -> decay.
    elif not spec.higher_is_better:
        score = math.exp(-value / sc)

    # 4. Higher is better, unbounded -> squash.
    else:
        score = 1.0 / (1.0 + math.exp(-value / sc))

    return max(0.0, min(1.0, score))


def normalize_metrics(
    values: dict[str, float],
    *,
    registry: MetricRegistry,
    scale: float | None = None,
) -> dict[str, float]:
    """Normalise a whole dict of metric values, preserving key order."""
    return {
        name: normalize_metric(value, name, registry=registry, scale=scale)
        for name, value in values.items()
    }
