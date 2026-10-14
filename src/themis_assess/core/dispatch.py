"""Calling metric functions with whatever inputs happen to be available.

Ported from repo D's ``RobustnessAssessor._missing_metric_args`` /
``_compute_metrics``, and from ``ClassificationAccuracyAssessment._call_metric`` on repo
A's ``main``. Both had independently arrived at the same idea: inspect a metric's
signature, supply the arguments it asks for from a pool of available values, and skip the
metrics whose inputs are not there.

This complements — it does not replace — the declared
:class:`~themis_assess.core.types.AccessLevel` on each metric spec. The declaration is
what lets the runner tell the user *before* a long evaluation which metrics it will not
be able to compute; this module is what dispatches each call once the run is under way.
One is a promise, the other is the mechanics.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "required_parameters",
    "optional_parameters",
    "missing_arguments",
    "can_call",
    "call_with_available",
    "compute_available",
    "unavailable",
]

_SELF_NAMES = frozenset({"self", "cls"})


def _parameters(fn: Callable[..., Any]) -> Mapping[str, inspect.Parameter]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Cannot inspect the signature of {fn!r}. Wrap it in a plain function so "
            f"its parameters can be discovered."
        ) from exc
    return signature.parameters


def _accepts_var_keyword(parameters: Mapping[str, inspect.Parameter]) -> bool:
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def required_parameters(fn: Callable[..., Any]) -> list[str]:
    """Names of parameters that have no default and must therefore be supplied.

    ``*args`` and ``**kwargs`` are not requirements, and ``self``/``cls`` are skipped so
    bound and unbound methods behave the same.
    """
    return [
        name
        for name, p in _parameters(fn).items()
        if name not in _SELF_NAMES
        and p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def optional_parameters(fn: Callable[..., Any]) -> list[str]:
    """Names of parameters that have a default and may be supplied."""
    return [
        name
        for name, p in _parameters(fn).items()
        if name not in _SELF_NAMES
        and p.default is not inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def missing_arguments(fn: Callable[..., Any], available: Mapping[str, Any]) -> list[str]:
    """Required parameters of ``fn`` that ``available`` cannot supply."""
    return [name for name in required_parameters(fn) if name not in available]


def can_call(fn: Callable[..., Any], available: Mapping[str, Any]) -> bool:
    """Whether ``available`` covers every required parameter of ``fn``."""
    return not missing_arguments(fn, available)


def call_with_available(fn: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Call ``fn`` with the subset of ``available`` that it actually accepts.

    A function declaring ``**kwargs`` receives everything, since it has said it can take
    it. Anything else receives only its named parameters, so an unexpected key in the
    pool does not become a ``TypeError``.

    Raises
    ------
    TypeError
        If a required parameter is missing, naming which — the failure mode this is
        meant to make legible.
    """
    missing = missing_arguments(fn, available)
    if missing:
        name = getattr(fn, "__name__", repr(fn))
        raise TypeError(
            f"Cannot call {name}: missing required argument(s) {missing}. "
            f"Available: {sorted(available)}."
        )
    parameters = _parameters(fn)
    if _accepts_var_keyword(parameters):
        kwargs = dict(available)
        kwargs.pop("self", None)
        kwargs.pop("cls", None)
    else:
        accepted = set(required_parameters(fn)) | set(optional_parameters(fn))
        kwargs = {k: v for k, v in available.items() if k in accepted}
    return fn(**kwargs)


def compute_available(
    functions: Mapping[str, Callable[..., Any]],
    available: Mapping[str, Any],
    *,
    skip_unavailable: bool = False,
) -> dict[str, Any]:
    """Evaluate every function whose inputs are present.

    Parameters
    ----------
    functions:
        ``{result_name: callable}``.
    available:
        Pool of values to draw arguments from.
    skip_unavailable:
        When ``True``, functions with missing inputs are silently omitted from the
        result. When ``False`` (the default), the first such function raises.

        The default is strict on purpose. Silently omitting a metric produces a report
        that is quietly incomplete, and a reader has no way to tell an absent metric
        from an unmeasured one. Set this ``True`` only where the caller then reports
        which metrics were skipped and why.

    Returns
    -------
    dict
        ``{result_name: value}``, in the order the functions were given.
    """
    out: dict[str, Any] = {}
    for name, fn in functions.items():
        if skip_unavailable and not can_call(fn, available):
            continue
        out[name] = call_with_available(fn, available)
    return out


def unavailable(
    functions: Mapping[str, Callable[..., Any]],
    available: Mapping[str, Any],
) -> dict[str, list[str]]:
    """``{result_name: missing argument names}`` for every function that cannot be called.

    The companion to ``skip_unavailable=True``: use it to say what was skipped.
    """
    out: dict[str, list[str]] = {}
    for name, fn in functions.items():
        missing = missing_arguments(fn, available)
        if missing:
            out[name] = missing
    return out
