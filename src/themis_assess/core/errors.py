"""Exception types and the guarded-division helper used across the metric layer.

Two rules the metric modules follow:

1. A metric divides by a quantity that can legitimately be zero only where the
   survey itself introduces a stabilising ``eps`` (Equation 13). Everywhere else a
   zero denominator is a real condition the caller needs to know about, and raises
   :class:`ZeroDenominatorError` naming the quantity rather than returning ``inf``
   or ``nan``.
2. Optional dependencies are imported inside the function that needs them, through
   :func:`require_extra`, so a missing extra produces an actionable message instead
   of an ``ImportError`` at package import.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "ThemisAssessError",
    "ZeroDenominatorError",
    "MissingExtraError",
    "safe_divide",
    "require_extra",
]


class ThemisAssessError(Exception):
    """Base class for every error raised by this library."""


class ZeroDenominatorError(ThemisAssessError, ZeroDivisionError):
    """A metric's denominator was zero and the survey specifies no stabiliser.

    Subclasses :class:`ZeroDivisionError` so existing ``except ZeroDivisionError``
    handlers keep working.
    """


class MissingExtraError(ThemisAssessError, ImportError):
    """An optional dependency is needed but not installed."""


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    eps: float = 0.0,
    what: str = "denominator",
) -> float:
    """Divide, stabilising with ``eps`` if given and refusing to divide by zero.

    Parameters
    ----------
    numerator, denominator:
        The quantities to divide.
    eps:
        Added to *both* numerator and denominator, following the survey's
        Equation 13. Defaults to ``0.0``, which is the faithful reading of every
        other equation; pass a small positive value where a zero denominator is
        expected and should be absorbed rather than reported.
    what:
        Name of the denominator quantity, used in the error message.

    Raises
    ------
    ZeroDenominatorError
        If ``denominator + eps`` is zero.
    """
    den = denominator + eps
    if den == 0.0:
        raise ZeroDenominatorError(
            f"{what} is zero, so this metric is undefined here. "
            f"Pass a small eps to stabilise the ratio, or check that the clean "
            f"evaluation actually produced a non-zero value."
        )
    return (numerator + eps) / den


def require_extra(module: str, extra: str) -> ModuleType:
    """Import ``module``, or raise a message naming the extra that provides it.

    >>> darts = require_extra("darts", "timeseries")   # doctest: +SKIP
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise MissingExtraError(
            f"'{module}' is required for this feature but is not installed. "
            f"Install it with:  pip install themis-assess[{extra}]"
        ) from exc
