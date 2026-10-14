"""Tests for the shared helpers: guarded division and the optional-dependency guard."""

from __future__ import annotations

import pytest

import themis_assess
from themis_assess.core.errors import (
    MissingExtraError,
    ThemisAssessError,
    ZeroDenominatorError,
    require_extra,
    safe_divide,
)


def test_safe_divide_without_eps_is_plain_division():
    assert safe_divide(1.0, 4.0) == pytest.approx(0.25)


def test_safe_divide_adds_eps_to_both_sides():
    """Following Eq. 13, where eps stabilises the ratio rather than just the denominator."""
    assert safe_divide(0.0, 0.0, eps=1e-6) == pytest.approx(1.0)
    assert safe_divide(1.0, 1.0, eps=0.5) == pytest.approx(1.0)


def test_safe_divide_raises_on_a_zero_denominator():
    with pytest.raises(ZeroDenominatorError):
        safe_divide(1.0, 0.0)


def test_zero_denominator_error_names_the_quantity():
    with pytest.raises(ZeroDenominatorError, match="clean accuracy"):
        safe_divide(1.0, 0.0, what="clean accuracy")


def test_zero_denominator_error_is_catchable_as_zero_division_error():
    """So existing `except ZeroDivisionError` handlers keep working."""
    with pytest.raises(ZeroDivisionError):
        safe_divide(1.0, 0.0)


def test_all_library_errors_share_a_base():
    assert issubclass(ZeroDenominatorError, ThemisAssessError)
    assert issubclass(MissingExtraError, ThemisAssessError)
    assert issubclass(MissingExtraError, ImportError)


def test_require_extra_returns_an_installed_module():
    assert require_extra("json", "core").dumps({"a": 1}) == '{"a": 1}'


def test_require_extra_names_the_extra_to_install():
    with pytest.raises(MissingExtraError, match=r"themis-assess\[timeseries\]"):
        require_extra("definitely_not_installed_xyz", "timeseries")


def test_top_level_exports_are_importable():
    assert themis_assess.__version__
    for name in themis_assess.__all__:
        assert hasattr(themis_assess, name), name


def test_no_core_module_imports_an_optional_dependency():
    """Core must stay importable without darts, torch, transformers or safetycage.

    Checked in a fresh interpreter rather than by inspecting this one's ``sys.modules``:
    other tests in the session legitimately import the optional extras, so a same-process
    check passes or fails depending on test ordering.
    """
    from tests.conftest import run_child

    forbidden = ("darts", "torch", "transformers", "textattack", "safetycage", "matplotlib")
    code = (
        "import sys\n"
        "import themis_assess\n"
        "import themis_assess.core\n"
        "import themis_assess.metrics.robustness\n"
        "import themis_assess.perturbations\n"
        f"forbidden = {forbidden!r}\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "leaked = sorted(set(forbidden) & loaded)\n"
        "assert not leaked, f'core pulled in: {leaked}'\n"
    )
    completed = run_child(code)
    assert completed.returncode == 0, completed.stderr
