"""Shared fixtures for the whole suite.

There is one thing here, and it exists because four tests spawn a child interpreter.

Those four defend properties a same-process test cannot see: that a derived seed is
stable *between* runs (an earlier revision used ``hash()``, which randomises string
hashing per process, so a recorded seed did not reproduce a sweep), and that importing the
core does not pull in an optional dependency. Both need a fresh interpreter.

A child interpreter, though, does not inherit pytest's ``pythonpath`` setting — so on the
"explore it without installing the package" path (``pip install -r requirements-dev.txt``
then ``pytest``) those four failed with ``No module named 'themis_assess'`` while every
other test passed. The failure was in the test harness, not the library, which is the
worst kind: it looks like the property is broken.

:func:`run_child` puts this repository's ``src/`` at the front of the child's
``PYTHONPATH``, so the child imports the same source tree as the parent whether or not
anything is installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def child_env() -> dict[str, str]:
    """The parent environment with this repository's ``src/`` prepended to PYTHONPATH."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    )
    return env


def run_child(code: str, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter that can import the package.

    Parameters
    ----------
    check:
        Raise on a non-zero exit. Left ``False`` by default because most callers want to
        assert on ``returncode`` with ``stderr`` in the message, which is a far more
        useful failure than a traceback from ``check=True``.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=check,
        env=child_env(),
    )


@pytest.fixture
def child_python():
    """:func:`run_child`, as a fixture, for tests that prefer to request it."""
    return run_child
