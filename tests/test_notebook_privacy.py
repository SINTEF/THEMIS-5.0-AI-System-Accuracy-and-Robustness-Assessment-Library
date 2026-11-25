"""No committed notebook may carry an absolute local path.

The notebooks are committed with their outputs and published to GitHub Pages, so a path
in a cell output is on the open web — and the usual way one gets there is not a mistake
anyone would notice while making it. A dependency raises a warning, Python prints it with
the full path of the file that raised, the cell captures it as stderr, and the notebook is
committed with a username, an operating system and a directory layout inside it.

That happened: `docs/tutorial_classification.ipynb` published a scipy `RuntimeWarning`
prefixed with a home directory, and it was found by a reader of the public site rather
than by anything here. This test is the thing that should have found it.

The fix is `python tools/notebook_privacy.py`, which redacts the path and keeps the
warning — the warning is information, the path is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import notebook_privacy  # noqa: E402

NOTEBOOKS = notebook_privacy.notebooks()


def test_there_are_notebooks_to_check():
    """Guards the test below: a glob that matches nothing passes vacuously."""
    assert NOTEBOOKS, f"no notebooks found under {notebook_privacy.DOCS}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_output_carries_an_absolute_local_path(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    found = notebook_privacy.offenders(notebook)
    assert not found, (
        f"{path.name} publishes {len(found)} line(s) containing an absolute path:\n"
        + "\n".join(f"  {line[:200]}" for line in found[:5])
        + "\n\nRun: python tools/notebook_privacy.py"
    )


def test_redaction_keeps_the_message_and_drops_the_machine():
    line = (
        r"c:\Users\someone\AppData\Local\envs\x\Lib\site-packages\scipy\stats\mod.py:601"
        ": RuntimeWarning: Precision loss occurred."
    )
    cleaned = notebook_privacy.redact(line)
    assert "someone" not in cleaned
    assert "AppData" not in cleaned
    assert cleaned.startswith(".../scipy/stats/mod.py:601")
    assert "RuntimeWarning: Precision loss occurred." in cleaned


def test_a_posix_home_directory_is_redacted_too():
    cleaned = notebook_privacy.redact("/home/someone/envs/x/lib/pkg/mod.py:12: Warning")
    assert "someone" not in cleaned
    assert cleaned.startswith(".../lib/pkg/mod.py:12")


def test_a_relative_path_is_left_alone():
    """The notebooks refer to `docs/tutorial.ipynb` constantly; that is not a leak."""
    for text in ("docs/tutorial_classification.ipynb", "src/themis_assess/viz/theme.py"):
        assert notebook_privacy.redact(text) == text


def test_minified_javascript_is_not_mistaken_for_a_path():
    """A plotly figure embeds plotly.js, which is full of path-shaped substrings.

    `text/html` is excluded from the scan for exactly this reason; the check would
    otherwise fail on every notebook containing a figure.
    """
    notebook = {
        "cells": [
            {
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"text/html": ['e.csv=e.dsv(",","text/csv"),x:\\a\\b\\c']},
                    }
                ]
            }
        ]
    }
    assert notebook_privacy.offenders(notebook) == []
