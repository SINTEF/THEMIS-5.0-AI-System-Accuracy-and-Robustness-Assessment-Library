"""Refuse to overwrite a generated notebook that has been edited by hand.

The notebooks in this directory are generated (docs/DECISIONS.md, D57), which is what
keeps their prose honest about their own outputs — and what makes them easy to destroy. A
rebuild rewrites every cell, including markdown someone has improved by hand *in the
notebook*, and there is no undo: these files are untracked, Jupyter's checkpoints only
exist if Jupyter wrote them, and VS Code's local history skips anything over its 256 kB
default. That is not hypothetical. It happened, and the text was gone.

So a builder calls :func:`refuse_if_hand_edited` before it writes anything. It compares
the cells already in the target against the ones about to replace them, prints a diff of
exactly what would be lost, and stops.

The resolution is not ``--force``. It is to move the improved text **into the builder**,
which is a tracked, diffable file that no rebuild can clobber — and then the next build
reports no difference because there is none. ``--force`` exists for the case where the
notebook's version is the one you want to discard, and it says so out loud.
"""

from __future__ import annotations

import difflib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nbformat as nbf

__all__ = ["cell_sources", "differences", "refuse_if_hand_edited"]


def cell_sources(cells: Sequence[Any], kind: str) -> list[str]:
    """The source of every ``kind`` cell, as text, in order.

    ``nbformat`` stores a source as either a string or a list of lines depending on how
    the notebook was written, so both are flattened here rather than at each call site.
    """
    out = []
    for cell in cells:
        if cell.get("cell_type") != kind:
            continue
        source = cell.get("source", "")
        out.append("".join(source) if isinstance(source, list) else source)
    return out


def differences(
    existing: Sequence[Any], proposed: Sequence[Any], kind: str
) -> list[str]:
    """Unified diffs, one per ``kind`` cell whose text would change.

    Compared position by position and stripped of surrounding blank lines, so a cell that
    merely moved shows up as two changes — which is the honest reading: the builder is
    about to put different text in both places.
    """
    before, after = cell_sources(existing, kind), cell_sources(proposed, kind)
    out = []
    for index in range(max(len(before), len(after))):
        mine = before[index].strip() if index < len(before) else ""
        theirs = after[index].strip() if index < len(after) else ""
        if mine == theirs:
            continue
        diff = difflib.unified_diff(
            mine.splitlines(),
            theirs.splitlines(),
            fromfile=f"{kind} cell {index} — in the notebook (yours)",
            tofile=f"{kind} cell {index} — in the builder (about to be written)",
            lineterm="",
        )
        out.append("\n".join(diff))
    return out


def refuse_if_hand_edited(
    target: Path, cells: Sequence[Any], *, force: bool = False
) -> None:
    """Stop the build if ``target`` holds text the builder is about to replace.

    Raises ``SystemExit`` with a non-zero status, so a builder needs no error handling
    around it and a CI job fails rather than silently discarding someone's work.
    """
    if not target.exists():
        return

    existing = nbf.read(target, as_version=4).cells
    found = {kind: differences(existing, cells, kind) for kind in ("markdown", "code")}
    total = sum(len(v) for v in found.values())
    if not total:
        print(f"guard: {target.name} matches the builder — nothing to lose.")
        return

    counts = ", ".join(f"{len(v)} {k}" for k, v in found.items() if v)
    if force:
        print(f"guard: --force — discarding hand edits in {counts} cell(s) of {target.name}.")
        return

    print(f"\n{'=' * 78}")
    print(f"REFUSING TO WRITE {target.name}: it differs from the builder in {counts} cell(s).")
    print(f"{'=' * 78}\n")
    for kind in ("markdown", "code"):
        for diff in found[kind]:
            print(diff)
            print()
    builder = Path(sys.argv[0]).name or "the builder"
    print(
        f"Nothing has been written. Those edits exist only in {target.name}, which is\n"
        f"untracked and has no undo — so copy the text you want to keep into {builder},\n"
        "where it is tracked and the next build reproduces it rather than fighting it.\n"
        "Pass --force only if the notebook's version is the one you mean to lose."
    )
    raise SystemExit(2)
