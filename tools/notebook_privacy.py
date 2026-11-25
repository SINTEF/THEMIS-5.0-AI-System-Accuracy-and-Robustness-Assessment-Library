"""Keep local filesystem paths out of the notebooks, which are published.

A cell output can carry the absolute path of the machine that executed it — most often a
warning raised from inside a dependency, which Python prints with the full path of the
file that raised it:

    c:\\Users\\<name>\\AppData\\Local\\...\\site-packages\\scipy\\stats\\_axis_nan_policy.py:601:
    RuntimeWarning: Precision loss occurred in moment calculation ...

The notebooks are committed and published to GitHub Pages, so that line puts a username,
an operating system and a directory layout on the open web. It tells a reader nothing
they need: the warning is worth keeping, the path to the installed dependency is not.

So this module *redacts* rather than deletes. ``.../scipy/stats/_axis_nan_policy.py:601``
says as much about where the warning came from as the absolute path did. Dropping the
warning entirely would be the dishonest fix — a warning that says "results may be
unreliable" is information the reader is entitled to.

    python tools/notebook_privacy.py                # redact docs/*.ipynb in place
    python tools/notebook_privacy.py --check        # report and exit 1; change nothing

``tests/test_notebook_privacy.py`` runs the check, so a notebook rebuilt on someone's
laptop cannot reach the published site with their home directory in it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# A Windows path with a drive letter, or a POSIX home directory. Both are anchored on
# something that cannot appear in a relative path, so this does not match the repo's own
# `docs/tutorial.ipynb`-style references.
PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/](?:[^\s\\/:*?\"<>|]+[\\/])+[^\s\\/:*?\"<>|]*"),
    re.compile(r"/(?:home|Users|root)/[^\s:,'\"]+"),
)

# How many trailing components to keep. Three is enough to name the module that raised —
# `scipy/stats/_axis_nan_policy.py` — without naming the machine.
KEEP = 3

# Only these carry warnings, tracebacks and repr output — the places a path realistically
# leaks from. `text/html` is excluded deliberately: a plotly figure embeds the whole
# minified plotly.js, and minified JavaScript is full of substrings that look like
# Windows paths to any regex simple enough to be worth having.
SCANNED_MIMES = ("text/plain",)


def _shorten(match: re.Match[str]) -> str:
    parts = [p for p in re.split(r"[\\/]", match.group(0)) if p]
    if len(parts) <= KEEP:
        return match.group(0)
    return ".../" + "/".join(parts[-KEEP:])


def redact(text: str) -> str:
    """Replace absolute local paths with their last few components."""
    for pattern in PATTERNS:
        text = pattern.sub(_shorten, text)
    return text


def _outputs_of(notebook: dict) -> list:
    return [out for cell in notebook["cells"] for out in cell.get("outputs", [])]


def scrub(notebook: dict) -> list[str]:
    """Redact every output in place. Returns the lines that were changed."""
    changed: list[str] = []

    def fix(text: str) -> str:
        cleaned = redact(text)
        if cleaned != text:
            for before, after in zip(text.splitlines(), cleaned.splitlines()):
                if before != after:
                    changed.append(after)
        return cleaned

    for out in _outputs_of(notebook):
        if "text" in out:  # a stream: stdout or stderr
            out["text"] = [fix(line) for line in out["text"]]
        for mime, payload in (out.get("data") or {}).items():
            if mime not in SCANNED_MIMES:
                continue
            if isinstance(payload, list):
                out["data"][mime] = [fix(line) for line in payload]
            else:
                out["data"][mime] = fix(payload)
        # A traceback names every frame's file, which is the worst case of all.
        if "traceback" in out:
            out["traceback"] = [fix(frame) for frame in out["traceback"]]
        if "evalue" in out:
            out["evalue"] = fix(out["evalue"])
    return changed


def offenders(notebook: dict) -> list[str]:
    """Every output line that still carries an absolute local path."""
    found = []
    for out in _outputs_of(notebook):
        chunks = list(out.get("text", [])) + list(out.get("traceback", []))
        for mime, payload in (out.get("data") or {}).items():
            if mime in SCANNED_MIMES:
                chunks += payload if isinstance(payload, list) else [payload]
        for chunk in chunks:
            for line in chunk.splitlines():
                if any(p.search(line) for p in PATTERNS):
                    found.append(line.strip())
    return found


def notebooks() -> list[Path]:
    return sorted(DOCS.glob("*.ipynb"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report and exit 1; change nothing"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="default: docs/*.ipynb")
    args = parser.parse_args(argv)

    targets = args.paths or notebooks()
    failed = False
    for path in targets:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if args.check:
            found = offenders(notebook)
            if found:
                failed = True
                print(f"{path.name}: {len(found)} output line(s) carry a local path:")
                for line in found[:5]:
                    print(f"    {line[:160]}")
            continue
        found = offenders(notebook)
        if not found:
            print(f"{path.name}: clean")
            continue

        # Rewriting the parsed notebook would re-serialise all of it: a one-line fix came
        # out as 2314 changed lines, which buries the change it was supposed to show. So
        # substitute in the raw file text instead, escaping each string the way JSON
        # already holds it. The diff is then exactly the lines that were redacted.
        text = path.read_text(encoding="utf-8")
        for line in found:
            replacement = redact(line)
            if replacement == line:
                continue
            before, after = json.dumps(line)[1:-1], json.dumps(replacement)[1:-1]
            if before not in text:
                print(f"{path.name}: could not place {line[:60]!r}", file=sys.stderr)
                failed = True
                continue
            text = text.replace(before, after)
        path.write_text(text, encoding="utf-8", newline="")

        # Parse it back, so a broken substitution fails here and not in someone's Jupyter.
        remaining = offenders(json.loads(path.read_text(encoding="utf-8")))
        if remaining:
            print(f"{path.name}: {len(remaining)} line(s) still carry a path", file=sys.stderr)
            failed = True
        else:
            print(f"{path.name}: redacted {len(found)} line(s)")
            for line in found[:5]:
                print(f"    {redact(line)[:160]}")

    if failed:
        print("\nRun `python tools/notebook_privacy.py` to redact them.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
