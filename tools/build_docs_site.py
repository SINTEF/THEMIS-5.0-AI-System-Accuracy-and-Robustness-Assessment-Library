"""Export the tutorial notebooks to a small static site.

The notebooks are committed *with* their outputs, so a reader who clones the repository
can open them and see every figure. This script is for the reader who has not cloned
anything: it converts each notebook to a standalone HTML page and writes an index, which
can be served from anywhere — GitHub Pages, an internal web server, or a local folder.

    python tools/build_docs_site.py              # -> site/
    python tools/build_docs_site.py --out public # for a CI job that publishes `public/`

It does **not** execute anything. The outputs it publishes are whatever is stored in the
notebook — for the two tutorials with builders in ``docs/``, put there by executing every
cell (D57); for ``safetycage_tutorial.ipynb``, which has no builder, whatever its author
last ran. A notebook that is not present is skipped with a warning rather than failing
the build, so that a CI job still publishes whatever is committed.
"""

from __future__ import annotations

import argparse
import html
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


@dataclass(frozen=True)
class Tutorial:
    notebook: str
    title: str
    blurb: str


TUTORIALS = (
    Tutorial(
        "tutorial_classification.ipynb",
        "Assessing a classifier end to end",
        "A random forest on the breast cancer Wisconsin dataset, taken through every "
        "layer of the library: accuracy against declared criteria, calibration, "
        "robustness under perturbation, error attribution, subgroup parity, and the "
        "report that collects them.",
    ),
    Tutorial(
        "forecasting_tutorial.ipynb",
        "Assessing a forecaster end to end",
        "Seven days of Spanish solar generation, forecast by XGBoost from three days of "
        "history and the weather expected over those seven days — so that “what "
        "happens when the weather forecast is wrong” becomes a robustness question "
        "with an answer someone can act on.",
    ),
    Tutorial(
        "safetycage_tutorial.ipynb",
        "Misclassification detection under distribution shift",
        "Four safetycage detectors — MSP, DOCTOR, RED and SPARDACUS — fitted to one CNN "
        "on histology tiles, tuned on the slide collection it was trained on, and then "
        "asked whether they still catch its mistakes on tiles from patients it has "
        "never seen. A random baseline says how much of that is skill.",
    ),
)

# The library's own palette, so the index does not look like it came from somewhere else.
INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>themis-assess · tutorials</title>
<style>
  :root {{
    --bg: #fcfcfb; --fg: #0b0b0b; --muted: #52514e; --line: #e6e5e1; --link: #2a78d6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #141414; --fg: #f2f1ee; --muted: #a8a6a1; --line: #2e2e2c; --link: #6aa8ea; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 48px 16px 80px; background: var(--bg); color: var(--fg);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  main {{ max-width: 720px; margin: 0 auto; }}
  h1 {{ font-size: 28px; line-height: 1.25; margin: 0 0 8px; letter-spacing: -0.01em; }}
  .lede {{ color: var(--muted); margin: 0 0 40px; }}
  a {{ color: var(--link); }}
  .card {{
    display: block; padding: 20px 22px; margin-bottom: 16px; border: 1px solid var(--line);
    border-radius: 10px; text-decoration: none; color: inherit; background: transparent;
  }}
  .card:hover {{ border-color: var(--link); }}
  .card h2 {{ font-size: 18px; margin: 0 0 6px; color: var(--link); }}
  .card p {{ margin: 0; color: var(--muted); font-size: 15px; }}
  footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line);
            color: var(--muted); font-size: 14px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.92em; }}
</style>
</head>
<body>
<main>
  <h1>themis-assess · tutorials</h1>
  <p class="lede">Worked examples, each taking one model end to end on real data. Every
  page is the notebook exactly as it was executed, figures included — nothing here
  is re-run in your browser.</p>
{cards}
  <footer>
    Part of <strong>THEMIS 5.0</strong> (Horizon Europe grant 101121042). The library,
    the decision log and the notebooks themselves are in the
    <a href="{repo}">repository</a>; the methodology is the survey the code cites
    section by section.
  </footer>
</main>
</body>
</html>
"""

CARD = """  <a class="card" href="{href}">
    <h2>{title}</h2>
    <p>{blurb}</p>
  </a>
"""

REPO = "https://github.com/SINTEF/THEMIS-5.0-AI-System-Accuracy-and-Robustness-Assessment-Library"


def _shown(path: Path) -> str:
    """The path as a reader of the log would write it — relative where that is possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def export(notebook: Path, out_dir: Path, title: str) -> Path:
    """Convert one notebook to a standalone HTML page in ``out_dir``."""
    from nbconvert import HTMLExporter
    import nbformat

    exporter = HTMLExporter()
    # "lab" keeps the notebook's own styling; the outputs are embedded as they are
    # stored, so a figure drawn by plotly stays interactive.
    exporter.template_name = "lab"
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True

    # Without this every page's tab reads "Notebook", which is useless once a reader has
    # three of them open. nbconvert takes the title from the resources metadata.
    body, _ = exporter.from_notebook_node(
        nbformat.read(notebook, as_version=4),
        resources={"metadata": {"name": title, "path": str(notebook.parent)}},
    )
    target = out_dir / (notebook.stem + ".html")
    target.write_text(body, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="site", help="directory to write (default: site/); it is replaced"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if a tutorial notebook is missing, instead of skipping it",
    )
    args = parser.parse_args(argv)

    out_dir = (ROOT / args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cards, missing = [], []
    for tutorial in TUTORIALS:
        notebook = DOCS / tutorial.notebook
        if not notebook.exists():
            missing.append(tutorial.notebook)
            print(f"skipped {tutorial.notebook}: not present", file=sys.stderr)
            continue
        page = export(notebook, out_dir, tutorial.title)
        size = page.stat().st_size / 1e6
        print(f"wrote {_shown(page)} ({size:.2f} MB)")
        cards.append(
            CARD.format(
                href=page.name,
                title=html.escape(tutorial.title),
                blurb=html.escape(tutorial.blurb),
            )
        )

    if missing and args.strict:
        print(f"missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    if not cards:
        print("no notebooks to publish", file=sys.stderr)
        return 1

    index = out_dir / "index.html"
    index.write_text(INDEX.format(cards="".join(cards), repo=REPO), encoding="utf-8")
    print(f"wrote {_shown(index)} ({len(cards)} tutorial(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
