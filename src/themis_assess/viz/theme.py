"""The one place colour and layout are decided.

Every figure in this package reads its palette and layout from here, so a project can
substitute its own house style by editing one file rather than by hunting through chart
code. That is the arrangement the four repos did not have: repo A's
``plot_confusion_matrix`` took a ``colormap="viridis"`` argument, repo C's used a
hand-written list of hex codes, repo D's plotting module carried its own, and repo B's
heatmap took a ``scale="log"`` switch — four independent colour decisions, none of them
comparable, all of them per-call.

Four rules the palette here follows, each of which one of the repos broke.

**Sequential encoding uses one hue, light to dark.** Repo A's confusion matrix defaulted
to ``viridis``, which sweeps purple through green to yellow. A multi-hue ramp makes the
*hue* look meaningful when only the lightness is, and readers reliably see boundaries in
it that are not in the data. The ramp here is a single blue.

**Diverging encoding uses two hues with a neutral grey midpoint.** For a signed quantity
— a SHAP value pushing error up or down — the midpoint must read as "nothing", which is
what rules out a hue there.

**Categorical hues are assigned in a fixed order and never cycled.** A ninth series is
not a generated colour; it folds into "other" or the chart becomes small multiples. The
eight slots below are validated for colour-vision deficiency separation as an adjacent
pairlist, which is the relevant test for lines and bars.

**Status colours are reserved and never carry meaning alone.** Pass and fail get a
symbol and a word as well as a colour, because three of the eight categorical slots and
two of the four status colours sit below 3:1 contrast on a light surface — a figure that
distinguishes pass from fail by colour alone is unreadable to some readers and invisible
in print.

Every figure builder also accepts ``template`` and returns a plain
:class:`plotly.graph_objects.Figure`, so a caller who dislikes these decisions can
override any of them afterwards rather than being stuck with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Palette",
    "DEFAULT_PALETTE",
    "apply_layout",
    "status_colour",
    "status_symbol",
]


@dataclass(frozen=True, slots=True)
class Palette:
    """Colour and layout parameters. Substitute an instance to re-skin every figure.

    Parameters
    ----------
    categorical:
        Fixed hue order for identity encoding. Assigned in order, never cycled.
    sequential:
        A single-hue ramp, lightest first, for magnitude encoding.
    diverging_low, diverging_mid, diverging_high:
        Two poles and a neutral midpoint, for signed quantities.
    status:
        Reserved for pass/fail state. Never reused as a series colour.
    """

    categorical: tuple[str, ...] = (
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    )
    sequential: tuple[str, ...] = (
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    )
    diverging_low: str = "#2a78d6"
    diverging_mid: str = "#f0efec"
    diverging_high: str = "#d03b3b"
    status: dict[str, str] = field(
        default_factory=lambda: {
            "good": "#0ca30c",
            "warning": "#fab219",
            "serious": "#ec835a",
            "critical": "#d03b3b",
        }
    )
    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    grid: str = "#e6e5e1"
    reference: str = "#52514e"

    def series(self, index: int) -> str:
        """The categorical colour for series ``index``.

        Raises past the eighth slot rather than cycling. A cycled palette means two
        series share a colour, and the reader has no way to know which two — so the
        honest options are folding the tail into "other" or facetting, and the caller
        has to choose one.
        """
        if index >= len(self.categorical):
            raise ValueError(
                f"Series {index} exceeds the {len(self.categorical)} categorical slots. "
                f"A ninth colour would repeat an earlier one, which makes two series "
                f"indistinguishable. Fold the tail into an 'other' group, or draw small "
                f"multiples."
            )
        return self.categorical[index]

    def colorscale(self) -> list[list[Any]]:
        """The sequential ramp as a plotly colorscale."""
        n = len(self.sequential)
        return [[i / (n - 1), colour] for i, colour in enumerate(self.sequential)]

    def diverging_colorscale(self) -> list[list[Any]]:
        """The diverging ramp, with the neutral midpoint at 0.5."""
        return [
            [0.0, self.diverging_low],
            [0.5, self.diverging_mid],
            [1.0, self.diverging_high],
        ]


DEFAULT_PALETTE = Palette()


def status_colour(passed: bool | None, *, palette: Palette | None = None) -> str:
    """The colour for a three-valued verdict.

    ``None`` — nothing was checked — gets ``warning`` rather than a neutral grey, because
    "no criteria supplied" is a gap in the evaluation and should draw the eye, not blend
    into the background. It is the state survey §8 calls an incomplete claim.
    """
    p = palette or DEFAULT_PALETTE
    if passed is None:
        return p.status["warning"]
    return p.status["good"] if passed else p.status["critical"]


def status_symbol(passed: bool | None) -> str:
    """A symbol for a verdict, so colour is never the only encoding of state."""
    return {True: "✓", False: "✗", None: "?"}[passed]


def apply_layout(
    figure: Any,
    *,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    palette: Palette | None = None,
    template: str = "simple_white",
    height: int | None = None,
    width: int | None = None,
    showlegend: bool | None = None,
) -> Any:
    """Apply the shared layout: recessive axes, a readable title, a legend when needed.

    Called by every builder here, so the figures compose in one report without one of
    them shouting. The grid is deliberately faint: a grid competing with the data is the
    most common way a correct chart becomes hard to read.
    """
    p = palette or DEFAULT_PALETTE
    figure.update_layout(
        template=template,
        title=dict(text=title, font=dict(size=16, color=p.text_primary)),
        font=dict(color=p.text_primary, size=12),
        paper_bgcolor=p.surface,
        plot_bgcolor=p.surface,
        margin=dict(l=70, r=30, t=60 if title else 30, b=60),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=p.text_secondary, size=11),
        ),
    )
    if showlegend is not None:
        figure.update_layout(showlegend=showlegend)
    if height is not None:
        figure.update_layout(height=height)
    if width is not None:
        figure.update_layout(width=width)
    figure.update_xaxes(
        title=dict(text=x_title, font=dict(size=12, color=p.text_secondary)),
        gridcolor=p.grid,
        zeroline=False,
        tickfont=dict(size=11, color=p.text_secondary),
    )
    figure.update_yaxes(
        title=dict(text=y_title, font=dict(size=12, color=p.text_secondary)),
        gridcolor=p.grid,
        zeroline=False,
        tickfont=dict(size=11, color=p.text_secondary),
    )
    return figure
