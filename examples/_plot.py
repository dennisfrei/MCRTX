"""Tiny plotting helpers for the examples — ASCII always, PNG if matplotlib is present.

Run the scripts with ``uv run --group examples python examples/<name>.py`` to get
PNGs; without the group they still print sparklines to the terminal.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], lo: float | None = None, hi: float | None = None) -> str:
    """Render a sequence as a one-line Unicode sparkline."""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = hi - lo or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, max(0, int((v - lo) / span * (len(_BLOCKS) - 1))))] for v in values)


def show_series(x: Sequence[float], series: dict[str, Sequence[float]], *, lo: float, hi: float) -> None:
    """Print each named series as a sparkline sharing one vertical scale."""
    print(f"    x: {float(x[0]):+.2f} {'.' * 20} {float(x[-1]):+.2f}   (scale {lo:.2f}..{hi:.2f})")
    for name, values in series.items():
        print(f"  {name:>18}: {sparkline(values, lo, hi)}")


def save_png(
    path: str | Path,
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path | None:
    """Save an overlay line plot if matplotlib is available; return the path or None."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in series.items():
        ax.plot(x, values, label=name, lw=1.6)
    ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
