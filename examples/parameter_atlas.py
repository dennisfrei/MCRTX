"""Parameter-atlas playground: scan line strength and wind acceleration.

A small grid of reference profiles shows how Sobolev optical depth and the
velocity-law exponent jointly shape a P-Cygni line.

    uv run --group examples python examples/parameter_atlas.py   # + PNG
    uv run python examples/parameter_atlas.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
BETAS = [0.5, 1.0, 2.0]
TAU_SCALES = [1.0, 5.0, 20.0]
# -----------------------------------------------------------------------------


def main() -> None:
    x = jnp.linspace(-1.2, 1.2, 41)
    series = {}
    for beta in BETAS:
        wind = BetaLawWind(beta=jnp.asarray(beta), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
        for tau_scale in TAU_SCALES:
            line = LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))
            flux = solve_profile(wind, line, x, n_p=140, source=SourceModel.THIN)
            label = rf"$\beta={beta:g},\tau={tau_scale:g}$"
            series[label] = [float(v) for v in flux]
            console.print(f"  beta={beta:g}, tau={tau_scale:g}: flux range [{float(flux.min()):.2f}, {float(flux.max()):.2f}]", markup=False)

    xs = [float(v) for v in x]
    show_series(xs, series, lo=0.0, hi=max(max(values) for values in series.values()))
    png = save_png(
        "examples/output/parameter_atlas.png",
        xs,
        series,
        xlabel=r"Doppler velocity $v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=r"P-Cygni parameter atlas",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
