"""Line-force playground: the radiative acceleration g_rad(r) from a line list.

This is the force that drives the wind (the seed of the dynamics track). Weak
lines give the optically-thin force (∝ 1/r^2); strong lines saturate into the
CAK optically-thick force (∝ (dv/dr)/(rho r^2)). Each curve below is g_rad(r)
normalised to its peak, to compare shapes.

    uv run --group examples python examples/line_force.py   # + PNG
    uv run python examples/line_force.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.dynamics import sobolev_line_force
from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData

# --- CONSTANTS: edit me ------------------------------------------------------
BETA = 0.8
V_PHOT = 0.01
CASES = {"thin (weak lines)": 1e-3, "intermediate": 5.0, "thick (strong lines)": 1e3}
N_LINES = 5
# -----------------------------------------------------------------------------


def main():
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
    r = jnp.exp(jnp.linspace(jnp.log(1.02), jnp.log(20.0), 200))

    series = {}
    for label, tau_scale in CASES.items():
        lines = [LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))] * N_LINES
        g = sobolev_line_force(wind, lines, r)
        series[label] = [float(v) for v in g / jnp.max(g)]  # normalised shape
        console.print(f"  {label:>20}: g_rad peaks at r = {float(r[int(jnp.argmax(g))]):.2f}", markup=False)

    rs = [float(v) for v in r]
    show_series(rs, series, lo=0.0, hi=1.0)

    png = save_png(
        "examples/output/line_force.png",
        rs,
        series,
        xlabel=r"Radius $r / R_\star$",
        ylabel=r"$g_\mathrm{rad}(r) / g_{\mathrm{rad,max}}$",
        title=rf"Radiative line force ($N_\mathrm{{lines}} = {N_LINES}$)",
    )
    console.print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
