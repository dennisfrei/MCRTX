"""Wind + NLTE playground: watch a line thermalise as collisions grow.

The line source function comes from the two-level rate equations coupled to the
Sobolev escape probabilities (`SourceModel.NLTE`). With no collisions it is the
pure-scattering P-Cygni line; as the base collision rate grows, the dense inner
wind thermalises (LTE) and the line turns into thermal emission.

    uv run --group examples python examples/nlte_wind.py   # + PNG
    uv run python examples/nlte_wind.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers.reference import SourceModel, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
BETA = 1.0  # velocity-law exponent
V_PHOT = 0.01  # photospheric speed / v_inf
TAU_SCALE = 10.0  # line strength
W_CORE = 0.6  # core radiation occupation number
BOLTZMANN = 0.5  # exp(-h nu / k T_gas)
COLLISIONS = [0.0, 1.0, 5.0, 20.0]  # base collision rates C_ul/A_ul to sweep
# -----------------------------------------------------------------------------


def main():
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
    x = jnp.linspace(-1.3, 1.3, 81)
    dx = float(x[1] - x[0])

    series = {}
    for collision in COLLISIONS:
        line = LineData(
            tau_scale=jnp.asarray(TAU_SCALE),
            n_l_exponent=jnp.asarray(0.0),
            collision=jnp.asarray(collision),
            w_core=jnp.asarray(W_CORE),
            boltzmann=jnp.asarray(BOLTZMANN),
        )
        flux = solve_profile(wind, line, x, source=SourceModel.NLTE)
        ew = float(jnp.sum(1.0 - flux) * dx)
        label = "scattering" if collision == 0.0 else f"collision={collision:g}"
        series[label] = [float(v) for v in flux]
        console.print(
            f"  {label:>16}: R in [{float(flux.min()):.2f}, {float(flux.max()):.2f}]  equiv.width={ew:+.3f}",
            markup=False,
        )

    xs = [float(v) for v in x]
    hi = max(max(s) for s in series.values())
    console.print()
    show_series(xs, series, lo=0.0, hi=hi)

    png = save_png(
        "examples/output/nlte_wind.png",
        xs,
        series,
        xlabel=r"Normalised velocity $x = v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=rf"Line thermalising with collisions ($\tau_\mathrm{{scale}} = {TAU_SCALE:g}$)",
    )
    console.print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
