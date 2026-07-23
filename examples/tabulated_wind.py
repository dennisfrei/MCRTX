"""Tabulated-medium playground: replace an analytic wind with sampled data.

A beta-law wind is sampled onto a radial grid and passed to the same reference
solver. This mirrors the interface a hydrodynamic or radiation-driven model
would provide.

    uv run --group examples python examples/tabulated_wind.py   # + PNG
    uv run python examples/tabulated_wind.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind, TabulatedWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
BETA = 0.8
V_PHOT = 0.01
TAU_SCALE = 5.0
N_GRID = 96
# -----------------------------------------------------------------------------


def main() -> None:
    analytic = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
    r_grid = jnp.exp(jnp.linspace(0.0, jnp.log(20.0), N_GRID))
    tabulated = TabulatedWind.sample(analytic, r_grid)
    probe = jnp.exp(jnp.linspace(0.0, jnp.log(20.0), 200))
    velocity_error = jnp.max(jnp.abs(analytic.v(probe) - tabulated.v(probe)))
    density_error = jnp.max(jnp.abs(analytic.rho(probe) - tabulated.rho(probe)))
    console.print(f"{N_GRID} radial samples: max |delta v|={float(velocity_error):.3e}", markup=False)
    console.print(f"                 max |delta rho|={float(density_error):.3e}", markup=False)

    line = LineData(tau_scale=jnp.asarray(TAU_SCALE), n_l_exponent=jnp.asarray(0.0))
    x = jnp.linspace(-1.3, 1.3, 61)
    analytic_profile = solve_profile(analytic, line, x, n_p=180, z_far=20.0, source=SourceModel.THIN)
    tabulated_profile = solve_profile(tabulated, line, x, n_p=180, z_far=20.0, source=SourceModel.THIN)
    profile_error = jnp.max(jnp.abs(analytic_profile - tabulated_profile))
    console.print(f"max profile difference: {float(profile_error):.3e}", markup=False)

    series = {
        "analytic wind": [float(v) for v in analytic_profile],
        "tabulated wind": [float(v) for v in tabulated_profile],
    }
    show_series([float(v) for v in x], series, lo=0.0, hi=max(max(values) for values in series.values()))
    png = save_png(
        "examples/output/tabulated_wind.png",
        [float(v) for v in x],
        series,
        xlabel=r"Doppler velocity $v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=rf"Analytic vs tabulated wind ($N_r = {N_GRID}$)",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
