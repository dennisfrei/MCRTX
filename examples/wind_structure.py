"""Wind-structure playground: plot the velocity field and density vs radius.

The velocity law v(r) is currently a *prescription* (the beta law). Changing
beta changes how fast the wind accelerates — and thus every line profile.
(Physically, v(r) is set by the line-driving force; see the module discussion.)

    uv run --group examples python examples/wind_structure.py   # + PNG
    uv run python examples/wind_structure.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind

# --- CONSTANTS: edit me ------------------------------------------------------
V_PHOT = 0.01  # photospheric speed / v_inf
BETAS = [0.5, 1.0, 2.0]  # velocity-law exponents to compare
R_MAX = 20.0  # outer radius (units of R_star)
# -----------------------------------------------------------------------------


def main():
    r = jnp.exp(jnp.linspace(0.0, jnp.log(R_MAX), 200))  # log-spaced 1 .. R_MAX
    velocities = {}
    for beta in BETAS:
        wind = BetaLawWind(beta=jnp.asarray(beta), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
        velocities[f"beta={beta:g}"] = [float(v) for v in wind.v(r)]

    console.print("velocity field v(r) / v_inf  (lower beta accelerates faster):", markup=False)
    show_series([float(v) for v in r], velocities, lo=0.0, hi=1.0)

    # Structure table for the middle beta.
    wind = BetaLawWind(
        beta=jnp.asarray(BETAS[len(BETAS) // 2]), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0)
    )
    console.print(f"\nstructure for beta={BETAS[len(BETAS) // 2]:g}:", markup=False)
    console.print(f"  {'r':>5} {'v/v_inf':>9} {'dv/dr':>9} {'rho':>11} {'W(r)':>7}", markup=False)
    for ri in (1.0, 1.5, 2.0, 5.0, 10.0, 20.0):
        rj = jnp.asarray(ri)
        console.print(
            f"  {ri:5.1f} {float(wind.v(rj)):9.4f} {float(wind.dv_dr(rj)):9.4f} "
            f"{float(wind.rho(rj)):11.4e} {float(wind.dilution(rj)):7.4f}",
            markup=False,
        )

    png = save_png(
        "examples/output/wind_structure.png",
        [float(v) for v in r],
        velocities,
        xlabel=r"Radius $r / R_\star$",
        ylabel=r"$v(r) / v_\infty$",
        title=r"$\beta$-law velocity field",
    )
    console.print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
