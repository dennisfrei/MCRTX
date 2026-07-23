"""P-Cygni playground: solve one wind-line scenario every which way and compare.

Tweak the CONSTANTS block and re-run. Shows the reference vs Monte Carlo solvers
and the M0 (thin) vs M1 (self-consistent) source functions on one axis.

    uv run --group examples python examples/pcygni.py   # + PNG
    uv run python examples/pcygni.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import unxt as u
from _plot import save_png, show_series
from unxt import Quantity

from mcrtx.scenarios import Method, RunConfig, SourceModel, WindLineScenario, solve

# --- CONSTANTS: edit me ------------------------------------------------------
R_STAR_CM = 7.0e11  # stellar radius [cm]
V_INF_KMS = 2000.0  # terminal wind speed [km/s]
V_PHOT_KMS = 20.0  # photospheric speed [km/s]
WAVELENGTH_A = 1548.0  # line rest wavelength [Angstrom] (C IV)
MDOT_G_S = 6.3e19  # mass-loss rate [g/s]  (~1e-6 Msun/yr)
F_LU = 0.19  # oscillator strength
LOWER_PER_MASS = 6.3e19  # lower-level absorbers per gram
BETA = 1.0  # velocity-law exponent
N_PACKETS = 400_000  # Monte Carlo sample count
# -----------------------------------------------------------------------------

_C_CM_S = 2.99792458e10


def _q(value, unit):
    return Quantity(value, u.unit(unit))


def main():
    scenario = WindLineScenario.from_physical(
        r_star=_q(R_STAR_CM, "cm"),
        v_inf=_q(V_INF_KMS, "km / s"),
        v_phot=_q(V_PHOT_KMS, "km / s"),
        nu_0=_q(_C_CM_S / (WAVELENGTH_A * 1e-8), "Hz"),
        mdot=_q(MDOT_G_S, "g / s"),
        f_lu=F_LU,
        lower_level_per_mass=_q(LOWER_PER_MASS, "g**-1"),
        beta=BETA,
    )
    n_ref = float(scenario.density_ref.ustrip(u.unit("cm**-3")))
    print(f"scenario: tau_scale={scenario.tau_scale:.1f}  n_ref={n_ref:.2e} cm^-3")

    runs = {
        "reference thin": (Method.REFERENCE, SourceModel.THIN),
        "reference self-cons.": (Method.REFERENCE, SourceModel.SELF_CONSISTENT),
        "MC thin": (Method.MC, SourceModel.THIN),
        "MC self-cons.": (Method.MC, SourceModel.SELF_CONSISTENT),
    }
    series = {}
    velocity = None
    for name, (method, source) in runs.items():
        result = solve(scenario, RunConfig(n_bins=81, n_packets=N_PACKETS, source=source), method)
        velocity = result.velocity.ustrip(u.unit("km / s"))
        flux = result.flux
        dx = float(velocity[1] - velocity[0]) / V_INF_KMS  # in v_inf units
        ew = float(jnp.sum(1.0 - flux) * dx)
        series[name] = [float(v) for v in flux]
        print(f"  {name:>22}: R in [{float(flux.min()):.2f}, {float(flux.max()):.2f}]  equiv.width={ew:+.3f}")

    velocities = [float(v) for v in velocity]
    hi = max(max(s) for s in series.values())
    print()
    show_series(velocities, series, lo=0.0, hi=hi)

    png = save_png(
        "examples/output/pcygni.png",
        velocities,
        series,
        xlabel="Doppler velocity [km/s]",
        ylabel="F / F_continuum",
        title=f"P-Cygni profile  (tau_scale={scenario.tau_scale:.0f})",
    )
    print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)")


if __name__ == "__main__":
    main()
