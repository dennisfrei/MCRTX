"""Units playground: solve the same physical scenario in SI and CGS.

The scenario layer converts physical quantities to the same dimensionless kernel
values regardless of whether inputs use SI or CGS units.

    uv run --group examples python examples/units_roundtrip.py   # + PNG
    uv run python examples/units_roundtrip.py                     # terminal only
"""

from __future__ import annotations

import jax.numpy as jnp
import unxt as u
from _plot import console, save_png, show_series
from unxt import Quantity

from mcrtx.scenarios import Method, RunConfig, SourceModel, WindLineScenario, solve

_C_CM_S = 2.99792458e10
_WAVELENGTH_A = 1548.0


def q(value: float, unit: str) -> Quantity:
    return Quantity(value, u.unit(unit))


def make_scenario(length: str, velocity: str, mass_rate: str, absorber: str) -> WindLineScenario:
    return WindLineScenario.from_physical(
        r_star=q(7.0e11 if length == "cm" else 7.0e9, length),
        v_inf=q(2.0e8 if velocity == "cm / s" else 2.0e6, velocity),
        v_phot=q(2.0e6 if velocity == "cm / s" else 2.0e4, velocity),
        nu_0=q(_C_CM_S / (_WAVELENGTH_A * 1e-8), "Hz"),
        mdot=q(6.3e19 if mass_rate == "g / s" else 6.3e16, mass_rate),
        f_lu=0.19,
        lower_level_per_mass=q(6.3e19 if absorber == "g**-1" else 6.3e22, absorber),
        beta=1.0,
    )


def main() -> None:
    cgs = make_scenario("cm", "cm / s", "g / s", "g**-1")
    si = make_scenario("m", "m / s", "kg / s", "kg**-1")
    config = RunConfig(n_bins=41, source=SourceModel.THIN)
    cgs_result = solve(cgs, config, Method.REFERENCE)
    si_result = solve(si, config, Method.REFERENCE)
    velocity_cgs = cgs_result.velocity.ustrip(u.unit("km / s"))
    velocity_si = si_result.velocity.ustrip(u.unit("km / s"))
    velocity_error = jnp.max(jnp.abs(velocity_cgs - velocity_si))
    flux_error = jnp.max(jnp.abs(cgs_result.flux - si_result.flux))
    console.print(f"max velocity-axis difference: {float(velocity_error):.3e} km/s", markup=False)
    console.print(f"max flux difference:          {float(flux_error):.3e}", markup=False)
    console.print(f"reference time scale: {cgs_result.scales.time:.3e} s", markup=False)

    series = {"CGS input": [float(v) for v in cgs_result.flux], "SI input": [float(v) for v in si_result.flux]}
    show_series([float(v) for v in velocity_cgs], series, lo=0.0, hi=max(max(values) for values in series.values()))
    png = save_png(
        "examples/output/units_roundtrip.png",
        [float(v) for v in velocity_cgs],
        series,
        xlabel=r"Doppler velocity $v$ [km s$^{-1}$]",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title="SI/CGS unit-boundary round trip",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
