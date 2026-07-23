"""Real atomic data to a synthetic spectrum: the C IV 1548/1550 doublet.

The whole path with no hand-tuned line strength anywhere: NIST wavelengths and
oscillator strengths -> build_line_list -> the coupled multi-line solver and the
Sobolev line force.

    uv run --group examples python examples/line_list.py   # + PNG
    uv run python examples/line_list.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import astropy.units as apu
import jax.numpy as jnp
from _plot import save_png, show_series
from astropy import constants as const
from unxt import Quantity

from mcrtx.dynamics import sobolev_line_force
from mcrtx.media import BetaLawWind
from mcrtx.physics import build_line_list
from mcrtx.solvers.multiline import multiline_profile_coupled
from mcrtx.solvers.reference import SourceModel
from mcrtx.units import ReferenceScales

# --- CONSTANTS: edit me ------------------------------------------------------
# C IV resonance doublet from NIST: (rest wavelength in Angstrom, f_lu).
LINES = [(1548.19, 0.190), (1550.77, 0.0952)]
R_STAR = Quantity(1e12, "cm")
V_INF = Quantity(2000.0, "km/s")
N_LOWER = Quantity(3e3, "cm**-3")  # C IV lower-level density at the reference radius
V_PHOT = 0.01
BETA = 0.8
SOURCE = SourceModel.SELF_CONSISTENT
# -----------------------------------------------------------------------------


def main():
    wavelengths = Quantity(jnp.asarray([lam for lam, _ in LINES]), "angstrom")
    f_values = [f for _, f in LINES]
    nu_0 = Quantity(float((const.c / (LINES[0][0] * apu.AA)).to_value(apu.Hz)), "Hz")

    scales = ReferenceScales.from_quantities(R_STAR, V_INF, nu_0, N_LOWER)
    centers, lines = build_line_list(wavelengths, f_values, scales)
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))

    print(f"reference: R_star={R_STAR}, v_inf={V_INF}, n_l={N_LOWER}")
    for (lam, f), center, line in zip(LINES, centers, lines, strict=True):
        print(f"  {lam:8.2f} A  f={f:<7.4f} -> x={float(center):+.4f} v_inf, tau_scale={float(line.tau_scale):.3f}")

    x = jnp.linspace(float(centers.min()) - 2.0, float(centers.max()) + 1.5, 241)
    flux = multiline_profile_coupled(wind, [float(c) for c in centers], lines, x, source=SOURCE, n_p=400)
    print(f"\nprofile: flux in [{float(flux.min()):.3f}, {float(flux.max()):.3f}]")

    r = jnp.asarray([1.5, 2.0, 5.0, 10.0])
    g = sobolev_line_force(wind, lines, r)
    forces = "  ".join(f"r={float(ri):.1f}: {float(gi):.3e}" for ri, gi in zip(r, g, strict=True))
    print(f"line force: {forces}")

    xs = [float(v) for v in x]
    series = {"C IV doublet": [float(v) for v in flux]}
    show_series(xs, series, lo=0.0, hi=max(1.1, float(flux.max())))

    png = save_png(
        "examples/output/line_list.png",
        xs,
        series,
        xlabel="velocity  (v_inf units)",
        ylabel="F / F_continuum",
        title="C IV 1548/1550 from NIST atomic data",
    )
    print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)")


if __name__ == "__main__":
    main()
