"""Source-function playground: compare thin, scattering, and NLTE closures.

The source functions are shown in units of the stellar core intensity. Increasing
line optical depth or collisions moves the source away from simple geometric
dilution.

    uv run --group examples python examples/source_functions.py   # + PNG
    uv run python examples/source_functions.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, source_at

# --- CONSTANTS: edit me ------------------------------------------------------
BETA = 1.0
TAU_SCALE = 10.0
COLLISION = 5.0
# -----------------------------------------------------------------------------


def main() -> None:
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
    r = jnp.exp(jnp.linspace(0.0, jnp.log(20.0), 100))
    line = LineData(
        tau_scale=jnp.asarray(TAU_SCALE),
        n_l_exponent=jnp.asarray(0.0),
        collision=jnp.asarray(COLLISION),
        w_core=jnp.asarray(0.6),
        boltzmann=jnp.asarray(0.5),
    )
    series = {}
    for model, label in (
        (SourceModel.THIN, "thin dilution"),
        (SourceModel.SELF_CONSISTENT, "self-consistent"),
        (SourceModel.NLTE, "NLTE collisions"),
    ):
        values = source_at(wind, line, r, model, p_max=20.0, n_r=128, n_mu=48)
        series[label] = [float(v) for v in values]
        console.print(f"  {label:>18}: S/I_core in [{float(values.min()):.3f}, {float(values.max()):.3f}]", markup=False)

    rs = [float(v) for v in r]
    show_series(rs, series, lo=0.0, hi=max(max(values) for values in series.values()))
    png = save_png(
        "examples/output/source_functions.png",
        rs,
        series,
        xlabel=r"Radius $r / R_\star$",
        ylabel=r"$S / I_\mathrm{core}$",
        title=rf"Source-function closures ($\tau = {TAU_SCALE:g}$)",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
