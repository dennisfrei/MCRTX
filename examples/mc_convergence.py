"""Monte Carlo convergence playground: compare packet counts to the reference.

The deterministic reference profile is compared with increasingly large Monte
Carlo samples. The RMS error should decrease approximately as N_packets^-1/2.

    uv run --group examples python examples/mc_convergence.py   # + PNG
    uv run python examples/mc_convergence.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, run_profile, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
TAU_SCALE = 5.0
V_PHOT = 0.01
BETA = 1.0
PACKET_COUNTS = [1_000, 4_000, 16_000]
N_BINS = 41
N_P = 240
# -----------------------------------------------------------------------------


def main() -> None:
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
    line = LineData(tau_scale=jnp.asarray(TAU_SCALE), n_l_exponent=jnp.asarray(0.0))
    x = jnp.linspace(-1.3, 1.3, N_BINS)
    edges = jnp.linspace(-1.3, 1.3, N_BINS + 1)
    reference = solve_profile(wind, line, x, n_p=N_P, source=SourceModel.THIN)
    series = {"reference": [float(v) for v in reference]}
    errors = {}

    console.print(f"Reference profile: tau_scale={TAU_SCALE:g}, beta={BETA:g}", markup=False)
    for n_packets in PACKET_COUNTS:
        mc = run_profile(
            wind,
            line,
            edges,
            jax.random.key(0),
            n_packets=n_packets,
            band=(-1.3, 1.3),
            p_max=20.0,
            source=SourceModel.THIN,
        )
        rms = float(jnp.sqrt(jnp.mean((mc - reference) ** 2)))
        errors[str(n_packets)] = rms
        series[f"MC ({n_packets:,})"] = [float(v) for v in mc]
        console.print(f"  {n_packets:>7,} packets: RMS error={rms:.4f}", markup=False)

    show_series([float(v) for v in x], series, lo=0.0, hi=max(max(values) for values in series.values()))
    png = save_png(
        "examples/output/mc_convergence.png",
        [float(v) for v in x],
        series,
        xlabel=r"Doppler velocity $v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=rf"Monte Carlo convergence ($\tau = {TAU_SCALE:g}$)",
    )
    console.print(f"\nRMS errors: {errors}", markup=False)
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
