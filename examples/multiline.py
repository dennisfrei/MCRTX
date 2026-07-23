"""Multi-line playground: build a small spectrum from several resonance lines.

Each line sits at its own centre on a shared velocity axis (units of v_inf) and
contributes its own P-Cygni profile. Well-separated lines superpose exactly;
overlapping ones are combined to first order (see solvers/multiline.py).

    uv run --group examples python examples/multiline.py   # + PNG
    uv run python examples/multiline.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers.multiline import multiline_profile, multiline_profile_coupled
from mcrtx.solvers.reference import SourceModel

# --- CONSTANTS: edit me ------------------------------------------------------
# A close doublet (centres within ~a Doppler width) shows the difference between
# the cheap additive combination and the exact coupled formal solution.
# (line centre in v_inf units, line strength tau_scale)
LINES = [(-0.3, 20.0), (0.3, 20.0)]
V_PHOT = 0.01
BETA = 1.0
SOURCE = SourceModel.SELF_CONSISTENT
# -----------------------------------------------------------------------------


def main():
    wind = BetaLawWind(beta=jnp.asarray(BETA), v_phot=jnp.asarray(V_PHOT), mdot_scale=jnp.asarray(1.0))
    centers = [c for c, _ in LINES]
    lines = [LineData(tau_scale=jnp.asarray(t), n_l_exponent=jnp.asarray(0.0)) for _, t in LINES]

    x = jnp.linspace(min(centers) - 2.0, max(centers) + 2.0, 241)
    additive = multiline_profile(wind, centers, lines, x, source=SOURCE, n_p=400)
    coupled = multiline_profile_coupled(wind, centers, lines, x, source=SOURCE, n_p=400)
    print(f"{len(LINES)} lines at v/v_inf = {centers}")
    print(f"  additive: flux in [{float(additive.min()):.2f}, {float(additive.max()):.2f}]  (unphysical if < 0)")
    print(f"  coupled : flux in [{float(coupled.min()):.2f}, {float(coupled.max()):.2f}]")

    xs = [float(v) for v in x]
    series = {"additive": [float(v) for v in additive], "coupled (exact)": [float(v) for v in coupled]}
    lo = min(0.0, float(additive.min()))
    hi = max(float(additive.max()), float(coupled.max()))
    show_series(xs, series, lo=lo, hi=hi)

    png = save_png(
        "examples/output/multiline.png",
        xs,
        series,
        xlabel="velocity  (v_inf units)",
        ylabel="F / F_continuum",
        title=f"{len(LINES)}-line doublet: additive vs coupled",
    )
    print(f"\nPNG: {png}" if png else "\n(install the 'examples' group for a PNG)")


if __name__ == "__main__":
    main()
