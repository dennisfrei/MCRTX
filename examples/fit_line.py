"""Differentiable fitting playground: recover line strength with JAX.

A synthetic reference profile is generated, then gradient descent fits its
Sobolev line strength while keeping the wind geometry fixed. The forward model
remains a plain differentiable JAX function.

    uv run --group examples python examples/fit_line.py   # + PNG
    uv run python examples/fit_line.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
TRUE_TAU = 7.0
INITIAL_TAU = 3.0
LEARNING_RATE = 12.0
N_STEPS = 80
# -----------------------------------------------------------------------------


def profile(tau_scale: jax.Array, x: jax.Array) -> jax.Array:
    wind = BetaLawWind(beta=jnp.asarray(0.8), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
    line = LineData(tau_scale=tau_scale, n_l_exponent=jnp.asarray(0.0))
    return solve_profile(wind, line, x, n_p=140, source=SourceModel.THIN)


def main() -> None:
    x = jnp.linspace(-1.2, 1.2, 41)
    target = profile(jnp.asarray(TRUE_TAU), x)

    def loss(tau_scale: jax.Array) -> jax.Array:
        residual = profile(tau_scale, x) - target
        return jnp.mean(residual**2)

    tau_scale = jnp.asarray(INITIAL_TAU)
    for step in range(N_STEPS):
        value, gradient = jax.value_and_grad(loss)(tau_scale)
        tau_scale = jnp.clip(tau_scale - LEARNING_RATE * gradient, 0.1, 30.0)
        if step % 10 == 0 or step == N_STEPS - 1:
            console.print(f"  step {step:>2}: loss={float(value):.5f} tau={float(tau_scale):.3f}", markup=False)

    fitted = profile(tau_scale, x)
    series = {"target": [float(v) for v in target], "fitted": [float(v) for v in fitted]}
    show_series([float(v) for v in x], series, lo=0.0, hi=max(max(values) for values in series.values()))
    console.print(f"recovered tau={float(tau_scale):.3f} (true tau={TRUE_TAU:g})", markup=False)
    png = save_png(
        "examples/output/fit_line.png",
        [float(v) for v in x],
        series,
        xlabel=r"Doppler velocity $v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=r"Differentiable line-profile fit",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
