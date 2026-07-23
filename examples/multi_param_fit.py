"""Multi-parameter fitting playground: fit a wind-line profile with Adam.

A synthetic profile is generated from four named parameters. Adam then updates
all four together: the velocity-law exponent, photospheric speed, line strength,
and radial lower-level population exponent. A single normalized profile is
intentionally under-constrained, so the recovered combination need not equal
the synthetic parameters exactly.

    uv run --group examples python examples/multi_param_fit.py   # + PNG
    uv run python examples/multi_param_fit.py                     # terminal only
"""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, solve_profile

# --- CONSTANTS: edit me ------------------------------------------------------
TRUE_BETA = 0.8
TRUE_LOG_V_PHOT = -2.0
TRUE_TAU = 7.0
TRUE_N_L_EXPONENT = 0.2
INITIAL_BETA = 1.2
INITIAL_LOG_V_PHOT = -1.5
INITIAL_TAU = 3.0
INITIAL_N_L_EXPONENT = -0.2
LEARNING_RATE = 0.03
N_STEPS = 60
# -----------------------------------------------------------------------------


class FitParameters(NamedTuple):
    """Named optimization parameters; positive quantities use log coordinates."""

    beta: jax.Array
    log_v_phot: jax.Array
    log_tau: jax.Array
    n_l_exponent: jax.Array

    def physical(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return ``(beta, v_phot, tau, n_l_exponent)`` in physical coordinates."""
        return self.beta, 10.0**self.log_v_phot, jnp.exp(self.log_tau), self.n_l_exponent

    def clipped(self) -> FitParameters:
        """Apply the physically useful bounds for this demonstration."""
        return self._replace(
            beta=jnp.clip(self.beta, 0.2, 2.5),
            log_v_phot=jnp.clip(self.log_v_phot, -2.5, -0.5),
            log_tau=jnp.clip(self.log_tau, jnp.log(0.1), jnp.log(30.0)),
            n_l_exponent=jnp.clip(self.n_l_exponent, -1.0, 1.0),
        )


def profile(parameters: FitParameters, x: jax.Array) -> jax.Array:
    """Evaluate a profile from explicitly named fitting parameters."""
    beta, v_phot, tau_scale, n_l_exponent = parameters.physical()
    wind = BetaLawWind(beta=beta, v_phot=v_phot, mdot_scale=jnp.asarray(1.0))
    line = LineData(tau_scale=tau_scale, n_l_exponent=n_l_exponent)
    return solve_profile(wind, line, x, n_p=140, source=SourceModel.THIN)


def make_parameters(beta: float, log_v_phot: float, tau: float, n_l_exponent: float) -> FitParameters:
    """Construct named optimizer parameters from readable physical inputs."""
    return FitParameters(
        beta=jnp.asarray(beta),
        log_v_phot=jnp.asarray(log_v_phot),
        log_tau=jnp.log(jnp.asarray(tau)),
        n_l_exponent=jnp.asarray(n_l_exponent),
    )


def format_parameters(parameters: FitParameters) -> str:
    """Format named physical parameters for terminal output."""
    beta, v_phot, tau_scale, n_l_exponent = parameters.physical()
    return (
        f"beta={float(beta):.3f} v_phot={float(v_phot):.4f} "
        f"tau={float(tau_scale):.3f} n_l_exp={float(n_l_exponent):+.3f}"
    )


def main() -> None:
    x = jnp.linspace(-1.2, 1.2, 41)
    target_parameters = make_parameters(TRUE_BETA, TRUE_LOG_V_PHOT, TRUE_TAU, TRUE_N_L_EXPONENT)
    target = profile(target_parameters, x)

    def loss(parameters: FitParameters) -> jax.Array:
        return jnp.mean((profile(parameters, x) - target) ** 2)

    parameters = make_parameters(INITIAL_BETA, INITIAL_LOG_V_PHOT, INITIAL_TAU, INITIAL_N_L_EXPONENT)
    moments = jax.tree.map(jnp.zeros_like, parameters)
    squared_moments = jax.tree.map(jnp.zeros_like, parameters)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    for step in range(N_STEPS):
        value, gradient = value_and_grad(parameters)
        moments = jax.tree.map(lambda moment, grad: 0.9 * moment + 0.1 * grad, moments, gradient)
        squared_moments = jax.tree.map(
            lambda moment, grad: 0.999 * moment + 0.001 * grad**2,
            squared_moments,
            gradient,
        )
        bias_correction = 1.0 - 0.9 ** (step + 1)
        squared_bias_correction = 1.0 - 0.999 ** (step + 1)
        parameters = jax.tree.map(
            lambda parameter, moment, squared_moment, bias_correction=bias_correction, squared_bias_correction=squared_bias_correction: (
                parameter
                - LEARNING_RATE * (moment / bias_correction)
                / (jnp.sqrt(squared_moment / squared_bias_correction) + 1e-8)
            ),
            parameters,
            moments,
            squared_moments,
        ).clipped()
        if step % 10 == 0 or step == N_STEPS - 1:
            console.print(f"  step {step:>2}: loss={float(value):.5f} {format_parameters(parameters)}", markup=False)

    fitted = profile(parameters, x)
    series = {"target": [float(v) for v in target], "fitted": [float(v) for v in fitted]}
    show_series([float(v) for v in x], series, lo=0.0, hi=max(max(values) for values in series.values()))
    console.print(f"target:   {format_parameters(target_parameters)}", markup=False)
    console.print(f"recovered: {format_parameters(parameters)}", markup=False)
    png = save_png(
        "examples/output/multi_param_fit.png",
        [float(v) for v in x],
        series,
        xlabel=r"Doppler velocity $v / v_\infty$",
        ylabel=r"$F / F_\mathrm{continuum}$",
        title=r"Multi-parameter differentiable line-profile fit",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
