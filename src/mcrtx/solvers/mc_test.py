"""Monte Carlo solver: continuum, reproducibility, agreement with the reference.

The acceptance test uses an RMS metric over bins (concept sec. 5): a single steep
central bin makes the max-difference noisy, but the RMS is stable across seeds.
A shared grid shape and ``n_packets`` let the tests reuse one XLA compilation.
"""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers.mc import run_profile
from mcrtx.solvers.reference import solve_profile

XC = jnp.linspace(-1.15, 1.15, 24)
_DX = XC[1] - XC[0]
EDGES = jnp.concatenate([XC - _DX / 2, XC[-1:] + _DX / 2])
BAND = (-1.2, 1.2)
N_PK = 200_000


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _line(tau_scale):
    return LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))


def test_zero_optical_depth_is_exactly_continuum(wind):
    # tau = 0 => full transmission, zero emission => R == 1 in every bin.
    r = run_profile(wind, _line(0.0), EDGES, jax.random.key(0), n_packets=N_PK, band=BAND)
    assert jnp.allclose(r, 1.0, atol=1e-9)


def test_reproducible_for_fixed_seed(wind):
    a = run_profile(wind, _line(5.0), EDGES, jax.random.key(3), n_packets=N_PK, band=BAND)
    b = run_profile(wind, _line(5.0), EDGES, jax.random.key(3), n_packets=N_PK, band=BAND)
    assert jnp.array_equal(a, b)


def test_matches_reference_within_mc_noise(wind):
    line = _line(5.0)
    ref = solve_profile(wind, line, XC, n_p=800)
    mc = run_profile(wind, line, EDGES, jax.random.key(0), n_packets=N_PK, band=BAND)
    rms = jnp.sqrt(jnp.mean((mc - ref) ** 2))
    assert float(rms) < 0.05
    # And it is a genuine P-Cygni profile, not a flat match.
    assert jnp.min(mc[XC < 0.0]) < 0.9  # blueshifted absorption
    assert jnp.max(mc[XC > 0.2]) > 1.1  # redshifted emission


def test_differentiable_in_optical_depth(wind):
    # Gradients flow through the smooth weights (D3), not the random draws.
    def trough(tau_scale):
        r = run_profile(wind, _line(tau_scale), EDGES, jax.random.key(0), n_packets=N_PK, band=BAND)
        return jnp.min(r)

    g = jax.grad(trough)(jnp.asarray(5.0))
    assert jnp.isfinite(g)
    assert g < 0.0  # deeper absorption trough as opacity grows


@pytest.mark.slow
def test_gradient_matches_finite_difference(wind):
    # D3 acceptance (concept sec. 5.3): the AD gradient at a fixed seed is an
    # unbiased estimator of d E[R] / d(tau_scale). With common random numbers it
    # matches a central finite difference of the seed-expectation to O(eps^2).
    keys = jax.random.split(jax.random.key(1), 12)
    tau0 = jnp.asarray(5.0)

    def observable(tau_scale, key):
        r = run_profile(wind, _line(tau_scale), EDGES, key, n_packets=40_000, band=BAND)
        return jnp.sum(r)  # smooth linear functional of the profile

    g_ad = jnp.mean(jax.vmap(jax.grad(observable), in_axes=(None, 0))(tau0, keys))

    def mean_obs(tau_scale):
        return jnp.mean(jax.vmap(lambda k: observable(tau_scale, k))(keys))

    eps = 1e-3
    g_fd = (mean_obs(tau0 + eps) - mean_obs(tau0 - eps)) / (2.0 * eps)
    assert jnp.abs(g_ad - g_fd) < 1e-3 * jnp.abs(g_fd)
