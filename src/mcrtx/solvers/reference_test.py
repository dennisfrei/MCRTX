"""Reference solver: continuum limit, P-Cygni asymmetry, optical-depth trends.

All tests share one frequency-grid shape and ``n_p`` so they reuse a single XLA
compilation (``tau_scale`` is a traced input, so its value does not recompile).
"""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import solve_profile

X = jnp.linspace(-1.5, 1.5, 101)
N_P = 200


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _solve(wind, tau_scale):
    line = LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))
    return solve_profile(wind, line, X, n_p=N_P)


def test_zero_optical_depth_is_flat_continuum(wind):
    assert jnp.allclose(_solve(wind, 0.0), 1.0, atol=1e-6)


def test_beyond_terminal_velocity_is_continuum(wind):
    # No resonance exists past |x| = v_inf = 1, so the profile returns to unity.
    r = _solve(wind, 20.0)
    assert jnp.allclose(r[jnp.abs(X) >= 1.1], 1.0, atol=1e-6)


def test_pcygni_asymmetry(wind):
    r = _solve(wind, 5.0)
    assert jnp.min(r[X < 0.0]) < 0.9  # blueshifted absorption trough
    assert jnp.max(r[X > 0.2]) > 1.1  # redshifted emission
    # Absorption is a blue-side feature; the red side never dips below continuum.
    assert jnp.min(r[X > 0.2]) >= 1.0 - 1e-6


def test_deeper_absorption_with_optical_depth(wind):
    depths = [jnp.min(_solve(wind, ts)) for ts in (1.0, 5.0, 30.0)]
    assert depths[0] > depths[1] > depths[2]


def test_emission_grows_with_optical_depth(wind):
    peaks = [jnp.max(_solve(wind, ts)) for ts in (1.0, 5.0, 30.0)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_jit_matches_eager(wind):
    line = LineData(tau_scale=jnp.asarray(5.0), n_l_exponent=jnp.asarray(0.0))
    eager = solve_profile(wind, line, X, n_p=N_P)
    jitted = jax.jit(solve_profile, static_argnames=("n_p", "n_bisect"))(wind, line, X, n_p=N_P)
    assert jnp.allclose(eager, jitted, atol=1e-10)
