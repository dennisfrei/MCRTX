"""Reference solver: continuum limit, P-Cygni asymmetry, optical-depth trends.

All tests share one frequency-grid shape and ``n_p`` so they reuse a single XLA
compilation (``tau_scale`` is a traced input, so its value does not recompile).
"""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers import SourceModel, solve_profile

# The reference solver is XLA-compilation-heavy (~8s/compile on a Pi); the whole
# module is 'slow' so a plain `pytest` skips it (CI runs `-m "not gpu"`).
pytestmark = pytest.mark.slow

X = jnp.linspace(-1.5, 1.5, 101)
N_P = 200


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _solve(wind, tau_scale, source=SourceModel.THIN):
    line = LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))
    return solve_profile(wind, line, X, n_p=N_P, source=source)


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


# --- M1: self-consistent two-level source -----------------------------------


def test_self_consistent_recovers_thin_when_optically_thin(wind):
    # Agreement is limited by the angular quadrature / radius interpolation.
    thin = _solve(wind, 1e-4, SourceModel.THIN)
    sc = _solve(wind, 1e-4, SourceModel.SELF_CONSISTENT)
    assert jnp.max(jnp.abs(sc - thin)) < 5e-3


def test_self_consistent_continuum_and_pcygni(wind):
    assert jnp.allclose(_solve(wind, 0.0, SourceModel.SELF_CONSISTENT), 1.0, atol=1e-6)
    sc = _solve(wind, 5.0, SourceModel.SELF_CONSISTENT)
    assert jnp.min(sc[X < 0.0]) < 0.9  # absorption
    assert jnp.max(sc[X > 0.2]) > 1.1  # emission


def test_self_consistent_conserves_photons_better_than_thin(wind):
    # A pure-scattering line cannot create photons: the thin source over-emits
    # badly for thick lines (equivalent width runs away), the self-consistent
    # source stays bounded (residual = physical loss into the star).
    dx = X[1] - X[0]
    ew_thin = jnp.sum(1.0 - _solve(wind, 50.0, SourceModel.THIN)) * dx
    ew_sc = jnp.sum(1.0 - _solve(wind, 50.0, SourceModel.SELF_CONSISTENT)) * dx
    assert jnp.abs(ew_sc) < 0.3
    assert jnp.abs(ew_sc) < 0.4 * jnp.abs(ew_thin)


def test_nlte_reduces_to_self_consistent_without_collisions(wind):
    # collision = 0 -> the rate equations give the pure-scattering source.
    line = LineData(tau_scale=jnp.asarray(10.0), n_l_exponent=jnp.asarray(0.0))
    nlte = solve_profile(wind, line, X, n_p=N_P, source=SourceModel.NLTE)
    scattering = solve_profile(wind, line, X, n_p=N_P, source=SourceModel.SELF_CONSISTENT)
    assert jnp.allclose(nlte, scattering, atol=1e-9)


def test_nlte_collisions_thermalise_into_emission(wind):
    # Density-scaled collisions thermalise the inner wind, adding thermal emission.
    dx = X[1] - X[0]
    scattering = LineData(tau_scale=jnp.asarray(10.0), n_l_exponent=jnp.asarray(0.0))
    collisional = LineData(
        tau_scale=jnp.asarray(10.0),
        n_l_exponent=jnp.asarray(0.0),
        collision=jnp.asarray(5.0),
        w_core=jnp.asarray(0.6),
        boltzmann=jnp.asarray(0.5),
    )
    ew_scattering = jnp.sum(1.0 - solve_profile(wind, scattering, X, n_p=N_P, source=SourceModel.SELF_CONSISTENT)) * dx
    ew_collisional = jnp.sum(1.0 - solve_profile(wind, collisional, X, n_p=N_P, source=SourceModel.NLTE)) * dx
    assert float(ew_collisional) < float(ew_scattering)  # more (thermal) emission
