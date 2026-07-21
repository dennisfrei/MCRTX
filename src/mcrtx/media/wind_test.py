"""BetaLawWind: analytic invariants of the MVP medium."""

import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def test_boundary_values(wind):
    assert wind.v(jnp.asarray(1.0)) == pytest.approx(0.01)
    assert wind.v(jnp.asarray(1e6)) == pytest.approx(1.0, rel=1e-5)


def test_velocity_monotonically_increasing(wind):
    r = jnp.linspace(1.0, 50.0, 500)
    assert jnp.all(jnp.diff(wind.v(r)) > 0)
    assert jnp.all(wind.dv_dr(r) > 0)


def test_dv_dr_matches_finite_difference(wind):
    r = jnp.linspace(1.5, 20.0, 100)
    eps = 1e-6
    fd = (wind.v(r + eps) - wind.v(r - eps)) / (2 * eps)
    assert jnp.allclose(wind.dv_dr(r), fd, rtol=1e-5)


def test_continuity(wind):
    r = jnp.linspace(1.1, 30.0, 100)
    assert jnp.allclose(wind.rho(r) * r**2 * wind.v(r), wind.mdot_scale)


def test_dilution_limits(wind):
    assert wind.dilution(jnp.asarray(1.0)) == pytest.approx(0.5)
    assert wind.dilution(jnp.asarray(1e4)) == pytest.approx(0.0, abs=1e-7)
