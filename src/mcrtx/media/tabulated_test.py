"""TabulatedWind: satisfies the Medium interface and reproduces the analytic wind."""

import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind, Medium, TabulatedWind
from mcrtx.physics.source import LineData
from mcrtx.solvers.reference import solve_profile


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def test_media_satisfy_the_protocol(wind):
    assert isinstance(wind, Medium)
    tab = TabulatedWind.sample(wind, jnp.linspace(1.0, 50.0, 100))
    assert isinstance(tab, Medium)


def test_tabulated_matches_analytic_at_grid_points(wind):
    r_grid = jnp.exp(jnp.linspace(0.0, jnp.log(50.0), 200))
    tab = TabulatedWind.sample(wind, r_grid)
    assert jnp.allclose(tab.v(r_grid), wind.v(r_grid))
    assert jnp.allclose(tab.rho(r_grid), wind.rho(r_grid))
    assert jnp.allclose(tab.dv_dr(r_grid), wind.dv_dr(r_grid))
    # dilution is purely geometric, so exact everywhere.
    r = jnp.asarray([1.5, 3.0, 10.0])
    assert jnp.allclose(tab.dilution(r), wind.dilution(r))


@pytest.mark.slow
def test_tabulated_reproduces_analytic_profile(wind):
    # A solved medium plugs into the solver exactly like the analytic wind.
    r_grid = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), 1024))
    tab = TabulatedWind.sample(wind, r_grid)
    line = LineData(tau_scale=jnp.asarray(5.0), n_l_exponent=jnp.asarray(0.0))
    x = jnp.linspace(-1.3, 1.3, 41)
    analytic = solve_profile(wind, line, x, n_p=200)
    tabulated = solve_profile(tab, line, x, n_p=200)
    assert jnp.max(jnp.abs(tabulated - analytic)) < 1e-3
