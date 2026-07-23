"""Sobolev line force: optically-thin and thick limits, differentiability, media."""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.dynamics import sobolev_line_force
from mcrtx.media import BetaLawWind, TabulatedWind
from mcrtx.physics.source import LineData

R = jnp.asarray([1.05, 1.5, 2.0, 5.0, 10.0])


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(0.8), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _line(tau_scale):
    return LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))


def test_optically_thin_force_scales_as_r_minus_two(wind):
    # Weak lines: g_rad ∝ 1/r^2 (the density cancels), so g * r^2 is constant.
    g = sobolev_line_force(wind, [_line(1e-6)] * 3, R)
    assert jnp.allclose(g * R**2, (g * R**2)[0], rtol=1e-4)


def test_optically_thick_force_is_cak_form(wind):
    # Strong lines saturate: g_rad = force_scale * N * |dv/dr| / (rho r^2).
    g = sobolev_line_force(wind, [_line(1e4)] * 3, R)
    metric = g * wind.rho(R) * R**2 / jnp.abs(wind.dv_dr(R))
    assert jnp.allclose(metric, 3.0, rtol=1e-3)


def test_more_lines_give_more_force(wind):
    one = sobolev_line_force(wind, [_line(5.0)], R)
    three = sobolev_line_force(wind, [_line(5.0)] * 3, R)
    assert jnp.all(three > one)


def test_force_is_differentiable(wind):
    g = jax.grad(lambda ts: jnp.sum(sobolev_line_force(wind, [_line(ts)], R)))(jnp.asarray(5.0))
    assert jnp.isfinite(g)
    assert g > 0.0  # stronger lines drive harder


def test_works_with_a_tabulated_medium(wind):
    # Swappability: a tabulated medium gives the same force (away from the base,
    # where interpolating the fast-varying dv/dr and rho costs some accuracy).
    tab = TabulatedWind.sample(wind, jnp.exp(jnp.linspace(0.0, jnp.log(50.0), 1024)))
    lines = [_line(5.0), _line(20.0)]
    r = jnp.asarray([1.5, 2.0, 5.0, 10.0])
    assert jnp.allclose(sobolev_line_force(tab, lines, r), sobolev_line_force(wind, lines, r), rtol=1e-2)
