"""Escape probability and self-consistent source: limits, gradients, recovery."""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import escape_probability, sobolev_nlte_source, two_level_source


def test_limits():
    assert float(escape_probability(jnp.asarray(1e-10))) == pytest.approx(1.0, abs=1e-9)
    assert float(escape_probability(jnp.asarray(1e4))) < 1.5e-4


def test_continuous_at_switch():
    below = escape_probability(jnp.asarray(0.99e-4))
    above = escape_probability(jnp.asarray(1.01e-4))
    assert abs(float(below - above)) < 1e-6


def test_gradient_finite_everywhere():
    taus = jnp.asarray([1e-8, 1e-4, 1e-2, 1.0, 50.0])
    grads = jax.vmap(jax.grad(escape_probability))(taus)
    assert jnp.all(jnp.isfinite(grads))
    assert jnp.all(grads < 0)  # beta strictly decreasing


def _source_on_grid(tau_scale, r):
    wind = BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
    n_l = wind.rho(r)
    mu_star = jnp.sqrt(1.0 - 1.0 / r**2)
    fn = jax.vmap(two_level_source, in_axes=(None, 0, 0, 0, 0))
    return fn(jnp.asarray(tau_scale), n_l, wind.v(r) / r, wind.dv_dr(r), mu_star)


@pytest.mark.slow
def test_two_level_source_recovers_dilution_when_thin():
    wind = BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
    r = jnp.asarray([1.5, 2.0, 5.0, 20.0])
    s = _source_on_grid(1e-6, r)
    assert jnp.allclose(s, wind.dilution(r), atol=1e-4)


@pytest.mark.slow
def test_two_level_source_is_half_at_photosphere():
    # At r = 1 the star fills half the sky, so S = 0.5 for any optical depth.
    s = _source_on_grid(30.0, jnp.asarray([1.0]))
    assert float(s[0]) == pytest.approx(0.5, abs=1e-3)


@pytest.mark.slow
def test_two_level_source_differentiable_in_tau_scale():
    r = jnp.asarray([1.5, 3.0])
    g = jax.grad(lambda ts: jnp.sum(_source_on_grid(ts, r)))(jnp.asarray(5.0))
    assert jnp.isfinite(g)


def _nlte_on_grid(r, **kw):
    wind = BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))
    n_l = wind.rho(r)
    mu_star = jnp.sqrt(1.0 - 1.0 / r**2)
    fn = jax.vmap(lambda nl, vo, dv, ms: sobolev_nlte_source(jnp.asarray(5.0), nl, vo, dv, ms, **kw))
    return fn(n_l, wind.v(r) / r, wind.dv_dr(r), mu_star)


@pytest.mark.slow
def test_nlte_source_reduces_to_two_level_without_collisions():
    r = jnp.asarray([1.5, 2.0, 5.0])
    scattering = _source_on_grid(5.0, r)
    nlte = _nlte_on_grid(r, collision=0.0, w_core=0.5)  # w_core irrelevant without collisions
    assert jnp.allclose(scattering, nlte, atol=1e-9)


@pytest.mark.slow
def test_nlte_source_thermalises_to_lte():
    r = jnp.asarray([1.5, 3.0, 10.0])
    w_core, boltzmann = 0.5, 0.3
    s = _nlte_on_grid(r, collision=1.0e3, w_core=w_core, boltzmann=boltzmann)
    lte = (1.0 / w_core) / (1.0 / boltzmann - 1.0)  # Boltzmann-ratio source, radius-independent
    assert jnp.allclose(s, lte, rtol=1e-3)


@pytest.mark.slow
def test_nlte_source_differentiable_in_collision():
    r = jnp.asarray([1.5, 3.0])
    g = jax.grad(lambda c: jnp.sum(_nlte_on_grid(r, collision=c, w_core=0.5, boltzmann=0.3)))(jnp.asarray(1.0))
    assert jnp.isfinite(g)
