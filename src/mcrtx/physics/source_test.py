"""Escape probability: limits, continuity at the Taylor switch, gradients."""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.physics.source import escape_probability


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
