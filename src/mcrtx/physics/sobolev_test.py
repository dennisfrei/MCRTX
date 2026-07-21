"""Sobolev optical depth: geometry and gradient sanity."""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.physics import sobolev_tau
from mcrtx.physics.sobolev import projected_velocity_gradient


def test_projected_gradient_limits():
    # Radial ray: pure dv/dr; tangential ray: pure v/r.
    assert projected_velocity_gradient(jnp.asarray(1.0), 0.3, 0.7) == pytest.approx(0.7)
    assert projected_velocity_gradient(jnp.asarray(0.0), 0.3, 0.7) == pytest.approx(0.3)


def test_tau_scales_linearly_with_strength_and_population():
    tau = sobolev_tau(jnp.asarray(2.0), jnp.asarray(3.0), jnp.asarray(0.5), 0.2, 0.4)
    assert 2 * tau == pytest.approx(float(sobolev_tau(jnp.asarray(4.0), jnp.asarray(3.0), jnp.asarray(0.5), 0.2, 0.4)))


def test_tau_differentiable_in_strength():
    grad = jax.grad(lambda s: sobolev_tau(s, jnp.asarray(1.0), jnp.asarray(0.7), 0.2, 0.4))(jnp.asarray(1.5))
    assert jnp.isfinite(grad) and grad > 0
