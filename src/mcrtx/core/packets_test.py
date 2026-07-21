"""PacketState.launch_photospheric: shapes, ranges, determinism."""

import jax.numpy as jnp

from mcrtx.core import PacketState


def test_launch_shapes_and_initial_values(key):
    n = 1000
    p = PacketState.launch_photospheric(key, n, band=(0.9, 1.1))
    for field in (p.r, p.mu, p.nu, p.weight, p.alive):
        assert field.shape == (n,)
    assert jnp.all(p.r == 1.0)
    assert jnp.all(p.weight == 1.0)
    assert jnp.all(p.alive)


def test_launch_directions_outward_and_frequencies_in_band(key):
    p = PacketState.launch_photospheric(key, 5000, band=(0.9, 1.1))
    assert jnp.all(p.mu >= 0.0) and jnp.all(p.mu <= 1.0)
    assert jnp.all(p.nu >= 0.9) and jnp.all(p.nu <= 1.1)


def test_launch_is_deterministic(key):
    a = PacketState.launch_photospheric(key, 128, band=(0.9, 1.1))
    b = PacketState.launch_photospheric(key, 128, band=(0.9, 1.1))
    assert jnp.array_equal(a.mu, b.mu)
    assert jnp.array_equal(a.nu, b.nu)
