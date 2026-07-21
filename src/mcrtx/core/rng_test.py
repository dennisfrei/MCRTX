"""PRNG discipline: determinism and stream independence."""

import jax
import jax.numpy as jnp

from mcrtx.core import interaction_key


def test_deterministic(key):
    a = jax.random.uniform(interaction_key(key, 3, 0))
    b = jax.random.uniform(interaction_key(key, 3, 0))
    assert a == b


def test_streams_and_steps_differ(key):
    keys = [interaction_key(key, s, t) for s in range(3) for t in range(2)]
    draws = jnp.stack([jax.random.uniform(k) for k in keys])
    assert len(jnp.unique(draws)) == len(draws)
