"""NLTE statistical equilibrium: two-level ratio, LTE limit, batching, gradients."""

import jax
import jax.numpy as jnp
import pytest

from mcrtx.physics.nlte import AtomicModel, assemble_rate_matrix, statistical_equilibrium


def test_two_level_population_ratio():
    model = AtomicModel(g=jnp.asarray([1.0, 3.0]), lower=jnp.asarray([0]), upper=jnp.asarray([1]))
    rate = assemble_rate_matrix(model, r_up=jnp.asarray([2.0]), r_down=jnp.asarray([5.0]))
    n = statistical_equilibrium(rate)
    assert float(n[1] / n[0]) == pytest.approx(2.0 / 5.0)
    assert float(jnp.sum(n)) == pytest.approx(1.0)


def test_lte_recovers_boltzmann():
    # Detailed balance on every transition => the stationary state is Boltzmann.
    g = jnp.asarray([1.0, 3.0, 5.0])
    energy = jnp.asarray([0.0, 1.0, 2.5])
    lower = jnp.asarray([0, 1, 0])
    upper = jnp.asarray([1, 2, 2])
    model = AtomicModel(g=g, lower=lower, upper=upper)
    kt = 0.7
    r_down = jnp.asarray([1.0, 2.0, 0.5])
    boltz_factor = (g[upper] / g[lower]) * jnp.exp(-(energy[upper] - energy[lower]) / kt)
    r_up = r_down * boltz_factor

    n = statistical_equilibrium(assemble_rate_matrix(model, r_up, r_down), n_total=1.0)
    boltzmann = g * jnp.exp(-energy / kt)
    boltzmann = boltzmann / jnp.sum(boltzmann)
    assert jnp.allclose(n, boltzmann, rtol=1e-6)


def test_populations_are_positive_and_normalised():
    model = AtomicModel(
        g=jnp.asarray([2.0, 2.0, 4.0, 1.0]),
        lower=jnp.asarray([0, 0, 1, 2]),
        upper=jnp.asarray([1, 2, 3, 3]),
    )
    r_up = jnp.asarray([1.0, 0.4, 0.8, 0.2])
    r_down = jnp.asarray([3.0, 2.0, 1.5, 1.0])
    n = statistical_equilibrium(assemble_rate_matrix(model, r_up, r_down), n_total=7.0)
    assert jnp.all(n > 0.0)
    assert float(jnp.sum(n)) == pytest.approx(7.0)


def test_batched_over_shells():
    model = AtomicModel(g=jnp.asarray([1.0, 3.0]), lower=jnp.asarray([0]), upper=jnp.asarray([1]))
    r_up = jnp.asarray([[1.0], [2.0], [4.0]])  # 3 shells, 1 transition
    r_down = jnp.asarray([[4.0], [4.0], [4.0]])
    rates = jax.vmap(assemble_rate_matrix, in_axes=(None, 0, 0))(model, r_up, r_down)
    n = jax.vmap(statistical_equilibrium)(rates)
    assert n.shape == (3, 2)
    assert jnp.allclose(n[:, 1] / n[:, 0], jnp.asarray([0.25, 0.5, 1.0]))


def test_differentiable_in_rates():
    model = AtomicModel(g=jnp.asarray([1.0, 3.0]), lower=jnp.asarray([0]), upper=jnp.asarray([1]))

    def upper_fraction(r_up_scalar):
        rate = assemble_rate_matrix(model, jnp.asarray([r_up_scalar]), jnp.asarray([4.0]))
        return statistical_equilibrium(rate)[1]

    g = jax.grad(upper_fraction)(jnp.asarray(2.0))
    assert jnp.isfinite(g)
    assert g > 0.0  # more pumping -> more upper-level population
