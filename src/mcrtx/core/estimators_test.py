"""Estimators: deposit binning, alive/out-of-range exclusion, scatter-add."""

import jax.numpy as jnp

from mcrtx.core import Estimators, deposit_escape, deposit_resonance
from mcrtx.core.packets import PacketState


def _packets(nu, alive, weight):
    n = len(nu)
    zeros = jnp.zeros((n,))
    return PacketState(r=zeros, mu=zeros, nu=jnp.asarray(nu), weight=jnp.asarray(weight), alive=jnp.asarray(alive))


def test_zeros_shapes():
    est = Estimators.zeros(n_nu=4, n_shells=3)
    assert est.spectrum.shape == (4,)
    assert est.j_bar.shape == (3,)


def test_deposit_escape_bins_escaped_in_range_only():
    est = Estimators.zeros(n_nu=3, n_shells=1)
    edges = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    packets = _packets(
        nu=[0.5, 1.5, 2.5, 1.5, -1.0, 5.0],
        alive=[False, False, False, True, False, False],  # 4th escaped-but-alive excluded
        weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    out = deposit_escape(est, packets, edges)
    # one escaped in-range packet per bin; alive and out-of-range dropped.
    assert jnp.array_equal(out.spectrum, jnp.asarray([1.0, 1.0, 1.0]))


def test_deposit_escape_accumulates_weights():
    est = Estimators.zeros(n_nu=2, n_shells=1)
    edges = jnp.asarray([0.0, 1.0, 2.0])
    packets = _packets(nu=[0.2, 0.7, 1.5], alive=[False, False, False], weight=[2.0, 3.0, 4.0])
    out = deposit_escape(est, packets, edges)
    assert jnp.array_equal(out.spectrum, jnp.asarray([5.0, 4.0]))


def test_deposit_resonance_scatter_add():
    est = Estimators.zeros(n_nu=1, n_shells=4)
    shell_idx = jnp.asarray([0, 2, 2, 3])
    weight = jnp.asarray([1.0, 1.0, 2.0, 5.0])
    out = deposit_resonance(est, shell_idx, weight)
    assert jnp.array_equal(out.j_bar, jnp.asarray([1.0, 0.0, 3.0, 5.0]))
