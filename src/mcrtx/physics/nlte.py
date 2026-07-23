"""Multi-level NLTE statistical equilibrium (domain-agnostic, D2).

The reusable core the concept calls for (C4/M1): per shell a single linear
system for the level populations, batched over shells with ``vmap`` +
``jnp.linalg.solve`` and fully differentiable. It knows nothing about winds,
Sobolev, or geometry — it takes an atom's transition list and the *total*
up/down rate coefficients (radiative + collisional) and returns populations.
Callers assemble those rates from whatever physics applies (Sobolev-beta and
``j_bar`` for the wind; something else for a disk or an SN ejecta).

Statistical equilibrium for level ``i``::

    sum_{j != i} R[i, j] n_j  =  n_i  sum_{j != i} R[j, i]

with ``R[i, j]`` the rate from ``j`` to ``i``. The system is rank-deficient by
one (probability is conserved), closed by number conservation ``sum_i n_i = n_total``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

__all__ = ["AtomicModel", "assemble_rate_matrix", "statistical_equilibrium"]


class AtomicModel(NamedTuple):
    """Static structure of a bound-bound multi-level atom.

    Attributes:
        g: Statistical weights per level, shape ``(L,)``.
        lower: Lower-level index of each transition, shape ``(T,)``.
        upper: Upper-level index of each transition, shape ``(T,)``.
    """

    g: Array
    lower: Array
    upper: Array


def assemble_rate_matrix(model: AtomicModel, r_up: Array, r_down: Array) -> Array:
    """Assemble the ``(L, L)`` rate matrix from per-transition rate coefficients.

    Args:
        model: The atom's level and transition structure.
        r_up: Total lower -> upper rate per transition (radiative + collisional), shape ``(T,)``.
        r_down: Total upper -> lower rate per transition, shape ``(T,)``.

    Returns:
        ``R`` with ``R[i, j]`` the rate from level ``j`` to level ``i``, shape ``(L, L)``.
    """
    length = model.g.shape[0]
    rate = jnp.zeros((length, length))
    rate = rate.at[model.upper, model.lower].add(r_up)  # into upper from lower
    rate = rate.at[model.lower, model.upper].add(r_down)  # into lower from upper
    return rate


def statistical_equilibrium(rate_matrix: Array, n_total: Array | float = 1.0) -> Array:
    """Solve multi-level statistical equilibrium for the level populations.

    Args:
        rate_matrix: ``R[i, j]`` = rate from level ``j`` to level ``i`` (>= 0 off-diagonal),
            shape ``(L, L)``. ``vmap`` this function over a leading shell axis.
        n_total: Total population the levels must sum to.

    Returns:
        Level populations summing to ``n_total``, shape ``(L,)``.
    """
    length = rate_matrix.shape[-1]
    off = rate_matrix * (1.0 - jnp.eye(length))  # drop self-rates
    out = jnp.sum(off, axis=-2)  # total rate leaving each level
    operator = off - jnp.diag(out)
    # Replace the (redundant) last balance equation with number conservation.
    operator = operator.at[-1, :].set(1.0)
    rhs = jnp.zeros((length,)).at[-1].set(n_total)
    return jnp.linalg.solve(operator, rhs)
