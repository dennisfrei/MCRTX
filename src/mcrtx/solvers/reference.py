"""Deterministic SEI-style reference solver (validation baseline, D6).

Escape-probability line formation on a radial grid following Lamers,
Cerruti-Sola & Perinotto (1987), sharing the L2/L3 modules with the MC
solver — the first proof of the multi-method design. Target size: small
(~200 lines), clarity over speed.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(M0): p-z ray grid, source function S = W(r) * beta-weighted core
#   intensity, formal integral per (p, nu), flux integration to a profile.
