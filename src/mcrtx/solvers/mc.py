"""Monte Carlo Sobolev packet propagator (MVP solver).

Fixed-length ``lax.scan`` over interaction steps with alive-masking (D5);
gradients flow only through smooth weight updates and resonance geometry,
never through discrete path topology (D3). Re-emission directions are
sampled isotropically in the comoving frame — a parameter-free
distribution, hence no gradient needed (concept sec. 3.4).
"""

from __future__ import annotations

__all__: list[str] = []

# TODO(M0): step function
#   (PacketState, Estimators) -> (PacketState, Estimators)
#   1. resonance radius per packet from observer-frame nu and v(r) (analytic
#      for monotonic v)
#   2. tau_sob at resonance; weight split: transmit exp(-tau) / scatter rest
#   3. scattered branch: isotropic CMF re-emission, frame transform, new nu
#   4. escape / star-hit masking; deposit escaped weight into spectrum
# TODO(M0): driver ``run(config, wind, line, key) -> Estimators`` wrapping the
#   step in ``jax.lax.scan`` with ``config.max_interactions``.
