# MCRTX

> GPU-native, differentiable radiative transfer building blocks in JAX.

MCRTX is a *base library* for radiative transfer: composable, solver-agnostic
physics modules (opacities, source functions), exchangeable media and geometry
models, and multiple transfer solvers. Everything is pure JAX, so the whole
stack is `jit`/`vmap`/`grad`-compatible and runs unchanged on CPU or GPU.

## Installation

Requires Python ≥ 3.13.

```bash
uv add mcrtx           # add to an existing uv project
pip install mcrtx      # or plain pip
```

For GPU (CUDA 12) support:

```bash
uv add "mcrtx[cuda]"
```

## Usage

The physics and media modules are plain JAX functions and pytrees — compose,
`jit`, `vmap`, and differentiate them freely.

```python
import jax
import jax.numpy as jnp
from mcrtx.media import BetaLawWind
from mcrtx.physics import sobolev_tau

# A spherically symmetric beta-law wind (dimensionless units).
wind = BetaLawWind(beta=jnp.asarray(0.8), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))

r = jnp.linspace(1.0, 20.0, 4)
wind.v(r)         # velocity profile
wind.dv_dr(r)     # radial gradient

# Sobolev optical depth at a resonance point, jit-compiled.
tau = jax.jit(sobolev_tau)
tau(jnp.asarray(1.0), wind.rho(jnp.asarray(2.0)), jnp.asarray(0.6),
    wind.v(jnp.asarray(2.0)) / 2.0, wind.dv_dr(jnp.asarray(2.0)))

# Differentiate straight through the model — e.g. d v(r=2) / d beta.
jax.grad(lambda b: BetaLawWind(b, jnp.asarray(0.01), jnp.asarray(1.0)).v(jnp.asarray(2.0)))(jnp.asarray(0.8))
```

## Documentation

- [`docs/physics.md`](docs/physics.md) — the physical concepts behind the simulations
- [`docs/architecture.md`](docs/architecture.md) — layer model, project status, MVP scope
- [`docs/concept.md`](docs/concept.md) — full design document (German)
- [`docs/development.md`](docs/development.md) — running tests, linting, type checks

## License

MIT
