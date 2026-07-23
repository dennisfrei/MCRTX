"""Packet-launch playground: inspect the Monte Carlo initial distribution.

Photospheric packets are launched with isotropic outward directions, flat
frequencies across a chosen band, and unit continuous weights.

    uv run --group examples python examples/packet_launch.py   # + PNG
    uv run python examples/packet_launch.py                     # terminal only
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from _plot import console, save_png, show_series

from mcrtx.core import PacketState
from mcrtx.core.rng import interaction_key

# --- CONSTANTS: edit me ------------------------------------------------------
N_PACKETS = 10_000
BAND = (-1.3, 1.3)
# -----------------------------------------------------------------------------


def main() -> None:
    root = jax.random.key(42)
    packets = PacketState.launch_photospheric(root, N_PACKETS, BAND)
    mu = jnp.sort(packets.mu)
    nu = jnp.sort(packets.nu)
    rank = jnp.linspace(0.0, 1.0, N_PACKETS)
    console.print(f"packets: {N_PACKETS:,}, alive={int(jnp.sum(packets.alive)):,}", markup=False)
    console.print(f"mu mean={float(jnp.mean(packets.mu)):.3f}, frequency mean={float(jnp.mean(packets.nu)):.3f}", markup=False)
    console.print(f"weight range=[{float(packets.weight.min()):.1f}, {float(packets.weight.max()):.1f}]", markup=False)
    console.print(
        f"interaction keys differ by stream: {bool(jnp.any(interaction_key(root, 0, 0) != interaction_key(root, 0, 1)))}",
        markup=False,
    )

    series = {"mu empirical CDF": [float(v) for v in mu], "nu empirical CDF": [float(v) for v in nu]}
    show_series([float(v) for v in rank], series, lo=min(BAND), hi=max(BAND))
    png = save_png(
        "examples/output/packet_launch.png",
        [float(v) for v in rank],
        series,
        xlabel="Empirical quantile",
        ylabel="Sorted packet value",
        title="Photospheric packet launch distributions",
    )
    console.print(f"PNG: {png}" if png else "(install the 'examples' group for a PNG)", markup=False)


if __name__ == "__main__":
    main()
