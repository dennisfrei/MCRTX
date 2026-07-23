# Examples

Runnable, editable playgrounds for the capabilities implemented so far. Each
script has a `CONSTANTS` block at the top — tweak it and re-run.

Plots need matplotlib (the `examples` dependency group); without it the scripts
still print Unicode sparklines to the terminal.

```bash
uv sync --group examples                          # once, for PNG output
uv run python examples/pcygni.py                  # P-Cygni profile playground
uv run python examples/nlte_demo.py               # NLTE statistical equilibrium
```

PNGs are written to `examples/output/` (git-ignored).

## `pcygni.py`

Builds one physical wind-line scenario (`WindLineScenario.from_physical`) and
solves it four ways — reference vs Monte Carlo solver × thin (M0) vs
self-consistent (M1) source — overlaid on a Doppler-velocity axis, with the
equivalent width of each printed. Good for seeing the solvers agree and how the
self-consistent source keeps the line photon-conserving.

## `nlte_demo.py`

Exercises the domain-agnostic NLTE core (`mcrtx.physics.nlte`): a 3-level toy
atom in collisional (LTE/Boltzmann) equilibrium, then a growing radiation field
pumps one transition out of LTE. Shows populations shifting as the radiation
field strengthens.
