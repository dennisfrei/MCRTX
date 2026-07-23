# Examples

Runnable, editable playgrounds for the capabilities implemented so far. Each
script has a `CONSTANTS` block at the top — tweak it and re-run.

Plots need matplotlib (the `examples` dependency group); without it the scripts
still print Rich-formatted Unicode sparklines to the terminal.

```bash
uv sync --group examples                          # once, for PNG output
uv run python examples/pcygni.py                  # P-Cygni profile playground
uv run python examples/line_force.py              # radiative line-force shapes
uv run python examples/multiline.py               # additive vs coupled line profiles
uv run python examples/nlte_demo.py               # NLTE statistical equilibrium
uv run python examples/nlte_wind.py               # line thermalisation in a wind
uv run python examples/wind_structure.py          # beta-law wind structure
uv run python examples/mc_convergence.py          # Monte Carlo convergence
uv run python examples/tabulated_wind.py          # analytic vs tabulated medium
uv run python examples/fit_line.py                # simple differentiable fit
uv run python examples/multi_param_fit.py        # named multi-parameter fit
uv run python examples/units_roundtrip.py         # SI/CGS unit invariance
uv run python examples/source_functions.py        # source-function closures
uv run python examples/packet_launch.py           # Monte Carlo packet launch
uv run python examples/parameter_atlas.py         # line-profile parameter scan
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

## `line_force.py`

Compares optically thin, intermediate, and optically thick line-driving forces
through a beta-law wind. Each curve is normalised to its peak to show how line
saturation changes the radial acceleration profile.

## `multiline.py`

Builds a close two-line spectrum and compares the additive approximation with
the coupled formal solution. This shows how overlapping resonance lines differ
from well-separated lines.

## `nlte_wind.py`

Sweeps the collision rate in a two-level wind-line model. The profiles show the
transition from pure scattering to thermalised emission as collisions increase.

## `wind_structure.py`

Plots beta-law velocity fields for several velocity-law exponents and prints a
small table of velocity, velocity gradient, density, and dilution factor.

## `mc_convergence.py`

Compares Monte Carlo profiles with the deterministic reference solver at
increasing packet counts. Reports the RMS error to show the expected reduction
in statistical noise.

## `tabulated_wind.py`

Samples an analytic beta-law wind onto a radial grid and passes the resulting
`TabulatedWind` through the same reference solver. This mirrors how simulated
hydrodynamic or radiation-driven wind data can be consumed.

## `fit_line.py`

Generates a synthetic profile and uses `jax.grad` with gradient descent to fit
its Sobolev line strength while keeping the wind geometry fixed. This is the
smallest differentiable fitting example.

## `multi_param_fit.py`

Fits four explicitly named parameters at once with Adam: $\beta$,
$v_\mathrm{phot}$, Sobolev line strength $\tau$, and the radial lower-level
exponent. It also shows that one continuum-normalised profile does not uniquely
identify every physical parameter: different combinations can produce similar
profiles.

## `units_roundtrip.py`

Builds the same physical scenario from SI and CGS quantities, then compares the
returned velocity axes and profiles. It also prints the derived reference time
scale $t_\mathrm{ref} = R_\star / v_\infty$.

## `source_functions.py`

Plots the optically thin, self-consistent scattering, and collisionally coupled
NLTE source functions as functions of radius.

## `packet_launch.py`

Inspects the initial photospheric packet distribution: directions, frequencies,
unit weights, and deterministic interaction-key stream separation.

## `parameter_atlas.py`

Scans three velocity-law exponents and three line strengths to show how $\beta$
and $\tau$ jointly shape the emergent P-Cygni profile.
