"""Scenario layer: unit boundary, cgs/SI roundtrip, solver dispatch."""

import jax.numpy as jnp
import pytest
import unxt as u
from unxt import Quantity

from mcrtx.scenarios import (
    Method,
    RunConfig,
    WindLineResult,
    WindLineScenario,
    continuity_number_density,
    sobolev_tau_scale,
    solve,
)

KMS = u.unit("km / s")


def q(value, unit):
    return Quantity(value, u.unit(unit))


@pytest.fixture
def scenario():
    return WindLineScenario(
        r_star=q(7.0e10, "cm"),
        v_inf=q(2.0e8, "cm / s"),
        v_phot=q(2.0e6, "cm / s"),
        nu_0=q(2.0e15, "Hz"),
        density_ref=q(1.0e10, "cm**-3"),
        beta=1.0,
        mdot_scale=1.0,
        tau_scale=5.0,
    )


def test_reference_produces_pcygni(scenario):
    res = solve(scenario, RunConfig(n_bins=41))
    assert isinstance(res, WindLineResult)
    assert res.method is Method.REFERENCE
    v = res.velocity.ustrip(KMS)
    assert jnp.min(res.flux[v < 0]) < 0.9  # blueshifted absorption
    assert jnp.max(res.flux[v > 0]) > 1.1  # redshifted emission


def test_velocity_axis_is_x_times_vinf(scenario):
    res = solve(scenario, RunConfig(x_min=-1.0, x_max=1.0, n_bins=5))
    # v_inf = 2e8 cm/s = 2000 km/s.
    expected = jnp.asarray([-2000.0, -1000.0, 0.0, 1000.0, 2000.0])
    assert jnp.allclose(res.velocity.ustrip(KMS), expected)


def test_cgs_si_roundtrip(scenario):
    si = WindLineScenario(
        r_star=q(7.0e8, "m"),
        v_inf=q(2.0e6, "m / s"),
        v_phot=q(2.0e4, "m / s"),
        nu_0=q(2.0e15, "Hz"),
        density_ref=q(1.0e16, "m**-3"),
        beta=1.0,
        mdot_scale=1.0,
        tau_scale=5.0,
    )
    cfg = RunConfig(n_bins=31)
    cgs = solve(scenario, cfg)
    other = solve(si, cfg)
    assert jnp.allclose(cgs.flux, other.flux)  # identical dimensionless core result
    assert jnp.allclose(cgs.velocity.ustrip(KMS), other.velocity.ustrip(KMS))


def test_default_config(scenario):
    res = solve(scenario)
    assert res.flux.shape == (RunConfig().n_bins,)


def test_mc_dispatch_runs_and_is_pcygni(scenario):
    res = solve(scenario, RunConfig(n_bins=21, n_packets=100_000), Method.MC)
    assert res.method is Method.MC
    assert not bool(jnp.isnan(res.flux).any())
    v = res.velocity.ustrip(KMS)
    assert jnp.min(res.flux[v < 0]) < 0.9
    assert jnp.max(res.flux[v > 0]) > 1.1


def _ostar():
    # O-star C IV 1548 A resonance line.
    return WindLineScenario.from_physical(
        r_star=q(7.0e11, "cm"),
        v_inf=q(2.0e8, "cm / s"),
        v_phot=q(2.0e6, "cm / s"),
        nu_0=q(1.936e15, "Hz"),
        mdot=q(6.3e19, "g / s"),
        f_lu=0.19,
        lower_level_per_mass=q(6.3e19, "g**-1"),
        beta=1.0,
    )


def test_sobolev_tau_scale_is_dimensionless_and_linear_in_f_lu():
    kw = {
        "nu_0": q(2.0e15, "Hz"),
        "density_ref": q(1.0e6, "cm**-3"),
        "r_star": q(7.0e11, "cm"),
        "v_inf": q(2.0e8, "cm / s"),
    }
    t1 = sobolev_tau_scale(0.2, **kw)
    t2 = sobolev_tau_scale(0.4, **kw)
    assert isinstance(t1, float) and t1 > 0.0
    assert t2 == pytest.approx(2.0 * t1)


def test_continuity_density_scales_with_mdot_and_has_number_units():
    kw = {"r_star": q(7.0e11, "cm"), "v_inf": q(2.0e8, "cm / s"), "lower_level_per_mass": q(6.0e19, "g**-1")}
    n1 = continuity_number_density(q(1.0e19, "g / s"), **kw)
    n2 = continuity_number_density(q(2.0e19, "g / s"), **kw)
    assert float(n1.ustrip(u.unit("cm**-3"))) > 0.0
    assert float(n2.ustrip(u.unit("cm**-3"))) == pytest.approx(2.0 * float(n1.ustrip(u.unit("cm**-3"))))


def test_from_physical_sets_unit_mdot_scale_and_positive_tau():
    sc = _ostar()
    assert sc.mdot_scale == 1.0
    assert sc.tau_scale > 0.0
    assert float(sc.density_ref.ustrip(u.unit("cm**-3"))) > 0.0


def test_from_physical_produces_saturated_pcygni():
    res = solve(_ostar(), RunConfig(n_bins=41))
    v = res.velocity.ustrip(KMS)
    assert jnp.min(res.flux[v < 0]) < 0.5  # optically thick -> deep trough
    assert jnp.max(res.flux[v > 0]) > 1.1
