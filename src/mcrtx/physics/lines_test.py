"""Atomic data to LineData: Doppler centres, tau_scale, unit invariance, validation."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import constants as const
from unxt import Quantity

from mcrtx.dynamics import sobolev_line_force
from mcrtx.media import BetaLawWind
from mcrtx.physics import build_line_list
from mcrtx.physics.lines import _OSCILLATOR_CONSTANT, _SPEED_OF_LIGHT
from mcrtx.solvers.multiline import multiline_profile_coupled
from mcrtx.units import ReferenceScales

# C IV resonance doublet (NIST): rest wavelengths and absorption oscillator strengths.
CIV_LAMBDA = [1548.19, 1550.77]
CIV_F = [0.190, 0.0952]


@pytest.fixture
def scales():
    # R_star = 1e12 cm, v_inf = 1000 km/s, nu_0 of C IV 1548, n_l = 1e8 cm^-3.
    return ReferenceScales.from_quantities(
        Quantity(1e12, "cm"),
        Quantity(1e8, "cm/s"),
        Quantity(1.936e15, "Hz"),
        Quantity(1e8, "cm**-3"),
    )


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(0.8), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _civ(scales, **kwargs):
    return build_line_list(Quantity(jnp.asarray(CIV_LAMBDA), "angstrom"), CIV_F, scales, **kwargs)


def test_centers_follow_the_doppler_formula(scales):
    centers, _ = _civ(scales)
    expected = (CIV_LAMBDA[1] - CIV_LAMBDA[0]) / CIV_LAMBDA[0] * 2.99792458e10 / 1e8
    assert float(centers[0]) == 0.0  # the first line is the default zero point
    assert float(centers[1]) == pytest.approx(expected, rel=1e-12)


def test_redder_lines_sit_at_positive_velocity(scales):
    centers, _ = _civ(scales)
    assert centers[1] > centers[0]


def test_reference_wavelength_moves_the_zero_point(scales):
    # The axis is the classical (lam - lam_ref) / lam_ref, so re-referencing also
    # rescales the spacing by lam_ref0 / lam_ref1 - a ~0.2% effect for this doublet.
    centers, _ = _civ(scales)
    shifted, _ = _civ(scales, reference_wavelength=Quantity(CIV_LAMBDA[1], "angstrom"))
    assert float(shifted[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(shifted[0]) < 0.0  # the bluer line now sits left of zero
    rescaled = jnp.diff(centers) * CIV_LAMBDA[0] / CIV_LAMBDA[1]
    assert jnp.allclose(jnp.abs(jnp.diff(shifted)), rescaled, rtol=1e-12)


def test_astropy_constants_are_the_cgs_values():
    # Guards the unit system: pi e^2 / (m_e c) = 0.02654 cm^2/s in CGS, and is off
    # by many orders of magnitude if astropy's SI value of e leaks in instead.
    assert pytest.approx(0.026540, rel=1e-5) == _OSCILLATOR_CONSTANT
    assert _SPEED_OF_LIGHT == 2.99792458e10


def test_tau_scale_matches_the_sobolev_prefactor(scales):
    _, lines = _civ(scales)
    prefactor = (np.pi * const.e.esu**2 / (const.m_e.cgs * const.c.cgs)).to_value("cm2 / s")
    expected = prefactor * CIV_F[0] * CIV_LAMBDA[0] * 1e-8 * scales.density * scales.time
    assert float(lines[0].tau_scale) == pytest.approx(expected, rel=1e-12)


def test_tau_scale_is_linear_in_oscillator_strength(scales):
    _, lines = _civ(scales)
    ratio = float(lines[0].tau_scale) / float(lines[1].tau_scale)
    assert ratio == pytest.approx(CIV_F[0] * CIV_LAMBDA[0] / (CIV_F[1] * CIV_LAMBDA[1]), rel=1e-12)


def test_input_unit_system_does_not_matter(scales):
    centers, lines = _civ(scales)
    nm = Quantity(jnp.asarray([lam / 10.0 for lam in CIV_LAMBDA]), "nm")
    centers_nm, lines_nm = build_line_list(nm, CIV_F, scales)
    assert jnp.allclose(centers, centers_nm, rtol=1e-12)
    assert jnp.allclose(
        jnp.asarray([line.tau_scale for line in lines]),
        jnp.asarray([line.tau_scale for line in lines_nm]),
        rtol=1e-12,
    )


def test_stimulated_correction_reduces_tau(scales):
    _, plain = _civ(scales)
    _, corrected = _civ(scales, stimulated=0.25)
    assert float(corrected[0].tau_scale) == pytest.approx(0.25 * float(plain[0].tau_scale), rel=1e-12)


def test_per_line_fields_broadcast(scales):
    _, lines = _civ(scales, n_l_exponent=[0.0, -1.0], g_lower=2.0, g_upper=[4.0, 2.0])
    assert [float(line.n_l_exponent) for line in lines] == [0.0, -1.0]
    assert [line.g_lower for line in lines] == [2.0, 2.0]
    assert [line.g_upper for line in lines] == [4.0, 2.0]


def test_rejects_a_non_length_wavelength(scales):
    with pytest.raises(ValueError, match="dimension"):
        build_line_list(Quantity(jnp.asarray([1.0, 2.0]), "Hz"), CIV_F, scales)


def test_rejects_mismatched_f_values(scales):
    with pytest.raises(ValueError, match="expected"):
        build_line_list(Quantity(jnp.asarray(CIV_LAMBDA), "angstrom"), [0.1], scales)


def test_rejects_a_bad_per_line_field(scales):
    with pytest.raises(ValueError, match="n_l_exponent"):
        _civ(scales, n_l_exponent=[0.0, 1.0, 2.0])


def test_feeds_the_line_force(scales, wind):
    _, lines = _civ(scales)
    r = jnp.asarray([1.5, 2.0, 5.0])
    g = sobolev_line_force(wind, lines, r)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.all(g > 0.0)


def test_feeds_the_coupled_solver(scales, wind):
    # A thinner wind keeps the doublet off full saturation so the profile has structure.
    thin = ReferenceScales.from_quantities(
        Quantity(1e12, "cm"), Quantity(1e8, "cm/s"), Quantity(1.936e15, "Hz"), Quantity(1e3, "cm**-3")
    )
    centers, lines = _civ(thin)
    x = jnp.linspace(-2.0, 2.5, 41)
    flux = multiline_profile_coupled(wind, [float(c) for c in centers], lines, x, n_p=200)
    assert jnp.all(jnp.isfinite(flux))
    assert jnp.any(flux < 0.99)  # the doublet leaves an absorption trough
