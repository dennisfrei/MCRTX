"""ReferenceScales: dimension validation, unit-system invariance, roundtrip."""

import pytest
import unxt as u
from unxt import Quantity

from mcrtx.units import ReferenceScales, Scale


def q(value, unit):
    return Quantity(value, u.unit(unit))


def test_from_quantities_stores_cgs_magnitudes():
    scales = ReferenceScales.from_quantities(q(7.0e10, "cm"), q(2000.0, "km / s"), q(1.0e15, "Hz"), q(1.0e10, "cm**-3"))
    assert scales.length == pytest.approx(7.0e10)
    assert scales.velocity == pytest.approx(2.0e8)  # 2000 km/s in cm/s
    assert scales.frequency == pytest.approx(1.0e15)
    assert scales.density == pytest.approx(1.0e10)
    assert scales.time == pytest.approx(7.0e10 / 2.0e8)


def test_input_unit_system_invariance():
    # Same physical scales expressed in CGS vs SI must store identically.
    cgs = ReferenceScales.from_quantities(q(7.0e10, "cm"), q(2.0e8, "cm / s"), q(1.0e15, "Hz"), q(1.0e10, "cm**-3"))
    si = ReferenceScales.from_quantities(q(7.0e8, "m"), q(2.0e6, "m / s"), q(1.0e15, "Hz"), q(1.0e16, "m**-3"))
    assert cgs.length == pytest.approx(si.length)
    assert cgs.velocity == pytest.approx(si.velocity)
    assert cgs.density == pytest.approx(si.density)


def test_nondimensionalize_is_input_unit_invariant():
    scales = ReferenceScales.from_quantities(q(7.0e10, "cm"), q(2.0e8, "cm / s"), q(1.0e15, "Hz"), q(1.0e10, "cm**-3"))
    in_cm = scales.nondimensionalize(q(1.4e11, "cm"), Scale.LENGTH)
    in_m = scales.nondimensionalize(q(1.4e9, "m"), Scale.LENGTH)
    assert float(in_cm) == pytest.approx(2.0)
    assert float(in_m) == pytest.approx(2.0)


def test_redimensionalize_roundtrip():
    scales = ReferenceScales.from_quantities(q(7.0e10, "cm"), q(2.0e8, "cm / s"), q(1.0e15, "Hz"), q(1.0e10, "cm**-3"))
    back = scales.redimensionalize(2.0, Scale.LENGTH, unit="cm")
    assert float(back.ustrip(u.unit("cm"))) == pytest.approx(1.4e11)
    again = scales.nondimensionalize(back, Scale.LENGTH)
    assert float(again) == pytest.approx(2.0)


def test_si_preset_yields_si_magnitudes():
    si = ReferenceScales.si()
    # A 1 km length is 1000 m in SI-scaled dimensionless units.
    assert float(si.nondimensionalize(q(1.0, "km"), Scale.LENGTH)) == pytest.approx(1000.0)


def test_wrong_dimension_raises():
    with pytest.raises(ValueError, match="length"):
        ReferenceScales.from_quantities(
            q(1.0, "km / s"),  # velocity where length expected
            q(1.0, "cm / s"),
            q(1.0, "Hz"),
            q(1.0, "cm**-3"),
        )


def test_nonpositive_scale_raises():
    with pytest.raises(ValueError, match="positive"):
        ReferenceScales.from_quantities(q(-1.0, "cm"), q(1.0, "cm / s"), q(1.0, "Hz"), q(1.0, "cm**-3"))
