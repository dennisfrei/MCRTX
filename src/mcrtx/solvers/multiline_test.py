"""Multi-line composite spectrum: superposition, separation, continuum."""

import jax.numpy as jnp
import pytest

from mcrtx.media import BetaLawWind
from mcrtx.physics.source import LineData
from mcrtx.solvers.multiline import multiline_profile, multiline_profile_coupled
from mcrtx.solvers.reference import SourceModel, solve_profile

# Each line invokes the reference solver (compilation-heavy on a Pi).
pytestmark = pytest.mark.slow


@pytest.fixture
def wind():
    return BetaLawWind(beta=jnp.asarray(1.0), v_phot=jnp.asarray(0.01), mdot_scale=jnp.asarray(1.0))


def _line(tau_scale):
    return LineData(tau_scale=jnp.asarray(tau_scale), n_l_exponent=jnp.asarray(0.0))


def test_single_line_equals_reference(wind):
    x = jnp.linspace(-1.3, 1.3, 41)
    multi = multiline_profile(wind, [0.0], [_line(5.0)], x, n_p=200)
    single = solve_profile(wind, _line(5.0), x, n_p=200)
    assert jnp.allclose(multi, single, atol=1e-9)


def test_separated_lines_superpose_independently(wind):
    x = jnp.linspace(-9.0, 9.0, 181)
    r = multiline_profile(wind, [-6.0, 6.0], [_line(5.0), _line(20.0)], x, n_p=200)
    # Continuum between the lines and in the far wings.
    assert float(r[int(jnp.argmin(jnp.abs(x)))]) == pytest.approx(1.0, abs=1e-6)
    assert jnp.allclose(r[jnp.abs(x) > 8.0], 1.0, atol=1e-6)
    # Each line region is a genuine P-Cygni (absorption + emission).
    for center in (-6.0, 6.0):
        window = jnp.abs(x - center) < 1.3
        assert jnp.min(r[window]) < 0.9
        assert jnp.max(r[window]) > 1.1


def test_coupled_single_line_equals_reference(wind):
    x = jnp.linspace(-1.3, 1.3, 41)
    coupled = multiline_profile_coupled(wind, [0.0], [_line(5.0)], x, n_p=200)
    single = solve_profile(wind, _line(5.0), x, n_p=200)
    assert jnp.allclose(coupled, single, atol=1e-9)


def test_coupled_matches_additive_when_separated(wind):
    x = jnp.linspace(-9.0, 9.0, 121)
    lines = [_line(5.0), _line(20.0)]
    additive = multiline_profile(wind, [-6.0, 6.0], lines, x, n_p=200)
    coupled = multiline_profile_coupled(wind, [-6.0, 6.0], lines, x, n_p=200)
    assert jnp.allclose(coupled, additive, atol=1e-6)


def test_coupled_overlap_stays_physical(wind):
    # Two overlapping thick lines: the additive combination double-counts the
    # absorption and drops below zero; the coupled formal solution multiplies
    # transmissions, so the flux stays non-negative — and the two genuinely differ.
    x = jnp.linspace(-2.0, 2.0, 121)
    lines = [_line(20.0), _line(20.0)]
    additive = multiline_profile(wind, [-0.3, 0.3], lines, x, n_p=300, source=SourceModel.SELF_CONSISTENT)
    coupled = multiline_profile_coupled(wind, [-0.3, 0.3], lines, x, n_p=300, source=SourceModel.SELF_CONSISTENT)
    assert jnp.min(coupled) >= -1e-9  # physical: non-negative flux
    assert jnp.min(additive) < 0.0  # the additive approximation fails here
    assert jnp.max(jnp.abs(coupled - additive)) > 0.1  # genuinely different
