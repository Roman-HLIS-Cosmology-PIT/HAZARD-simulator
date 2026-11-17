"""Tests for F(T) functions."""

from hazard_simulator.atomic_excitation import ft_calculation


def test_sample_material():
    """Make a sample material."""

    ne = 1e28  # electron density, m^-3
    quality = 10  # Q-factor
    omega_naught = 1e6

    sample_material = ft_calculation.simple_material(ne, quality, omega_naught)
    print(len(sample_material["T_ARRAY"]))
    assert len(sample_material["T_ARRAY"]) == 50
