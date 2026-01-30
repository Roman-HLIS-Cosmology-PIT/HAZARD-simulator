"""Test functions for material properties."""

import gcrsim
import numpy as np


def test_mean_excitation_energy_in_bounds():
    """Bragg-log average should lie between min/max elemental I values used in the function
    (Hg=800, Cd=469, Te=485) for x in [0,1]."""
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        I_mean = gcrsim.mean_excitation_energy_HgCdTe(x)
        assert np.isfinite(I_mean)
        assert 469.0 <= I_mean <= 800.0


def test_radiation_length_positive():
    """Test to ensure physically sensible radiation length"""
    for x in [0.0, 0.445, 1.0]:
        X0 = gcrsim.radiation_length_HgCdTe(x)
        assert np.isfinite(X0)
        assert X0 > 0


def test_density_positive():
    """Test to ensure physically sensible density"""
    for x in [0.0, 0.445, 1.0]:
        rho = gcrsim.density_HgCdTe(x)
        assert np.isfinite(rho)
        assert rho > 0


def test_mean_Z_A_reasonable():
    """Test to ensure reasonable values for mean Z and mean Z"""
    Z_mean, A_mean = gcrsim.mean_Z_A_HgCdTe(0.445)
    assert np.isfinite(Z_mean) and np.isfinite(A_mean)
    assert 40 < Z_mean < 80  # rough sanity: between Cd/Te/Hg scale
    assert 100 < A_mean < 220

