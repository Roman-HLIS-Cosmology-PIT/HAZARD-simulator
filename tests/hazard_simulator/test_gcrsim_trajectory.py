"""Tests for gcrsim trajectory functions"""

import numpy as np
from hazard_simulator import gcrsim


def test_compute_curvature_straight_line_zero():
    """Test to ensure curvature gives sensible answers"""
    CRS = gcrsim.CosmicRaySimulation
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    kappa = CRS.compute_curvature(positions)
    assert np.allclose(kappa, 0.0)


def test_transform_angles_identity_when_primary_along_z():
    """Test for proper angle identity for primary"""
    CRS = gcrsim.CosmicRaySimulation
    theta_p, phi_p = 0.0, 0.0  # along +z
    theta_d, phi_d = 0.7, 1.2
    tg, pg = CRS.transform_angles(theta_p, phi_p, theta_d, phi_d)
    assert np.isclose(tg, theta_d)
    assert np.isclose(pg, phi_d)
