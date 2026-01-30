import builtins
import importlib.util
import io

import numpy as np
import pandas as pd
import pytest


def _import_gcrsim_with_stubs(monkeypatch):
    """Test for gcrsim driver script"""
    real_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("rgb_color_list.txt"):
            return io.StringIO("red\t#ff0000\n")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open, raising=True)
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *a, **k: pd.DataFrame(
            {
                "year": [2018],
                "month": [1],
                "date": [2018.04],
                "mean": [10.0],
                "std_dev": [1.0],
                "num_obs": [30],
                "marker": ["D"],
            }
        ),
        raising=True,
    )

    spec = importlib.util.spec_from_file_location("gcrsim", "gcrsim.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gcrsim(monkeypatch):
    """Test for  gcrsim  monkey business"""
    return _import_gcrsim_with_stubs(monkeypatch)


def test_compute_curvature_straight_line_zero(gcrsim):
    """Test to ensure curvature gives sensible answers"""
    CRS = gcrsim.CosmicRaySimulation
    positions = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    kappa = CRS.compute_curvature(positions)
    assert np.allclose(kappa, 0.0)


def test_transform_angles_identity_when_primary_along_z(gcrsim):
    """Test for proper angle identity for primary"""
    CRS = gcrsim.CosmicRaySimulation
    theta_p, phi_p = 0.0, 0.0  # along +z
    theta_d, phi_d = 0.7, 1.2
    tg, pg = CRS.transform_angles(theta_p, phi_p, theta_d, phi_d)
    assert np.isclose(tg, theta_d)
    assert np.isclose(pg, phi_d)
