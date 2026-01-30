import builtins
import importlib.util
import io

import numpy as np
import pandas as pd
import pytest


def _import_gcrsim_with_stubs(monkeypatch):
    """
    Import gcrsim.py while stubbing:
      - open("rgb_color_list.txt")  -> tiny in-memory file
      - pd.read_csv("SN_m_tot_V2.0.csv") -> tiny DataFrame with required columns
    """
    # Stub rgb_color_list.txt
    rgb_text = "red\t#ff0000\nblue\t#0000ff\n"
    real_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("rgb_color_list.txt"):
            return io.StringIO(rgb_text)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open, raising=True)

    # Stub sunspot CSV
    def fake_read_csv(path, sep=";", engine="python", *args, **kwargs):
        # Minimal set of rows/cols the module expects after it renames columns:
        # ["year","month","date","mean","std_dev","num_obs","marker"]
        # and then filters date >= 1986.707, assigns solar_cycle, etc.
        df = pd.DataFrame(
            {
                "year": [2018, 2018, 2019],
                "month": [1, 2, 1],
                "date": [2018.04, 2018.12, 2019.04],
                "mean": [10.0, 20.0, 15.0],
                "std_dev": [1.0, 1.0, 1.0],
                "num_obs": [30, 30, 30],
                "marker": ["D", "D", "D"],
            }
        )
        return df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    # Import from local path
    spec = importlib.util.spec_from_file_location("gcrsim", "gcrsim.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gcrsim(monkeypatch):
    """Test for gcrsim wrapper script"""
    return _import_gcrsim_with_stubs(monkeypatch)


def test_mean_excitation_energy_in_bounds(gcrsim):
    """Bragg-log average should lie between min/max elemental I values used in the function
    (Hg=800, Cd=469, Te=485) for x in [0,1]."""
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        I_mean = gcrsim.mean_excitation_energy_HgCdTe(x)
        assert np.isfinite(I_mean)
        assert 469.0 <= I_mean <= 800.0


def test_radiation_length_positive(gcrsim):
    """Test to ensure physically sensible radiation length"""
    for x in [0.0, 0.445, 1.0]:
        X0 = gcrsim.radiation_length_HgCdTe(x)
        assert np.isfinite(X0)
        assert X0 > 0


def test_density_positive(gcrsim):
    """Test to ensure physically sensible density"""
    for x in [0.0, 0.445, 1.0]:
        rho = gcrsim.density_HgCdTe(x)
        assert np.isfinite(rho)
        assert rho > 0


def test_mean_Z_A_reasonable(gcrsim):
    """Test to ensure reasonable values for mean Z and mean Z"""
    Z_mean, A_mean = gcrsim.mean_Z_A_HgCdTe(0.445)
    assert np.isfinite(Z_mean) and np.isfinite(A_mean)
    assert 40 < Z_mean < 80  # rough sanity: between Cd/Te/Hg scale
    assert 100 < A_mean < 220
