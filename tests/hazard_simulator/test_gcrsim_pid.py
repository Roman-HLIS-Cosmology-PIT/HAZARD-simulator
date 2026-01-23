import importlib.util
import io
import builtins
import pandas as pd
import pytest


def _import_gcrsim_with_stubs(monkeypatch):
    rgb_text = "red\t#ff0000\n"
    real_open = builtins.open

    def fake_open(path, mode="r", *args, **kwargs):
        if str(path).endswith("rgb_color_list.txt"):
            return io.StringIO(rgb_text)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open, raising=True)

    def fake_read_csv(*args, **kwargs):
        return pd.DataFrame(
            {
                "year": [2018],
                "month": [1],
                "date": [2018.04],
                "mean": [10.0],
                "std_dev": [1.0],
                "num_obs": [30],
                "marker": ["D"],
            }
        )

    monkeypatch.setattr(pd, "read_csv", fake_read_csv, raising=True)

    spec = importlib.util.spec_from_file_location("gcrsim", "gcrsim.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gcrsim(monkeypatch):
    return _import_gcrsim_with_stubs(monkeypatch)


def test_pid_roundtrip_encode_decode(gcrsim):
    CRS = gcrsim.CosmicRaySimulation
    enc = CRS.encode_pid(species_idx=1, primary_idx=45, delta_idx=23)
    s = CRS.decode_pid(enc)
    assert s.startswith("H-")  # species_idx=1 is "H" in your mapping
    enc2 = CRS.encode_pid_string(s)
    assert enc2 == enc


def test_get_parent_pid_clears_delta_bits(gcrsim):
    CRS = gcrsim.CosmicRaySimulation
    enc = CRS.encode_pid(species_idx=2, primary_idx=7, delta_idx=999)
    parent = CRS.get_parent_pid(enc)
    # delta bits are the lowest 14 bits; parent must have them zero
    assert (parent & ((1 << 14) - 1)) == 0
    # species and primary should remain the same
    assert (parent >> 14) == (enc >> 14)


def test_encode_pid_string_rejects_bad_format(gcrsim):
    CRS = gcrsim.CosmicRaySimulation
    with pytest.raises(ValueError):
        CRS.encode_pid_string("not-a-pid")
    with pytest.raises(ValueError):
        CRS.encode_pid_string("H-X0045-D00023")
