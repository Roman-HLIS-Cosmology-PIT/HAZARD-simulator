"""PID encoding tests."""

import pytest
from hazard_simulator import gcrsim


def test_pid_roundtrip_encode_decode():
    """Test to verify pids can be encoded and decoded without error"""
    CRS = gcrsim.CosmicRaySimulation
    enc = CRS.encode_pid(species_idx=1, primary_idx=45, delta_idx=23)
    s = CRS.decode_pid(enc)
    assert s.startswith("H-")  # species_idx=1 is "H" in your mapping
    enc2 = CRS.encode_pid_string(s)
    assert enc2 == enc


def test_get_parent_pid_clears_delta_bits():
    """Test to make sure that binary delta bits are cleared when calling get_parent"""
    CRS = gcrsim.CosmicRaySimulation
    enc = CRS.encode_pid(species_idx=2, primary_idx=7, delta_idx=999)
    parent = CRS.get_parent_pid(enc)
    # delta bits are the lowest 14 bits; parent must have them zero
    assert (parent & ((1 << 14) - 1)) == 0
    # species and primary should remain the same
    assert (parent >> 14) == (enc >> 14)


def test_encode_pid_string_rejects_bad_format():
    """Test to make sure pid functions reject improperly formarted pids"""
    CRS = gcrsim.CosmicRaySimulation
    with pytest.raises(ValueError):
        CRS.encode_pid_string("not-a-pid")
    with pytest.raises(ValueError):
        CRS.encode_pid_string("H-X0045-D00023")
