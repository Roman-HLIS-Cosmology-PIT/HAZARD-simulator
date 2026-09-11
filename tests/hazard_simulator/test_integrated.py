"""Integrated test for making a frame."""


import numpy as np
from hazard_simulator.roman_pipeline_interface import generate_singleframe_cr


def test_frame():
    """Test for making a frame."""

    fr = generate_singleframe_cr(seed=2000, date=2027.0, dt=3.16)
    assert np.shape(fr) == (4088, 4088)
    assert 900 < np.count_nonzero(fr > 100.0) < 1300
    assert 500 < np.count_nonzero(fr > 1.0e3) < 900

    fr = generate_singleframe_cr(seed=2000, date=2027.0, dt=30.0)
    assert np.shape(fr) == (4088, 4088)
    assert 8000 < np.count_nonzero(fr > 100.0) < 11000
    assert 6000 < np.count_nonzero(fr > 1.0e3) < 8000
