import hazard_simulator


def test_version():
    """Check to see that we can get the package version"""
    assert hazard_simulator.__version__ is not None
