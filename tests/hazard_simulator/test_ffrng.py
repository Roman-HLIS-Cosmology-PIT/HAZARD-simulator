"""Tests of the ffrng functionality."""

from hazard_simulator.ffrng import FastForwardRNG


def _adv(x):
    x.advance(1)


def test_ffrng():
    """Simple test of ffrng --- just a stub now."""

    rng = FastForwardRNG(676767)
    for _ in range(5):
        _adv(rng)
    assert rng.position == 5
