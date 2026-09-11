"""Tests of the ffrng functionality."""

import numpy as np
from hazard_simulator.ffrng import FastForwardRNG


def _adv(x):
    x.advance(1)


def test_ffrng():
    """Simple test of ffrng --- just a stub now."""

    rng = FastForwardRNG(676767)
    for _ in range(8):
        _adv(rng)
    assert rng.position == 8

    rng2 = FastForwardRNG(676767)
    rng2.jump_power_of_two(k=2)
    rng2.advance(8)

    # check lack of correlation
    n = 100000
    x = np.zeros(n)
    y = np.zeros(n)
    for j in range(n):
        x[j] = rng.random()
        y[j] = rng2.random()
    assert 0.245 < np.mean(x * y) < 0.255

    d = rng.get_state()
    assert d["position"] == 100008
    assert d["bitgen_state"]["bit_generator"] == "PCG64"

    rng.goto(12000)
    rng2 = FastForwardRNG(676767)
    rng2.advance(12000)

    q1 = rng.random()
    q2 = rng2.random()
    assert np.abs(q1 - q2) < 1.0e-7

    rng2 = FastForwardRNG(676767)
    q1 = rng.random()
    q2 = rng2.random()
    assert np.abs(q1 - q2) > 1.0e-7
    rng2.set_state(rng.get_state())
    q1 = rng.random()
    q2 = rng2.random()
    assert np.abs(q1 - q2) < 1.0e-7

    arr = rng2.multinomial(1000, np.ones(4) / 4.0, size=1)
    assert np.all(arr > 200)
    assert np.all(arr < 300)
