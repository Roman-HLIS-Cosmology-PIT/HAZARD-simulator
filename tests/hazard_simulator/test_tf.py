"""Thomas-Fermi test function"""

import numpy as np
from hazard_simulator.atomic_excitation.thomas_fermi import Potential


def test_potential():
    """Tests Thoman-Fermi potential for mercury (Hg, Z=80)."""

    Z = 80
    V = []
    for r in np.logspace(-13, -7, 13).tolist():
        V.append(Potential(Z, r))
    V = np.array(V)

    err = np.log(
        V
        / np.array(
            [
                1.1364954e06,
                3.4983894e05,
                1.0249952e05,
                2.6474454e04,
                5.1485469e03,
                5.9199949e02,
                3.2755749e01,
                8.8522288e-01,
                1.4506361e-02,
                1.8118277e-04,
                1.9926858e-06,
                2.0735748e-08,
                2.1080230e-10,
            ]
        )
    )

    assert np.all(np.abs(err) < 0.02)
