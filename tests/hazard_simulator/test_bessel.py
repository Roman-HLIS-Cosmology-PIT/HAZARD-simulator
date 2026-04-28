"""Tests for Bessel function generator."""

import os

import asdf
import numpy as np
from hazard_simulator.atomic_excitation import bessel_function_generation


def test_bessel(tmp_path):
    """
    Test for Bessel function generator.

    This makes a table of Bessel functions, calling
    ``hazard_simulator.atomic_excitation.bessel_function_generation.bessel_main``
    and writing the results to a file. Then it checks that the correct spherical
    Bessel functions are in the file and it is formatted correctly.

    Parameters
    ----------
    tmp_path : str or str-like
        Directory to write the test ASDF file.

    Returns
    -------
    None

    """

    dzeta = 1.0e10 / 5.067730719e9  # convert to keV^-1

    tr = bessel_function_generation.bessel_main(
        outfile=str(tmp_path) + "/bessel.asdf",
        dzeta=dzeta,
        nzeta=50,
        nmax=30,
        r_min=1.0e-11,
        r_max=1.0e-10,
        nr=10,
    )

    assert os.path.exists(str(tmp_path) + "/bessel.asdf")
    with asdf.open(str(tmp_path) + "/bessel.asdf") as f:
        n = np.copy(f["parameters"]["n"])
        r = np.copy(f["parameters"]["r"])
        zeta = np.copy(f["parameters"]["zeta"])

        assert np.all(n - np.arange(1, 31) == 0)
        assert np.allclose(r / 1.0e-11, np.arange(1, 11))
        assert np.allclose(zeta / 1.0e10, np.arange(1, 51))

        # check j_7(18), etc.
        z = 18.0
        j6 = -0.018529564606574975
        j7 = 0.043527771833608467
        j8 = 0.054802707801248698
        assert np.abs(f["functions"]["bessel_j"][6, 17, 9] - j7) < 1.0e-9
        assert np.abs(f["functions"]["bessel_j_11"][6, 17, 9] - j7 / z * np.sqrt(28)) < 1.0e-9
        assert np.abs(f["functions"]["bessel_j_prime"][6, 17, 9] - (7 * j6 - 8 * j8) / 15) < 1.0e-9
        assert np.abs(f["functions"]["bessel_pi"][6, 17, 9] - (8 * j6 - 7 * j8) / 15) < 1.0e-9

    # check that the dictionary was returned
    assert np.all(tr["parameters"]["n"] == n)
