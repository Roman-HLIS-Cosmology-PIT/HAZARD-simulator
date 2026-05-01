"""
This is my first try at defining the library.
It includes the definition of all of the parameters as well as the Bessel Functions themselves.

Date Created: 4/5/26
Last Updated: 4/22/26

"""

import asdf
import numpy as np
from scipy.special import spherical_jn


def bessel_j(n, zeta, r):
    """
    Computes the spherical Bessel function of the first kind, j_n(zeta * r).

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    zeta : float
        Zeta paramter as defined: in the paper
    r : float
        Radial distance.

    Returns
    -------
    result : float
        Value of the function
    Notes
    -----
    Does not vectorize normally over n, so we loop over unique n values
    """
    result = np.empty_like(r, dtype=float)
    for ni in np.unique(n):
        mask = n == ni
        result[mask] = spherical_jn(ni, zeta[mask] * r[mask])
    return result


def bessel_j_prime(n, zeta, r):
    """
    Computes the derivative of the spherical Bessel function of the first kind with respect to r.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    zeta : float
        Zeta paramter as defined in the paper
    r : float
        Radial distance.

    Returns
    -------
    result : float
        returns the value of the function

    Notes
    -----
    Does not store the result as a variable
    """
    bessel_lower = bessel_j(n - 1, zeta, r)
    bessel_raised = bessel_j(n + 1, zeta, r)
    return (1 / (2 * n + 1)) * (n * bessel_lower - (n + 1) * bessel_raised)


def bessel_j_11(n, zeta, r):
    """
    Computes the second derivative of the spherical Bessel function of the first kind with respect to r.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    zeta : float
        Zeta paramter as defined in the paper
    r : float
        Radial distance.

    Returns
    -------
    result : float
    """
    bessel_regular = bessel_j(n, zeta, r)
    return np.sqrt((n * (n + 1)) / 2) * (bessel_regular / (zeta * r))


def bessel_pi(n, zeta, r):
    """
    Computes the pi function as defined in EqC224 in GCR Paper.

    Parameters
    ----------
    n : int
        Order of the Bessel function.
    zeta : float
        Zeta paramter as defined in the paper
    r : float
        Radial distance.

    Returns
    -------
    result : value of the function
    """
    bessel_regular = bessel_j(n, zeta, r)
    bessel_prime = bessel_j_prime(n, zeta, r)
    return (bessel_regular / (zeta * r)) + bessel_prime


def bessel_main(outfile=None, nmax=480, dzeta=1.54, nzeta=311, r_min=2.206105875e-13, r_max=2.0e-10, nr=500):
    """
    Main function to output the results to an ASDF file given a name

    Parameters
    ----------
    outfile : str, optional
        File name for where to store the table of results. (If None, then no output.)
    nmax : int, optional
        The maximum value of total angular momentum to build.
    dzeta : float, optional
        The spacing of wavenumber samples in keV^-1.
    nzeta : int, optional
        The number of wavenumbers to generate (so spaced as `dzeta`, 2 * `dzeta`, ...
        up to `nzeta` * `dzeta`).
    r_min, r_max : float, optional
        The minimum and maximum radius for the table in meters (inclusive).
    nr : int, optional
        The number of radius samples.

    Returns
    -------
    dict
        A dictionary (that could be converted to an ASDF tree if saved to a file).

    """

    # Define parameters n, zeta, and r
    n = np.arange(1, nmax + 1)
    zeta_nonSI = dzeta * np.arange(1, nzeta + 1)  # noqa: N816
    zeta = zeta_nonSI * 5067730719.0  # Converts from keV/H_bar*c to 1/m

    r = np.linspace(r_min, r_max, nr)

    # Update so each function calls a number of rs with a delta r and so on
    # Next step is to plot one slice for a given zeta value; should peak around l = r times zeta

    # Build 3D meshgrid for n, zeta, and r
    n_mesh, zeta_mesh, r_mesh = np.meshgrid(n, zeta, r, indexing="ij")

    # Evaluate each of the Bessel functions for the defined parameters
    bessel_j_values = bessel_j(n_mesh, zeta_mesh, r_mesh)
    bessel_j_prime_values = bessel_j_prime(n_mesh, zeta_mesh, r_mesh)
    bessel_j_11_values = bessel_j_11(n_mesh, zeta_mesh, r_mesh)
    bessel_pi_values = bessel_pi(n_mesh, zeta_mesh, r_mesh)

    # Build data tree for ASDF file
    tree = {
        "metadata": {
            "description": "Spherical Bessel function values for n, zeta, and r parameters.",
            "date_created": "4/5/26",
            "last_updated": "4/5/26",
        },
        "parameters": {"n": n, "zeta": zeta, "r": r},
        "functions": {
            "bessel_j": bessel_j_values,
            "bessel_j_prime": bessel_j_prime_values,
            "bessel_j_11": bessel_j_11_values,
            "bessel_pi": bessel_pi_values,
        },
    }

    # Save the data to an ASDF file
    if outfile is not None:
        af = asdf.AsdfFile(tree)
        af.write_to(outfile, all_array_compression="zlib")
    # print("Library saved to bessel_library.asdf")
    # print(f" n shape: {bessel_j_values.shape}")
    # print(f" Array size: {bessel_j_values.nbytes / 1e6:.2f} MB per function")

    return tree


"""
# For test, do a quick check of the file
with asdf.open("bessel_library.asdf") as af:
    loaded_tree = af.tree
    print("Loaded library from ASDF file:")
    print(loaded_tree["metadata"])
    print(f" n shape: {loaded_tree['functions']['bessel_j'].shape}")
"""


"""
#Added time block to check how long it takes to compute the Bessel functions
start = time.time()
bessel_j_values = bessel_j(n_mesh, zeta_mesh, r_mesh)
bessel_j_prime_values = bessel_j_prime(n_mesh, zeta_mesh, r_mesh)
bessel_j_11_values = bessel_j_11(n_mesh, zeta_mesh, r_mesh)
bessel_pi_values = bessel_pi(n_mesh, zeta_mesh, r_mesh)
end = time.time()
print(f"Time taken to compute Bessel functions: {end - start:.2f} seconds")
"""
